// SPDX-License-Identifier: Apache-2.0
//
// A named list of local models, so adding one is a command rather than a code change.
//
// WHY THIS EXISTS
//     Until now every model reached the engine as a filesystem path typed on a command
//     line, and whether it would work depended on someone having read its header. That
//     makes the maintainer a dependency for every model anyone wants to run, which is the
//     difference between a research project and a tool.
//
//     `add` inspects before it records. A model that cannot run is still registered --
//     hiding it would only move the surprise later -- but its blockers are stored with it,
//     so `list` says what is missing without re-reading 74 GB of header every time.
//
// FORMAT
//     ~/.rustlm/registry.json, a plain JSON object of name -> entry. Hand-built with
//     serde_json::Value rather than derived: the crate deliberately carries no `serde`
//     derive dependency, and a manifest this small does not justify adding one.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::capability::Report;

pub struct Entry {
    pub name: String,
    pub path: PathBuf,
    pub format: String,
    pub arch: String,
    pub bytes: u64,
    pub blockers: Vec<String>,
}

impl Entry {
    pub fn runnable(&self) -> bool {
        self.blockers.is_empty()
    }
}

pub struct Registry {
    file: PathBuf,
    pub entries: Vec<Entry>,
}

/// `$RUSTLM_HOME`, else `~/.rustlm`, falling back to the old `~/.rustmlhub` if that is
/// where an existing registry actually lives.
///
/// The override exists so tests never touch a real user's registry. The fallback exists
/// because renaming a tool must not silently orphan the models someone already registered
/// -- a rename that loses state is indistinguishable, from the user's side, from the tool
/// forgetting. `$RUSTMLHUB_HOME` stays honoured for the same reason.
pub fn home() -> PathBuf {
    for k in ["RUSTLM_HOME", "RUSTMLHUB_HOME"] {
        if let Some(h) = std::env::var_os(k) {
            return PathBuf::from(h);
        }
    }
    let base = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    let new = base.join(".rustlm");
    // Prefer the new location, but only adopt it once it exists or the old one does not:
    // an installation that predates the rename keeps working with no migration step.
    let old = base.join(".rustmlhub");
    if !new.join("registry.json").exists() && old.join("registry.json").exists() {
        return old;
    }
    new
}

impl Registry {
    pub fn open() -> Registry {
        Registry::open_at(&home())
    }

    /// Open a registry rooted at an explicit directory.
    ///
    /// This, not `open`, is the primitive. `std::env::set_var` is process-global and Rust
    /// runs tests in parallel threads, so a test that pointed `RUSTMLHUB_HOME` at a
    /// scratch directory could have it unset by a sibling test mid-write -- and the save
    /// would land in the real user's registry. That is not hypothetical: it happened, and
    /// it wrote a fake entry into a live manifest.
    pub fn open_at(dir: &Path) -> Registry {
        let file = dir.join("registry.json");
        let mut entries = Vec::new();
        if let Ok(s) = std::fs::read_to_string(&file) {
            if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&s) {
                for (name, v) in map {
                    entries.push(Entry {
                        name,
                        path: PathBuf::from(v["path"].as_str().unwrap_or_default()),
                        format: v["format"].as_str().unwrap_or_default().to_string(),
                        arch: v["arch"].as_str().unwrap_or_default().to_string(),
                        bytes: v["bytes"].as_u64().unwrap_or(0),
                        blockers: v["blockers"]
                            .as_array()
                            .map(|a| {
                                a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()
                            })
                            .unwrap_or_default(),
                    });
                }
            }
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Registry { file, entries }
    }

    /// Inspect `path` and record it under `name`, replacing any existing entry.
    ///
    /// Inspection is not optional: a registry that records paths without checking them
    /// just relocates the moment of discovery from `add` to the first generation attempt,
    /// which is a much worse place to find out.
    pub fn add(&mut self, name: &str, path: &Path) -> Result<Report, String> {
        if name.is_empty() || name.contains('/') {
            return Err(format!("{name:?} is not a usable model name"));
        }
        let abs = path
            .canonicalize()
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let r = Report::inspect(&abs)?;
        self.entries.retain(|e| e.name != name);
        self.entries.push(Entry {
            name: name.to_string(),
            path: r.path.clone(),
            format: r.format.as_str().to_string(),
            arch: r.arch.clone(),
            bytes: r.bytes,
            blockers: r.blockers.clone(),
        });
        self.entries.sort_by(|a, b| a.name.cmp(&b.name));
        self.save()?;
        Ok(r)
    }

    pub fn remove(&mut self, name: &str) -> Result<bool, String> {
        let before = self.entries.len();
        self.entries.retain(|e| e.name != name);
        let removed = self.entries.len() != before;
        if removed {
            self.save()?;
        }
        Ok(removed)
    }

    /// A registered name, or a path that happens to exist. Taking both means a caller
    /// never has to care which they were given.
    pub fn resolve(&self, what: &str) -> Option<PathBuf> {
        if let Some(e) = self.entries.iter().find(|e| e.name == what) {
            return Some(e.path.clone());
        }
        let p = PathBuf::from(what);
        p.exists().then_some(p)
    }

    pub fn save(&self) -> Result<(), String> {
        let mut map = serde_json::Map::new();
        for e in &self.entries {
            map.insert(
                e.name.clone(),
                json!({
                    "path": e.path.to_string_lossy(),
                    "format": e.format,
                    "arch": e.arch,
                    "bytes": e.bytes,
                    "blockers": e.blockers,
                }),
            );
        }
        if let Some(d) = self.file.parent() {
            std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
        }
        std::fs::write(&self.file, format!("{:#}\n", Value::Object(map)))
            .map_err(|e| format!("{}: {e}", self.file.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("rustlm-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_registry_round_trips_through_its_file() {
        let d = scratch("rt");
        let mut r = Registry::open_at(&d);
        assert!(r.entries.is_empty(), "a fresh registry starts empty");
        r.entries.push(Entry {
            name: "demo".into(),
            path: PathBuf::from("/models/demo"),
            format: "gguf".into(),
            arch: "qwen35moe".into(),
            bytes: 21_166_758_016,
            blockers: vec!["no layer implementation yet".into()],
        });
        r.save().unwrap();

        let back = Registry::open_at(&d);
        assert_eq!(back.entries.len(), 1);
        let e = &back.entries[0];
        assert_eq!(e.name, "demo");
        assert_eq!(e.arch, "qwen35moe");
        assert_eq!(e.bytes, 21_166_758_016, "a 21 GB size must survive as u64");
        // Blockers are persisted so `list` can report them without re-reading a 74 GB
        // header on every invocation.
        assert_eq!(e.blockers.len(), 1);
        assert!(!e.runnable());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn resolve_accepts_a_name_or_a_path_and_rejects_neither() {
        let d = scratch("resolve");
        let mut r = Registry::open_at(&d);
        r.entries.push(Entry {
            name: "big".into(),
            path: d.clone(),
            format: "gguf".into(),
            arch: "x".into(),
            bytes: 1,
            blockers: vec![],
        });
        assert_eq!(r.resolve("big"), Some(d.clone()), "by registered name");
        assert_eq!(r.resolve(d.to_str().unwrap()), Some(d.clone()), "by existing path");
        assert_eq!(r.resolve("neither"), None, "an unknown name is not silently a path");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_name_that_would_break_the_manifest_is_refused() {
        let d = scratch("names");
        let mut r = Registry::open_at(&d);
        assert!(r.add("", Path::new(".")).is_err(), "empty name");
        assert!(r.add("a/b", Path::new(".")).is_err(), "a name must not look like a path");
        let _ = std::fs::remove_dir_all(&d);
    }
}
