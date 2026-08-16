// SPDX-License-Identifier: Apache-2.0
//
// Installing a model, without a person in the middle.
//
// WHY `curl` AND NOT AN HTTP CRATE
//     `curl` already solves resumable transfers over flaky links (`-C -`), redirects,
//     TLS, and proxy configuration, and it is on every machine this runs on. The Rust
//     alternative is `reqwest`, which drags in a TLS stack -- the same class of dependency
//     that made claurst's build require cmake and BoringSSL. A 40-line subprocess wrapper
//     keeps this crate's dependency list something a person can still read.
//
//     The shell script this replaces (`hffetch.sh`) proved the mechanics, including one
//     lesson kept below: a token passed as a command-line argument is visible to any
//     process on the machine via `pgrep -af`. It goes in a 0600 config file instead.
//
// THE CHECK THAT MAKES THIS WORTH HAVING
//     A GGUF file puts its architecture and its ENTIRE tensor list in the header, before
//     any weights. So one range request for the first megabyte answers "is this a
//     Mixture-of-Experts model this engine can run?" -- for a 21 GB download, that is a
//     0.005% sample. Refusing early costs a second; finding out afterwards costs the whole
//     transfer, and on a metered or slow link that is the difference that matters.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Head-sample sizes to try, in order.
///
/// The first attempt is usually enough, and the retries exist because the TOKENIZER lives
/// in the metadata: a 250k-token vocabulary plus its merge list is several megabytes of
/// strings sitting between the header and the tensor table. Measured on Qwen2.5-0.5B, 4 MB
/// was not enough to reach the tensors -- so this grows rather than guessing a constant.
///
/// Even the largest is 0.3% of a 21 GB checkpoint, which is the whole point.
const HEAD_TRIES: [u64; 3] = [8 << 20, 32 << 20, 96 << 20];

pub struct Plan {
    pub repo: String,
    /// Every file to fetch, in shard order. One entry for an ordinary checkpoint; `N` for a
    /// split one. `St::open` reads a directory of shards directly, so the parts are stored
    /// side by side rather than concatenated.
    pub files: Vec<(String, u64)>,
    pub dest: PathBuf,
}

impl Plan {
    pub fn url(&self, file: &str) -> String {
        format!("https://huggingface.co/{}/resolve/main/{file}", self.repo)
    }
    pub fn bytes(&self) -> u64 {
        self.files.iter().map(|(_, s)| *s).sum()
    }
}

/// Split a shard filename into its set key and its part numbers.
///
/// llama.cpp names the parts `<base>-00001-of-00062.gguf`. The key has to include the
/// DIRECTORY, because one repo can publish several quantisations as separate shard sets --
/// `Q2_K/…-00001-of-00062.gguf` and `Q3_K_M/…-00001-of-00094.gguf` live in the same repo,
/// and keying on the basename alone would interleave two different models into one set.
fn shard_of(name: &str) -> Option<(String, usize, usize)> {
    let stem = name.strip_suffix(".gguf").or_else(|| name.strip_suffix(".GGUF"))?;
    let (head, total) = stem.rsplit_once("-of-")?;
    let (base, idx) = head.rsplit_once('-')?;
    // Both halves must be all digits, or `-of-` in a model's own name would parse as a
    // shard marker and turn an ordinary file into a one-part set with a truncated key.
    if idx.is_empty() || total.is_empty() {
        return None;
    }
    let i: usize = idx.parse().ok()?;
    let n: usize = total.parse().ok()?;
    (n > 0 && i >= 1 && i <= n).then(|| (base.to_string(), i, n))
}

fn curl(args: &[&str]) -> Result<Vec<u8>, String> {
    let out = Command::new("curl")
        .args(["-sSL", "--fail", "--retry", "3", "--retry-delay", "2"])
        .args(args)
        .output()
        .map_err(|e| format!("curl: {e} -- pull needs curl on PATH"))?;
    if !out.status.success() {
        return Err(format!(
            "curl failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(out.stdout)
}

/// Every file in a Hugging Face repo, as (path, size).
pub fn list_repo(repo: &str) -> Result<Vec<(String, u64)>, String> {
    let url = format!("https://huggingface.co/api/models/{repo}?blobs=true");
    let body = curl(&[&url])?;
    let v: serde_json::Value =
        serde_json::from_slice(&body).map_err(|e| format!("{repo}: not JSON ({e})"))?;
    if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
        return Err(format!("{repo}: {err}"));
    }
    let files = v["siblings"].as_array().ok_or_else(|| format!("{repo}: no file list"))?;
    Ok(files
        .iter()
        .filter_map(|f| {
            let name = f["rfilename"].as_str()?.to_string();
            let size = f.get("size").and_then(serde_json::Value::as_u64).unwrap_or(0);
            Some((name, size))
        })
        .collect())
}

/// Choose the GGUF, or the set of GGUF shards, to fetch.
///
/// Quantisation is matched case-insensitively on the filename, which is the only place it
/// is recorded -- and a filename is a CLAIM, not a fact. It is checked against the header
/// after download, because a `Q4_K_M` build has already been observed to contain 132 Q5_0
/// tensors and half its `ffn_down_exps` at Q4_K rather than Q6_K.
///
/// SPLIT CHECKPOINTS
/// ```text
///     Past a few tens of gigabytes every publisher splits, because the Hub's own tooling
///     does -- Qwen3.8-2.4T ships as 62 or 94 parts. This used to refuse them outright,
///     which put the entire large end of the ecosystem out of reach for the one command
///     that is supposed to install things.
///
///     Nothing about the READER needed changing: `St::open` already sorts a directory of
///     `.gguf` files, scans each with its own shard index and merges the tensor tables,
///     taking metadata from the first. That is exactly the split layout, since part 1
///     carries the architecture, hparams and tokenizer. Only the download refused.
///
///     A set is returned only when it is COMPLETE. A missing part is the dangerous case:
///     the tensors in it would simply be absent, and while `Trunk::load` does fail on the
///     first name it cannot find, it fails after the whole transfer rather than before it.
/// ```
pub fn choose(files: &[(String, u64)], quant: Option<&str>) -> Result<Vec<(String, u64)>, String> {
    let ggufs: Vec<&(String, u64)> =
        files.iter().filter(|(n, _)| n.to_lowercase().ends_with(".gguf")).collect();
    if ggufs.is_empty() {
        return Err("this repo has no .gguf files -- pull handles gguf only".into());
    }
    // Group the parts of every split set, and keep whole files as sets of one, so the
    // quantisation match below sees one candidate per MODEL rather than one per file.
    let mut sets: Vec<Set> = Vec::new();
    for (n, s) in ggufs.iter().copied() {
        let (key, total) = match shard_of(n) {
            Some((base, _, t)) => (base, t),
            None => (n.clone(), 1),
        };
        match sets.iter_mut().find(|c| c.key == key) {
            Some(c) => c.parts.push((n.clone(), *s)),
            None => sets.push(Set { key, parts: vec![(n.clone(), *s)], total }),
        }
    }
    for c in sets.iter_mut() {
        c.parts.sort();
    }

    let pick = match quant {
        Some(q) => {
            let ql = q.to_lowercase();
            sets.iter()
                .find(|c| c.key.to_lowercase().contains(&ql))
                .ok_or_else(|| format!("no gguf matching {q:?}. Available:\n  {}", names(&sets)))?
        }
        None if sets.len() == 1 => &sets[0],
        None => {
            return Err(format!(
                "several quantisations available; pick one with --quant:\n  {}",
                names(&sets)
            ))
        }
    };

    if pick.total > 1 && pick.parts.len() != pick.total {
        return Err(format!(
            "{} is a {}-part split gguf but the repo lists only {} of them. \
             Downloading an incomplete set would produce a directory that loads and is \
             missing tensors, so this refuses rather than starting the transfer.",
            pick.key,
            pick.total,
            pick.parts.len()
        ));
    }
    Ok(pick.parts.clone())
}

/// One candidate model: a whole file, or every part of one split checkpoint.
struct Set {
    /// What `--quant` matches against: the filename, or the shared prefix of the parts.
    key: String,
    parts: Vec<(String, u64)>,
    /// Parts the NAMES claim exist, which is what `parts.len()` is checked against.
    total: usize,
}

fn names(sets: &[Set]) -> String {
    sets.iter()
        .map(|c| {
            if c.total > 1 {
                let gb = c.parts.iter().map(|(_, s)| *s).sum::<u64>() as f64 / 1e9;
                format!("{}  ({} parts, {gb:.1} GB)", c.key, c.total)
            } else {
                c.key.clone()
            }
        })
        .collect::<Vec<_>>()
        .join("\n  ")
}

/// What the header says, before a single weight has been transferred.
pub struct Probe {
    pub arch: String,
    pub n_tensors: usize,
    /// Routed-expert tensors seen across the parts inspected.
    pub expert_tensors: usize,
    pub n_experts: usize,
    /// How many parts were read to get here. More than one means part one's tensor list
    /// was not enough on its own -- see `probe_set`.
    pub parts_probed: usize,
}

/// Read a split checkpoint's header, following on to later parts if part one is not
/// conclusive.
///
/// WHY ONE PART IS NOT ENOUGH
/// ```text
///     A split GGUF gives every part its own header listing only ITS OWN tensors. Part one
///     carries the architecture, hparams and tokenizer, but it need not carry any weights
///     at all: mmnga-o's Qwen3.8 build has a 10.9 MB part one and nine 20 GB parts after
///     it. Judging "is this MoE?" on part one's tensor list alone therefore reported
///
///         "qwen35moe" is not a Mixture-of-Experts model: 0 expert tensors,
///          expert_count 358
///
///     which is not merely a false refusal but a self-contradicting one -- 358 experts is
///     right there in the same sentence. Absence in part one is absence of evidence.
///
///     So when part one shows no expert tensors and more parts exist, this reads the next
///     few. That keeps the two-independent-signals rule intact rather than weakening it to
///     "trust the metadata": the metadata still has to be confirmed by a real tensor name,
///     it is just allowed to be confirmed by a part other than the first.
/// ```
pub fn probe_set(urls: &[String], token: Option<&Path>) -> Result<Probe, String> {
    let mut p = probe_remote(&urls[0], token)?;
    p.parts_probed = 1;
    // Three is enough for any layout that puts metadata alone in part one; past that the
    // checkpoint is odd enough that the full-file `inspect` should be the one to judge it.
    for u in urls.iter().skip(1).take(3) {
        if p.expert_tensors > 0 {
            break;
        }
        println!("  part 1 lists no weights; reading part {} as well ...", p.parts_probed + 1);
        match probe_remote(u, token) {
            // Later parts carry tensors but not the architecture, so only the counts are
            // merged -- `arch` and `n_experts` stay as part one declared them.
            Ok(q) => {
                p.expert_tensors += q.expert_tensors;
                p.n_tensors += q.n_tensors;
                p.parts_probed += 1;
            }
            Err(e) => return Err(format!("part {} of the set: {e}", p.parts_probed + 1)),
        }
    }
    Ok(p)
}

/// Range-GET the head of a remote GGUF and read its header.
///
/// This is the whole point of the command. The header carries `general.architecture` and
/// one entry per tensor, so a megabyte decides whether a 21 GB download is worth starting.
pub fn probe_remote(url: &str, token: Option<&Path>) -> Result<Probe, String> {
    let mut last = String::new();
    for (i, n) in HEAD_TRIES.iter().enumerate() {
        match probe_head(url, token, *n) {
            Ok(p) => return Ok(p),
            Err(e) => {
                last = e;
                if i + 1 < HEAD_TRIES.len() {
                    println!("  header needs more than {:.0} MB (the tokenizer lives in it); \
                              retrying with {:.0} MB", *n as f64 / 1e6,
                             HEAD_TRIES[i + 1] as f64 / 1e6);
                }
            }
        }
    }
    Err(last)
}

fn probe_head(url: &str, token: Option<&Path>, head_bytes: u64) -> Result<Probe, String> {
    let tmp = std::env::temp_dir().join(format!("rustlm-probe-{}.gguf", std::process::id()));
    let range = format!("0-{}", head_bytes - 1);
    let mut args: Vec<String> =
        vec!["-r".into(), range, "-o".into(), tmp.to_string_lossy().into_owned(), url.into()];
    if let Some(k) = token {
        args.push("-K".into());
        args.push(k.to_string_lossy().into_owned());
    }
    let r = curl(&args.iter().map(String::as_str).collect::<Vec<_>>());
    let out = (|| {
        r?;
        let f = std::fs::File::open(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
        // The header precedes all tensor data, so a truncated file parses cleanly. A
        // request for a range the server ignored would give a short read here, not a
        // wrong answer.
        let (meta, tensors) = crate::gguf::scan_header(0, &f)
            .map_err(|e| {
                format!("the first {:.0} MB are not a readable gguf header: {e}",
                        head_bytes as f64 / 1e6)
            })?;
        let arch = meta
            .get("general.architecture")
            .and_then(crate::gguf::Value::as_str)
            .unwrap_or("")
            .to_string();
        let n_experts = meta
            .get(&format!("{arch}.expert_count"))
            .and_then(crate::gguf::Value::as_u)
            .unwrap_or(0) as usize;
        Ok(Probe {
            arch,
            n_tensors: tensors.len(),
            expert_tensors: tensors.iter().filter(|t| is_expert(&t.name)).count(),
            n_experts,
            parts_probed: 1,
        })
    })();
    let _ = std::fs::remove_file(&tmp);
    out
}

/// A routed-expert tensor, in either naming convention this engine reads.
fn is_expert(name: &str) -> bool {
    name.contains("_exps.") || name.contains(".experts.")
}

impl Probe {
    /// Is this a Mixture-of-Experts model this engine knows the shape of?
    ///
    /// Two independent signals must agree, because either alone can lie: the metadata may
    /// declare `expert_count` on a model whose experts were merged away, and a stray
    /// tensor name proves nothing about the architecture. Requiring both means a dense
    /// model cannot slip through on a name.
    pub fn verdict(&self) -> Result<&'static crate::moearch::Arch, String> {
        // The family table decides FIRST, because whether the expert signals below are
        // even meaningful depends on the answer. `qwen35` is dense by construction, so
        // demanding routed experts of it would refuse a model this build runs -- and the
        // refusal would have read "is not a Mixture-of-Experts model", which is true and
        // beside the point.
        //
        // The original rule ("a dense model would gain nothing from streaming") is no
        // longer universal: Qwen3.8-27B's feed-forward is 62% of the checkpoint and streams
        // through the same cache as any expert. What has not changed is that a family this
        // build has no block for gains nothing, and that is what the `None` arm says.
        if let Some(a) = crate::moearch::find(&self.arch) {
            if a.dense {
                return if a.verified {
                    Ok(a)
                } else {
                    Err(format!(
                        "{:?} is a dense family whose layout is known, but its forward pass \
                         has not been diffed against a reference yet. Refusing to spend the \
                         download.",
                        self.arch
                    ))
                };
            }
        }
        // Which signal is missing changes what the answer MEANS, so it changes the
        // message. Saying "not a Mixture-of-Experts model" while printing a non-zero
        // expert count contradicts itself in one sentence, and a user who believes it
        // gives up on a model that was fine.
        if self.n_experts == 0 && self.expert_tensors == 0 {
            return Err(format!(
                "{:?} is not a Mixture-of-Experts model: {} expert tensors, expert_count {}. \
                 This engine streams routed experts off disk to run models larger than RAM; \
                 a dense model would gain nothing from it and is not handled.",
                self.arch, self.expert_tensors, self.n_experts
            ));
        }
        if self.expert_tensors == 0 {
            return Err(format!(
                "{:?} declares expert_count {} but no routed-expert tensor appears in the \
                 {} part(s) read. The two signals must agree -- metadata can outlive the \
                 experts it describes -- so this refuses rather than guessing. \
                 `rustlm inspect` on a local copy reads every tensor and will settle it.",
                self.arch, self.n_experts, self.parts_probed
            ));
        }
        if self.n_experts == 0 {
            return Err(format!(
                "{:?} has {} routed-expert tensors but declares expert_count 0, so the \
                 router width is unknown and nothing can be loaded from it.",
                self.arch, self.expert_tensors
            ));
        }
        match crate::moearch::find(&self.arch) {
            Some(a) if a.verified => Ok(a),
            Some(_) => Err(format!(
                "{:?} is a Mixture-of-Experts model with {} experts and its layout is known, \
                 but its forward pass has not been diffed against a reference yet, so it \
                 would load and produce unverified output. Refusing to spend the download.",
                self.arch, self.n_experts
            )),
            None => Err(format!(
                "{:?} is not a MoE family this build knows ({} expert tensors found). \
                 `rustlm inspect` on a local copy will say what is missing.",
                self.arch, self.expert_tensors
            )),
        }
    }
}

/// Where a token lives, if the user has one. Never passed on the command line: an argument
/// is visible to every process on the machine through `pgrep -af`, and this one is a
/// credential.
pub fn token_file() -> Option<PathBuf> {
    let p = crate::registry::home().join("hf-token");
    p.exists().then_some(p)
}

/// Resolve a repo and quantisation into a concrete download, checking the header first.
///
/// The header check reads PART ONE, which is where a split checkpoint keeps its
/// architecture, hparams and tokenizer -- so the refusal is just as early for a 62-part
/// 886 GB model as for a single 21 GB file, which is when it matters most.
pub fn plan(repo: &str, quant: Option<&str>, dir: &Path) -> Result<(Plan, Probe), String> {
    let files = list_repo(repo)?;
    let parts = choose(&files, quant)?;
    let name = repo.rsplit('/').next().unwrap_or(repo);
    let plan = Plan { repo: repo.to_string(), files: parts, dest: dir.join(name) };
    if plan.files.len() > 1 {
        println!(
            "  {}  ({} parts, {:.1} GB total)",
            plan.files[0].0,
            plan.files.len(),
            plan.bytes() as f64 / 1e9
        );
    } else {
        println!("  {}  ({:.1} GB)", plan.files[0].0, plan.bytes() as f64 / 1e9);
    }
    println!("  reading the header before downloading the weights ...");
    let urls: Vec<String> = plan.files.iter().map(|(f, _)| plan.url(f)).collect();
    let probe = probe_set(&urls, token_file().as_deref())?;
    Ok((plan, probe))
}

/// Refuse a download that cannot finish. On a 62-part checkpoint this is not a formality:
/// the difference between noticing now and noticing at part 61 is days of transfer.
pub fn check_space(p: &Plan) -> Result<(), String> {
    let free = crate::io::free_bytes(&p.dest).unwrap_or(u64::MAX);
    // A tenth over, so the transfer does not fill the filesystem to the last byte.
    let need = p.bytes() + p.bytes() / 10;
    if free < need {
        return Err(format!(
            "{:.0} GB needed ({:.0} GB of parts plus headroom) and {:.0} GB free on {}",
            need as f64 / 1e9,
            p.bytes() as f64 / 1e9,
            free as f64 / 1e9,
            p.dest.display()
        ));
    }
    Ok(())
}

/// Fetch every part, resumably.
///
/// `curl -C -` continues a partial file, which is what makes a dropped connection on a
/// 21 GB transfer an inconvenience rather than a restart -- and on an 886 GB one, the
/// difference between possible and not. Parts already at their full listed size are
/// skipped outright, so re-running after an interruption resumes at the part it stopped on
/// instead of re-verifying the ones behind it.
///
/// Shards are flattened into the destination directory by BASENAME, because the repo may
/// keep them under a quantisation folder (`Q2_K/…gguf`) and `St::open` reads one flat
/// directory. Two sets are never mixed: `choose` returns exactly one set.
pub fn download(p: &Plan) -> Result<PathBuf, String> {
    std::fs::create_dir_all(&p.dest).map_err(|e| format!("{}: {e}", p.dest.display()))?;
    let n = p.files.len();
    let mut last = p.dest.clone();
    for (i, (file, size)) in p.files.iter().enumerate() {
        let base = file.rsplit('/').next().unwrap_or(file);
        let out = p.dest.join(base);
        last = out.clone();
        // `size` is the Hub's own blob size. Equal means complete; anything short is a
        // partial that `-C -` will continue.
        if *size > 0 && std::fs::metadata(&out).is_ok_and(|m| m.len() == *size) {
            if n > 1 {
                println!("  [{}/{n}] {base} -- already complete", i + 1);
            }
            continue;
        }
        if n > 1 {
            println!("  [{}/{n}] {base}  ({:.1} GB)", i + 1, *size as f64 / 1e9);
        }
        let mut args: Vec<String> = vec![
            "-C".into(), "-".into(),
            "--progress-bar".into(),
            "-o".into(), out.to_string_lossy().into_owned(),
            p.url(file),
        ];
        if let Some(k) = token_file() {
            args.push("-K".into());
            args.push(k.to_string_lossy().into_owned());
        }
        // Inherit stdio so curl's progress bar reaches the terminal.
        let st = Command::new("curl")
            .args(["-SL", "--fail", "--retry", "5", "--retry-delay", "3"])
            .args(&args)
            .status()
            .map_err(|e| format!("curl: {e}"))?;
        if !st.success() {
            return Err(format!(
                "part {} of {n} failed ({st}); rerun to resume from where it stopped",
                i + 1
            ));
        }
    }
    if n > 1 {
        verify_parts(p)?;
    }
    Ok(last)
}

/// Every part present, at the size the Hub listed.
///
/// Worth doing separately from the transfer loop because the loop skips what looks
/// finished, and "looks finished" is exactly the judgement that should be re-checked
/// before a directory is handed to the loader. A short part does not fail loudly at load
/// time -- its tensors are simply absent from the merged table.
fn verify_parts(p: &Plan) -> Result<(), String> {
    let mut bad = Vec::new();
    for (file, size) in &p.files {
        let base = file.rsplit('/').next().unwrap_or(file);
        let out = p.dest.join(base);
        match std::fs::metadata(&out) {
            Ok(m) if *size == 0 || m.len() == *size => {}
            Ok(m) => bad.push(format!("{base}: {} bytes, expected {size}", m.len())),
            Err(e) => bad.push(format!("{base}: {e}")),
        }
    }
    if !bad.is_empty() {
        return Err(format!(
            "the shard set is incomplete after downloading, so it must not be loaded:\n  {}",
            bad.join("\n  ")
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files() -> Vec<(String, u64)> {
        vec![
            ("README.md".into(), 100),
            ("Model-Q4_K_M.gguf".into(), 21_000_000_000),
            ("Model-Q6_K.gguf".into(), 30_000_000_000),
        ]
    }

    fn only(files: &[(String, u64)], q: Option<&str>) -> String {
        let v = choose(files, q).unwrap();
        assert_eq!(v.len(), 1, "expected a single file, got {}", v.len());
        v[0].0.clone()
    }

    #[test]
    fn a_quantisation_is_matched_case_insensitively() {
        assert_eq!(only(&files(), Some("q4_k_m")), "Model-Q4_K_M.gguf");
        assert_eq!(only(&files(), Some("Q6_K")), "Model-Q6_K.gguf");
    }

    /// With several to choose from, guessing would download tens of gigabytes of the wrong
    /// thing. The error lists what is available so the next attempt succeeds.
    #[test]
    fn an_ambiguous_repo_asks_rather_than_guessing() {
        let e = choose(&files(), None).unwrap_err();
        assert!(e.contains("--quant") && e.contains("Q4_K_M") && e.contains("Q6_K"), "{e}");
    }

    #[test]
    fn a_single_gguf_needs_no_quant_flag() {
        let one = vec![("only.gguf".into(), 1)];
        assert_eq!(only(&one, None), "only.gguf");
    }

    #[test]
    fn a_repo_without_gguf_says_so() {
        let e = choose(&[("model.safetensors".into(), 1)], None).unwrap_err();
        assert!(e.contains("no .gguf"), "{e}");
    }

    fn split(base: &str, n: usize) -> Vec<(String, u64)> {
        (1..=n).map(|i| (format!("{base}-{i:05}-of-{n:05}.gguf"), 1_000_000_000)).collect()
    }

    /// The whole point of the change: a split checkpoint resolves to its parts, in order.
    #[test]
    fn a_split_checkpoint_resolves_to_every_part_in_order() {
        let v = choose(&split("M-Q4_K_M", 3), None).unwrap();
        assert_eq!(v.len(), 3);
        assert_eq!(v[0].0, "M-Q4_K_M-00001-of-00003.gguf", "part one must come first");
        assert_eq!(v[2].0, "M-Q4_K_M-00003-of-00003.gguf");
        // Order matters beyond tidiness: `plan` probes v[0], and only part one carries the
        // architecture, hparams and tokenizer.
        assert!(v.windows(2).all(|w| w[0].0 < w[1].0), "sorted");
    }

    /// Listing order is not guaranteed by the Hub, and part 10 sorts before part 2 as text
    /// only because the names are zero-padded. Assert the padding is what carries it.
    #[test]
    fn parts_are_ordered_even_when_the_listing_is_shuffled() {
        let mut f = split("M", 12);
        f.reverse();
        let v = choose(&f, None).unwrap();
        assert_eq!(v[0].0, "M-00001-of-00012.gguf");
        assert_eq!(v[1].0, "M-00002-of-00012.gguf");
        assert_eq!(v[11].0, "M-00012-of-00012.gguf");
    }

    /// An incomplete set must be refused BEFORE the transfer. The parts that arrived would
    /// merge into a tensor table that is simply missing rows.
    #[test]
    fn an_incomplete_shard_set_is_refused() {
        let mut f = split("M", 5);
        f.remove(2);
        let e = choose(&f, None).unwrap_err();
        assert!(e.contains("only 4 of them"), "{e}");
        assert!(e.contains("missing tensors"), "{e}");
    }

    /// DevQuasar's Qwen3.8 repo, in miniature: two shard SETS in one repo. Keying on the
    /// basename alone would interleave 62 Q2_K parts with 94 Q3_K_M ones.
    #[test]
    fn two_shard_sets_in_one_repo_stay_separate() {
        let mut f = split("Q2_K/Model", 4);
        f.extend(split("Q3_K_M/Model", 6));
        let v = choose(&f, Some("q2_k")).unwrap();
        assert_eq!(v.len(), 4, "picked the Q2_K set");
        assert!(v.iter().all(|(n, _)| n.starts_with("Q2_K/")), "no Q3_K_M parts leaked in");
        assert_eq!(choose(&f, Some("Q3_K_M")).unwrap().len(), 6);
        // And with no --quant it must ask rather than guess, naming the sets and sizes.
        let e = choose(&f, None).unwrap_err();
        assert!(e.contains("--quant") && e.contains("4 parts") && e.contains("6 parts"), "{e}");
    }

    /// A set counts as ONE candidate, so a repo holding a single split model needs no flag.
    #[test]
    fn a_lone_shard_set_needs_no_quant_flag() {
        assert_eq!(choose(&split("M", 7), None).unwrap().len(), 7);
    }

    /// `-of-` inside a model's own name must not be read as a shard marker: that would turn
    /// one ordinary file into a truncated set key and, with a second file present, produce
    /// a bogus "incomplete set" refusal for a repo that is perfectly fine.
    #[test]
    fn a_filename_that_merely_contains_of_is_not_a_shard() {
        assert_eq!(shard_of("Mixture-of-Experts.gguf"), None);
        assert_eq!(shard_of("M-0001a-of-0003.gguf"), None);
        assert_eq!(shard_of("M-00000-of-00003.gguf"), None, "parts are 1-based");
        assert_eq!(shard_of("M-00004-of-00003.gguf"), None, "part past the total");
        assert_eq!(shard_of("M-00001-of-00003.gguf"), Some(("M".into(), 1, 3)));
        let f = vec![("Mixture-of-Experts.gguf".into(), 1u64)];
        assert_eq!(only(&f, None), "Mixture-of-Experts.gguf");
    }

    fn probe(arch: &str, exps: usize, n: usize) -> Probe {
        Probe {
            arch: arch.into(),
            n_tensors: 100,
            expert_tensors: exps,
            n_experts: n,
            parts_probed: 1,
        }
    }

    /// The whole reason the header is read first. A dense model must be refused BEFORE the
    /// weights are transferred.
    #[test]
    fn a_dense_model_is_refused_on_the_header_alone() {
        let e = probe("llama", 0, 0).verdict().unwrap_err();
        assert!(e.contains("not a Mixture-of-Experts"), "{e}");
    }

    /// Both signals are required. Metadata can declare experts on a model whose experts
    /// were merged away, and a stray tensor name proves nothing on its own.
    #[test]
    fn one_signal_alone_is_not_enough_to_call_something_moe() {
        assert!(probe("qwen3moe", 0, 128).verdict().is_err(), "count without tensors");
        assert!(probe("qwen3moe", 40, 0).verdict().is_err(), "tensors without a count");
        assert!(probe("qwen3moe", 40, 128).verdict().is_ok(), "both together");
    }

    /// A refusal must not contradict itself. Reporting "not a Mixture-of-Experts model"
    /// while printing `expert_count 358` in the same sentence is what a metadata-only part
    /// one used to produce, and a user who believes it abandons a model that was fine.
    #[test]
    fn a_refusal_names_the_signal_that_is_actually_missing() {
        let e = probe("qwen35moe", 0, 358).verdict().unwrap_err();
        assert!(!e.contains("is not a Mixture-of-Experts model"), "self-contradicting: {e}");
        assert!(e.contains("declares expert_count 358"), "{e}");
        assert!(e.contains("part(s) read"), "says how much was inspected: {e}");

        // The genuinely dense case keeps the blunt wording, because there it is true.
        let d = probe("llama", 0, 0).verdict().unwrap_err();
        assert!(d.contains("is not a Mixture-of-Experts model"), "{d}");

        // And experts without a router width is its own distinct failure.
        let r = probe("qwen3moe", 40, 0).verdict().unwrap_err();
        assert!(r.contains("router width is unknown"), "{r}");
    }

    /// A real MoE whose forward pass is unverified must not be downloaded on the strength
    /// of its layout being known -- it would load and produce output nobody has checked.
    #[test]
    fn an_unverified_moe_family_is_refused_with_its_reason() {
        let e = probe("glm4moe", 40, 128).verdict().unwrap_err();
        assert!(e.contains("not been diffed"), "{e}");
        assert!(e.contains("Refusing to spend the download"), "{e}");
    }

    #[test]
    fn an_unknown_architecture_is_refused() {
        assert!(probe("mamba", 40, 8).verdict().unwrap_err().contains("not a MoE family"));
    }

    #[test]
    fn a_verified_family_is_accepted() {
        assert_eq!(probe("qwen35moe", 120, 256).verdict().unwrap().name, "qwen35moe");
    }
}

#[cfg(test)]
mod dense_verdict_tests {
    use super::*;

    fn probe(arch: &str, exps: usize, n: usize) -> Probe {
        Probe {
            arch: arch.into(),
            n_tensors: 100,
            expert_tensors: exps,
            n_experts: n,
            parts_probed: 1,
        }
    }

    /// A verified DENSE family must be accepted on its architecture alone. It reports zero
    /// experts because it has none, and the old rule read that as "refuse".
    #[test]
    fn a_verified_dense_family_is_accepted_with_no_experts() {
        let a = probe("qwen35", 0, 0).verdict().expect("qwen35 is runnable");
        assert_eq!(a.name, "qwen35");
        assert!(a.dense);
    }

    /// The refusal for a genuinely unsupported dense model is unchanged -- that one really
    /// does gain nothing, because there is no block to run it with.
    #[test]
    fn an_unknown_dense_model_is_still_refused() {
        for arch in ["llama", "mamba"] {
            let e = probe(arch, 0, 0).verdict().unwrap_err();
            assert!(e.contains("not a Mixture-of-Experts"), "{arch}: {e}");
        }
    }

    /// The MoE path must not have been loosened by the dense branch: both signals are still
    /// required for a routed family.
    #[test]
    fn the_moe_requirements_are_unchanged() {
        assert!(probe("qwen35moe", 0, 256).verdict().is_err(), "count without tensors");
        assert!(probe("qwen35moe", 40, 0).verdict().is_err(), "tensors without a count");
        assert!(probe("qwen35moe", 40, 256).verdict().is_ok(), "both together");
        assert!(probe("glm4moe", 40, 128).verdict().unwrap_err().contains("not been diffed"));
    }
}
