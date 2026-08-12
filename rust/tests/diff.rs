// SPDX-License-Identifier: Apache-2.0
//
// The differential gate for the readers, made self-contained.
//
// `make rust-diff` ran the C and Rust safetensors readers and the C and Rust config
// readers over the same fixtures and required byte-identical output. Those C binaries are
// going away with the rest of the non-Rust tree, so their answers were frozen into
// `tests/golden/diff/` on 2026-08-12 -- the fixtures themselves, the C reader's index and
// widened values, and the C config reader's stdout.
//
// This is not a weaker check than the original. The C reader was never the *authority*; it
// was an INDEPENDENT implementation, and independence is what a differential test buys. A
// recording of what it produced preserves exactly that, and unlike the binary it cannot
// drift.

use std::path::{Path, PathBuf};

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/diff")
}

fn bin(name: &str) -> PathBuf {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/release").join(name);
    assert!(p.exists(), "{} is not built; run `cargo build --release`", p.display());
    p
}

/// The six tensors the original gate covered: two dtypes wide, a 1-D and a 2-D case, a
/// scalar, and one that lives in the second shard.
const TENSORS: [&str; 6] = [
    "plain.f32.2d",
    "plain.bf16.1d",
    "tricky.f16.1d",
    "packed.u8.2d",
    "scalar.f32",
    "second.shard.f32",
];

#[test]
fn the_safetensors_reader_matches_the_frozen_c_reader() {
    let d = dir();
    let work = std::env::temp_dir().join(format!("k3-diff-{}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();

    let out = std::process::Command::new(bin("test_st"))
        .current_dir(&work)
        .arg(d.join("st"))
        .arg("r_index.json")
        .args(TENSORS)
        .output()
        .expect("running test_st");
    assert!(out.status.success(), "test_st failed: {}", String::from_utf8_lossy(&out.stderr));

    // The index: every tensor's dtype, shape, shard and byte range.
    let got = std::fs::read_to_string(work.join("r_index.json")).expect("no index written");
    let want = std::fs::read_to_string(d.join("c_index.json")).expect("frozen index missing");
    assert_eq!(got, want, "the index diverged from the frozen C reader");

    // The widened values. This is the half that catches a wrong dtype conversion, which an
    // index comparison alone would pass.
    let got = std::fs::read_to_string(work.join("st_values.json")).expect("no values written");
    let want = std::fs::read_to_string(d.join("c_values.json")).expect("frozen values missing");
    assert_eq!(got, want, "widened values diverged from the frozen C reader");

    let _ = std::fs::remove_dir_all(&work);
}

#[test]
fn the_config_reader_matches_the_frozen_c_reader() {
    let d = dir();
    let out = std::process::Command::new(bin("test_cfg"))
        .arg("fixture")
        .arg(d.join("ref_k3.json"))
        .output()
        .expect("running test_cfg");
    // The C gate captured stdout and stderr together and compared the pair.
    let mut got = String::from_utf8_lossy(&out.stdout).into_owned();
    got.push_str(&String::from_utf8_lossy(&out.stderr));
    let want = std::fs::read_to_string(d.join("cfg_expected.txt")).expect("frozen output missing");

    // The first line echoes the config's PATH, and the frozen capture used the relative
    // one the Makefile passed. Reducing both to the basename compares what the reader
    // parsed rather than where it was invoked from -- every other line is untouched, so
    // this narrows the comparison by exactly one filename and nothing else.
    let strip = |s: &str| {
        s.lines()
            .map(|l| match l.strip_prefix("config: ") {
                Some(rest) => match rest.split_once(' ') {
                    Some((path, tail)) => format!(
                        "config: {} {tail}",
                        Path::new(path).file_name().unwrap_or_default().to_string_lossy()
                    ),
                    None => l.to_string(),
                },
                None => l.to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert_eq!(strip(&got), strip(&want), "config reader output diverged from the frozen C reader");
}

/// Same guard as the golden gate: deleting the fixtures must fail loudly rather than
/// turning these into vacuous passes.
#[test]
fn the_frozen_reader_fixtures_are_present() {
    let d = dir();
    for f in ["c_index.json", "c_values.json", "cfg_expected.txt", "ref_k3.json"] {
        let p = d.join(f);
        assert!(
            std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0) > 0,
            "{} is missing or empty -- the independent reader reference is gone",
            p.display()
        );
    }
    assert!(d.join("st").is_dir(), "the safetensors fixture shards are missing");
}
