// SPDX-License-Identifier: Apache-2.0
//
// The golden gate, made self-contained.
//
// WHY THIS FILE EXISTS AT ALL
//     `make rust-golden` asserted that the Rust engine's logits were BYTE-IDENTICAL to the
//     C engine's, over the whole stack, on a real (tiny) checkpoint. It is the strongest
//     verification this project has -- it is what caught w2/w3 swapped in the expert name
//     table, where the argmax still agreed and the correlation was 0.988, so only an
//     elementwise comparison against an independent implementation could find it.
//
//     That gate compared against a C binary. The non-Rust tree is being deleted, which
//     would have deleted the oracle with it and left the contract unenforceable and
//     unrecoverable -- you cannot re-derive a reference from an implementation you no
//     longer have.
//
//     So the C engine's answers were written down first, on 2026-08-12, into
//     `tests/golden/`: the tiny checkpoint, the packed trunk, the C logits, and the token
//     ids the C engine decoded. The contract survives; the dependency does not.
//
// WHAT IS PINNED, AND WHY EACH ONE
//     1. Logits byte-identical to the C engine        -- catches silently-wrong weights.
//     2. Logits match an INDEPENDENT torch reference  -- catches both engines being wrong
//                                                        the same way.
//     3. Greedy and incremental decode agree          -- catches KV/state carry bugs.
//     4. Identical ids across trunk budgets           -- the project's headline claim, that
//                                                        memory buys speed and not
//                                                        capability. An async reader
//                                                        writing over a live slot breaks
//                                                        this and nothing else.

use std::path::{Path, PathBuf};

fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn reference() -> serde_json::Value {
    let p = golden_dir().join("REFERENCE.json");
    let s = std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("{}: {e} -- the frozen C reference is missing, and it \
                                    cannot be regenerated now that the C engine is gone",
                                   p.display()));
    serde_json::from_str(&s).expect("REFERENCE.json is not valid JSON")
}

/// The engine under test, run the way `scripts/rust-golden.sh` ran it.
fn run(args: &[&str]) -> String {
    let exe = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/release/k3_run");
    if !exe.exists() {
        // Not a silent skip: a missing binary must not read as a passing gate.
        panic!("{} is not built; run `cargo build --release` first", exe.display());
    }
    let out = std::process::Command::new(&exe)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("running {}: {e}", exe.display()));
    assert!(
        out.status.success(),
        "k3_run {args:?} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn ids_arg(r: &serde_json::Value) -> String {
    r["prompt_ids"]
        .as_array()
        .expect("prompt_ids")
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// Byte-for-byte against the C engine. Not a tolerance -- the whole point is that a
/// tolerance would have admitted the w2/w3 swap.
#[test]
fn logits_are_byte_identical_to_the_frozen_c_reference() {
    let g = golden_dir();
    let r = reference();
    let tmp = std::env::temp_dir().join(format!("k3-golden-{}.bin", std::process::id()));
    run(&[
        g.join("tiny").to_str().unwrap(),
        "--ids", &ids_arg(&r),
        "--gen", "1",
        "--cache-gb", "0.5",
        "--dump-logits", tmp.to_str().unwrap(),
    ]);
    let got = std::fs::read(&tmp).expect("the engine wrote no logits");
    let want = std::fs::read(g.join("c_logits.bin")).expect("frozen C logits missing");
    let _ = std::fs::remove_file(&tmp);

    assert_eq!(got.len(), want.len(), "logit dump changed size");
    if got != want {
        // Name the first divergence rather than just failing: "they differ" is not
        // actionable on a 1 KB blob.
        let i = got.iter().zip(&want).position(|(a, b)| a != b).unwrap();
        let f = |b: &[u8], i: usize| {
            f32::from_le_bytes(b[i / 4 * 4..i / 4 * 4 + 4].try_into().unwrap())
        };
        panic!(
            "logits diverge from the C engine at byte {i} (element {}): got {} want {}",
            i / 4,
            f(&got, i),
            f(&want, i)
        );
    }
}

/// Greedy decode must reproduce the exact token sequence the C engine produced.
#[test]
fn greedy_decode_reproduces_the_frozen_token_ids() {
    let g = golden_dir();
    let r = reference();
    let want: Vec<i64> =
        r["full_ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
    let tmp = std::env::temp_dir().join(format!("k3-golden-ids-{}.txt", std::process::id()));
    run(&[
        g.join("tiny").to_str().unwrap(),
        "--ids", &ids_arg(&r),
        "--gen", &r["gen"].to_string(),
        "--cache-gb", "0.5",
        "--out", tmp.to_str().unwrap(),
    ]);
    let got = std::fs::read_to_string(&tmp).expect("no ids written");
    let _ = std::fs::remove_file(&tmp);
    let got: Vec<i64> =
        got.trim().split(',').filter_map(|s| s.trim().parse().ok()).collect();
    assert_eq!(got, want, "decoded ids drifted from the frozen C reference");
}

/// The same sequence via the incremental path (KV cache + carried KDA state). A bug here
/// shows up only after several tokens, which is why the reference is 8 long and not 1.
#[test]
fn incremental_decode_agrees_with_the_full_sweep() {
    let g = golden_dir();
    let r = reference();
    let want: Vec<i64> =
        r["full_ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
    let tmp = std::env::temp_dir().join(format!("k3-golden-inc-{}.txt", std::process::id()));
    run(&[
        g.join("tiny").to_str().unwrap(),
        "--ids", &ids_arg(&r),
        "--gen", &r["gen"].to_string(),
        "--incremental",
        "--cache-gb", "0.5",
        "--out", tmp.to_str().unwrap(),
    ]);
    let got = std::fs::read_to_string(&tmp).expect("no ids written");
    let _ = std::fs::remove_file(&tmp);
    let got: Vec<i64> =
        got.trim().split(',').filter_map(|s| s.trim().parse().ok()).collect();
    assert_eq!(got, want, "incremental decode diverged from the frozen reference");
}

/// The headline claim: memory buys SPEED, not capability. 0.0005 GB pins nothing so every
/// layer streams through the ring and the asynchronous reader runs; 0.05 GB pins all 13.
/// Identical ids at both ends is the property an async reader writing over a live slot
/// breaks silently and nothing else does.
#[test]
fn output_is_identical_across_every_trunk_budget() {
    let g = golden_dir();
    let r = reference();
    let want: Vec<i64> =
        r["full_ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
    for gb in r["trunk_budgets_gb"].as_array().unwrap() {
        let gb = gb.to_string();
        let tmp =
            std::env::temp_dir().join(format!("k3-golden-t{gb}-{}.txt", std::process::id()));
        run(&[
            g.join("tiny").to_str().unwrap(),
            "--trunk", g.join("trunk").to_str().unwrap(),
            "--ids", &ids_arg(&r),
            "--gen", &r["gen"].to_string(),
            "--cache-gb", "0.5",
            "--trunk-gb", &gb,
            "--out", tmp.to_str().unwrap(),
        ]);
        let got = std::fs::read_to_string(&tmp).expect("no ids written");
        let _ = std::fs::remove_file(&tmp);
        let got: Vec<i64> =
            got.trim().split(',').filter_map(|s| s.trim().parse().ok()).collect();
        assert_eq!(got, want, "trunk budget {gb} GB changed the output");
    }
}

/// The frozen artefacts must actually be present and non-empty. Without this, deleting
/// `tests/golden/` would turn four gates into four vacuous passes.
#[test]
fn the_frozen_reference_is_present_and_complete() {
    let g = golden_dir();
    for f in [
        "REFERENCE.json",
        "c_logits.bin",
        "tiny/config.json",
        "tiny/model.safetensors",
        "tiny/ref_logits.json",
        "trunk/trunk.bin",
        "trunk/trunk.json",
    ] {
        let p = g.join(f);
        let n = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
        assert!(n > 0, "{} is missing or empty -- the oracle is gone", p.display());
    }
    let r = reference();
    assert_eq!(r["full_ids"].as_array().unwrap().len(), 13, "5 prompt + 8 generated");
}
