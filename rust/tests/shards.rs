// SPDX-License-Identifier: Apache-2.0
//
// A split checkpoint must load as one model.
//
// WHY THIS IS A GATE AND NOT A NOTE
//     `fetch::choose` learned to download split GGUFs on the strength of a claim about the
//     READER: that `St::open` already sorts a directory of `.gguf` files, scans each with
//     its own shard index and merges the tensor tables. That claim was read off the code,
//     and the failure it guards is the one this project refuses to accept anywhere else --
//     a loader that binds SOME of a model and reports success. Missing tensors do not
//     announce themselves; they surface as a model that runs and is wrong, or as a
//     "missing blk.37..." after a 62-part transfer has already finished.
//
//     Every real split checkpoint is hundreds of gigabytes, so the fixture is built here:
//     two shards, four tensors, no model and no external file.
//
//     `tests/tools/split_gguf.py` does the same thing to a real tiny qwen35moe checkpoint,
//     which is how the stronger property was checked by hand -- a forward pass over three
//     shards is bit-identical to the whole file, same routing, `logits sum -0.402005`.
//     That needs a generated model; this needs nothing, so this is the one that runs on
//     every commit.

use std::path::Path;

const ALIGN: usize = 32;

fn str_bytes(s: &str) -> Vec<u8> {
    let mut v = (s.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(s.as_bytes());
    v
}

/// Write one GGUF holding `tensors`, each a named 1-D F32 vector.
///
/// Offsets are relative to this file's OWN data section, which is the whole point: a shard
/// is a complete GGUF, not a byte range of a larger one, so a reader that treated part
/// two's offsets as absolute would read the wrong bytes rather than fail.
fn write_gguf(path: &Path, tensors: &[(&str, Vec<f32>)], part: u16, total: u16) {
    let mut head = Vec::new();
    head.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // "GGUF"
    head.extend_from_slice(&3u32.to_le_bytes());
    head.extend_from_slice(&(tensors.len() as u64).to_le_bytes());

    // Only part one carries the architecture, exactly as llama.cpp's splitter emits it.
    let mut kv = Vec::new();
    let mut n_kv = 0u64;
    if part == 1 {
        kv.extend_from_slice(&str_bytes("general.architecture"));
        kv.extend_from_slice(&8u32.to_le_bytes());
        kv.extend_from_slice(&str_bytes("qwen35moe"));
        n_kv += 1;
    }
    for (k, v) in [("split.no", part - 1), ("split.count", total)] {
        kv.extend_from_slice(&str_bytes(k));
        kv.extend_from_slice(&2u32.to_le_bytes()); // u16
        kv.extend_from_slice(&v.to_le_bytes());
        n_kv += 1;
    }
    head.extend_from_slice(&n_kv.to_le_bytes());
    head.extend_from_slice(&kv);

    let mut off = 0usize;
    let mut body = Vec::new();
    for (name, data) in tensors {
        head.extend_from_slice(&str_bytes(name));
        head.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        head.extend_from_slice(&(data.len() as u64).to_le_bytes());
        head.extend_from_slice(&0u32.to_le_bytes()); // F32
        head.extend_from_slice(&(off as u64).to_le_bytes());
        body.resize(off, 0);
        for f in data {
            body.extend_from_slice(&f.to_le_bytes());
        }
        off = body.len().div_ceil(ALIGN) * ALIGN;
    }
    let pad = head.len().div_ceil(ALIGN) * ALIGN - head.len();
    head.resize(head.len() + pad, 0);
    head.extend_from_slice(&body);
    std::fs::write(path, &head).unwrap();
}

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("k3-shards-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Two shards, four tensors, one model -- with the DATA checked, not just the count.
///
/// A merge that got the offsets wrong would still report four tensors.
#[test]
fn a_split_checkpoint_loads_as_one_model() {
    let d = tmpdir("merge");
    let a: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..8).map(|i| 100.0 + i as f32).collect();
    let c: Vec<f32> = (0..8).map(|i| 200.0 + i as f32).collect();
    let e: Vec<f32> = (0..8).map(|i| 300.0 + i as f32).collect();
    write_gguf(&d.join("m-00001-of-00002.gguf"), &[("token_embd.weight", a.clone()),
                                                  ("blk.0.attn_norm.weight", b.clone())], 1, 2);
    // Part two's tensors sit at ITS offsets 0 and 32 -- the same numbers part one used.
    write_gguf(&d.join("m-00002-of-00002.gguf"), &[("blk.1.attn_norm.weight", c.clone()),
                                                  ("output.weight", e.clone())], 2, 2);

    let st = k3::st::St::open(&d).expect("a directory of shards must open");
    assert_eq!(st.tensors.len(), 4, "every part's tensors must appear");
    // Metadata comes from part one, which is the only part that has any.
    assert_eq!(
        st.meta.as_ref().and_then(|m| m.get("general.architecture")).and_then(k3::gguf::Value::as_str),
        Some("qwen35moe"),
        "part one's metadata must survive the merge"
    );

    for (name, want) in [
        ("token_embd.weight", &a),
        ("blk.0.attn_norm.weight", &b),
        ("blk.1.attn_norm.weight", &c),
        ("output.weight", &e),
    ] {
        let t = st.find(name).unwrap_or_else(|| panic!("{name} missing after merge"));
        let mut got = vec![0f32; want.len()];
        st.read_f32(t, &mut got);
        assert_eq!(&got, want, "{name} read the wrong bytes -- shard offsets are not absolute");
    }
    let _ = std::fs::remove_dir_all(&d);
}

/// The dangerous case, stated as a test: a set with a part missing must not look complete.
///
/// `fetch::choose` refuses this before the download, but a directory can also be assembled
/// by hand, so the loader's own behaviour is worth pinning: it binds what is there. Nothing
/// makes that safe except the caller having checked -- which is why `choose` counts parts
/// and `download` re-verifies every size afterwards.
#[test]
fn a_missing_part_shows_up_as_missing_tensors_not_as_an_error() {
    let d = tmpdir("gap");
    write_gguf(&d.join("m-00001-of-00002.gguf"), &[("token_embd.weight", vec![1.0; 8])], 1, 2);
    let st = k3::st::St::open(&d).expect("one part alone still parses");
    assert_eq!(st.tensors.len(), 1);
    assert!(st.find("output.weight").is_none(), "the absent part's tensors are simply absent");
    let _ = std::fs::remove_dir_all(&d);
}

/// Ordering is load-bearing: `St::open` sorts paths, and zero-padded part numbers are what
/// make lexicographic order the same as numeric order past nine parts.
#[test]
fn parts_past_nine_still_sort_numerically() {
    let d = tmpdir("order");
    for i in 1..=12u16 {
        write_gguf(
            &d.join(format!("m-{i:05}-of-00012.gguf")),
            &[(Box::leak(format!("blk.{}.w", i - 1).into_boxed_str()), vec![i as f32; 4])],
            i,
            12,
        );
    }
    let st = k3::st::St::open(&d).unwrap();
    assert_eq!(st.tensors.len(), 12);
    // Part one sorts first, so its metadata is the metadata -- if part 10 sorted before
    // part 1 the model would take its architecture from a part that has none.
    assert!(st.meta.as_ref().is_some_and(|m| m.contains_key("general.architecture")));
    for i in 1..=12u16 {
        let t = st.find(&format!("blk.{}.w", i - 1)).unwrap();
        let mut got = vec![0f32; 4];
        st.read_f32(t, &mut got);
        assert_eq!(got, vec![i as f32; 4], "part {i} bound the wrong data");
    }
    let _ = std::fs::remove_dir_all(&d);
}
