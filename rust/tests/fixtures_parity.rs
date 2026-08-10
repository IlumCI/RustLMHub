// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

use k3::cfg::{load, load_file, Error};
use serde_json::Value;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/fixtures")
        .canonicalize()
        .expect("tests/fixtures must exist relative to the crate")
}

#[test]
fn flat_fixture_loads_with_the_oracles_numbers() {
    let path = fixtures().join("ref_k3.json");
    let txt = std::fs::read_to_string(&path).expect("ref_k3.json");
    let root: Value = serde_json::from_str(&txt).expect("valid JSON");
    let cfg = root.get("config").expect("ref_k3.json has a \"config\" member");

    let c = load(cfg, "ref_k3.json").expect("the flat fixture must load");

    assert_eq!(c.hidden, 128);
    assert_eq!(c.n_layers, 13);
    assert_eq!(c.vocab, 256);
    assert_eq!(c.kda_heads, 4);
    assert_eq!(c.kda_head_dim, 16);
    assert_eq!(c.n_experts, 8);
    assert_eq!(c.topk, 2);
    assert_eq!(c.latent, 64);
    assert_eq!(c.attn_res_block, 3);
    assert_eq!(c.situ_b1, 4.0);
    assert_eq!(c.situ_b2, 25.0);
    assert_eq!(c.full_attn.len(), 4);
    assert_eq!(c.gate_lb, -5.0);
    assert!(!c.nested, "ref_k3.json is the flat shape");

    let map: String = (0..c.n_layers)
        .map(|l| if c.is_mla(l) { 'M' } else { 'K' })
        .collect();
    assert_eq!(map, "KKKMKKKMKKKMM", "layer map must match the oracle's");
}

#[test]
fn the_negative_fixtures_are_all_rejected() {
    for name in ["bad_layer_index", "bad_topk", "no_layermap"] {
        let path = fixtures().join("cfg").join(format!("{name}.json"));
        match load_file(&path) {
            Ok(_) => panic!("{name}.json must not load"),
            Err(Error::Io(m)) | Err(Error::Parse(m)) => {
                panic!("{name}.json should fail a real check, not I/O or parse: {m}")
            }
            Err(e @ Error::Missing { .. }) | Err(e @ Error::Structural(_)) => {
                eprintln!("{name}.json rejected: {e}");
            }
        }
    }
}

#[test]
fn a_config_without_a_layer_map_fails_rather_than_running_all_kda() {
    let path = fixtures().join("cfg/no_layermap.json");
    match load_file(&path) {
        Err(Error::Missing { keys, .. }) => {
            assert!(
                keys.iter().any(|k| k == "full_attn_layers"),
                "the missing key must be named, got {keys:?}"
            );
        }
        other => panic!("expected a Missing error naming full_attn_layers, got {other:?}"),
    }
}
