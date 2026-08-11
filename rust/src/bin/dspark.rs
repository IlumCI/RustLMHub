// SPDX-License-Identifier: Apache-2.0
//
// Run DSpark on real checkpoint weights from a SYNTHETIC main_hidden and dump every
// intermediate an independent implementation can be compared against.
//
// Feeding a synthetic main_hidden rather than running the 304B main model is what makes
// this checkable at all: it exercises the whole drafter -- main_proj, main_norm, the
// per-stage KV ring, the bidirectional block attention, all three MoE stages, hc_head,
// the Markov head and the confidence head -- in about a minute, against a numpy reference
// that never has to load 166 GB.
//
// usage: dspark <model_dir> <T> <seed_token> <cache_gb> <out.bin>
//
// out.bin, little-endian:
//   u32 n_stages, u32 T, u32 block, u32 hidden, u32 head_dim, u32 rank, u32 vocab
//   f32[hidden]                       main_x for the LAST position
//   f32[n_stages][T][head_dim]        the per-stage KV rows
//   f32[block][hidden]                hc_head output
//   f32[block]                        confidence
//   u32[block]                        drafted token ids

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

use k3::arch;
use k3::cache::{v4_expert_names, Cache};
use k3::dspark::DSpark;
use k3::st::St;
use k3::v4::{precompute_rope, MoeDimsV4};
use k3::v4run::Trunk;

/// The same deterministic main_hidden the numpy reference builds. Sine rather than a PRNG
/// so both sides can generate it independently without agreeing on a generator.
fn synth(t_len: usize, width: usize) -> Vec<f32> {
    (0..t_len * width).map(|i| (i as f64 * 0.001).sin() as f32 * 0.05).collect()
}

fn run(a: &[String]) -> Result<(), String> {
    let dir = Path::new(&a[1]);
    let t_len: usize = a[2].parse().map_err(|e| format!("T: {e}"))?;
    let seed: u32 = a[3].parse().map_err(|e| format!("seed: {e}"))?;
    let cache_gb: f64 = a[4].parse().map_err(|e| format!("cache-gb: {e}"))?;

    let st = St::open(dir).map_err(|e| e.to_string())?;
    let spec = arch::spec_file(&dir.join("config.json"))?;
    let (e, hd, eps) = (spec.hidden, spec.head_dim, spec.rms_eps);

    let trunk = Trunk::load_io(&st, &spec)?;
    let n_stages = k3::dspark::n_stages(&st);
    println!("dspark: {n_stages} stages");
    let mut ds = DSpark::load(&st, &spec, n_stages, 5, 128799)?;
    let slot = {
        let r = k3::cache::locate(&st, &v4_expert_names(2, 0)).ok_or("cannot locate an expert")?;
        k3::cache::slot_need(&r)
    };
    let mut cache = Cache::new((cache_gb * 1e9) as i64, slot, spec.n_experts, spec.topk)?;
    let md = MoeDimsV4 {
        hidden: e,
        moe_inter: spec.moe_inter,
        n_experts: spec.n_experts,
        topk: spec.topk,
        route_scale: spec.routed_scale,
        swiglu_limit: match spec.glu {
            k3::ops::Glu::SwigluClamped { limit } => limit,
            _ => 10.0,
        },
        n_hash_layers: spec.n_hash_layers,
    };

    let ntgt = k3::dspark::TARGET_LAYERS.len();
    let mh = synth(t_len, ntgt * e);
    let rope = precompute_rope(64, t_len + 16, 0, spec.rope_theta, 16.0, 32.0, 1.0);
    for p in 0..t_len {
        ds.push(&mh[p * ntgt * e..][..ntgt * e], p, &rope, e, hd, 64, eps);
    }
    let main_x = ds.main_x(&mh[(t_len - 1) * ntgt * e..][..ntgt * e], e, eps);
    let d = ds.draft(&trunk, &spec, &md, seed, t_len - 1, &rope, &st, &mut cache, eps)?;

    let mut o: Vec<u8> = Vec::new();
    for v in [n_stages, t_len, ds.block, e, hd, ds.rank, spec.vocab] {
        o.extend_from_slice(&(v as u32).to_le_bytes());
    }
    let mut push = |v: &[f32]| {
        for x in v {
            o.extend_from_slice(&x.to_le_bytes());
        }
    };
    push(&main_x);
    for s in 0..n_stages {
        push(&ds.kv_rows(s)[..t_len * hd]);
    }
    push(&d.hx);
    push(&d.conf);
    for id in &d.ids {
        o.extend_from_slice(&id.to_le_bytes());
    }
    std::fs::File::create(&a[5])
        .and_then(|mut f| f.write_all(&o))
        .map_err(|e| format!("{}: {e}", a[5]))?;
    println!("draft {:?}", d.ids);
    println!("conf  {:?}", d.conf);
    cache.report("dspark");
    Ok(())
}

fn main() -> ExitCode {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 6 {
        eprintln!("usage: dspark <model_dir> <T> <seed_token> <cache_gb> <out.bin>");
        return ExitCode::from(2);
    }
    match run(&a) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
