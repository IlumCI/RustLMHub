// SPDX-License-Identifier: Apache-2.0
//
// Exercise the streaming expert cache against a real DeepSeek-V4 shard, under enough
// pressure to force eviction, and check every byte it hands back against a direct read.
//
// usage: v4_cache <shard_dir> <layer> <cache_mb> <requests>

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use k3::cache::{locate, v4_expert_names, Cache};
use k3::st::St;

fn main() -> ExitCode {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        eprintln!("usage: v4_cache <dir> <layer> <cache_mb> <requests>");
        return ExitCode::from(2);
    }
    let layer: usize = a[2].parse().unwrap();
    let cache_mb: i64 = a[3].parse().unwrap();
    let nreq: usize = a[4].parse().unwrap();

    let s = match St::open(Path::new(&a[1])) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let n_experts = (0..)
        .take_while(|&e| s.find(v4_expert_names(layer, e).probe_name()).is_some())
        .count();
    if n_experts == 0 {
        eprintln!("no routed experts for layer {layer}");
        return ExitCode::FAILURE;
    }
    let r0 = locate(&s, &v4_expert_names(layer, 0)).expect("expert 0");
    // A slot holds every coalesced run with its O_DIRECT slack, not the enclosing span.
    let slot_bytes = k3::cache::slot_need(&r0);

    let topk = 6;
    let mut c = match Cache::new(cache_mb * 1_000_000, slot_bytes, n_experts, topk) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("v4_cache: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "layer {layer}: {n_experts} experts of {:.2} MB in {} run(s), cache {} slots ({} MB)",
        r0.nbytes as f64 / 1e6,
        r0.runs.len(),
        c.nslot(),
        cache_mb
    );
    if c.nslot() >= n_experts {
        eprintln!("  NOTE: cache holds every expert; nothing will be evicted");
    }

    // Two streams. "sweep" strides across the whole pool and is deliberately
    // cache-hostile: it exercises eviction, not caching. "hot" draws 80% of requests
    // from a tenth of the experts, which is what skewed MoE routing actually looks
    // like, and is where a hit rate means something.
    let hot_set = (n_experts / 10).max(1);
    let hostile = std::env::var("SWEEP").is_ok();
    let ids: Vec<usize> = (0..nreq)
        .map(|i| {
            if hostile {
                (i * 37 + i / n_experts) % n_experts
            } else if i % 5 == 0 {
                (i * 37) % n_experts
            } else {
                (i * 13) % hot_set
            }
        })
        .collect();
    println!("  stream       : {}", if hostile { "sweep (eviction stress)" } else { "hot set" });

    let t0 = Instant::now();
    let mut bad = 0usize;
    let mut checked = 0usize;
    let mut direct = k3::st::Aligned::new(slot_bytes);
    for (i, &e) in ids.iter().enumerate() {
        let names = v4_expert_names(layer, e);
        let Some(slot) = c.get(&s, layer, e, &names) else {
            eprintln!("  fetch failed for expert {e}");
            return ExitCode::FAILURE;
        };
        // Check a sample against an independent read of the same tensors. Every
        // request would be correct but slow; a fixed fraction still catches a slot
        // handed out before its read landed, or a stale eviction.
        if i % 7 == 0 {
            let q = c.expert(slot);
            let r = locate(&s, &names).unwrap();
            let mut bases = Vec::new();
            let mut cur = 0usize;
            for run in &r.runs {
                let region = k3::cache::run_region(run.len);
                let (avail, pad) =
                    s.read_aligned(r.shard, run.off, run.len, &mut direct[cur..cur + region]);
                assert_eq!(avail, run.len, "direct read short");
                bases.push(cur + pad as usize);
                cur += region;
            }
            let parts = [q.p1, q.s1, q.p3, q.s3, q.p2, q.s2];
            for (k, part) in parts.iter().enumerate() {
                let (run, o, n) = r.parts[k];
                let b = bases[run] + o as usize;
                let want = &direct[b..b + n as usize];
                if *part != want {
                    bad += 1;
                    eprintln!("  MISMATCH expert {e} tensor {k}");
                    break;
                }
            }
            checked += 1;
        }
    }
    let dt = t0.elapsed().as_secs_f64();

    c.report(&format!("\ncache after {nreq} requests"));
    println!("  byte check   : {}/{checked} sampled experts identical to a direct read",
             checked - bad);
    println!("  wall         : {dt:.2} s  ({:.1} MB/s effective)",
             c.bytes_read as f64 / 1e6 / dt.max(1e-9));
    let hot: usize = c.hist.values().filter(|&&v| v > 1).count();
    println!("  histogram    : {} of {} experts requested more than once", hot, c.hist.len());

    if bad == 0 {
        println!("\nCACHE VERIFIED: every sampled expert matches a direct read.");
        ExitCode::SUCCESS
    } else {
        println!("\nCACHE FAILED: {bad} mismatches");
        ExitCode::FAILURE
    }
}
