// SPDX-License-Identifier: Apache-2.0
//
// What does a prompt token actually cost, and what is it spending the time on?
//
// The serve harness reports seconds per token, which is the number the user feels but not
// the one that explains anything. Prefill on this engine is a race between two costs that
// scale in OPPOSITE directions with chunk width:
//
//     expert I/O   falls with width -- k tokens share one union of experts per layer
//     activations  rise with width  -- k * hidden floats held for the whole pass
//
// and the current width is pinned at 8 by a constant, not by either of those. This
// measures both so the constant can be replaced by arithmetic. Usage:
//
//     prefillbench <gguf-dir-or-file> <n_tokens> <width> [<width> ...] [--cache-gb G]
//
// Each width prefills the SAME synthetic token sequence from a cold session, so the only
// thing that varies is the chunking.

use k3::model::Model;

fn main() {
    if let Err(e) = run() {
        eprintln!("prefillbench: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err("usage: prefillbench <path> <n_tokens> <width>...  [--cache-gb G]".into());
    }
    let path = args[0].clone();
    let n: usize = args[1].parse().map_err(|_| "n_tokens must be a number")?;
    let mut cache_gb = 5.0f64;
    let mut widths: Vec<usize> = Vec::new();
    let mut i = 2;
    while i < args.len() {
        if args[i] == "--cache-gb" {
            cache_gb = args.get(i + 1).ok_or("--cache-gb needs a value")?.parse().map_err(|_| "bad --cache-gb")?;
            i += 2;
        } else {
            widths.push(args[i].parse().map_err(|_| "width must be a number")?);
            i += 1;
        }
    }
    if widths.is_empty() {
        return Err("give at least one width".into());
    }

    let st = k3::st::St::open(std::path::Path::new(&path)).map_err(|e| e.to_string())?;
    let mut m = k3::model::Qwen35::load(&st, cache_gb, 0.0, n + 64)?;
    {
        let c = m.cfg();
        println!(
            "model  : {} layers, hidden {}, {} experts, topk {}, moe_inter {}",
            c.n_layers, c.hidden, c.n_experts, c.topk, c.moe_inter
        );
    }
    println!("cache  : {} slots ({cache_gb:.2} GB), derived width {}", m.nslot(), m.width());
    println!();

    // A deterministic pseudo-prompt. Real text routes MORE coherently than this, so every
    // union measured here is a pessimistic bound on a real prompt -- which is the side to
    // be wrong on when sizing a cache.
    let vocab = m.vocab() as u64;
    let ids: Vec<u32> = (0..n as u64)
        .map(|i| {
            let mut x = i.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            x ^= x >> 33;
            (x % vocab.min(150_000)) as u32
        })
        .collect();

    let mut logits = vec![0f32; m.vocab()];
    println!(
        "{:>6}  {:>9}  {:>9}  {:>9}  {:>8}  {:>7}  {:>6}  {:>10}  {:>5}",
        "width", "total s", "s/token", "GB read", "MB/tok", "MB/s", "share", "max|d|", "argmax"
    );
    let mut first: Option<Vec<f32>> = None;
    for &w in &widths {
        m.reset();
        m.reset_io_stats();
        k3::qwen35run::reset_sharing();
        m.set_width(w);
        let t0 = std::time::Instant::now();
        m.feed(&ids, &mut logits)?;
        let dt = t0.elapsed().as_secs_f64();
        let (bytes, ..) = m.io_stats();
        let gb = bytes as f64 / 1e9;
        let (routed, unique) = k3::qwen35run::sharing();
        // The gate, not a statistic. Chunking changes only the ORDER work is issued in, so
        // every width must land on the same logits; a drift here means batching has
        // changed the arithmetic, which is exactly the failure that still produces fluent
        // text and no error.
        let (dmax, same) = match &first {
            None => {
                first = Some(logits.clone());
                (0.0f32, true)
            }
            Some(f) => (
                f.iter().zip(&logits).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max),
                argmax(f) == argmax(&logits),
            ),
        };
        println!(
            "{w:>6}  {dt:>9.2}  {:>9.3}  {gb:>9.2}  {:>8.1}  {:>7.0}  {:>5.1}%  {dmax:>10.3e}  {:>5}",
            dt / n as f64,
            gb * 1e3 / n as f64,
            gb * 1e9 / dt / 1e6,
            100.0 * (1.0 - unique as f64 / routed.max(1) as f64),
            if same { "same" } else { "DIFF" },
        );
    }
    Ok(())
}

fn argmax(v: &[f32]) -> usize {
    let mut b = 0;
    for i in 1..v.len() {
        if v[i] > v[b] {
            b = i;
        }
    }
    b
}
