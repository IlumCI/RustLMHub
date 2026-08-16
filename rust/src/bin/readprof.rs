// Per-token / per-layer / per-tensor read profiler for the dense qwen35 decode path.
//
// Answers "which bytes are read per token?" with MEASURED numbers, not bandwidth*time:
//   - /proc/self/io read_bytes delta per token  (ground truth: actual bytes off the block dev)
//   - the cache's own per-layer byte accounting  (cross-check + attribution)
//   - a residency table: every tensor category, its size, resident?, reads/token
//
// No prefetcher is attached, so every streamed byte flows through `admit` and is attributed.
//
//   readprof MODEL_DIR [n_tokens] [cache_gb|auto] [prompt]
use std::collections::BTreeMap;

fn proc_read_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/io")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("read_bytes:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0)
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let p = a.get(1).ok_or("usage: readprof MODEL_DIR [n_tokens] [cache_gb|auto] [prompt]")?;
    let n: usize = a.get(2).map_or(Ok(8), |v| v.parse()).map_err(|e| format!("{e}"))?;
    let gb_arg = a.get(3).cloned().unwrap_or_else(|| "auto".into());
    let prompt = a.get(4).cloned().unwrap_or_else(|| "Write a long story about a robot.".into());

    let st = k3::st::St::open(std::path::Path::new(p)).map_err(|e| e.to_string())?;
    let meta = st.meta.as_ref().ok_or("no gguf metadata")?;
    let cfg = k3::qwen35::Cfg::from_meta(meta)?;
    if !cfg.is_dense() {
        return Err("readprof targets a DENSE qwen35".into());
    }
    let max_ctx: usize = std::env::var("Q35_MAX_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(8192);
    let tr = k3::qwen35run::Trunk::load(&st, &cfg, max_ctx)?;

    // ---- residency map from the tensor list ----
    // category -> (total bytes, is it streamed through the arena?)
    let classify = |name: &str| -> &'static str {
        if name.starts_with("token_embd") { "embedding" }
        else if name.starts_with("output.") { "lm_head" }
        else if name.contains("ffn_gate") { "ffn_gate" }
        else if name.contains("ffn_up") { "ffn_up" }
        else if name.contains("ffn_down") { "ffn_down" }
        else if name.contains("attn_qkv") || name.contains("attn_q.") || name.contains("attn_k.")
             || name.contains("attn_v.") || name.contains("attn_output") || name.contains("attn_gate") { "attention" }
        else if name.contains("ssm_") { "ssm/deltanet" }
        else if name.contains("_norm") { "norms" }
        else { "other" }
    };
    let streamed = |cat: &str| matches!(cat, "ffn_gate" | "ffn_up" | "ffn_down") ;
    let mut cat_bytes: BTreeMap<&str, u64> = BTreeMap::new();
    for t in &st.tensors {
        *cat_bytes.entry(classify(&t.name)).or_insert(0) += t.nbytes as u64;
    }
    let per_layer_ffn: u64 = st.tensors.iter()
        .filter(|t| t.name.starts_with("blk.0.") && classify(&t.name).starts_with("ffn"))
        .map(|t| t.nbytes as u64).sum();

    // ---- cache (no prefetcher; every read attributed) ----
    let mut slot = 0usize;
    for l in 0..cfg.n_layers {
        let r = k3::cache::locate(&st, &k3::cache::gguf_dense_ffn_src(l, 0))
            .ok_or_else(|| format!("cannot locate layer {l}"))?;
        slot = slot.max(k3::cache::slot_need(&r));
    }
    let budget: i64 = if gb_arg.eq_ignore_ascii_case("auto") {
        k3::cache::auto_budget_bytes(2_500_000_000, (slot * 2) as i64, (slot * cfg.n_layers) as i64, 5_000_000_000)
    } else {
        (gb_arg.parse::<f64>().map_err(|e| format!("{e}"))? * 1e9) as i64
    };
    let mut cache = k3::cache::Cache::new(budget, slot, 1, 1)?;
    let resident_ffn_layers = cache.nslot();
    println!("model {}  | trunk {:.2} GB resident | arena {:.2} GB = {} of {} FFN layers | slot {:.0} MB | per-layer FFN {:.0} MB",
             p, tr.bytes as f64 / 1e9, budget as f64 / 1e9, resident_ffn_layers, cfg.n_layers,
             slot as f64 / 1e6, per_layer_ffn as f64 / 1e6);

    let tok = k3::tok::Tok::from_gguf(meta)?;
    let ids = tok.encode(&prompt, false)?;
    let mut sess = k3::qwen35run::Session::new(&tr);
    let mut logits = vec![0f32; cfg.vocab.max(tr.io.vocab)];

    // prefill (not profiled per-token; warms the arena as a real conversation would)
    if ids.len() > 1 {
        k3::qwen35run::step_many(&tr, &mut sess, &st, &mut cache, &ids[..ids.len() - 1], &mut logits)?;
    }
    let mut cur = *ids.last().unwrap();

    println!("\nper-token disk reads (decode):");
    let argmax = |v: &[f32]| v.iter().enumerate().fold((0, f32::MIN), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0;
    let mut totals: BTreeMap<usize, u64> = BTreeMap::new();
    let mut proc_sum = 0u64;
    let mut cache_sum = 0u64;
    for step_i in 0..n {
        cache.reset_stats();
        let pr0 = proc_read_bytes();
        let t0 = std::time::Instant::now();
        k3::qwen35run::step(&tr, &mut sess, &st, &mut cache, cur, &mut logits)?;
        let dt = t0.elapsed().as_secs_f64();
        let pr = proc_read_bytes().saturating_sub(pr0);
        cur = argmax(&logits) as u32;
        proc_sum += pr;
        cache_sum += cache.bytes_read;
        for (&l, &b) in cache.layer_bytes.iter() {
            *totals.entry(l).or_insert(0) += b;
        }
        let misses = cache.misses;
        println!("  token {step_i:2}: /proc read {:6.0} MB | cache read {:6.0} MB ({} miss / {} hit) | {:.2} s -> {:.2} GB/s",
                 pr as f64 / 1e6, cache.bytes_read as f64 / 1e6, misses, cache.hits, dt, pr as f64 / 1e9 / dt.max(1e-9));
        // On a representative warm token, dump the full per-layer map.
        if step_i == n.saturating_sub(1) {
            println!("    per-layer (last token):");
            for l in 0..cfg.n_layers {
                let b = cache.layer_bytes.get(&l).copied().unwrap_or(0);
                let hit = cache.layer_hits.get(&l).copied().unwrap_or(0) > 0;
                println!("      layer {l:2}: {:6.1} MB  {}", b as f64 / 1e6, if b > 0 { "READ" } else if hit { "resident(hit)" } else { "-" });
            }
        }
    }

    // ---- residency table ----
    println!("\nresidency map (avg over {n} decode tokens):");
    println!("  {:<16} {:>10} {:>10} {:>14}", "tensor", "size", "resident?", "reads/token");
    let avg_ffn_layers_read: f64 = totals.values().sum::<u64>() as f64 / per_layer_ffn as f64 / n as f64;
    for (cat, &bytes) in &cat_bytes {
        let (res, rpt) = if streamed(cat) {
            ("STREAMED", format!("{avg_ffn_layers_read:.1} layers"))
        } else if *cat == "embedding" {
            ("streamed", "~1 row".into())
        } else {
            ("RESIDENT", "0".into())
        };
        println!("  {:<16} {:>7.2} GB {:>10} {:>14}", cat, bytes as f64 / 1e9, res, rpt);
    }
    // ---- batch economics: does a K-wide forward read the FFN ONCE (amortizable across K
    // accepted tokens, i.e. speculative decoding) or K times? This is the decisive test. ----
    println!("\nbatch economics (bytes for a K-wide forward = the speculative-decode ceiling):");
    let tail: Vec<u32> = ids.iter().rev().take(16).rev().copied().collect();
    for k in [1usize, 2, 4, 8] {
        let chunk: Vec<u32> = (0..k).map(|i| tail[tail.len().saturating_sub(k) + i]).collect();
        let mut s2 = k3::qwen35run::Session::new(&tr);
        cache.reset_stats();
        let pr0 = proc_read_bytes();
        let t0 = std::time::Instant::now();
        k3::qwen35run::step_many(&tr, &mut s2, &st, &mut cache, &chunk, &mut logits)?;
        let dt = t0.elapsed().as_secs_f64();
        let pr = proc_read_bytes().saturating_sub(pr0);
        println!("  K={k}: read {:5.2} GB  in {:.2}s  ->  {:.2} GB per token-position (if all K accepted)  [{:.2}x vs K=1]",
                 pr as f64 / 1e9, dt, pr as f64 / 1e9 / k as f64, dt / k as f64 / 2.5 * 2.5);
    }
    println!("  If GB stays flat as K grows, one weight traversal serves K tokens: accept a of K");
    println!("  drafted tokens and you pay ~1/a of the disk traffic per accepted token.");

    println!("\nSUMMARY: /proc {:.2} GB/tok  cache {:.2} GB/tok  (should match; delta = embedding rows + misc)",
             proc_sum as f64 / n as f64 / 1e9, cache_sum as f64 / n as f64 / 1e9);
    println!("  FFN is {:.0}% of the checkpoint; {} of {} FFN layers resident -> {:.0}% of FFN re-streamed/token",
             (cat_bytes.get("ffn_gate").unwrap_or(&0) + cat_bytes.get("ffn_up").unwrap_or(&0) + cat_bytes.get("ffn_down").unwrap_or(&0)) as f64
                 / st.tensors.iter().map(|t| t.nbytes as u64).sum::<u64>() as f64 * 100.0,
             resident_ffn_layers, cfg.n_layers,
             (1.0 - resident_ffn_layers as f64 / cfg.n_layers as f64) * 100.0);
    Ok(())
}
