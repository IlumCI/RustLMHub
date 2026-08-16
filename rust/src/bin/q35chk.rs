// Smallest end-to-end path for qwen35moe: load the trunk, stream experts, emit tokens.
fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let p = a.get(1).ok_or("usage: q35chk MODEL_DIR [n_tokens] [cache_gb] [first_id]")?;
    let n: usize = a.get(2).map_or(Ok(4), |v| v.parse()).map_err(|e| format!("{e}"))?;
    // cache_gb accepts a number, or "auto" to size the arena to swap-safe available RAM.
    let gb_arg = a.get(3).cloned().unwrap_or_else(|| "2".into());
    let auto = gb_arg.eq_ignore_ascii_case("auto");
    let gb: f64 = if auto { 0.0 } else { gb_arg.parse().map_err(|e| format!("{e}"))? };
    let prompt = a.get(4).cloned().unwrap_or_else(|| "The capital of France is".into());

    let st = k3::st::St::open(std::path::Path::new(p)).map_err(|e| e.to_string())?;
    let cfg = k3::qwen35::Cfg::from_meta(st.meta.as_ref().ok_or("no gguf metadata")?)?;
    let t0 = std::time::Instant::now();
    // Size the rope table for the whole conversation this run can reach.
    let max_ctx: usize = std::env::var("Q35_MAX_CTX").ok().and_then(|v| v.parse().ok())
        .unwrap_or(8192);
    let tr = k3::qwen35run::Trunk::load(&st, &cfg, max_ctx)?;
    println!("trunk {:.2} GB in {:.1} s | {} experts top-{} | expert dtypes {:?}",
             tr.bytes as f64 / 1e9, t0.elapsed().as_secs_f64(), cfg.n_experts, cfg.topk,
             tr.expert_dt[0].map(|d| d.name()));
    let mixed: Vec<usize> =
        (0..cfg.n_layers).filter(|&l| tr.expert_dt[l] != tr.expert_dt[0]).collect();
    if !mixed.is_empty() {
        println!("mixed quantisation: layers {mixed:?} differ -> {:?}",
                 tr.expert_dt[mixed[0]].map(|d| d.name()));
    }

    // Slot sizing is NOT arithmetic on tensor sizes. `locate` resolves the expert into
    // the actual byte runs the reader will issue, and `slot_need` pads them the way the
    // arena lays them out. Summing the three tensor sizes myself put p2's offset inside
    // p3's region, so the Q6_K kernel was handed the Q4_K gate -- same bytes, wrong shape.
    // The MAX over every layer, not layer 0's. This build quantises ffn_down_exps to Q6_K
    // in most layers and Q4_K in layers 5-6, so a slot sized from layer 0 is 270 KB too
    // small for nothing and a slot sized from layer 5 is too small for everything else.
    // A dense checkpoint streams one whole FFN per layer where an MoE streams one expert
    // of many; both go through the same cache, addressed differently.
    let src = if cfg.is_dense() {
        k3::cache::gguf_dense_ffn_src
    } else {
        k3::cache::gguf_expert_src
    };
    let mut slot = 0usize;
    for l in 0..cfg.n_layers {
        let r = k3::cache::locate(&st, &src(l, 0))
            .ok_or_else(|| format!("cannot locate layer {l} expert 0"))?;
        slot = slot.max(k3::cache::slot_need(&r));
    }
    let (n_exp, topk) = if cfg.is_dense() { (1, 1) } else { (cfg.n_experts, cfg.topk) };
    // Read MemAvailable AFTER the trunk is resident, so it already excludes those bytes.
    // The whole streamed weight set is the useful ceiling (a bigger arena caches nothing
    // more); a 2.5 GB margin leaves room for activations, KV growth and other apps.
    let max_budget = (slot as i64) * (cfg.n_layers as i64);
    let budget: i64 = if auto {
        let b = k3::cache::auto_budget_bytes(2_500_000_000, (slot * 2) as i64, max_budget, (gb * 1e9) as i64);
        let avail = k3::cache::mem_available_bytes().unwrap_or(0);
        println!("auto arena: {:.2} GB (MemAvailable {:.2} GB - 2.5 GB margin, capped at the {:.1} GB weight set)",
                 b as f64 / 1e9, avail as f64 / 1e9, max_budget as f64 / 1e9);
        b
    } else {
        (gb * 1e9) as i64
    };
    let mut cache = k3::cache::Cache::new(budget, slot, n_exp, topk)?;
    println!("expert cache: {} slots of {:.2} MB ({:.1} GB budget)",
             cache.nslot(), slot as f64 / 1e6, budget as f64 / 1e9);
    // Prefetch (opt-in, Q35_PREFETCH): overlap the next layer's FFN read with this layer's
    // compute. MEASURED NET-NEGATIVE for the dense 27B on this machine — decode is disk-
    // BANDWIDTH-bound (3.1 GB/s, reads back-to-back), so a prefetch read only contends with
    // the critical read, and the Prefetcher's read-into-own-buffer-then-copy-into-slot adds a
    // 165 MB memcpy per layer that costs more than the overlap saves. Kept opt-in for configs
    // with a large arena (few misses -> idle disk during compute, where overlap could pay).
    if std::env::var_os("Q35_PREFETCH").is_some() {
        match k3::cache::Prefetcher::new(std::path::Path::new(p), slot) {
            Ok(pf) => { cache.attach_prefetcher(pf); println!("prefetch: ON (layer L+1 overlaps L's compute)"); }
            Err(e) => eprintln!("prefetch: off ({e})"),
        }
    }

    let tok = k3::tok::Tok::from_gguf(st.meta.as_ref().unwrap())?;
    let ids = tok.encode(&prompt, false)?;
    println!("prompt {:?} -> {} tokens", prompt, ids.len());

    let mut sess = k3::qwen35run::Session::new(&tr);
    let mut logits = vec![0f32; tr.cfg.vocab];
    let mut text = String::new();
    let tg = std::time::Instant::now();
    // Prefill one token at a time: correct, and the honest baseline before any batching.
    for (i, id) in ids.iter().enumerate() {
        let s = std::time::Instant::now();
        k3::qwen35run::step(&tr, &mut sess, &st, &mut cache, *id, &mut logits)?;
        if std::env::var_os("Q35_SUMS").is_some() {
            let ls: f64 = logits.iter().map(|v| *v as f64).sum();
            eprintln!("== after prompt token {i}: logits sum {ls:.6}");
        }
        if i + 1 == ids.len() {
            println!("  prefill {} tokens done ({:.2} s for the last)", ids.len(),
                     s.elapsed().as_secs_f64());
        }
    }
    for t in 0..n {
        let id = logits.iter().enumerate()
            .max_by(|x, y| x.1.total_cmp(y.1)).map(|(i, _)| i as u32).unwrap();
        if id == tok.eos { println!("  [eos]"); break; }
        let piece = tok.piece(id)?;
        text.push_str(&piece);
        // Time the STEP, not the argmax before it. The first version measured the wrong
        // side of the call and reported 0.00 s a token.
        let s = std::time::Instant::now();
        k3::qwen35run::step(&tr, &mut sess, &st, &mut cache, id, &mut logits)?;
        println!("  token {t}: {:?} (id {id}) in {:.2} s", piece, s.elapsed().as_secs_f64());
        if !logits.iter().all(|v| v.is_finite()) { return Err(format!("step {t}: non-finite")); }
    }
    println!("\n=== {:?}{:?}\n{} tokens total in {:.1} s", prompt, text,
             ids.len() + n, tg.elapsed().as_secs_f64());
    if std::env::var_os("Q35_CERT").is_some() {
        let (skipped, total) = k3::qwen35run::cert_report();
        if total > 0 {
            println!("certified sparsity: {skipped}/{total} FFN neurons skipped ({:.1}%)",
                     skipped as f64 / total as f64 * 100.0);
        }
    }
    Ok(())
}
