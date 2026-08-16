// End-to-end depth-1 LoRA fine-tune of a DENSE qwen35 (e.g. Qwen3.8-27B) on a prepared
// JSONL dataset. This is the working proof that the streamed training path runs on the real
// checkpoint: it streams every frozen weight off disk exactly as inference does, adapts the
// top block's feed-forward with a LoRA adapter, and backprops through the streamed head and
// FFN (ops::wt / out_prod_q) -- the two novel training kernels -- with Adam updating only the
// adapter. The loss printed is the mean cross-entropy over RESPONSE tokens; if the machine is
// correct it falls.
//
// It stops the backward at the top FFN, so it needs no attention backward (the gated
// delta-net backward is not written yet). That makes it a real but shallow fine-tune -- the
// scaffold every deeper configuration extends, not the final training run.
//
//   train_run MODEL_DIR prepared.jsonl [--limit N] [--rank R] [--lr F] [--accum A]
//             [--epochs E] [--cache-gb G] [--max-seq S] [--out adapter.loaa]
use k3::train::{self, FfnLora};

fn arg(a: &[String], key: &str) -> Option<String> {
    a.iter().position(|x| x == key).and_then(|i| a.get(i + 1)).cloned()
}
fn argf<T: std::str::FromStr>(a: &[String], key: &str, def: T) -> T {
    arg(a, key).and_then(|v| v.parse().ok()).unwrap_or(def)
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let model = a.get(1).ok_or("usage: train_run MODEL_DIR prepared.jsonl [opts]")?;
    let data = a.get(2).ok_or("usage: train_run MODEL_DIR prepared.jsonl [opts]")?;
    let limit: usize = argf(&a, "--limit", usize::MAX);
    let rank: usize = argf(&a, "--rank", 32);
    let lr: f32 = argf(&a, "--lr", 1e-4);
    let accum: usize = argf(&a, "--accum", 8);
    let epochs: usize = argf(&a, "--epochs", 1);
    let cache_gb: f64 = argf(&a, "--cache-gb", 4.0);
    let max_seq: usize = argf(&a, "--max-seq", 1024);
    let out = arg(&a, "--out").unwrap_or_else(|| "adapter.loaa".into());

    // ---- load the frozen model, exactly as inference does ----
    let st = k3::st::St::open(std::path::Path::new(model)).map_err(|e| e.to_string())?;
    let meta = st.meta.as_ref().ok_or("no gguf metadata")?;
    let cfg = k3::qwen35::Cfg::from_meta(meta)?;
    if !cfg.is_dense() {
        return Err("train_run targets DENSE qwen35 (the delta-net/expert backward is not written yet)".into());
    }
    let t0 = std::time::Instant::now();
    let tr = k3::qwen35run::Trunk::load(&st, &cfg, max_seq.max(64))?;
    println!("loaded {:.2} GB trunk in {:.1} s | {} layers, hidden {}, ffn {}",
             tr.bytes as f64 / 1e9, t0.elapsed().as_secs_f64(), cfg.n_layers, cfg.hidden, cfg.dense_inter);

    // slot sizing for the streamed FFN cache (max over layers -- mixed quant)
    let mut slot = 0usize;
    for l in 0..cfg.n_layers {
        let r = k3::cache::locate(&st, &k3::cache::gguf_dense_ffn_src(l, 0))
            .ok_or_else(|| format!("cannot locate layer {l} feed-forward"))?;
        slot = slot.max(k3::cache::slot_need(&r));
    }
    let mut cache = k3::cache::Cache::new((cache_gb * 1e9) as i64, slot, 1, 1)?;
    println!("expert cache: {} slots of {:.2} MB", cache.nslot(), slot as f64 / 1e6);

    let tok = k3::tok::Tok::from_gguf(meta)?;
    let tmpl = k3::chat::Template::from_gguf(meta);
    if !tmpl.is_jinja() {
        return Err("model has no chat template".into());
    }

    // ---- data: detect the format, normalise every row, tokenise ----
    // Any shape works — chat messages, system/user/response, instruction/input/output,
    // prompt/completion, or raw text — auto-detected (override with --format).
    let force = arg(&a, "--format").and_then(|f| match f.as_str() {
        "chat" => Some(k3::dataset::Fmt::Chat),
        "sysuser" => Some(k3::dataset::Fmt::SysUserResp),
        "instruction" => Some(k3::dataset::Fmt::Instruction),
        "prompt" => Some(k3::dataset::Fmt::PromptCompletion),
        "text" => Some(k3::dataset::Fmt::TextOnly),
        _ => None,
    });
    let (fmt, rows) = k3::dataset::load(data, force)?;
    let mut examples = Vec::new();
    for r in &rows {
        if let Ok(ex) = train::example_from_record(&tmpl, &tok, r) {
            if ex.scored_tokens() > 0 && ex.ids.len() <= max_seq {
                examples.push(ex);
            }
        }
    }
    println!("format: {} | usable examples: {} (of {} rows, capped at {max_seq} tokens)",
             fmt.label(), examples.len(), rows.len());
    if examples.is_empty() {
        return Err("no usable examples".into());
    }

    // ---- the adapter: LoRA on the last layer's gate/up/down ----
    let mut ffn = FfnLora::new(cfg.hidden, cfg.dense_inter, rank, (2 * rank) as f32, 0xC0FFEE);
    println!("adapter: rank {rank} on the top FFN, {:.2} MB resident (params + Adam + grad)",
             ffn.gate.resident_bytes() as f64 * 3.0 / 1e6);

    // deterministic shuffle: a fixed permutation via an LCG walk over indices.
    let permute = |n: usize, seed: u64| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..n).collect();
        let mut s = seed | 1;
        for i in (1..n).rev() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (s >> 33) as usize % (i + 1);
            idx.swap(i, j);
        }
        idx
    };

    // ---- train ----
    let mut step_t = 0u64;
    let mut seen = 0usize;
    let mut accum_loss = 0f64;
    let mut accum_n = 0usize;
    ffn.zero_grad();
    let start = std::time::Instant::now();

    // Frozen-feature cache: only the top FFN trains, so `h` (the frozen stack's output) is
    // identical every epoch for the same example. Compute it ONCE, reuse it on later epochs,
    // skipping the expensive 64-layer forward. Only worth it across epochs, and bounded so a
    // large set degrades gracefully to recompute instead of swapping.
    let fcache_budget: usize = argf(&a, "--fcache-gb", if epochs > 1 { 2 } else { 0 }) * 1_000_000_000;
    let mut hcache: std::collections::HashMap<usize, Vec<f32>> = std::collections::HashMap::new();
    let mut hcache_bytes = 0usize;
    let mut cache_hits = 0u64;

    for epoch in 0..epochs {
        let order = permute(examples.len(), 0xA11CE + epoch as u64);
        for (bi, &ei) in order.iter().enumerate() {
            if seen >= limit {
                break;
            }
            let ex = &examples[ei];
            let (input, target, mask) = ex.view();
            let es = std::time::Instant::now();
            let loss = if let Some(h) = hcache.get(&ei) {
                cache_hits += 1;
                k3::qwen35run::train_lastffn_from_hidden(&tr, &st, &mut cache, h, &target, &mask, &mut ffn)?
            } else {
                let mut sess = k3::qwen35run::Session::new(&tr);
                let h = k3::qwen35run::frozen_hidden(&tr, &mut sess, &st, &mut cache, &input)?;
                let loss = k3::qwen35run::train_lastffn_from_hidden(&tr, &st, &mut cache, &h, &target, &mask, &mut ffn)?;
                let hb = h.len() * 4;
                if fcache_budget > 0 && hcache_bytes + hb <= fcache_budget {
                    hcache_bytes += hb;
                    hcache.insert(ei, h);
                }
                loss
            };
            accum_loss += loss as f64 * ex.scored_tokens() as f64;
            accum_n += ex.scored_tokens();
            seen += 1;

            if seen % accum == 0 {
                step_t += 1;
                ffn.adam_step(lr, 0.9, 0.999, 1e-8, step_t, accum as f32);
                ffn.zero_grad();
                let mean = accum_loss / accum_n.max(1) as f64;
                let (h, m, e) = cache_stats(&cache);
                println!(
                    "epoch {epoch} step {step_t:>4} (ex {}/{})  loss {mean:6.3}  \
                     {:.1}s/ex  fcache {cache_hits}hit  expert {h}/{m} ev {e}",
                    bi + 1, order.len(), es.elapsed().as_secs_f64()
                );
                accum_loss = 0.0;
                accum_n = 0;
                // Checkpoint every step so a long or interrupted run never loses the adapter.
                if let Err(e) = ffn.save(&out) {
                    eprintln!("checkpoint save failed: {e}");
                }
            }
        }
        if seen >= limit {
            break;
        }
    }

    ffn.save(&out)?;
    println!("saved adapter -> {out}  ({} examples in {:.1}s)", seen, start.elapsed().as_secs_f64());
    Ok(())
}

fn cache_stats(c: &k3::cache::Cache) -> (u64, u64, u64) {
    (c.hits, c.misses, c.evictions)
}
