// Measure the native MTP head's acceptance rate: how often does the block-64 next-n head's
// 2-ahead draft equal what the main 27B model itself greedily produces? That fraction, times
// the K-wide-forward ceiling proven by `readprof`, is the real speculative-decode speedup.
//
// Teacher-forced over a coherent passage (one batched forward), which is a fast proxy for the
// deployment acceptance. `draft[t]` predicts token t+2; the main model's greedy for t+2 is
// `main_greedy[t+1]`. Also reports agreement with the real text (a sanity/quality view).
//
//   mtpaccept MODEL_DIR [cache_gb|auto] [passage-file]
fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let p = a.get(1).ok_or("usage: mtpaccept MODEL_DIR [cache_gb|auto] [passage-file]")?;
    let gb_arg = a.get(2).cloned().unwrap_or_else(|| "auto".into());

    let default_passage = "The city of Venice is built on a group of small islands separated \
        by canals and linked by bridges. It has no roads, only canals, and people travel by \
        boat or on foot. The main waterway, the Grand Canal, winds through the centre of the \
        city. Venice was once a powerful maritime republic and a major centre of trade between \
        Europe and the East. Today it is one of the most visited cities in the world, famous \
        for its architecture, its art, and its annual carnival. Rising sea levels and the \
        weight of tourism now threaten its fragile foundations, and engineers have built a \
        system of mobile barriers to hold back the highest tides.";
    let passage = match a.get(3) {
        Some(f) => std::fs::read_to_string(f).map_err(|e| format!("read {f}: {e}"))?,
        None => default_passage.to_string(),
    };

    let st = k3::st::St::open(std::path::Path::new(p)).map_err(|e| e.to_string())?;
    let meta = st.meta.as_ref().ok_or("no gguf metadata")?;
    let cfg = k3::qwen35::Cfg::from_meta(meta)?;
    if !cfg.is_dense() {
        return Err("mtpaccept targets a DENSE qwen35 with an MTP head".into());
    }
    let tr = k3::qwen35run::Trunk::load(&st, &cfg, 4096)?;
    let mtp = k3::qwen35run::MtpHead::load(&st, &cfg)?;
    eprintln!("loaded trunk {:.2} GB + MTP head", tr.bytes as f64 / 1e9);

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

    let tok = k3::tok::Tok::from_gguf(meta)?;
    let ids = tok.encode(&passage, false)?;
    let l = ids.len();
    if l < 8 {
        return Err("passage too short".into());
    }
    eprintln!("passage: {l} tokens; running the main model once...");

    // main model: per-position final hidden, then per-position greedy token.
    let mut sess = k3::qwen35run::Session::new(&tr);
    let t0 = std::time::Instant::now();
    let hidden = k3::qwen35run::final_hiddens(&tr, &mut sess, &st, &mut cache, &ids)?;
    let hid = cfg.hidden;
    let vocab = tr.io.vocab;
    let mut main_greedy = vec![0u32; l];
    let mut logits = vec![0f32; vocab];
    for t in 0..l {
        tr.io.logits(&hidden[t * hid..][..hid], &mut logits)?;
        main_greedy[t] = logits.iter().enumerate().fold((0usize, f32::MIN), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) }).0 as u32;
    }
    eprintln!("main forward + head: {:.1}s", t0.elapsed().as_secs_f64());

    // MTP drafts: position t uses hidden_t + emb(ids[t+1]) to predict token t+2.
    let n = l - 1; // positions 0..l-2 have a t+1 token
    let draft = mtp.draft_many(&tr.io, &cfg, &tr.rope, &hidden[..n * hid], &ids[1..=n])?;

    // Acceptance: draft[t] (predicts t+2) vs main_greedy[t+1] (main's greedy for t+2).
    let (mut accept, mut real_match, mut total) = (0usize, 0usize, 0usize);
    for t in 0..n.saturating_sub(1) {
        total += 1;
        if draft[t] == main_greedy[t + 1] {
            accept += 1;
        }
        if (t + 2) < l && draft[t] == ids[t + 2] {
            real_match += 1;
        }
    }
    let order = std::env::var("MTP_ORDER").unwrap_or_else(|_| "eh".into());
    println!("\nMTP head acceptance  (eh_proj order = {order}; set MTP_ORDER=he to flip)");
    println!("  samples                : {total}");
    println!("  draft == main greedy   : {accept} / {total} = {:.1}%   <-- speculative-decode acceptance", 100.0 * accept as f64 / total as f64);
    println!("  draft == real next tok : {real_match} / {total} = {:.1}%   (quality vs the passage)", 100.0 * real_match as f64 / total as f64);

    // Translate to a speedup, using readprof's measured K=2 verify cost (3.7s) vs 2.55s decode.
    let a = accept as f64 / total as f64;
    let sp = 2.55 * (1.0 + a) / 3.70;
    println!("\n  With 1 MTP step (draft 1 extra token, verify K=2):");
    println!("    tokens/round = 1 + {:.2} = {:.2};  time/round ~= 3.70s (K=2 forward)", a, 1.0 + a);
    println!("    -> ~{:.2}x vs 2.55s/token baseline (accept>0.49 to beat 1.0x; chaining more MTP steps raises the ceiling toward the K=4/8 numbers)", sp);
    Ok(())
}
