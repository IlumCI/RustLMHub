// Full speculative decoder for the dense qwen35, using the native MTP head (block 64) as the
// drafter and a K=2 verify (`step_verify`) that reads the FFN ONCE for both tokens. Greedy
// verification makes it LOSSLESS: this binary decodes the same prompt both ways and ASSERTS the
// token sequences are identical, then reports the real end-to-end speedup.
//
//   specdec MODEL_DIR [n_tokens] [cache_gb|auto] [prompt]
use std::time::Instant;

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().fold((0usize, f32::MIN), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0 as u32
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let p = a.get(1).ok_or("usage: specdec MODEL_DIR [n_tokens] [cache_gb|auto] [prompt]")?;
    let n: usize = a.get(2).map_or(Ok(16), |v| v.parse()).map_err(|e| format!("{e}"))?;
    let gb_arg = a.get(3).cloned().unwrap_or_else(|| "auto".into());
    let prompt = a.get(4).cloned().unwrap_or_else(|| "Write a short story about a lighthouse keeper who discovers something unusual.".into());

    let st = k3::st::St::open(std::path::Path::new(p)).map_err(|e| e.to_string())?;
    let meta = st.meta.as_ref().ok_or("no gguf metadata")?;
    let cfg = k3::qwen35::Cfg::from_meta(meta)?;
    if !cfg.is_dense() {
        return Err("specdec targets a DENSE qwen35 with an MTP head".into());
    }
    let tr = k3::qwen35run::Trunk::load(&st, &cfg, 4096)?;
    let mtp = k3::qwen35run::MtpHead::load(&st, &cfg)?;
    let hid = cfg.hidden;
    let vocab = tr.io.vocab;
    eprintln!("loaded trunk {:.2} GB + MTP head; decoding {n} tokens", tr.bytes as f64 / 1e9);

    let mut slot = 0usize;
    for l in 0..cfg.n_layers {
        let r = k3::cache::locate(&st, &k3::cache::gguf_dense_ffn_src(l, 0)).ok_or("locate")?;
        slot = slot.max(k3::cache::slot_need(&r));
    }
    let budget: i64 = if gb_arg.eq_ignore_ascii_case("auto") {
        k3::cache::auto_budget_bytes(2_500_000_000, (slot * 2) as i64, (slot * cfg.n_layers) as i64, 5_000_000_000)
    } else {
        (gb_arg.parse::<f64>().map_err(|e| format!("{e}"))? * 1e9) as i64
    };
    let mut cache = k3::cache::Cache::new(budget, slot, 1, 1)?;

    let tok = k3::tok::Tok::from_gguf(meta)?;
    let ids = tok.encode(&prompt, false)?;
    let plen = ids.len();
    let mut logits = vec![0f32; vocab];

    // ---------- baseline greedy decode ----------
    let mut sess = k3::qwen35run::Session::new(&tr);
    let h0 = k3::qwen35run::final_hiddens(&tr, &mut sess, &st, &mut cache, &ids)?;
    tr.io.logits(&h0[(plen - 1) * hid..][..hid], &mut logits)?;
    let mut g = argmax(&logits);
    let t0 = Instant::now();
    let mut base = Vec::with_capacity(n);
    for _ in 0..n {
        base.push(g);
        k3::qwen35run::step(&tr, &mut sess, &st, &mut cache, g, &mut logits)?;
        g = argmax(&logits);
    }
    let base_dt = t0.elapsed().as_secs_f64();

    // ---------- speculative decode ----------
    let mut sess = k3::qwen35run::Session::new(&tr);
    let h0 = k3::qwen35run::final_hiddens(&tr, &mut sess, &st, &mut cache, &ids)?;
    let mut h = h0[(plen - 1) * hid..][..hid].to_vec();
    tr.io.logits(&h, &mut logits)?;
    let mut g = argmax(&logits);
    let t0 = Instant::now();
    let mut spec = Vec::with_capacity(n + 1);
    let (mut rounds, mut accepted, mut draft_s) = (0usize, 0usize, 0f64);
    while spec.len() < n {
        let td = Instant::now();
        let d = mtp.draft_many(&tr.io, &cfg, &tr.rope, &h, &[g])?[0];
        draft_s += td.elapsed().as_secs_f64();
        let (xs2, ckpt) = k3::qwen35run::step_verify(&tr, &mut sess, &st, &mut cache, g, d)?;
        rounds += 1;
        spec.push(g); // the main model's own token, always accepted
        tr.io.logits(&xs2[0..hid], &mut logits)?;
        let m1 = argmax(&logits); // the main model's true next token after g
        if d == m1 {
            accepted += 1;
            spec.push(d);
            h = xs2[hid..2 * hid].to_vec();
            tr.io.logits(&h, &mut logits)?;
            g = argmax(&logits);
        } else {
            k3::qwen35run::spec_rollback(&tr, &mut sess, &ckpt);
            h = xs2[0..hid].to_vec();
            g = m1;
        }
    }
    let spec_dt = t0.elapsed().as_secs_f64();
    spec.truncate(n);

    // ---------- verdict ----------
    let identical = spec == base;
    println!("\noutput: {:?}", tok.decode(&base, true).unwrap_or_default());
    println!("\nLOSSLESS CHECK: spec tokens {} baseline tokens", if identical { "==" } else { "!= (BUG)" });
    if !identical {
        for i in 0..n {
            if spec[i] != base[i] {
                println!("  first divergence at token {i}: spec {} != base {}", spec[i], base[i]);
                break;
            }
        }
    }
    println!("\nacceptance : {accepted}/{rounds} rounds = {:.1}%  ({:.2} tokens/round)",
             100.0 * accepted as f64 / rounds as f64, n as f64 / rounds as f64);
    println!("baseline   : {:.2}s = {:.3} s/token", base_dt, base_dt / n as f64);
    println!("speculative: {:.2}s = {:.3} s/token   (draft overhead {:.2}s total = {:.3} s/round)",
             spec_dt, spec_dt / n as f64, draft_s, draft_s / rounds as f64);
    println!("\nSPEEDUP    : {:.2}x", base_dt / spec_dt);
    if !identical {
        return Err("speculative output diverged from greedy — the decoder is not lossless".into());
    }
    Ok(())
}
