// Stage-0 MoEfication capture: run the dense qwen35 over a calibration corpus and dump, for a
// chosen set of layers, per-token FFN inputs `hs` and per-neuron contributions
// `c_j = |silu(gate·hs)_j · (up·hs)_j| · ‖down_col_j‖₂` — the artefacts the OFFLINE go/no-go
// (oracle top-k recall + arena-fit simulation) consumes. This is a MEASUREMENT pass: it streams
// every frozen weight exactly as inference does, changes nothing about the model, and produces
// no logits — only `{out}/layer_{l}_c.f32`, `layer_{l}_hs.f32`, and `meta.json`.
//
// Each calibration document/window is processed as an INDEPENDENT context (fresh session, pos 0),
// so the captured activations reflect realistic per-position context rather than one giant
// correlated sequence.
//
//   moefy_capture MODEL_DIR calib.txt [--layers 0,16,32,48,63] [--max-tokens 4000]
//                 [--seq-len 512] [--cache-gb 4] [--out DIR]
use std::io::Write;

fn arg(a: &[String], k: &str) -> Option<String> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned()
}
fn argf<T: std::str::FromStr>(a: &[String], k: &str, d: T) -> T {
    arg(a, k).and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let model = a.get(1).ok_or("usage: moefy_capture MODEL_DIR calib.txt [opts]")?;
    let calib = a.get(2).ok_or("usage: moefy_capture MODEL_DIR calib.txt [opts]")?;
    let layers: Vec<usize> = arg(&a, "--layers")
        .unwrap_or_else(|| "0,16,32,48,63".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let max_tokens: usize = argf(&a, "--max-tokens", 4000);
    let seq_len: usize = argf(&a, "--seq-len", 512);
    let cache_gb: f64 = argf(&a, "--cache-gb", 4.0);
    let out = arg(&a, "--out").unwrap_or_else(|| "moefy_capture".into());

    // ---- load the frozen model, exactly as inference does ----
    let st = k3::st::St::open(std::path::Path::new(model)).map_err(|e| e.to_string())?;
    let meta = st.meta.as_ref().ok_or("no gguf metadata")?;
    let cfg = k3::qwen35::Cfg::from_meta(meta)?;
    if !cfg.is_dense() {
        return Err("moefy_capture targets a DENSE qwen35 (e.g. Qwen3.8-27B)".into());
    }
    for &l in &layers {
        if l >= cfg.n_layers {
            return Err(format!("--layers: {l} >= n_layers {}", cfg.n_layers));
        }
    }
    let t0 = std::time::Instant::now();
    let tr = k3::qwen35run::Trunk::load(&st, &cfg, seq_len.max(64))?;
    eprintln!(
        "loaded {:.2} GB trunk in {:.1}s | {} layers, hidden {}, inter {}",
        tr.bytes as f64 / 1e9, t0.elapsed().as_secs_f64(), cfg.n_layers, cfg.hidden, cfg.dense_inter
    );

    // slot sizing for the streamed FFN cache (max over layers — mixed quant)
    let mut slot = 0usize;
    for l in 0..cfg.n_layers {
        let r = k3::cache::locate(&st, &k3::cache::gguf_dense_ffn_src(l, 0))
            .ok_or_else(|| format!("cannot locate layer {l} feed-forward"))?;
        slot = slot.max(k3::cache::slot_need(&r));
    }
    let mut cache = k3::cache::Cache::new((cache_gb * 1e9) as i64, slot, 1, 1)?;
    eprintln!("expert cache: {} slots of {:.2} MB", cache.nslot(), slot as f64 / 1e6);

    let tok = k3::tok::Tok::from_gguf(meta)?;
    let text = std::fs::read_to_string(calib).map_err(|e| format!("read {calib}: {e}"))?;
    let ids = tok.encode(&text, false)?;
    let goal = max_tokens.min(ids.len());
    eprintln!(
        "calibration: {} chars -> {} tokens; capturing {} across layers {:?} (seq_len {})",
        text.len(), ids.len(), goal, layers, seq_len
    );
    if ids.is_empty() {
        return Err("calibration corpus tokenised to nothing".into());
    }

    let dir = std::path::PathBuf::from(&out);
    let mut sink = k3::qwen35run::MoefyCapture::new(&dir, &layers, max_tokens)?;

    let start = std::time::Instant::now();
    let mut pos = 0usize;
    while pos < ids.len() && !sink.full() {
        let end = (pos + seq_len).min(ids.len());
        let chunk = &ids[pos..end];
        let mut sess = k3::qwen35run::Session::new(&tr);
        k3::qwen35run::capture_ffn_stats(&tr, &mut sess, &st, &mut cache, chunk, &mut sink)?;
        pos = end;
        eprintln!("  captured {} / {} tokens ({:.1}s)", sink.captured(), goal, start.elapsed().as_secs_f64());
    }
    sink.finish()?;

    // meta.json so the offline analysis reads the shapes without guessing.
    let meta_json = format!(
        "{{\n  \"n_captured\": {},\n  \"n_layers\": {},\n  \"hidden\": {},\n  \"inter\": {},\n  \"layers\": [{}],\n  \"seq_len\": {},\n  \"dtype\": \"f32-le\",\n  \"model\": {:?}\n}}\n",
        sink.captured(),
        cfg.n_layers,
        cfg.hidden,
        cfg.dense_inter,
        layers.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(", "),
        seq_len,
        model
    );
    std::fs::File::create(dir.join("meta.json"))
        .and_then(|mut f| f.write_all(meta_json.as_bytes()))
        .map_err(|e| e.to_string())?;
    eprintln!(
        "done: {} tokens captured in {:.1}s -> {}",
        sink.captured(), start.elapsed().as_secs_f64(), dir.display()
    );
    Ok(())
}
