// Diff the COMMON MoE block against llama.cpp on a synthetic qwen3moe fixture.
fn main() -> Result<(), String> {
    let p = std::env::args().nth(1).ok_or("need a model dir")?;
    let st = k3::st::St::open(std::path::Path::new(&p)).map_err(|e| e.to_string())?;
    let cfg = k3::moegen::Cfg::from_meta(st.meta.as_ref().ok_or("no meta")?)?;
    println!("arch {} | {} blocks hidden {} | {}h/{}kv x {} | {} experts top-{} | qk_norm {} shexp {}",
             cfg.arch.name, cfg.n_layers, cfg.hidden, cfg.n_heads, cfg.n_kv_heads,
             cfg.head_dim, cfg.n_experts, cfg.topk, cfg.arch.qk_norm, cfg.arch.shared_expert);
    let tr = k3::moegen::Trunk::load(&st, &cfg, 4096)?;
    let mut slot = 0usize;
    for l in 0..cfg.n_layers {
        let r = k3::cache::locate(&st, &k3::cache::gguf_expert_src(l, 0)).ok_or("locate")?;
        slot = slot.max(k3::cache::slot_need(&r));
    }
    let mut cache = k3::cache::Cache::new(200_000_000, slot, cfg.n_experts, cfg.topk)?;
    let mut s = k3::moegen::Session::new(&tr);
    let mut logits = vec![0f32; tr.cfg.vocab];
    for id in std::env::args().nth(2).unwrap_or_else(|| "32".into())
        .split(',').filter_map(|v| v.trim().parse::<u32>().ok()) {
        k3::moegen::step(&tr, &mut s, &st, &mut cache, id, &mut logits)?;
    }
    let sum: f64 = logits.iter().map(|v| *v as f64).sum();
    println!("logits sum {sum:.6}");
    if !logits.iter().all(|v| v.is_finite()) { return Err("non-finite logits".into()); }
    Ok(())
}
