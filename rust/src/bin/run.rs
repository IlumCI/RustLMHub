// SPDX-License-Identifier: Apache-2.0
//
// usage: run MODEL_DIR [--tokenizer F] [--prompt S | --ids a,b,c] [--gen N]
//            [--cache-gb G] [--trunk DIR] [--trunk-gb G] [--incremental] [--out F]
//
// One binary for every architecture. The model directory's own config.json decides which,
// so nothing here is per-model except the descriptor it produces.

use std::path::PathBuf;
use std::process::ExitCode;

use k3::arch::{self, Family, Spec};
use k3::st::St;

struct Args {
    model: PathBuf,
    tokenizer: Option<PathBuf>,
    prompt: Option<String>,
    ids: Option<Vec<u32>>,
    gen: usize,
    cache_gb: f64,
    trunk: Option<PathBuf>,
    trunk_gb: f64,
    incremental: bool,
    out: Option<PathBuf>,
    dry: bool,
    trace: Option<PathBuf>,
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        model: PathBuf::new(),
        tokenizer: None,
        prompt: None,
        ids: None,
        gen: 16,
        cache_gb: 4.0,
        trunk: None,
        trunk_gb: 4.0,
        incremental: false,
        out: None,
        dry: false,
        trace: None,
    };
    let v: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < v.len() {
        let need = |i: usize| -> Result<String, String> {
            v.get(i + 1).cloned().ok_or_else(|| format!("{} needs a value", v[i]))
        };
        match v[i].as_str() {
            "--tokenizer" | "--tok" => { a.tokenizer = Some(need(i)?.into()); i += 2 }
            "--prompt" => { a.prompt = Some(need(i)?); i += 2 }
            "--ids" => {
                a.ids = Some(
                    need(i)?
                        .split(',')
                        .map(|s| s.trim().parse().map_err(|_| "--ids wants integers".to_string()))
                        .collect::<Result<_, _>>()?,
                );
                i += 2
            }
            "--gen" | "-n" => { a.gen = need(i)?.parse().map_err(|_| "--gen wants a number")?; i += 2 }
            "--cache-gb" => { a.cache_gb = need(i)?.parse().map_err(|_| "--cache-gb")?; i += 2 }
            "--trunk" => { a.trunk = Some(need(i)?.into()); i += 2 }
            "--trunk-gb" => { a.trunk_gb = need(i)?.parse().map_err(|_| "--trunk-gb")?; i += 2 }
            "--incremental" => { a.incremental = true; i += 1 }
            "--out" => { a.out = Some(need(i)?.into()); i += 2 }
            "--dry-run" => { a.dry = true; i += 1 }
            "--dump-cache-trace" => { a.trace = Some(need(i)?.into()); i += 2 }
            s if s.starts_with('-') => return Err(format!("unknown flag {s}")),
            s => { a.model = s.into(); i += 1 }
        }
    }
    if a.model.as_os_str().is_empty() {
        return Err("a model directory is required".into());
    }
    Ok(a)
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("run: {e}");
            ExitCode::FAILURE
        }
    }
}

/// A missing layer reads as zeros and the model still emits fluent text, so an incomplete
/// checkpoint is refused by name rather than run. Reported all at once.
fn missing(st: &St, s: &Spec) -> Vec<String> {
    let mut out = Vec::new();
    for t in [s.embed(), s.final_norm(), s.head()] {
        if st.find(&t).is_none() {
            out.push(t);
        }
    }
    for l in 0..s.n_layers {
        // One probe per ROUTED layer: the first expert's packed weight. A layer whose
        // shard never arrived fails this, and checking every tensor of every layer would
        // print 40,000 lines for the same fact. Dense layers have no experts to probe.
        if s.is_dense(l) {
            continue;
        }
        let n = s.expert_names(l, 0);
        if st.find(n.probe_name()).is_none() {
            out.push(n.probe_name().to_string());
        }
    }
    out
}

fn run() -> Result<(), String> {
    let a = parse().map_err(|e| {
        format!(
            "{e}\nusage: run MODEL_DIR [--tokenizer F] [--prompt S | --ids a,b,c] [--gen N]\n\
             \x20            [--cache-gb G] [--trunk DIR] [--trunk-gb G] [--incremental]"
        )
    })?;

    let cfg = a.model.join("config.json");
    let spec = arch::spec_file(&cfg)?;
    println!("{}", spec.summary(&cfg.display().to_string()));

    let st = St::open(&a.model).map_err(|e| e.to_string())?;
    println!("checkpoint: {} shards, {} tensors", st.nshard(), st.tensors.len());

    let miss = missing(&st, &spec);
    if !miss.is_empty() {
        eprintln!(
            "\nrun: this checkpoint is incomplete: {} of the required tensors are absent.",
            miss.len()
        );
        for m in miss.iter().take(8) {
            eprintln!("    {m}");
        }
        if miss.len() > 8 {
            eprintln!("    ... and {} more", miss.len() - 8);
        }
        return Err(
            "refusing to generate. A missing layer reads as zeros and the model still \
             emits fluent, wrong text -- there is no error to notice at run time."
                .into(),
        );
    }
    println!("checkpoint is complete: all {} layers present", spec.n_layers);

    if a.dry {
        let one = spec.expert_bytes(4, 32) as f64 / 1e6;
        let gb = spec.expert_bytes_per_token(4, 32) as f64 / 1e9;
        println!("one routed expert (MXFP4 g32, with scales): {one:.2} MB");
        println!("routed-expert traffic per token: {gb:.2} GB");
        return Ok(());
    }

    match spec.family {
        Family::K3 => Err("K3 runs through the k3_run binary, which carries its trunk \
                           streaming and tiktoken loader"
            .into()),
        Family::V4 => {
            let params = k3::v4run::Params::from_env(a.gen, a.cache_gb);
            // The CLI's sink is the old inline print, moved out of the engine so a server
            // can supply its own without the loop knowing where the token goes.
            let mut sink = |_id: u32, piece: &str| {
                use std::io::Write as _;
                print!("{piece}");
                let _ = std::io::stdout().flush();
                // The CLI never stops early; a server returns false on a stop sequence or
                // a hung-up client.
                true
            };
            k3::v4run::generate(&st, &spec, &a.model, a.tokenizer.as_deref(),
                                a.prompt.as_deref(), a.ids.as_deref(), &params,
                                a.out.as_deref(), a.trace.as_deref(), &mut sink)
        }
        f => Err(format!(
            "{}: the descriptor parses and the checkpoint validates, but its layer \
             kernels are not implemented yet",
            f.as_str()
        )),
    }
}
