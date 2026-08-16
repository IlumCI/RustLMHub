// SPDX-License-Identifier: Apache-2.0
//
// rustlm -- install a model, ask what the engine can do with it, serve it.
//
// The point of this binary is that none of it depends on the maintainer. `inspect` reads
// the checkpoint's own bytes and reports what is missing by name; `add` refuses to record
// a model it has not inspected; `serve` puts the result behind the one HTTP interface
// every local-model client already speaks. Filenames lie -- a `Q4_K_M` build held 132 Q5_0
// tensors, a `Q2_K` build held no Q2_K at all, and a config advertised one MTP layer while
// three shipped -- so nothing here trusts a name for anything that matters.
//
//   rustlm inspect <path>            what this build can and cannot do with it
//   rustlm add <name> <path>         inspect, then register under a short name
//   rustlm list                      what is registered, and what is blocked
//   rustlm remove <name>
//   rustlm probe <name> [token_id]   smallest real forward path, for a GGUF model
//   rustlm serve [name] [--addr A] [--cache-gb G] [--max-tokens N]

use std::path::Path;
use std::process::ExitCode;

use k3::capability::Report;
use k3::registry::Registry;

fn usage() -> ExitCode {
    eprintln!(
        "usage:\n  \
         rustlm pull <repo> [name]          download a MoE model and register it\n  \
         rustlm inspect <path>              what this build can and cannot do with it\n  \
         rustlm add <name> <path>           inspect, then register under a short name\n  \
         rustlm list                        what is registered, and what is blocked\n  \
         rustlm remove <name>\n  \
         rustlm probe <name> [token_id]     smallest real forward path (gguf)\n  \
         rustlm serve [name] [options]      OpenAI-compatible HTTP server\n  \
         rustlm run <name> [--addr A]       interactive chat with a RUNNING server\n  \
         rustlm train                       fine-tune & eval workbench (interactive TUI)\n  \
         rustlm code [args...]              the RustLM Code TUI (separate binary)\n  \
         rustlm accel                       which optional accelerators are usable here\n\
         \n\
         pull options:\n  \
         --quant Q          which quantisation, e.g. Q4_K_M\n  \
         --dir PATH         where to put it, default ~/models\n  \
         --dry-run          resolve, read the header, report the plan; transfer nothing\n\
         \n\
         serve options:\n  \
         --addr HOST:PORT   default 127.0.0.1:11434 (Ollama's port, so clients find it)\n  \
         --cache-gb G|auto  expert cache budget in GB, or `auto` (default): the largest\n                     \
                     swap-safe size for the free RAM, re-measured each launch\n  \
         --max-tokens N     default cap per request, default 4096\n  \
         --max-ctx N        rope table size, default 16384\n  \
         --conv-gb G        cached conversation state, default 2 (makes turn 2 cheap)\n  \
         --prefill-width N  prompt tokens per chunk, default 0 = derive from the cache\n  \
         --no-int8          disable the int8 fast path (on by default; ~1.5x, near-bitwise)"
    );
    ExitCode::from(2)
}

/// `--flag value` pairs after the subcommand. Deliberately tiny: this binary has six
/// subcommands and no need for an argument-parsing dependency.
fn flag<'a>(a: &'a [String], name: &str) -> Option<&'a str> {
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).map(String::as_str)
}

fn num_flag<T: std::str::FromStr>(a: &[String], name: &str, dflt: T) -> Result<T, String> {
    match flag(a, name) {
        Some(v) => v.parse().map_err(|_| format!("{name}: {v:?} is not a number")),
        None => Ok(dflt),
    }
}

/// `--cache-gb` accepts a number or `auto`. `auto` (also the default when the flag is absent)
/// is passed down as `0.0`, the sentinel the model loaders size to swap-safe available RAM.
fn cache_gb_flag(a: &[String]) -> Result<f64, String> {
    match flag(a, "--cache-gb") {
        None => Ok(0.0),
        Some(s) if s.eq_ignore_ascii_case("auto") => Ok(0.0),
        Some(s) => s.parse().map_err(|_| format!("--cache-gb: {s:?} is not a number or \"auto\"")),
    }
}

/// Interactive chat REPL against a running server, like `ollama run`. Keeps the conversation
/// history so it is a real multi-turn chat, and `/bye` (or EOF) exits.
fn chat_repl(addr: &str, model: &str) -> std::io::Result<()> {
    use std::io::Write;
    println!("chatting with {model:?} at {addr}.  /bye or Ctrl-D to exit.\n");
    let mut history: Vec<serde_json::Value> = Vec::new();
    let stdin = std::io::stdin();
    loop {
        print!(">>> ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            println!();
            break; // EOF (Ctrl-D)
        }
        let msg = line.trim();
        if msg.is_empty() {
            continue;
        }
        if msg == "/bye" || msg == "/exit" || msg == "/quit" {
            break;
        }
        history.push(serde_json::json!({"role": "user", "content": msg}));
        match chat_turn(addr, model, &history) {
            Ok(reply) => history.push(serde_json::json!({"role": "assistant", "content": reply})),
            Err(e) => {
                eprintln!("\n(error: {e})");
                history.pop(); // drop the unanswered user turn so history stays consistent
            }
        }
    }
    Ok(())
}

/// One chat turn: POST the history to `/v1/chat/completions` (streaming), print the assistant
/// delta as it arrives, and return the full text. One connection per turn (the server sends
/// `Connection: close`), read until EOF.
fn chat_turn(addr: &str, model: &str, history: &[serde_json::Value]) -> std::io::Result<String> {
    use std::io::{BufRead, Write};
    let body = serde_json::json!({"model": model, "messages": history, "stream": true}).to_string();
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    let mut sock = std::net::TcpStream::connect(addr)?;
    sock.write_all(req.as_bytes())?;
    let reader = std::io::BufReader::new(sock);
    let mut lines = reader.lines();

    // status line + headers
    let status = lines.next().transpose()?.unwrap_or_default();
    if !status.contains("200") {
        // read the rest as the error body
        let mut err = status;
        for l in lines.by_ref() {
            err.push('\n');
            err.push_str(&l?);
        }
        return Err(std::io::Error::other(err.trim().to_string()));
    }
    for l in lines.by_ref() {
        let l = l?;
        if l.is_empty() || l == "\r" {
            break; // end of headers
        }
    }

    // SSE body: `data: {json}` frames, ending in `data: [DONE]`.
    let mut full = String::new();
    let mut out = std::io::stdout();
    for l in lines {
        let l = l?;
        let data = match l.strip_prefix("data: ") {
            Some(d) => d,
            None => continue,
        };
        if data == "[DONE]" {
            break;
        }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
            if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
                out.write_all(c.as_bytes())?;
                out.flush()?;
                full.push_str(c);
            }
        }
    }
    println!("\n");
    Ok(full)
}

fn run(a: &[String]) -> Result<bool, String> {
    match a.get(1).map(String::as_str) {
        Some("inspect") => {
            let p = a.get(2).ok_or("inspect needs a path")?;
            let r = Report::inspect(Path::new(p))?;
            print!("{r}");
            Ok(r.runnable())
        }
        Some("pull") => {
            // Install a model end to end: resolve the repo, READ ITS HEADER FIRST, refuse
            // anything that is not a Mixture-of-Experts family this build has verified,
            // then download resumably and register it.
            let repo = a.get(2).filter(|s| !s.starts_with("--"))
                .ok_or("pull needs a Hugging Face repo, e.g. rustlm pull owner/model --quant Q4_K_M")?;
            let quant = flag(a, "--quant");
            let dir = flag(a, "--dir").map(std::path::PathBuf::from)
                .unwrap_or_else(|| std::path::PathBuf::from(
                    std::env::var("HOME").unwrap_or_default()).join("models"));
            println!("resolving {repo} ...");
            let (plan, probe) = k3::fetch::plan(repo, quant, &dir)?;
            match probe.verdict() {
                Ok(arch) => println!(
                    "  {:?}: {} experts, {} expert tensors of {} -- runnable ({})",
                    probe.arch, probe.n_experts, probe.expert_tensors, probe.n_tensors,
                    if arch.uses_common_block() { "common MoE block" } else { "dedicated block" }),
                Err(e) => {
                    // Nothing has been downloaded past the header at this point.
                    return Err(format!("{e}
  (nothing beyond the header was downloaded)"));
                }
            }
            // Disk is checked only now, because the MoE verdict above is the cheaper
            // refusal and should come first. On a 62-part checkpoint the difference
            // between noticing here and noticing at part 61 is days of transfer.
            let space = k3::fetch::check_space(&plan);
            if a.iter().any(|s| s == "--dry-run") {
                // Everything above this line is free: one range request for part one's
                // header. Once a checkpoint can be 889 GB across 62 parts, "show me what
                // you would do" stops being a convenience.
                println!(
                    "dry run: {} part(s), {:.1} GB into {}",
                    plan.files.len(),
                    plan.bytes() as f64 / 1e9,
                    plan.dest.display()
                );
                match space {
                    Ok(()) => println!("  disk: enough free"),
                    Err(e) => println!("  disk: NOT enough -- {e}"),
                }
                return Ok(true);
            }
            space?;
            println!(
                "downloading {} to {} ...",
                if plan.files.len() > 1 {
                    format!("{} parts", plan.files.len())
                } else {
                    "1 file".into()
                },
                plan.dest.display()
            );
            k3::fetch::download(&plan)?;
            // Register it, which re-inspects the WHOLE file: the header check was a
            // 4 MB sample, and a filename is a claim, not a fact.
            let name = a.get(3).filter(|s| !s.starts_with("--")).cloned()
                .unwrap_or_else(|| repo.rsplit('/').next().unwrap_or(repo).to_lowercase());
            let mut reg = Registry::open();
            let r = reg.add(&name, &plan.dest)?;
            print!("{r}");
            println!("registered as {name:?} -- `rustlm serve {name}` to run it");
            Ok(r.runnable())
        }
        Some("add") => {
            let (name, p) = match (a.get(2), a.get(3)) {
                (Some(n), Some(p)) => (n, p),
                _ => return Err("add needs a name and a path".into()),
            };
            let mut reg = Registry::open();
            let r = reg.add(name, Path::new(p))?;
            print!("{r}");
            println!("registered as {name:?}");
            if !r.runnable() {
                // Recorded anyway: hiding it would only move the surprise to the first
                // generation attempt, which is a worse place to find out.
                println!("  (registered despite blockers -- `list` will keep showing them)");
            }
            Ok(r.runnable())
        }
        Some("probe") => {
            // Smallest real forward path: embed a token, final-norm it, run the vocabulary
            // head. No decoder layers, so the top tokens are not a prediction about
            // language -- they are proof that the container, the k-quant kernels and the
            // 21 GB checkpoint agree with each other on actual weights.
            let name = a.get(2).ok_or("probe needs a model name or path")?;
            let id: u32 = a.get(3).map(|s| s.parse()).transpose()
                .map_err(|e| format!("token id: {e}"))?.unwrap_or(0);
            let reg = Registry::open();
            let path = reg.resolve(name).ok_or_else(|| format!("no model {name:?}"))?;
            let st = k3::st::St::open(&path).map_err(|e| e.to_string())?;
            let meta = st.meta.as_ref().ok_or("probe currently reads gguf models")?;
            let cfg = k3::arch::gguf_to_json(meta)?;
            let g = |k: &str| cfg[k].as_u64().unwrap_or(0) as usize;
            let (hidden, vocab) = (g("hidden_size"), g("vocab_size"));
            let eps = cfg["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
            println!("hidden {hidden}  vocab {vocab}  rms_eps {eps:e}");

            let t0 = std::time::Instant::now();
            let io = k3::qwen35::Io::load(&st, hidden, vocab, eps)?;
            println!("io weights: {:.2} GB in {:.1} s", io.bytes() as f64 / 1e9,
                     t0.elapsed().as_secs_f64());

            let mut x = vec![0f32; hidden];
            io.embed_row(id, &mut x)?;
            let (mn, mx) = x.iter().fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
            let rms = (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / hidden as f64).sqrt();
            println!("embed[{id}]: min {mn:.5} max {mx:.5} rms {rms:.5}");

            let t1 = std::time::Instant::now();
            let mut logits = vec![0f32; vocab];
            io.logits(&x, &mut logits)?;
            let dt = t1.elapsed().as_secs_f64();
            println!("head: {vocab} logits in {dt:.3} s ({:.2} GMAC/s)",
                     vocab as f64 * hidden as f64 / dt / 1e9);
            let mut idx: Vec<usize> = (0..vocab).collect();
            idx.sort_by(|&i, &j| logits[j].total_cmp(&logits[i]));
            print!("top-8:");
            for &i in idx.iter().take(8) {
                print!(" {i}({:.3})", logits[i]);
            }
            println!();
            // All-zero or non-finite logits mean the weights were read wrong, and that is
            // the failure this probe exists to make loud rather than plausible.
            if !logits.iter().all(|v| v.is_finite()) {
                return Err("logits contain NaN or inf -- the head was read wrong".into());
            }
            if logits.iter().all(|v| *v == 0.0) {
                return Err("every logit is zero -- the head read no weights".into());
            }
            Ok(true)
        }
        Some("list") => {
            let reg = Registry::open();
            if reg.entries.is_empty() {
                println!("no models registered ({}/registry.json)", k3::registry::home().display());
                return Ok(true);
            }
            for e in &reg.entries {
                println!(
                    "{:<24} {:>8.1} GB  {:<12} {:<12} {}",
                    e.name,
                    e.bytes as f64 / 1e9,
                    e.format,
                    e.arch,
                    if e.runnable() { "runnable".to_string() } else { format!("{} blocker(s)", e.blockers.len()) }
                );
                for b in &e.blockers {
                    println!("    - {b}");
                }
            }
            Ok(true)
        }
        Some("code") => {
            // Dispatch to `rustlm-code` the way git dispatches to `git-*`: one command on
            // the user's path, two workspaces that stay separately buildable and, for the
            // fork, separately rebasable onto upstream.
            let exe = std::env::current_exe().ok();
            let sibling = exe.as_ref().and_then(|p| p.parent()).map(|d| d.join("rustlm-code"));
            let prog = match &sibling {
                // Prefer the one built next to this binary, so a working tree runs its own
                // build rather than whatever an older install left on PATH.
                Some(p) if p.exists() => p.clone(),
                _ => std::path::PathBuf::from("rustlm-code"),
            };
            match std::process::Command::new(&prog).args(&a[2..]).status() {
                Ok(st) => Ok(st.success()),
                Err(e) => Err(format!(
                    "cannot run {}: {e}\n  \
                     `rustlm code` runs the RustLM Code TUI, which is a separate binary. \
                     Build it with `cargo build --release --no-default-features -p \
                     rustlm-code` in the rustlm-code checkout, then put it on PATH or \
                     beside this binary.",
                    prog.display()
                )),
            }
        }
        Some("run") => {
            // Ollama's `run`: an interactive chat REPL against an ALREADY-RUNNING server. It
            // does not start the engine — `rustlm serve` does that, in its own terminal — it
            // just speaks OpenAI `/v1/chat/completions` to it over a socket.
            let model = a.get(2).filter(|s| !s.starts_with("--"))
                .ok_or("run needs a model name, e.g. rustlm run qwen3.8-27b")?;
            let addr = flag(a, "--addr").unwrap_or("127.0.0.1:11434").to_string();
            // The one precondition: the server has to be up. Fail clearly if it is not,
            // rather than hanging or printing a socket error mid-chat.
            if std::net::TcpStream::connect(&addr).is_err() {
                return Err(format!(
                    "no server at {addr}. Start it first, in another terminal:\n  \
                     rustlm serve {model}"
                ));
            }
            chat_repl(&addr, model).map_err(|e| e.to_string())?;
            Ok(true)
        }
        Some("train") | Some("tui") | Some("workbench") => {
            // Same git-style sibling dispatch as `code`: the fine-tune/eval workbench is the
            // `rustlm_tui` binary built next to this one. One tool, one entry point.
            let exe = std::env::current_exe().ok();
            let sibling = exe.as_ref().and_then(|p| p.parent()).map(|d| d.join("rustlm_tui"));
            let prog = match &sibling {
                Some(p) if p.exists() => p.clone(),
                _ => std::path::PathBuf::from("rustlm_tui"),
            };
            match std::process::Command::new(&prog).args(&a[2..]).status() {
                Ok(st) => Ok(st.success()),
                Err(e) => Err(format!(
                    "could not launch {}: {e}\n  \
                     `rustlm train` opens the fine-tune & eval workbench (the `rustlm_tui` \
                     binary). Build it with `cargo build --release --bin rustlm_tui`, then \
                     put it on PATH or beside this binary.",
                    prog.display()
                )),
            }
        }
        Some("accel") => {
            // Says what is compiled in, what actually works on THIS machine, and how to
            // turn on what is not. An accelerator that silently does nothing is worse than
            // one that is absent.
            let gb: f64 = num_flag(a, "--cache-gb", 5.0f64)?;
            println!("optional accelerators (default build uses none of them):");
            for s in k3::accel::status((gb * 1e9) as usize) {
                println!("  {s}");
            }
            Ok(true)
        }
        Some("serve") => {
            // Serving defaults to the int8 (Q8-activation) path: ~1.5x faster generation,
            // near-bitwise. An explicit `RUSTLM_INT8=0` or `--no-int8` turns it back off for a
            // bit-exact server.
            let int8 = !a.iter().any(|x| x == "--no-int8")
                && std::env::var_os("RUSTLM_INT8").map(|v| v != "0").unwrap_or(true);
            k3::ops::set_int8(int8);
            println!("int8 fast path: {}", if int8 { "on (~1.5x, near-bitwise; --no-int8 to disable)" } else { "off (bit-exact)" });
            // No model named: serve the only runnable one, rather than making the common
            // case type a name it could have looked up.
            let reg = Registry::open();
            let name = match a.get(2).filter(|s| !s.starts_with("--")) {
                Some(n) => n.clone(),
                None => {
                    let mut runnable = reg.entries.iter().filter(|e| e.runnable());
                    match (runnable.next(), runnable.next()) {
                        (Some(e), None) => e.name.clone(),
                        (None, _) => {
                            return Err("no runnable model is registered -- `rustlm list` \
                                        shows what is there and what blocks it"
                                .into())
                        }
                        (Some(_), Some(_)) => {
                            return Err(format!(
                                "several runnable models are registered ({}) -- name one",
                                reg.entries
                                    .iter()
                                    .filter(|e| e.runnable())
                                    .map(|e| e.name.as_str())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ))
                        }
                    }
                }
            };
            // Defaults tuned for speed/efficiency: the arena auto-sizes to swap-safe RAM, and
            // the per-request and context caps are generous enough for long tasks and agent
            // prompts out of the box.
            let mt = num_flag(a, "--max-tokens", 4096usize)?;
            let cg = cache_gb_flag(a)?;
            let mut params = k3::v4run::Params::from_env(mt, cg);
            params.max_tokens = mt;
            params.cache_gb = cg;
            let cfg = k3::serve::Cfg {
                addr: flag(a, "--addr").unwrap_or("127.0.0.1:11434").to_string(),
                model: name,
                max_ctx: num_flag(a, "--max-ctx", 16384usize)?,
                conv_gb: num_flag(a, "--conv-gb", 2.0f64)?,
                prefill_width: num_flag(a, "--prefill-width", 0usize)?,
                params,
            };
            k3::serve::run(&cfg)?;
            Ok(true)
        }
        Some("remove") => {
            let name = a.get(2).ok_or("remove needs a name")?;
            let mut reg = Registry::open();
            if reg.remove(name)? {
                println!("removed {name:?}");
                Ok(true)
            } else {
                Err(format!("{name:?} is not registered"))
            }
        }
        _ => Err(String::new()),
    }
}

fn main() -> ExitCode {
    let a: Vec<String> = std::env::args().collect();
    match run(&a) {
        Ok(true) => ExitCode::SUCCESS,
        // A model that cannot run is not a crash: the report is the answer, and a non-zero
        // exit lets a script branch on it.
        Ok(false) => ExitCode::from(1),
        Err(e) if e.is_empty() => usage(),
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
