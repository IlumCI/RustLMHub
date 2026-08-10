// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::process::ExitCode;

use k3::cfg::{load, load_file, Cfg};
use serde_json::Value;

struct Checker {
    fails: u32,
}

impl Checker {
    fn ck(&mut self, cond: bool, what: &str, got: i64, want: i64) {
        if cond {
            println!("  ok    {what:<28} {got}");
        } else {
            println!("  FAIL  {what:<28} got {got}, want {want}");
            self.fails += 1;
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: test_cfg <real|fixture|reject> <path>");
        return ExitCode::from(2);
    }
    let (mode, path) = (args[1].as_str(), Path::new(&args[2]));

    match mode {
        "reject" => match load_file(path) {
            Ok(_) => {
                println!("  FAIL  expected rejection, but the config loaded");
                ExitCode::FAILURE
            }
            Err(e) => {
                println!("  ok    correctly rejected {}", path.display());
                println!("{e}");
                ExitCode::SUCCESS
            }
        },
        "fixture" => {
            let txt = match std::fs::read_to_string(path) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("{}: {e}", path.display());
                    return ExitCode::from(2);
                }
            };
            let root: Value = match serde_json::from_str(&txt) {
                Ok(v) => v,
                Err(_) => {
                    eprintln!("{}: not valid JSON", path.display());
                    return ExitCode::from(2);
                }
            };
            let Some(cfg) = root.get("config") else {
                eprintln!("{} has no \"config\" member", path.display());
                return ExitCode::from(2);
            };
            let whence = path.display().to_string();
            let c = match load(cfg, &whence) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::FAILURE;
                }
            };
            println!("{}", c.summary(&whence));
            report(check_fixture(&c), "FIXTURE CONFIG")
        }
        "real" => {
            let c = match load_file(path) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::FAILURE;
                }
            };
            println!("{}", c.summary(&path.display().to_string()));
            report(check_real(&c), "REAL CONFIG")
        }
        other => {
            eprintln!("unknown mode {other}");
            ExitCode::from(2)
        }
    }
}

fn report(fails: u32, label: &str) -> ExitCode {
    println!("\n{label}: {}", if fails > 0 { "FAIL" } else { "PASS" });
    if fails > 0 { ExitCode::FAILURE } else { ExitCode::SUCCESS }
}

pub fn check_fixture(c: &Cfg) -> u32 {
    let mut k = Checker { fails: 0 };
    k.ck(c.hidden == 128, "hidden", c.hidden as i64, 128);
    k.ck(c.n_layers == 13, "layers", c.n_layers as i64, 13);
    k.ck(c.vocab == 256, "vocab", c.vocab as i64, 256);
    k.ck(c.kda_heads == 4, "kda heads", c.kda_heads as i64, 4);
    k.ck(c.kda_head_dim == 16, "kda head_dim", c.kda_head_dim as i64, 16);
    k.ck(c.n_experts == 8, "experts", c.n_experts as i64, 8);
    k.ck(c.topk == 2, "topk", c.topk as i64, 2);
    k.ck(c.latent == 64, "latent", c.latent as i64, 64);
    k.ck(c.attn_res_block == 3, "attn_res_block", c.attn_res_block as i64, 3);
    k.ck(c.situ_b1 == 4.0, "situ b1", c.situ_b1 as i64, 4);
    k.ck(c.situ_b2 == 25.0, "situ b2", c.situ_b2 as i64, 25);
    k.ck(c.full_attn.len() == 4, "full_attn count", c.full_attn.len() as i64, 4);
    k.ck(c.gate_lb == -5.0, "gate_lb (x-1)", (-c.gate_lb) as i64, 5);
    k.fails
}

pub fn check_real(c: &Cfg) -> u32 {
    let mut k = Checker { fails: 0 };
    k.ck(c.hidden == 7168, "hidden", c.hidden as i64, 7168);
    k.ck(c.n_layers == 93, "layers", c.n_layers as i64, 93);
    k.ck(c.vocab == 163840, "vocab", c.vocab as i64, 163840);
    k.ck(c.kda_heads == 96, "kda heads", c.kda_heads as i64, 96);
    k.ck(c.kda_head_dim == 128, "kda head_dim", c.kda_head_dim as i64, 128);
    k.ck(c.conv_k == 4, "short conv k", c.conv_k as i64, 4);
    k.ck(c.n_heads == 96, "attn heads", c.n_heads as i64, 96);
    k.ck(c.q_lora == 1536, "q_lora", c.q_lora as i64, 1536);
    k.ck(c.kv_lora == 512, "kv_lora", c.kv_lora as i64, 512);
    k.ck(c.qk_nope == 128, "qk_nope", c.qk_nope as i64, 128);
    k.ck(c.qk_rope == 64, "qk_rope", c.qk_rope as i64, 64);
    k.ck(c.v_head == 128, "v_head", c.v_head as i64, 128);
    k.ck(c.n_experts == 896, "experts", c.n_experts as i64, 896);
    k.ck(c.topk == 16, "topk", c.topk as i64, 16);
    k.ck(c.n_shared == 2, "shared experts", c.n_shared as i64, 2);
    k.ck(c.latent == 3584, "latent", c.latent as i64, 3584);
    k.ck(c.moe_inter == 3072, "moe_inter", c.moe_inter as i64, 3072);
    k.ck(c.first_dense == 1, "dense layers", c.first_dense as i64, 1);
    k.ck(c.dense_inter == 33792, "dense_inter", c.dense_inter as i64, 33792);
    k.ck(c.attn_res_block == 12, "attn_res_block", c.attn_res_block as i64, 12);
    k.ck(c.situ_b1 == 4.0, "situ b1", c.situ_b1 as i64, 4);
    k.ck(c.situ_b2 == 25.0, "situ b2", c.situ_b2 as i64, 25);
    k.ck(c.gate_lb == -5.0, "gate_lb (x-1)", (-c.gate_lb) as i64, 5);

    let fa = &c.full_attn;
    k.ck(fa.len() == 24, "MLA layer count", fa.len() as i64, 24);
    k.ck(fa.first() == Some(&4), "first MLA layer", fa.first().copied().unwrap_or(-1) as i64, 4);
    k.ck(fa.get(22) == Some(&92), "MLA layer 23", fa.get(22).copied().unwrap_or(-1) as i64, 92);
    k.ck(fa.get(23) == Some(&93), "last MLA layer", fa.get(23).copied().unwrap_or(-1) as i64, 93);

    let nmla = (0..c.n_layers).filter(|&l| c.is_mla(l)).count();
    let nkda = (0..c.n_layers).filter(|&l| c.is_kda(l)).count();
    k.ck(nmla == 24, "k3_is_mla says", nmla as i64, 24);
    k.ck(nkda == 69, "k3_is_kda says", nkda as i64, 69);
    k.ck(c.is_mla(91) && c.is_mla(92), "layers 91,92 both MLA", 1, 1);
    k.ck(!c.is_mla(0), "layer 0 is KDA", c.is_mla(0) as i64, 0);
    k.fails
}
