// SPDX-License-Identifier: Apache-2.0

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use k3::fmt::g;
use k3::st::St;

fn base_name(p: &Path) -> String {
    p.file_name().map_or_else(String::new, |s| s.to_string_lossy().into_owned())
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: test_st <dir> [index.json] [tensor ...]");
        return ExitCode::from(2);
    }
    let dir = Path::new(&args[1]);
    let out = args.get(2).map_or("st_index.json", |s| s.as_str());

    println!("opening {}", dir.display());
    let t0 = Instant::now();
    let s = match St::open(dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            eprintln!("OPEN FAILED");
            return ExitCode::FAILURE;
        }
    };
    let t_open = t0.elapsed().as_secs_f64();

    let total: i64 = s.tensors.iter().map(|t| t.nbytes).sum();
    let count = |d: k3::st::Dtype| s.tensors.iter().filter(|t| t.dtype == d).count();
    let nt = s.tensors.len();

    println!("  shards        : {}", s.nshard());
    println!("  tensors       : {nt}");
    println!("  indexed bytes : {:.2} GB", total as f64 / 1e9);
    println!(
        "  index built in: {t_open:.3} s  ({:.1} us/tensor)",
        1e6 * t_open / nt.max(1) as f64
    );
    println!(
        "  dtypes        : U8 {}, BF16 {}, F16 {}, F32 {}",
        count(k3::st::Dtype::U8),
        count(k3::st::Dtype::Bf16),
        count(k3::st::Dtype::F16),
        count(k3::st::Dtype::F32)
    );

    let t0 = Instant::now();
    let mut bad = 0;
    for (i, t) in s.tensors.iter().enumerate() {
        match s.find(&t.name) {
            Some(f) if std::ptr::eq(f, &s.tensors[i]) => {}
            _ => bad += 1,
        }
    }
    let t_look = t0.elapsed().as_secs_f64();
    println!(
        "  round trip    : {}/{nt} resolve to themselves{}",
        nt - bad,
        if bad > 0 { "   <-- COLLISION BUG" } else { "" }
    );
    println!(
        "  lookup cost   : {:.0} ns each ({nt} lookups in {t_look:.3} s)",
        1e9 * t_look / nt.max(1) as f64
    );

    let ghosts = ["", "no.such.tensor", "language_model.model.layers.999.self_attn.A_log"];
    let ghost_bad = ghosts.iter().filter(|g| s.find(g).is_some()).count();
    println!(
        "  absent names  : {}/{} correctly return NULL",
        ghosts.len() - ghost_bad,
        ghosts.len()
    );

    let mut f = match std::fs::File::create(out) {
        Ok(f) => std::io::BufWriter::new(f),
        Err(_) => {
            eprintln!("cannot write {out}");
            return ExitCode::FAILURE;
        }
    };
    let _ = write!(f, "{{\"nshard\":{},\"nt\":{nt},\"tensors\":{{", s.nshard());
    for (i, t) in s.tensors.iter().enumerate() {
        if i > 0 {
            let _ = write!(f, ",");
        }
        let shape: Vec<String> = t.shape.iter().map(|d| d.to_string()).collect();
        let _ = write!(
            f,
            "{}:{{\"shard\":\"{}\",\"dtype\":\"{}\",\"shape\":[{}],\"off\":{},\"nbytes\":{}}}",
            json_str(&t.name),
            base_name(&s.paths[t.shard]),
            t.dtype.name(),
            shape.join(","),
            t.off,
            t.nbytes
        );
    }
    let _ = writeln!(f, "}}}}");
    drop(f);
    println!("  wrote {out}");

    if args.len() > 3 {
        let mut v = std::io::BufWriter::new(std::fs::File::create("st_values.json").unwrap());
        let _ = write!(v, "{{");
        let mut first = true;
        for name in &args[3..] {
            let Some(t) = s.find(name) else {
                println!("  MISSING: {name}");
                continue;
            };
            let n = t.numel() as usize;
            let mut buf = vec![0f32; n];
            let t1 = Instant::now();
            let got = s.read_f32(t, &mut buf) as usize;
            let dt = t1.elapsed().as_secs_f64();

            let (mut mn, mut mx, mut sum, mut nonfinite) = (f64::MAX, f64::MIN, 0f64, 0usize);
            for &x in &buf[..got] {
                if !x.is_finite() {
                    nonfinite += 1;
                    continue;
                }
                mn = mn.min(x as f64);
                mx = mx.max(x as f64);
                sum += x as f64;
            }

            println!("  {name}");
            let mbps = t.nbytes as f64 / 1e6 / dt.max(1e-9);
            if nonfinite > 0 {
                let fin = got - nonfinite;
                println!(
                    "    {} {got} elems, {:.2} MB in {dt:.3} s ({mbps:.0} MB/s), \
                     min {} max {} mean {}  \
                     <-- {nonfinite} of {got} values are non-finite (stats cover the other {fin})",
                    t.dtype.name(),
                    t.nbytes as f64 / 1e6,
                    g(mn, 6), g(mx, 6),
                    g(if fin > 0 { sum / fin as f64 } else { 0.0 }, 6)
                );
            } else {
                println!(
                    "    {} {got} elems, {:.2} MB in {dt:.3} s ({mbps:.0} MB/s), \
                     min {} max {} mean {}",
                    t.dtype.name(),
                    t.nbytes as f64 / 1e6,
                    g(mn, 6), g(mx, 6),
                    g(if got > 0 { sum / got as f64 } else { 0.0 }, 6)
                );
            }

            if !first {
                let _ = write!(v, ",");
            }
            first = false;
            let nd = got.min(4096);
            let head: Vec<String> = buf[..nd].iter().map(|x| x.to_bits().to_string()).collect();
            let tail_start = got.saturating_sub(64);
            let tail: Vec<String> =
                buf[tail_start..got].iter().map(|x| x.to_bits().to_string()).collect();
            let _ = write!(
                v,
                "{}:{{\"n\":{got},\"first_bits\":[{}],\"last_bits\":[{}],\"tail_start\":{tail_start}}}",
                json_str(name),
                head.join(","),
                tail.join(",")
            );
        }
        let _ = writeln!(v, "}}");
        drop(v);
        println!("  wrote st_values.json");
    }

    println!();
    if bad == 0 && ghost_bad == 0 {
        println!("READER SELF-CHECKS PASSED (now run tools/verify_st.py for the external check)");
        ExitCode::SUCCESS
    } else {
        println!("READER SELF-CHECKS FAILED");
        ExitCode::FAILURE
    }
}
