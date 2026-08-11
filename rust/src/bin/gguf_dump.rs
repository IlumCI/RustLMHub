// SPDX-License-Identifier: Apache-2.0
//
// Dump what src/gguf.rs sees in a .gguf file, so the official `gguf` Python package can
// be asked the same questions independently.
//
// usage: gguf_dump <file.gguf> [tensor_name] [out.bin]
//
// With no tensor name it prints the container: alignment, a few metadata keys, and every
// tensor's dtype, shape, offset and length. With one, it dequantises that tensor and
// writes the f32 values to out.bin for a numeric comparison.

use std::fs::File;
use std::io::Write;
use std::process::ExitCode;

use k3::gguf;
use k3::st::Dtype;

fn main() -> ExitCode {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 {
        eprintln!("usage: gguf_dump <file.gguf> [tensor] [out.bin]");
        return ExitCode::from(2);
    }
    // A directory goes through St::open, which is the path the engine actually uses --
    // it exercises the extension filter, the mixed-directory refusal and the metadata
    // stash, none of which calling gguf::scan directly would touch.
    if std::path::Path::new(&a[1]).is_dir() {
        return match k3::st::St::open(std::path::Path::new(&a[1])) {
            Ok(st) => {
                println!(
                    "St::open: {} tensors from {} file(s), gguf metadata {}",
                    st.tensors.len(),
                    st.paths.len(),
                    if st.meta.is_some() { "present" } else { "ABSENT" }
                );
                if let Some(m) = &st.meta {
                    match k3::arch::gguf_to_json(m) {
                        Ok(cfg) => println!("  config bridge: {cfg}"),
                        Err(e) => println!("  config bridge FAILED: {e}"),
                    }
                }
                for t in st.tensors.iter().take(3) {
                    println!("  {} {} {:?} off={}", t.name, t.dtype.name(), t.shape, t.off);
                }
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{e}");
                ExitCode::FAILURE
            }
        };
    }
    let f = match File::open(&a[1]) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{}: {e}", a[1]);
            return ExitCode::FAILURE;
        }
    };
    let (meta, tensors) = match gguf::scan(0, &f) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    if a.len() < 3 {
        println!("tensors {}  kv {}", tensors.len(), meta.len());
        for k in ["general.architecture", "general.alignment", "general.name"] {
            if let Some(v) = meta.get(k) {
                println!("  {k} = {v:?}");
            }
        }
        // Scalar metadata only: a token vocabulary is hundreds of thousands of entries
        // and printing it would bury everything that matters.
        let mut keys: Vec<&String> = meta
            .keys()
            .filter(|k| !matches!(meta.get(*k), Some(gguf::Value::Arr(_))))
            .collect();
        keys.sort();
        for k in keys {
            println!("KV\t{k}\t{:?}", meta[k]);
        }
        for t in &tensors {
            println!(
                "T\t{}\t{}\t{:?}\t{}\t{}",
                t.name,
                t.dtype.name(),
                t.shape,
                t.off,
                t.nbytes
            );
        }
        return ExitCode::SUCCESS;
    }

    let Some(t) = tensors.iter().find(|t| t.name == a[2]) else {
        eprintln!("no tensor named {}", a[2]);
        return ExitCode::FAILURE;
    };
    let mut raw = vec![0u8; t.nbytes as usize];
    use std::os::unix::fs::FileExt;
    if let Err(e) = f.read_exact_at(&mut raw, t.off as u64) {
        eprintln!("read {}: {e}", t.name);
        return ExitCode::FAILURE;
    }
    let numel: i64 = t.shape.iter().product();

    // `bench` mode: what the fused kernel ACTUALLY achieves on real weights, in GMAC/s.
    //
    // The roofline needs a measured number, not a theoretical one. Peak AVX2 f64 FMA on
    // this CPU is ~8 MAC/cycle/core, but a k-quant kernel spends much of its time
    // unpacking nibbles and applying per-sub-block scales, so the delivered rate is what
    // decides whether a given model is compute-bound or bandwidth-bound.
    if a.len() > 4 && a[4] == "bench" {
        if t.shape.len() != 2 {
            eprintln!("bench needs a 2-D tensor, {} is {:?}", t.name, t.shape);
            return ExitCode::FAILURE;
        }
        let (rows, k_in) = (t.shape[0] as usize, t.shape[1] as usize);
        let x: Vec<f32> =
            (0..k_in).map(|i| ((i as f64 * 0.017).sin() * 0.4) as f32).collect();
        let mut y = vec![0f32; rows];
        let run = |y: &mut Vec<f32>| match t.dtype {
            Dtype::Q3K => gguf::matmul_q3k(y, &x, &raw, k_in, rows),
            Dtype::Q4K => gguf::matmul_q4k(y, &x, &raw, k_in, rows),
            Dtype::Q5K => gguf::matmul_q5k(y, &x, &raw, k_in, rows),
            Dtype::Q6K => gguf::matmul_q6k(y, &x, &raw, k_in, rows),
            _ => {}
        };
        run(&mut y); // warm the caches; the first pass reads the weights cold
        let iters: usize = a.get(5).and_then(|v| v.parse().ok()).unwrap_or(20);
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            run(&mut y);
        }
        let dt = t0.elapsed().as_secs_f64();
        let macs = rows as f64 * k_in as f64 * iters as f64;
        println!(
            "{} {} {rows}x{k_in}  {iters} iters in {dt:.3} s  ->  {:.2} GMAC/s  \
             ({:.2} GB/s of packed weights)",
            t.name,
            t.dtype.name(),
            macs / dt / 1e9,
            raw.len() as f64 * iters as f64 / dt / 1e9
        );
        return ExitCode::SUCCESS;
    }

    // `matmul` mode: run the FUSED kernel over the whole tensor against a deterministic
    // activation vector, so the reference can check `dequantise(W) @ x` independently.
    // This is the only way the fused path is exercised on real quantised weights rather
    // than on synthetic blocks.
    if a.len() > 4 && a[4] == "matmul" {
        if t.shape.len() != 2 {
            eprintln!("matmul mode needs a 2-D tensor, {} is {:?}", t.name, t.shape);
            return ExitCode::FAILURE;
        }
        let (rows, k_in) = (t.shape[0] as usize, t.shape[1] as usize);
        let x: Vec<f32> =
            (0..k_in).map(|i| ((i as f64 * 0.017).sin() * 0.4) as f32).collect();
        let mut y = vec![0f32; rows];
        match t.dtype {
            Dtype::Q3K => gguf::matmul_q3k(&mut y, &x, &raw, k_in, rows),
            Dtype::Q4K => gguf::matmul_q4k(&mut y, &x, &raw, k_in, rows),
            Dtype::Q5K => gguf::matmul_q5k(&mut y, &x, &raw, k_in, rows),
            Dtype::Q6K => gguf::matmul_q6k(&mut y, &x, &raw, k_in, rows),
            d => {
                eprintln!("{} is {}, no fused kernel", t.name, d.name());
                return ExitCode::FAILURE;
            }
        }
        let mut b: Vec<u8> = Vec::with_capacity(y.len() * 4);
        for v in &y {
            b.extend_from_slice(&v.to_le_bytes());
        }
        if let Err(e) = File::create(&a[3]).and_then(|mut w| w.write_all(&b)) {
            eprintln!("{}: {e}", a[3]);
            return ExitCode::FAILURE;
        }
        println!("{} matmul {rows}x{k_in}", t.name);
        return ExitCode::SUCCESS;
    }

    let mut out = vec![0f32; numel as usize];
    match t.dtype {
        Dtype::Q3K => gguf::q3k_dequant(&mut out, &raw, (numel as usize) / gguf::QK_K),
        Dtype::Q4K => gguf::q4k_dequant(&mut out, &raw, (numel as usize) / gguf::QK_K),
        Dtype::Q5K => gguf::q5k_dequant(&mut out, &raw, (numel as usize) / gguf::QK_K),
        Dtype::Q6K => gguf::q6k_dequant(&mut out, &raw, (numel as usize) / gguf::QK_K),
        Dtype::F32 => {
            for (i, v) in out.iter_mut().enumerate() {
                *v = f32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
            }
        }
        Dtype::F16 => {
            for (i, v) in out.iter_mut().enumerate() {
                *v = k3::st::f16_to_f32(u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]));
            }
        }
        d => {
            eprintln!("{} is {}, which gguf_dump cannot widen yet", t.name, d.name());
            return ExitCode::FAILURE;
        }
    }
    println!("{} {} {:?} {} values", t.name, t.dtype.name(), t.shape, out.len());
    if a.len() > 3 {
        let mut b: Vec<u8> = Vec::with_capacity(out.len() * 4);
        for v in &out {
            b.extend_from_slice(&v.to_le_bytes());
        }
        if let Err(e) = File::create(&a[3]).and_then(|mut w| w.write_all(&b)) {
            eprintln!("{}: {e}", a[3]);
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}
