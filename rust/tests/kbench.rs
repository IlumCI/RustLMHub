// Kernel microbenchmark. Not a gate -- `cargo test --release --test kbench -- --nocapture`.
use std::time::Instant;

fn bench(name: &str, macs: f64, mut f: impl FnMut()) {
    f();
    let t = Instant::now();
    for _ in 0..20 { f(); }
    let d = t.elapsed() / 20;
    println!("  {name:22} {:>9.2?}  {:6.2} GMAC/s", d, macs / d.as_secs_f64() / 1e9);
}

#[test]
fn kernels() {
    // The shapes DeepSeek-V4 actually uses: expert w1/w3 are [2048][4096], w2 is
    // [4096][2048]; the attention projections are 4096-wide.
    let (k, rows, block, g) = (4096usize, 2048usize, 128usize, 32usize);
    let macs = (k * rows) as f64;
    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01).sin()).collect();

    let packed: Vec<u8> = (0..k / 2 * rows).map(|i| (i * 7 % 251) as u8).collect();
    let ngrp = k.div_ceil(g);
    let msc: Vec<u8> = (0..ngrp * rows).map(|i| (120 + i % 13) as u8).collect();
    let mut y = vec![0f32; rows];
    bench("mxfp4 (expert)", macs, || {
        k3::ops::matmul_mxfp4(&mut y, &x, &packed, &msc, k, rows, g)
    });

    let w8: Vec<u8> = (0..k * rows).map(|i| ((i * 7) % 126) as u8).collect();
    let sb = k.div_ceil(block) * rows.div_ceil(block);
    let fsc: Vec<u8> = (0..sb).map(|i| (120 + i % 13) as u8).collect();
    let mut y2 = vec![0f32; rows];
    bench("fp8_block (trunk)", macs, || {
        k3::ops::matmul_fp8_block(&mut y2, &x, &w8, &fsc, k, rows, block)
    });

    let wb: Vec<u16> = (0..k * rows).map(|i| 0x3F00u16.wrapping_add(i as u16 % 97)).collect();
    let mut y3 = vec![0f32; rows];
    bench("bf16 (reference)", macs, || k3::ops::matmul_bf16(&mut y3, &x, &wb, k, rows));
}

/// A Q4_K tensor whose bytes are a VALID encoding.
///
/// Filling a k-quant buffer with pseudo-random bytes puts random bit patterns in the two
/// f16 scales, and an exponent of all ones is inf or NaN -- which then propagates through
/// every dot product and makes a bitwise comparison fail on data, not on arithmetic. The
/// scales are therefore pinned to real values and only the quantised payload is arbitrary.
fn q4k_fixture(k: usize, rows: usize) -> Vec<u8> {
    let nb = k / 256;
    let mut v = vec![0u8; nb * 144 * rows];
    for (i, blk) in v.chunks_mut(144).enumerate() {
        // f16 0.00994 and 0.00299: small, normal, and not powers of two.
        blk[0..2].copy_from_slice(&0x2116u16.to_le_bytes());
        blk[2..4].copy_from_slice(&0x1e21u16.to_le_bytes());
        for (j, b) in blk[4..].iter_mut().enumerate() {
            *b = ((i * 31 + j * 7) % 251) as u8;
        }
    }
    v
}

/// The shapes qwen3.6-35B spends its prefill in, and how far they are from the machine.
///
/// Prefill on that model measured 77% compute, so the question this answers is whether
/// that compute is near the hardware or nowhere near it. 16 cores of AVX2 FMA is roughly
/// 380 GMAC/s in f32; anything an order of magnitude under that is a kernel problem, not a
/// physics problem.
#[test]
fn qwen35_shapes() {
    let cases: [(usize, usize, &str); 4] = [
        (2048, 8192, "qkv proj  2048x8192"),
        (4096, 2048, "attn out  4096x2048"),
        (2048, 512, "expert up 2048x512"),
        (512, 2048, "expert dn  512x2048"),
    ];
    for (k, rows, name) in cases {
        let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01).sin()).collect();
        let q4 = q4k_fixture(k, rows);
        let mut y = vec![0f32; rows];
        bench(name, (k * rows) as f64, || k3::gguf::matmul_q4k(&mut y, &x, &q4, k, rows));
    }
}

/// Batched against serial: the same answer, and how much faster.
///
/// The identity assertion is the point. Batching only helps because one weight decode is
/// shared across the chunk, and a shared decode that rounded differently would be a model
/// that is wrong only on long prompts -- fluent, plausible, and untraceable.
#[test]
fn batched_kquant() {
    for &(k, rows, name) in
        &[(2048usize, 8192usize, "qkv  2048x8192"), (2048, 512, "up   2048x512")]
    {
        let q4 = q4k_fixture(k, rows);
        for &ntok in &[1usize, 2, 4, 8, 16, 32] {
            let x: Vec<f32> = (0..k * ntok).map(|i| ((i % 977) as f32 * 0.01).sin()).collect();
            let mut serial = vec![0f32; rows * ntok];
            for t in 0..ntok {
                k3::gguf::matmul_q4k(&mut serial[t * rows..][..rows], &x[t * k..][..k], &q4, k, rows);
            }
            let mut batched = vec![0f32; rows * ntok];
            k3::gguf::matmul_q4k_many(&mut batched, &x, &q4, k, rows, ntok);
            assert!(serial.iter().all(|v| v.is_finite()), "{name}: fixture produced non-finite");
            let bad = serial.iter().zip(&batched).position(|(a, b)| a.to_bits() != b.to_bits());
            assert!(bad.is_none(), "{name} at ntok {ntok}: element {:?} differs", bad);

            if ntok == 1 {
                continue;
            }
            let macs = (k * rows * ntok) as f64;
            let t0 = Instant::now();
            for _ in 0..5 {
                for t in 0..ntok {
                    k3::gguf::matmul_q4k(&mut serial[t * rows..][..rows], &x[t * k..][..k], &q4, k, rows);
                }
            }
            let ds = t0.elapsed() / 5;
            let t1 = Instant::now();
            for _ in 0..5 {
                k3::gguf::matmul_q4k_many(&mut batched, &x, &q4, k, rows, ntok);
            }
            let db = t1.elapsed() / 5;
            println!(
                "  {name} x{ntok:<3} serial {:6.2} GMAC/s  batched {:6.2} GMAC/s  {:.2}x",
                macs / ds.as_secs_f64() / 1e9,
                macs / db.as_secs_f64() / 1e9,
                ds.as_secs_f64() / db.as_secs_f64()
            );
        }
    }
}
