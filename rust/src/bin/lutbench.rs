// Characterise the LUT-GEMM path against the current Q4_K kernel on real qwen FFN shapes.
//
// This times three things per shape:
//   - matmul_q4k        : the current f32-dequant serial kernel (the baseline to beat)
//   - matmul_q4k_lut    : the SCALAR LUT reference (correctness-proven, not yet fast)
//   - build_lut (once)  : the activation-quantise + table-build cost, which amortises across
//                         all output rows
//
// The scalar LUT is expected to be SLOWER than the baseline -- the win comes only from the
// AVX2 `pshufb` kernel (32 lookups/instruction), which is the next build. This bench exists
// to (a) confirm the LUT path is wired and near-bitwise on real shapes, and (b) measure the
// gap the SIMD kernel must close, on THIS machine, rather than trusting the paper's 3.1x.
use k3::gguf;
use k3::lut;
use std::time::Instant;

fn rand_q4k(nblocks: usize, seed: u64) -> Vec<u8> {
    let mut v = vec![0u8; nblocks * 144];
    let mut s = seed | 1;
    for b in v.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *b = (s >> 33) as u8;
    }
    for i in 0..nblocks {
        let blk = &mut v[i * 144..][..144];
        blk[0..2].copy_from_slice(&0x3400u16.to_le_bytes());
        blk[2..4].copy_from_slice(&0x3000u16.to_le_bytes());
    }
    v
}

fn main() {
    // FFN shapes for Qwen3.8-27B: hidden=5120, inter=17408.
    let shapes: [(&str, usize, usize); 5] = [
        ("gate/up  h->inter", 5120, 17408),
        ("down     inter->h", 17408, 5120),
        ("attn qg  h->12288", 5120, 12288), // fused q/gate, full-attn block
        ("attn o   12288->h", 12288, 5120),  // output projection
        ("attn kv  h->2048 ", 5120, 2048),   // k or v projection (small)
    ];
    println!(
        "{:<20} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "shape", "baseline", "lut-avx2", "avx2+bld", "speedup", "rms_rel"
    );
    for (name, k_in, rows) in shapes {
        let x: Vec<f32> = (0..k_in).map(|i| ((i as f64 * 0.013).sin() * 0.7) as f32).collect();
        let src = rand_q4k(rows * (k_in / 256), 20260816);

        let mut y_ref = vec![0f32; rows];
        let mut y_lut = vec![0f32; rows];

        // correctness: near-bitwise vs the f32 kernel
        gguf::matmul_q4k(&mut y_ref, &x, &src, k_in, rows);
        let act = lut::ActLut::build(&x);
        let w = lut::Q4kLutW::repack(&src, k_in, rows); // offline, once
        lut::matmul_q4k_lut_avx2(&mut y_lut, &act, &w);
        let scale = y_ref.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-6);
        let rms = (y_ref.iter().zip(&y_lut).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum::<f64>()
            / rows as f64)
            .sqrt() as f32
            / scale;

        let iters = 50;
        let t = Instant::now();
        for _ in 0..iters {
            gguf::matmul_q4k(&mut y_ref, &x, &src, k_in, rows);
        }
        let base = t.elapsed().as_secs_f64() / iters as f64;

        // AVX2 LUT matmul with the weights prepacked (the realistic hot-path cost).
        let t = Instant::now();
        for _ in 0..iters {
            lut::matmul_q4k_lut_avx2(&mut y_lut, &act, &w);
        }
        let avx2 = t.elapsed().as_secs_f64() / iters as f64;

        // AVX2 LUT including the per-token activation LUT build (weights stay prepacked).
        let t = Instant::now();
        for _ in 0..iters {
            let a = lut::ActLut::build(&x);
            lut::matmul_q4k_lut_avx2(&mut y_lut, &a, &w);
        }
        let avx2b = t.elapsed().as_secs_f64() / iters as f64;

        // int8 (Q8 activation) path, INCLUDING the per-call activation quantise.
        let mut y_q8 = vec![0f32; rows];
        let q8a = gguf::Q8Act::quantize(&x, k_in);
        gguf::matmul_q4k_q8(&mut y_q8, &q8a, &src, k_in, rows);
        let rms8 = (y_ref.iter().zip(&y_q8).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum::<f64>()
            / rows as f64).sqrt() as f32 / scale;
        let t = Instant::now();
        for _ in 0..iters {
            let q = gguf::Q8Act::quantize(&x, k_in);
            gguf::matmul_q4k_q8(&mut y_q8, &q, &src, k_in, rows);
        }
        let int8 = t.elapsed().as_secs_f64() / iters as f64;

        let macs = (k_in * rows) as f64;
        let gs = |t: f64| macs / t / 1e9;
        println!(
            "{:<20} f32 {:>6.1}  lut {:>6.1}({:.2}x)  int8 {:>6.1}({:.2}x)  rms lut {:.4} int8 {:.4}",
            name,
            gs(base),
            gs(avx2),
            gs(avx2) / gs(base),
            gs(int8),
            gs(int8) / gs(base),
            rms,
            rms8,
        );
        let _ = avx2b;
    }
    println!("\n(G/s = GMAC/s, 16-thread. f32 = current dot_p8 baseline. lut = pshufb (near-bitwise).");
    println!(" int8 = Q8-activation maddubs incl. per-call quantise (near-bitwise). rms vs f32 kernel.)");
}
