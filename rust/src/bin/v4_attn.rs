// SPDX-License-Identifier: Apache-2.0
//
// Run DeepSeek-V4 layer 0's attention on real checkpoint weights and dump the result,
// so an independent numpy reference can be compared against it.
//
// usage: v4_attn <shard_dir> <T> <out.bin>

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

use k3::ops::W;
use k3::st::St;
use k3::v4;

/// Dequantise an FP8 e4m3 matrix with 128x128 E8M0 block scales into f32.
fn deq_fp8(s: &St, name: &str, out_dim: usize, in_dim: usize, block: usize) -> Vec<f32> {
    let wt = s.find(&format!("{name}.weight")).unwrap_or_else(|| panic!("{name}.weight"));
    let st = s.find(&format!("{name}.scale")).unwrap_or_else(|| panic!("{name}.scale"));
    let mut wb = vec![0u8; wt.nbytes as usize];
    let mut sb = vec![0u8; st.nbytes as usize];
    s.read(wt, &mut wb);
    s.read(st, &mut sb);
    let sb_in = in_dim.div_ceil(block);
    let mut o = vec![0f32; out_dim * in_dim];
    for r in 0..out_dim {
        for i in 0..in_dim {
            let sc = k3::st::e8m0_to_f32(sb[(r / block) * sb_in + i / block]);
            o[r * in_dim + i] = k3::st::f8_e4m3_to_f32(wb[r * in_dim + i]) * sc;
        }
    }
    o
}

fn f32_of(s: &St, name: &str) -> Vec<f32> {
    let t = s.find(name).unwrap_or_else(|| panic!("{name}"));
    let mut v = vec![0f32; t.numel() as usize];
    s.read_f32(t, &mut v);
    v
}

fn main() -> ExitCode {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        eprintln!("usage: v4_attn <dir> <layer> <T> <out.bin>");
        return ExitCode::from(2);
    }
    let layer = a[2].clone();
    let t_len: usize = a[3].parse().unwrap();
    let s = match St::open(Path::new(&a[1])) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let p = |n: &str| format!("layers.{layer}.{n}");
    let d = v4::AttnDims {
        hidden: 4096,
        n_heads: 64,
        head_dim: 512,
        rope_head_dim: 64,
        q_lora_rank: 1024,
        o_lora_rank: 1024,
        o_groups: 8,
        window: 128,
        compress_ratio: 0, // set below from the tensors present
        eps: 1e-6,
    };

    eprint!("dequantising layer 0 attention weights ... ");
    let wq_a = deq_fp8(&s, &p("attn.wq_a"), d.q_lora_rank, d.hidden, 128);
    let wq_b = deq_fp8(&s, &p("attn.wq_b"), d.n_heads * d.head_dim, d.q_lora_rank, 128);
    let wkv = deq_fp8(&s, &p("attn.wkv"), d.head_dim, d.hidden, 128);
    let gsz = d.n_heads * d.head_dim / d.o_groups;
    let wo_a = deq_fp8(&s, &p("attn.wo_a"), d.o_groups * d.o_lora_rank, gsz, 128);
    let wo_b = deq_fp8(&s, &p("attn.wo_b"), d.hidden, d.o_groups * d.o_lora_rank, 128);
    let q_norm = f32_of(&s, &p("attn.q_norm.weight"));
    let kv_norm = f32_of(&s, &p("attn.kv_norm.weight"));
    let sink = f32_of(&s, &p("attn.attn_sink"));
    eprintln!("done");

    let w = v4::AttnW {
        wq_a: W::F32(&wq_a),
        q_norm: &q_norm,
        wq_b: W::F32(&wq_b),
        wkv: W::F32(&wkv),
        kv_norm: &kv_norm,
        wo_a: W::F32(&wo_a),
        wo_b: W::F32(&wo_b),
        attn_sink: &sink,
    };

    // Which compressed machinery this layer has is visible in its tensors.
    let has_comp = s.find(&p("attn.compressor.wkv.weight")).is_some();
    let has_idx = s.find(&p("attn.indexer.wq_b.weight")).is_some();
    let ratio = if !has_comp { 0 } else if has_idx { 4 } else { 128 };
    let d = v4::AttnDims { compress_ratio: ratio, ..d };

    // A compressed layer enables YaRN on compress_rope_theta; ratio 0 uses base theta.
    let rope = if ratio == 0 {
        v4::precompute_rope(d.rope_head_dim, t_len.max(2), 0, 10000.0, 16.0, 32.0, 1.0)
    } else {
        v4::precompute_rope(d.rope_head_dim, t_len.max(2), 65536, 160000.0, 16.0, 32.0, 1.0)
    };

    let x: Vec<f32> = (0..t_len * d.hidden)
        .map(|i| ((i as f32) * 0.001).sin() * 0.05)
        .collect();
    let mut out = vec![0f32; t_len * d.hidden];
    let (ckv, cg, ape, cn, ikv, ig, iape, inn, iwq, iwp);
    let (cw, cd, iw, id, icw, icd);
    let comp = if has_comp {
        ckv = f32_of(&s, &p("attn.compressor.wkv.weight"));
        cg = f32_of(&s, &p("attn.compressor.wgate.weight"));
        ape = f32_of(&s, &p("attn.compressor.ape"));
        cn = f32_of(&s, &p("attn.compressor.norm.weight"));
        cw = v4::CompressorW { ape: &ape, wkv: W::F32(&ckv), wgate: W::F32(&cg), norm: &cn };
        cd = v4::CompressorDims { hidden: d.hidden, head_dim: d.head_dim,
            rope_head_dim: d.rope_head_dim, ratio, rotate: false, eps: d.eps };
        let indexer = if has_idx {
            ikv = f32_of(&s, &p("attn.indexer.compressor.wkv.weight"));
            ig = f32_of(&s, &p("attn.indexer.compressor.wgate.weight"));
            iape = f32_of(&s, &p("attn.indexer.compressor.ape"));
            inn = f32_of(&s, &p("attn.indexer.compressor.norm.weight"));
            iwq = deq_fp8(&s, &p("attn.indexer.wq_b"), 64 * 128, d.q_lora_rank, 128);
            iwp = f32_of(&s, &p("attn.indexer.weights_proj.weight"));
            icw = v4::CompressorW { ape: &iape, wkv: W::F32(&ikv), wgate: W::F32(&ig), norm: &inn };
            icd = v4::CompressorDims { hidden: d.hidden, head_dim: 128,
                rope_head_dim: d.rope_head_dim, ratio: 4, rotate: true, eps: d.eps };
            iw = v4::IndexerW { wq_b: W::F32(&iwq), weights_proj: W::F32(&iwp) };
            id = v4::IndexerDims { hidden: d.hidden, n_heads: 64, head_dim: 128,
                rope_head_dim: d.rope_head_dim, q_lora_rank: d.q_lora_rank,
                index_topk: 512, ratio: 4 };
            Some((&iw, &id, &icw, &icd))
        } else { None };
        Some(v4::Compressed { w: &cw, d: &cd, indexer })
    } else { None };
    println!("layer {layer}: compress_ratio {ratio}, indexer {has_idx}");
    v4::attention_prefill(&mut out, &x, &w, &d, t_len, &rope, comp.as_ref());

    let mn = out.iter().cloned().fold(f32::INFINITY, f32::min);
    let mx = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("layer 0 attention, T={t_len}: out range [{mn:.6}, {mx:.6}], all finite = {}",
             out.iter().all(|v| v.is_finite()));

    let mut f = std::io::BufWriter::new(std::fs::File::create(&a[4]).unwrap());
    for v in &out {
        f.write_all(&v.to_bits().to_le_bytes()).unwrap();
    }
    println!("wrote {}", a[4]);
    ExitCode::SUCCESS
}
