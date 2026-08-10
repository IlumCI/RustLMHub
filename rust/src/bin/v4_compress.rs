// SPDX-License-Identifier: Apache-2.0
//
// Run DeepSeek-V4 layer 2's Compressor and Indexer on real checkpoint weights and dump
// both, so tools/verify_v4_compress.py can compare against an independent numpy parse.
//
// usage: v4_compress <shard_dir> <layer> <T> <out_prefix>

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

use k3::ops::W;
use k3::st::St;
use k3::v4;

fn deq_fp8(s: &St, name: &str, out_dim: usize, in_dim: usize, block: usize) -> Vec<f32> {
    let wt = s.find(&format!("{name}.weight")).unwrap_or_else(|| panic!("{name}.weight"));
    let st = s.find(&format!("{name}.scale")).unwrap_or_else(|| panic!("{name}.scale"));
    let (mut wb, mut sb) = (vec![0u8; wt.nbytes as usize], vec![0u8; st.nbytes as usize]);
    s.read(wt, &mut wb);
    s.read(st, &mut sb);
    let sb_in = in_dim.div_ceil(block);
    let mut o = vec![0f32; out_dim * in_dim];
    for r in 0..out_dim {
        for i in 0..in_dim {
            o[r * in_dim + i] = k3::st::f8_e4m3_to_f32(wb[r * in_dim + i])
                * k3::st::e8m0_to_f32(sb[(r / block) * sb_in + i / block]);
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

fn dump(path: &str, v: &[f32]) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    for x in v {
        f.write_all(&x.to_bits().to_le_bytes()).unwrap();
    }
}

fn main() -> ExitCode {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        eprintln!("usage: v4_compress <dir> <layer> <T> <out_prefix>");
        return ExitCode::from(2);
    }
    let (layer, t_len, pfx) = (&a[2], a[3].parse::<usize>().unwrap(), &a[4]);
    let s = match St::open(Path::new(&a[1])) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let p = |n: &str| format!("layers.{layer}.{n}");
    let has_indexer = s.find(&p("attn.indexer.wq_b.weight")).is_some();
    let ratio = if has_indexer { 4 } else { 128 };

    let (hidden, hd, rd) = (4096usize, 512usize, 64usize);
    let cd = if ratio == 4 { 2 * hd } else { hd };

    // The compressor's wkv/wgate ship as BF16; only the attention matrices are FP8.
    let ckv = f32_of(&s, &p("attn.compressor.wkv.weight"));
    let cg = f32_of(&s, &p("attn.compressor.wgate.weight"));
    let ape = f32_of(&s, &p("attn.compressor.ape"));
    let cn = f32_of(&s, &p("attn.compressor.norm.weight"));

    let cw = v4::CompressorW {
        ape: &ape,
        wkv: W::F32(&ckv),
        wgate: W::F32(&cg),
        norm: &cn,
    };
    let cd_dims = v4::CompressorDims {
        hidden,
        head_dim: hd,
        rope_head_dim: rd,
        ratio,
        rotate: false,
        eps: 1e-6,
    };
    assert_eq!(cd_dims.coff() * hd, cd, "coff must follow the ratio");

    // compress_ratio != 0 enables YaRN on compress_rope_theta.
    let rope = v4::precompute_rope(rd, t_len.max(2), 65536, 160000.0, 16.0, 32.0, 1.0);

    let x: Vec<f32> = (0..t_len * hidden).map(|i| ((i as f32) * 0.001).sin() * 0.05).collect();
    let kvc = v4::compress_prefill(&x, &cw, &cd_dims, t_len, &rope);
    let nblk = if kvc.is_empty() { 0 } else { kvc.len() / hd };
    println!("layer {layer}: ratio {ratio}, T={t_len} -> {nblk} compressed blocks of {hd}");
    if nblk > 0 {
        let mn = kvc.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = kvc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("  compressed kv range [{mn:.6}, {mx:.6}]");
    }
    dump(&format!("{pfx}_kvc.bin"), &kvc);

    if has_indexer {
        let ihd = 128usize;
        let ikv = f32_of(&s, &p("attn.indexer.compressor.wkv.weight"));
        let ig = f32_of(&s, &p("attn.indexer.compressor.wgate.weight"));
        let iape = f32_of(&s, &p("attn.indexer.compressor.ape"));
        let inn = f32_of(&s, &p("attn.indexer.compressor.norm.weight"));
        let iw = v4::CompressorW {
            ape: &iape,
            wkv: W::F32(&ikv),
            wgate: W::F32(&ig),
            norm: &inn,
        };
        let idims = v4::CompressorDims {
            hidden,
            head_dim: ihd,
            rope_head_dim: rd,
            ratio: 4,
            rotate: true, // the Indexer's compressor rotates
            eps: 1e-6,
        };
        let ikvc = v4::compress_prefill(&x, &iw, &idims, t_len, &rope);
        dump(&format!("{pfx}_ikvc.bin"), &ikvc);

        // q-LoRA from the attention block feeds the indexer.
        let wq_a = deq_fp8(&s, &p("attn.wq_a"), 1024, hidden, 128);
        let q_norm = f32_of(&s, &p("attn.q_norm.weight"));
        let mut qr = vec![0f32; t_len * 1024];
        let mut tmp = vec![0f32; 1024];
        for t in 0..t_len {
            k3::ops::mmw(&mut tmp, &x[t * hidden..][..hidden], W::F32(&wq_a), hidden, 1024);
            k3::ops::rmsnorm(&mut qr[t * 1024..][..1024], &tmp, &q_norm, 1024, 1e-6);
        }
        let iwq_b = deq_fp8(&s, &p("attn.indexer.wq_b"), 64 * ihd, 1024, 128);
        let iwp = f32_of(&s, &p("attn.indexer.weights_proj.weight"));
        let inw = v4::IndexerW { wq_b: W::F32(&iwq_b), weights_proj: W::F32(&iwp) };
        let ind = v4::IndexerDims {
            hidden,
            n_heads: 64,
            head_dim: ihd,
            rope_head_dim: rd,
            q_lora_rank: 1024,
            index_topk: 512,
            ratio: 4,
        };
        let idxs = v4::indexer_prefill(&qr, &x, &ikvc, &inw, &ind, t_len, &rope, 0);
        println!("  indexer: {} blocks of {ihd}, top-k per token:", ikvc.len() / ihd);
        for (t, row) in idxs.iter().enumerate() {
            println!("    t={t}: {row:?}");
        }
        let flat: Vec<f32> = idxs.iter().flat_map(|r| r.iter().map(|&v| v as f32)).collect();
        dump(&format!("{pfx}_idx.bin"), &flat);
    }
    println!("wrote {pfx}_*.bin");
    ExitCode::SUCCESS
}
