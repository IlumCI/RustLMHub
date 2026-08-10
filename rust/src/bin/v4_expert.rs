// SPDX-License-Identifier: Apache-2.0
//
// Dequantise one DeepSeek-V4 routed expert out of a real shard and dump it, so an
// independent Python decode can be compared against it byte for byte.
//
// usage: v4_expert <shard_dir> <weight_tensor> <scale_tensor> <out.bin>

use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

use k3::ops;
use k3::st::St;

fn main() -> ExitCode {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        eprintln!("usage: v4_expert <dir> <weight> <scale> <out.bin>");
        return ExitCode::from(2);
    }
    let s = match St::open(Path::new(&a[1])) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let (Some(wt), Some(st)) = (s.find(&a[2]), s.find(&a[3])) else {
        eprintln!("missing {} or {}", a[2], a[3]);
        return ExitCode::FAILURE;
    };

    // The packed weight is [rows, in/2] bytes; the scale is [rows, in/group].
    let rows = wt.shape[0] as usize;
    let pcols = wt.shape[1] as usize;
    let ngrp = st.shape[1] as usize;
    let width = pcols * 2;
    let group = width / ngrp;
    println!(
        "{}\n  packed {:?} {} -> logical [{rows}, {width}], {ngrp} groups of {group}\n  scale  {:?} {}",
        a[2], wt.shape, wt.dtype.name(), st.shape, st.dtype.name()
    );
    if group != ops::MXFP4_GROUP {
        println!("  NOTE: group {group} is not the MXFP4 group {}", ops::MXFP4_GROUP);
    }

    let mut packed = vec![0u8; wt.nbytes as usize];
    let mut scales = vec![0u8; st.nbytes as usize];
    s.read(wt, &mut packed);
    s.read(st, &mut scales);

    let mut out = vec![0f32; rows * width];
    ops::mxfp4_dequant(&mut out, &packed, &scales, rows, pcols, group);

    // The quantiser picks the scale so that amax/scale lands at fp4_max = 6.0
    // (inference/kernel.py:134). Checking that per group is an independent test of the
    // SCALE-to-GROUP association -- getting the row stride or group stride wrong shows
    // up here immediately. It says nothing about nibble order, which permutes elements
    // within a pair and so leaves every group statistic unchanged.
    let (mut over, mut at_max, mut zero_scale) = (0usize, 0usize, 0usize);
    for r in 0..rows {
        for g in 0..ngrp {
            let sb = scales[r * ngrp + g];
            if sb == 255 {
                zero_scale += 1;
                continue;
            }
            let sc = k3::st::e8m0_to_f32(sb);
            let lo = r * width + g * group;
            let amax = out[lo..lo + group].iter().fold(0f32, |m, v| m.max(v.abs()));
            let ratio = amax / sc;
            if ratio > 6.0 + 1e-3 {
                over += 1;
            }
            if (ratio - 6.0).abs() < 1e-3 {
                at_max += 1;
            }
        }
    }
    let total = rows * ngrp;
    println!(
        "  groups {total}: {at_max} ({:.1}%) hit amax/scale == 6.0 exactly, {over} exceed 6.0, {zero_scale} NaN scales",
        100.0 * at_max as f64 / total as f64
    );
    if over > 0 {
        println!("  FAIL: a group exceeding fp4_max means the scale is paired with the wrong group");
        return ExitCode::FAILURE;
    }

    let mut f = std::io::BufWriter::new(std::fs::File::create(&a[4]).unwrap());
    for v in &out {
        f.write_all(&v.to_bits().to_le_bytes()).unwrap();
    }
    println!("  wrote {} ({} floats)", a[4], out.len());
    ExitCode::SUCCESS
}
