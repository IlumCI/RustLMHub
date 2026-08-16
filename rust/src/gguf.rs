// SPDX-License-Identifier: Apache-2.0
//
// GGUF container reader, and the Q4_K super-block format.
//
// WHY THIS EXISTS
//     The engine could previously only read what a lab happened to publish as
//     safetensors, which in practice means BF16 or FP8 -- the two formats least suited to
//     a machine that is bandwidth-bound on storage. Every community that actually runs
//     large models on small hardware publishes GGUF, and they do it for exactly the
//     reason this port measured: at 4.25 bits a DeepSeek-V4 expert costs 13.37 MB, and
//     the same expert in BF16 costs 88 MB. k-quants are that trade, calibrated.
//
//     Reading GGUF turns "can we run this one checkpoint" into "we can run any large
//     model somebody has quantised", which is what `docs/MULTI_MODEL.md` set out to do.
//
// HOW IT INTEGRATES
//     `scan` produces the same `st::Tensor` records the safetensors scanner does -- name,
//     dtype, shape, byte offset, byte length. Everything downstream is byte-range based
//     and does not care what produced the range: `St::find`, `St::read_aligned`, the
//     expert cache, the sweep-aware eviction and the VRAM tier all work unchanged.
//
// LAYOUT (v2 and v3, little-endian throughout)
//     magic "GGUF" u32 | version u32 | tensor_count u64 | kv_count u64
//     kv_count  x  { key: str, type: u32, value }
//     tensor_count x  { name: str, n_dims: u32, dims: [u64; n_dims], type: u32, off: u64 }
//     padding to `general.alignment` (default 32)
//     tensor data, each tensor's `off` relative to the START of this section
//
//     A string is a u64 length followed by that many bytes, NOT NUL-terminated.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};

use rayon::prelude::*;

use crate::st::{Dtype, Tensor};

const MAGIC: u32 = 0x4655_4747; // "GGUF" read as a little-endian u32

/// Metadata value. GGUF's type tags are dense and small, so this mirrors them directly
/// rather than collapsing to strings -- `arch.rs` needs the integers as integers.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U(u64),
    I(i64),
    F(f64),
    Bool(bool),
    Str(String),
    Arr(Vec<Value>),
}

impl Value {
    pub fn as_u(&self) -> Option<u64> {
        match self {
            Value::U(v) => Some(*v),
            Value::I(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_f(&self) -> Option<f64> {
        match self {
            Value::F(v) => Some(*v),
            Value::U(v) => Some(*v as f64),
            Value::I(v) => Some(*v as f64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

pub type Meta = HashMap<String, Value>;

/// One GGML block format: how many elements a block holds, and how many bytes it takes.
///
/// `nbytes = numel / block_elems * block_bytes`, which is why a wrong table entry is not
/// a rounding error -- it desynchronises every tensor offset after it.
fn ggml_type(t: u32) -> Option<(Dtype, i64, i64)> {
    Some(match t {
        0 => (Dtype::F32, 1, 4),
        1 => (Dtype::F16, 1, 2),
        // The legacy 32-element blocks. Listed so a mixed file SCANS -- llama.cpp's
        // "Q4_K_M" is a recipe, not a single type, and the one used for Qwen2.5-0.5B is
        // 132 Q5_0 tensors against 12 Q4_K. A type in this table can have its offsets
        // computed; whether it can be dequantised is a separate question the kernel
        // answers.
        2 => (Dtype::Q4_0, 32, 18),  // d + 16 nibble bytes
        3 => (Dtype::Q4_1, 32, 20),  // d + m + 16
        6 => (Dtype::Q5_0, 32, 22),  // d + qh[4] + 16
        7 => (Dtype::Q5_1, 32, 24),  // d + m + qh[4] + 16
        8 => (Dtype::Q8_0, 32, 34),  // d + 32 int8
        // Codebook format: same footprint as Q4_0, but the nibble indexes a 16-entry
        // non-linear table rather than scaling directly. Listed for offsets only -- the
        // codebook is not implemented, so a tensor of this type cannot be decoded.
        // Verified against the reference on 120 tensors of a real Q2_K build.
        20 => (Dtype::IQ4NL, 32, 18),
        // The k-quant super-blocks, all 256 elements.
        10 => (Dtype::Q2K, 256, 84),   // scales[16] + qs[64] + d + dmin
        11 => (Dtype::Q3K, 256, 110),  // hmask[32] + qs[64] + scales[12] + d
        12 => (Dtype::Q4K, 256, 144),  // d + dmin + scales[12] + qs[128]
        13 => (Dtype::Q5K, 256, 176),  // d + dmin + scales[12] + qh[32] + qs[128]
        14 => (Dtype::Q6K, 256, 210),  // ql[128] + qh[64] + scales[16] + d
        30 => (Dtype::Bf16, 1, 2),
        _ => return None,
    })
}

/// The name GGUF gives a type, for error messages that name what is missing rather than
/// printing a bare integer.
fn type_name(t: u32) -> &'static str {
    match t {
        2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1", 9 => "Q8_1",
        10 => "Q2_K", 11 => "Q3_K", 13 => "Q5_K", 15 => "Q8_K",
        16 => "IQ2_XXS", 17 => "IQ2_XS", 18 => "IQ3_XXS", 19 => "IQ1_S", 20 => "IQ4_NL",
        21 => "IQ3_S", 22 => "IQ2_S", 23 => "IQ4_XS", 29 => "IQ1_M",
        _ => "unknown",
    }
}

struct Rd {
    buf: Vec<u8>,
    pos: usize,
}

impl Rd {
    fn take(&mut self, n: usize) -> io::Result<&[u8]> {
        if self.pos + n > self.buf.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "gguf: header truncated"));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn str(&mut self) -> io::Result<String> {
        let n = self.u64()? as usize;
        // A string running past the buffer means the header is longer than the window
        // read so far, NOT that the file is corrupt -- so this must be the same
        // UnexpectedEof that `take` raises, or `scan`'s grow-and-retry never fires and a
        // perfectly good file is rejected. (It was: a Q2_K build of Qwen2.5-0.5B failed
        // with "string past end" while two other builds of the same model parsed fine,
        // purely because its metadata happened to straddle the 1 MiB initial window.)
        // Genuine corruption is caught by `scan` giving up once the whole file is
        // buffered, and by the per-tensor bounds check against the file length.
        if self.pos + n > self.buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("gguf: {n}-byte string runs past the {}-byte header window", self.buf.len()),
            ));
        }
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn value(&mut self, ty: u32) -> io::Result<Value> {
        Ok(match ty {
            0 => Value::U(self.take(1)?[0] as u64),
            1 => Value::I(self.take(1)?[0] as i8 as i64),
            2 => Value::U(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as u64),
            3 => Value::I(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => Value::U(self.u32()? as u64),
            5 => Value::I(self.u32()? as i32 as i64),
            6 => Value::F(f32::from_bits(self.u32()?) as f64),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.str()?),
            9 => {
                let it = self.u32()?;
                let n = self.u64()? as usize;
                // Token vocabularies run to hundreds of thousands of entries; that is
                // normal and must not be mistaken for corruption.
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.value(it)?);
                }
                Value::Arr(v)
            }
            10 => Value::U(self.u64()?),
            11 => Value::I(self.u64()? as i64),
            12 => Value::F(f64::from_bits(self.u64()?)),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("gguf: metadata value type {ty}"),
                ))
            }
        })
    }
}

/// Parse one `.gguf` file's header into metadata plus tensor records.
///
/// Reads the header region only -- never the tensor data, which for a 74 GB quant is the
/// entire point of streaming it later.
pub fn scan(shard: usize, f: &File) -> io::Result<(Meta, Vec<Tensor>)> {
    scan_inner(shard, f, true)
}

/// Parse the header of a file whose TENSOR DATA is absent -- a partial download.
///
/// The bounds check that `scan` applies is exactly right for a real checkpoint (a tensor
/// pointing past the end means a truncated or corrupt file) and exactly wrong for a
/// deliberate head sample, where every tensor points past the end by design. This skips
/// only that check; every other validation still runs, so a malformed header is still
/// caught.
pub fn scan_header(shard: usize, f: &File) -> io::Result<(Meta, Vec<Tensor>)> {
    scan_inner(shard, f, false)
}

fn scan_inner(shard: usize, f: &File, whole_file: bool) -> io::Result<(Meta, Vec<Tensor>)> {
    // The header is unbounded in principle (the token vocabulary lives in it), so grow
    // until the parse succeeds rather than guessing one size.
    let flen = f.metadata()?.len();
    // A head sample has every tensor pointing past its end BY DESIGN, so the bounds check
    // is suppressed by claiming an unbounded file. Every other validation still runs, so a
    // malformed header is still caught -- only "does the data fit" is skipped, and for a
    // partial download that question is meaningless rather than false.
    let limit = if whole_file { flen } else { u64::MAX };
    let mut cap = (1 << 20).min(flen as usize);
    loop {
        let mut buf = vec![0u8; cap];
        (&*f).seek(SeekFrom::Start(0))?;
        let got = read_full(f, &mut buf)?;
        buf.truncate(got);
        let mut r = Rd { buf, pos: 0 };
        match parse(&mut r, shard, limit) {
            Ok(v) => return Ok(v),
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof && (cap as u64) < flen => {
                cap = (cap * 4).min(flen as usize);
            }
            Err(e) => return Err(e),
        }
    }
}

fn read_full(mut f: &File, buf: &mut [u8]) -> io::Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        match f.read(&mut buf[n..])? {
            0 => break,
            k => n += k,
        }
    }
    Ok(n)
}

fn parse(r: &mut Rd, shard: usize, flen: u64) -> io::Result<(Meta, Vec<Tensor>)> {
    let magic = r.u32()?;
    if magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "gguf: bad magic"));
    }
    let version = r.u32()?;
    if version != 2 && version != 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("gguf: version {version} (only 2 and 3 have the u64 layout this parses)"),
        ));
    }
    let n_tensors = r.u64()? as usize;
    let n_kv = r.u64()? as usize;

    let mut meta = Meta::new();
    for _ in 0..n_kv {
        let k = r.str()?;
        let t = r.u32()?;
        let v = r.value(t)?;
        meta.insert(k, v);
    }

    let mut infos = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name = r.str()?;
        let nd = r.u32()? as usize;
        if nd > 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("gguf: {name} has {nd} dimensions"),
            ));
        }
        // GGUF stores dims fastest-varying first; safetensors and this engine both use
        // row-major outer-first, so reverse. Getting this wrong transposes every matrix
        // in the file and still produces finite activations.
        let mut dims: Vec<i64> = Vec::with_capacity(nd);
        for _ in 0..nd {
            dims.push(r.u64()? as i64);
        }
        dims.reverse();
        let ty = r.u32()?;
        let off = r.u64()? as i64;
        infos.push((name, dims, ty, off));
    }

    let align = meta.get("general.alignment").and_then(Value::as_u).unwrap_or(32) as usize;
    if !align.is_power_of_two() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("gguf: general.alignment {align} is not a power of two"),
        ));
    }
    // Tensor offsets are relative to the data section, which begins at the first aligned
    // boundary after the header.
    let data = r.pos.next_multiple_of(align) as i64;

    let mut tensors = Vec::with_capacity(infos.len());
    for (name, shape, ty, off) in infos {
        let Some((dtype, belems, bbytes)) = ggml_type(ty) else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("gguf: {name} is {} (type {ty}), which this build cannot read", type_name(ty)),
            ));
        };
        let numel: i64 = shape.iter().product();
        if numel % belems != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("gguf: {name} has {numel} elements, not a multiple of the {belems}-element block"),
            ));
        }
        let nbytes = numel / belems * bbytes;
        let abs = data + off;
        if abs < 0 || (abs + nbytes) as u64 > flen {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("gguf: {name} runs past the end of the file"),
            ));
        }
        tensors.push(Tensor { name, shard, dtype, shape, off: abs, nbytes });
    }
    Ok((meta, tensors))
}

// ---------------------------------------------------------------------------------------
// The k-quant super-blocks
// ---------------------------------------------------------------------------------------

/// Elements per k-quant super-block. Q4_K, Q5_K and Q6_K all use 256.
pub const QK_K: usize = 256;
/// Bytes per Q4_K super-block: fp16 d, fp16 dmin, 12 packed scale bytes, 128 nibble bytes.
pub const Q4K_BLOCK: usize = 144;
/// Bytes per Q5_K super-block: Q4_K's 144 plus 32 bytes holding one extra bit per element.
pub const Q5K_BLOCK: usize = 176;
/// Bytes per Q6_K super-block: ql[128] + qh[64] + 16 SIGNED scales + fp16 d.
pub const Q6K_BLOCK: usize = 210;

/// Unpack the 6-bit scale and 6-bit min for one of the eight 32-element sub-blocks.
///
/// This is the part of Q4_K that is easy to implement plausibly and wrongly. The twelve
/// bytes hold eight 6-bit scales and eight 6-bit mins, and the packing is NOT uniform:
/// sub-blocks 0-3 take a clean six bits from bytes 0-3 and 4-7, while sub-blocks 4-7 take
/// their low four bits from bytes 8-11 and their high two bits from the top of bytes 0-7.
/// A version that reads all eight uniformly still yields scales in range, still decodes,
/// and is wrong for half of every tensor.
#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Dequantise `nblocks` Q4_K super-blocks into `out` (`nblocks * QK_K` floats).
///
/// Per super-block: two fp16 scales (`d` for the quantised scales, `dmin` for the
/// quantised mins), then eight 32-element sub-blocks each with its own 6-bit scale and
/// 6-bit min. The nibble order within a byte is low-then-high across a 32-element half:
/// the first 32 outputs take the LOW nibbles of 32 bytes, the next 32 take the HIGH
/// nibbles of the same 32 bytes -- not alternating within each byte.
/// Q8_0: blocks of 32, each `d` (f16) then 32 `int8`; `w[i] = d * q[i]`. 34 bytes/block.
pub fn q8_0_dequant(out: &mut [f32], src: &[u8], nblocks: usize) {
    const B: usize = 34;
    assert!(src.len() >= nblocks * B, "q8_0: source too short");
    assert!(out.len() >= nblocks * 32, "q8_0: destination too short");
    for b in 0..nblocks {
        let blk = &src[b * B..][..B];
        let d = crate::st::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dst = &mut out[b * 32..][..32];
        for i in 0..32 {
            dst[i] = d * (blk[2 + i] as i8) as f32;
        }
    }
}

pub fn q4k_dequant(out: &mut [f32], src: &[u8], nblocks: usize) {
    assert!(src.len() >= nblocks * Q4K_BLOCK, "q4k: source too short");
    assert!(out.len() >= nblocks * QK_K, "q4k: destination too short");
    for b in 0..nblocks {
        let blk = &src[b * Q4K_BLOCK..][..Q4K_BLOCK];
        let d = crate::st::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = crate::st::f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let dst = &mut out[b * QK_K..][..QK_K];

        let mut o = 0usize; // output element
        let mut q = 0usize; // nibble byte
        let mut is = 0usize; // sub-block index
        while o < QK_K {
            let (sc1, m1) = scale_min_k4(is, scales);
            let (sc2, m2) = scale_min_k4(is + 1, scales);
            let (d1, mn1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, mn2) = (d * sc2 as f32, dmin * m2 as f32);
            for l in 0..32 {
                dst[o + l] = d1 * (qs[q + l] & 0xF) as f32 - mn1;
            }
            for l in 0..32 {
                dst[o + 32 + l] = d2 * (qs[q + l] >> 4) as f32 - mn2;
            }
            o += 64;
            q += 32;
            is += 2;
        }
    }
}

/// Bytes per Q3_K super-block: hmask[32] + qs[64] + 12 packed 6-bit scales + fp16 d.
pub const Q3K_BLOCK: usize = 110;

/// Unpack Q3_K's sixteen 6-bit scales from its twelve bytes, each biased by -32.
///
/// The packing is a different puzzle from Q4_K's. Twelve bytes hold sixteen 6-bit values:
/// bytes 0-7 carry the low four bits of each, and bytes 8-11 carry the high two bits of
/// all sixteen, two bits at a time across four shift positions. `ggml` does this with four
/// `u32` words and two masks; done here explicitly so the shape is visible.
fn q3k_scales(src: &[u8]) -> [i8; 16] {
    let mut aux = [0u32; 4];
    for (i, a) in aux.iter_mut().enumerate().take(3) {
        *a = u32::from_le_bytes(src[i * 4..i * 4 + 4].try_into().unwrap());
    }
    const KM1: u32 = 0x0303_0303;
    const KM2: u32 = 0x0f0f_0f0f;
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KM2) | (((tmp >> 4) & KM1) << 4);
    aux[3] = ((aux[1] >> 4) & KM2) | (((tmp >> 6) & KM1) << 4);
    aux[0] = (aux[0] & KM2) | ((tmp & KM1) << 4);
    aux[1] = (aux[1] & KM2) | (((tmp >> 2) & KM1) << 4);
    let mut out = [0i8; 16];
    for (i, o) in out.iter_mut().enumerate() {
        // The stored value is unsigned 0..63; the real scale is biased by -32.
        *o = ((aux[i / 4] >> (8 * (i % 4))) & 0xFF) as i8;
    }
    out
}

/// Dequantise `nblocks` Q3_K super-blocks into `out`.
///
/// Three bits per weight, assembled from two places: the low two bits live in `qs`, and
/// the third lives in `hmask` -- **inverted**. A set mask bit means "add nothing", a clear
/// one means "subtract 4", so the quant runs -4..3 rather than 0..7. Reading the mask the
/// natural way round produces in-range values that are wrong for every weight whose high
/// bit is clear, which is roughly half of them.
pub fn q3k_dequant(out: &mut [f32], src: &[u8], nblocks: usize) {
    assert!(src.len() >= nblocks * Q3K_BLOCK, "q3k: source too short");
    assert!(out.len() >= nblocks * QK_K, "q3k: destination too short");
    for b in 0..nblocks {
        let blk = &src[b * Q3K_BLOCK..][..Q3K_BLOCK];
        let (hm, qs) = (&blk[0..32], &blk[32..96]);
        let sc = q3k_scales(&blk[96..108]);
        let d = crate::st::f16_to_f32(u16::from_le_bytes([blk[108], blk[109]]));
        let dst = &mut out[b * QK_K..][..QK_K];
        let mut m = 1u8;
        let mut is = 0usize;
        let mut o = 0usize;
        for n in 0..2 {
            let q = &qs[n * 32..][..32];
            for j in 0..4 {
                let shift = 2 * j;
                for half in 0..2usize {
                    let dl = d * (sc[is] - 32) as f32;
                    is += 1;
                    for l in 0..16 {
                        let i = half * 16 + l;
                        let lo = ((q[i] >> shift) & 3) as i32;
                        let hi = if hm[i] & m != 0 { 0 } else { 4 };
                        dst[o + l] = dl * (lo - hi) as f32;
                    }
                    o += 16;
                }
                m <<= 1;
            }
        }
    }
}

/// Dequantise `nblocks` Q5_K super-blocks into `out`.
///
/// Q4_K with one extra bit per element. The layout is the SAME two fp16 scales and the
/// SAME twelve 6-bit scale/min bytes, then 32 bytes of high bits, then the 128 nibble
/// bytes. The high bits are the trap: `qh` is indexed by position within the 32-element
/// half and is NOT advanced between sub-block pairs -- instead the bit MASK shifts left by
/// two each time round, so sub-block pair `p` reads bits `2p` and `2p+1` of `qh[l]`.
/// Advancing `qh` like `qs` reads the right number of bytes in the wrong order.
pub fn q5k_dequant(out: &mut [f32], src: &[u8], nblocks: usize) {
    assert!(src.len() >= nblocks * Q5K_BLOCK, "q5k: source too short");
    assert!(out.len() >= nblocks * QK_K, "q5k: destination too short");
    for b in 0..nblocks {
        let blk = &src[b * Q5K_BLOCK..][..Q5K_BLOCK];
        let d = crate::st::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let dmin = crate::st::f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let scales = &blk[4..16];
        let qh = &blk[16..48];
        let qs = &blk[48..176];
        let dst = &mut out[b * QK_K..][..QK_K];

        let (mut o, mut q, mut is) = (0usize, 0usize, 0usize);
        let (mut u1, mut u2) = (1u8, 2u8);
        while o < QK_K {
            let (sc1, m1) = scale_min_k4(is, scales);
            let (sc2, m2) = scale_min_k4(is + 1, scales);
            let (d1, mn1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, mn2) = (d * sc2 as f32, dmin * m2 as f32);
            for l in 0..32 {
                let hi = if qh[l] & u1 != 0 { 16u8 } else { 0 };
                dst[o + l] = d1 * ((qs[q + l] & 0xF) + hi) as f32 - mn1;
            }
            for l in 0..32 {
                let hi = if qh[l] & u2 != 0 { 16u8 } else { 0 };
                dst[o + 32 + l] = d2 * ((qs[q + l] >> 4) + hi) as f32 - mn2;
            }
            o += 64;
            q += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

/// Dequantise `nblocks` Q6_K super-blocks into `out`.
///
/// Structurally unlike Q4_K/Q5_K: there is no min term, the scales are **signed** 8-bit
/// (not 6-bit unsigned packed), and the quants are recentred by subtracting 32 so they run
/// -32..31. It also strides in 128-element halves rather than 64, taking four interleaved
/// 32-element groups per half at output offsets 0, 32, 64 and 96 -- reading them as four
/// consecutive runs decodes every value into the wrong place.
pub fn q6k_dequant(out: &mut [f32], src: &[u8], nblocks: usize) {
    assert!(src.len() >= nblocks * Q6K_BLOCK, "q6k: source too short");
    assert!(out.len() >= nblocks * QK_K, "q6k: destination too short");
    for b in 0..nblocks {
        let blk = &src[b * Q6K_BLOCK..][..Q6K_BLOCK];
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let sc = &blk[192..208];
        let d = crate::st::f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));
        let dst = &mut out[b * QK_K..][..QK_K];

        for n in 0..2 {
            let (yo, qlo, qho, sco) = (n * 128, n * 64, n * 32, n * 8);
            for l in 0..32 {
                let is = l / 16;
                let h = qh[qho + l];
                let q1 = ((ql[qlo + l] & 0xF) | ((h & 3) << 4)) as i32 - 32;
                let q2 = ((ql[qlo + l + 32] & 0xF) | (((h >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[qlo + l] >> 4) | (((h >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[qlo + l + 32] >> 4) | (((h >> 6) & 3) << 4)) as i32 - 32;
                // The scales are i8: a negative scale is legal and flips the sign.
                dst[yo + l] = d * sc[sco + is] as i8 as f32 * q1 as f32;
                dst[yo + l + 32] = d * sc[sco + is + 2] as i8 as f32 * q2 as f32;
                dst[yo + l + 64] = d * sc[sco + is + 4] as i8 as f32 * q3 as f32;
                dst[yo + l + 96] = d * sc[sco + is + 6] as i8 as f32 * q4 as f32;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// Fused matmuls
// ---------------------------------------------------------------------------------------
//
// A row is dequantised straight into the dot product, never into a buffer. Dequantising
// first would cost 4 bytes per weight in traffic to save nothing: the whole reason
// k-quants exist is that the packed form is what fits down the pipe.
//
// THE AFFINE TERM, which is what makes these unlike `matmul_mxfp4`.
//     Q4_K and Q5_K reconstruct a weight as `d*sc*q - dmin*m`, so the min does NOT factor
//     out of a dot product the way a pure scale does:
//
//         sum_i (d*sc*q_i - dmin*m) * x_i  =  d*sc * sum_i(q_i * x_i)  -  dmin*m * sum_i(x_i)
//
//     Each 32-element sub-block therefore needs TWO running sums: the weighted one and a
//     plain sum of the activations. Dropping the second term leaves a matmul that is
//     finite, plausible and wrong -- exactly the failure this file's verifier exists for.
//     Q6_K has no min and is a pure scale, so it needs only the first.
//
// AVX2 IS DELIBERATELY NOT HERE YET. `matmul_mxfp4`'s vector path is bit-identical to its
// scalar one only because of a carefully chosen `i % 8` lane partition that reproduces the
// scalar fold tree exactly (ops.rs:592-653). Reproducing that for an affine form with two
// accumulators is a separate piece of work, and shipping a fast path that is merely CLOSE
// to the scalar one would silently break the bit-exactness contract every other kernel in
// this engine holds. Scalar first, measured, then vectorised if it is on the critical path.

/// Dot product with a FIXED eight-lane partition: lane `j` accumulates every element with
/// `i % 8 == j`, then the lanes fold as `(s0+s4 + s1+s5) + (s2+s6 + s3+s7)`. `n` must be a
/// multiple of 8.
///
/// The partition exists so the scalar and AVX2 paths are **bit-identical**, which they are
/// for a reason worth stating: both operands come from `f32`, and the product of two f32
/// values is EXACT in f64 (24-bit mantissas give 48 significant bits, under f64's 53). So
/// `fmadd_pd`'s single rounding and the scalar's multiply-then-add agree exactly. Feed this
/// anything that is not f32-derived and that guarantee is gone.
#[inline]
fn dot_p8(w: &[f32], x: &[f32], n: usize) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: features checked above; both slices are asserted >= n below.
            return unsafe { dot_p8_avx2(w, x, n) };
        }
    }
    dot_p8_scalar(w, x, n)
}

#[inline]
fn dot_p8_scalar(w: &[f32], x: &[f32], n: usize) -> f64 {
    let mut s = [0f64; 8];
    for i in 0..n {
        s[i & 7] += w[i] as f64 * x[i] as f64;
    }
    ((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7]))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_p8_avx2(w: &[f32], x: &[f32], n: usize) -> f64 {
    use std::arch::x86_64::*;
    debug_assert_eq!(n % 8, 0);
    debug_assert!(w.len() >= n && x.len() >= n);
    let (mut u0, mut u1) = (_mm256_setzero_pd(), _mm256_setzero_pd());
    let mut i = 0usize;
    while i < n {
        let wv = _mm256_loadu_ps(w.as_ptr().add(i));
        let xv = _mm256_loadu_ps(x.as_ptr().add(i));
        // u0 takes lanes 0-3 of each group of 8, u1 takes 4-7, so lane j of u0 holds
        // i % 8 == j and lane j of u1 holds i % 8 == j + 4.
        u0 = _mm256_fmadd_pd(
            _mm256_cvtps_pd(_mm256_castps256_ps128(wv)),
            _mm256_cvtps_pd(_mm256_castps256_ps128(xv)),
            u0,
        );
        u1 = _mm256_fmadd_pd(
            _mm256_cvtps_pd(_mm256_extractf128_ps(wv, 1)),
            _mm256_cvtps_pd(_mm256_extractf128_ps(xv, 1)),
            u1,
        );
        i += 8;
    }
    // lane j of u = s[j] + s[j+4]; hadd then gives (t0+t1, t2+t3).
    let u = _mm256_add_pd(u0, u1);
    let h = _mm_hadd_pd(_mm256_castpd256_pd128(u), _mm256_extractf128_pd(u, 1));
    _mm_cvtsd_f64(h) + _mm_cvtsd_f64(_mm_unpackhi_pd(h, h))
}

/// Per-32-element sums of the activations, computed ONCE for the whole matmul.
///
/// The affine term `dmin * m * sum(x)` depends only on `x`, not on the row, so hoisting it
/// out of the row loop removes 32 adds per sub-block per row -- on a 4096-row projection
/// that is the difference between doing this work once and 4096 times.
fn xsums(x: &[f32], n_sub: usize) -> Vec<f64> {
    (0..n_sub)
        .map(|s| x[s * 32..][..32].iter().map(|v| *v as f64).sum())
        .collect()
}

/// `y[r] = row r of the Q4_K matrix, dotted with x`. `k_in` must be a multiple of `QK_K`.
pub fn matmul_q4k(y: &mut [f32], x: &[f32], src: &[u8], k_in: usize, rows: usize) {
    kq_matmul(y, x, src, k_in, rows, Q4K_BLOCK, q4k_row)
}

/// `y[r] = row r of the Q5_K matrix, dotted with x`. `k_in` must be a multiple of `QK_K`.
pub fn matmul_q5k(y: &mut [f32], x: &[f32], src: &[u8], k_in: usize, rows: usize) {
    kq_matmul(y, x, src, k_in, rows, Q5K_BLOCK, q5k_row)
}

/// `y[r] = row r of the Q3_K matrix, dotted with x`. `k_in` must be a multiple of `QK_K`.
pub fn matmul_q3k(y: &mut [f32], x: &[f32], src: &[u8], k_in: usize, rows: usize) {
    kq_matmul(y, x, src, k_in, rows, Q3K_BLOCK, q3k_row)
}

/// `y[r] = row r of the Q6_K matrix, dotted with x`. `k_in` must be a multiple of `QK_K`.
pub fn matmul_q6k(y: &mut [f32], x: &[f32], src: &[u8], k_in: usize, rows: usize) {
    kq_matmul(y, x, src, k_in, rows, Q6K_BLOCK, q6k_row)
}

/// The same three matmuls against `ntok` activation vectors at once.
///
/// WHY THIS EXISTS
/// ```text
///     A prompt chunk of `k` tokens used to call `matmul_q4k` `k` times, which decodes every
///     weight row `k` times to use it once. Decoding is not a small share of the work: a
///     Q4_K super-block unpacks 256 nibbles, applies a 6-bit scale to each and writes 256
///     floats, against a dot product of the same 256 elements -- so the row is being
///     rebuilt at roughly the cost of using it.
///
///     Measured on this machine (16 cores, AVX2), single 2048x8192 Q4_K matvec:
///     17.4 GMAC/s against a ~380 GMAC/s f32 FMA peak. That gap is arithmetic intensity,
///     not threads -- the same kernel at one thread gives 3.6 GMAC/s, so rayon is already
///     doing its part. Sharing one decode across the chunk is what raises it.
/// ```
///
/// `x` and `y` are TOKEN-MAJOR: `x[t * k_in ..]` and `y[t * rows ..]`. Rows are the axis
/// that fans out across threads, so the kernel accumulates into a row-major scratch and
/// transposes once at the end; at `rows * ntok` floats that transpose is noise next to the
/// matmul it follows.
///
/// Bit-identical to `ntok` separate calls, by construction rather than by tolerance --
/// both paths reduce through `kq_row`'s block-major order over the same `BlockFn`.
pub fn matmul_q4k_many(
    y: &mut [f32],
    x: &[f32],
    src: &[u8],
    k_in: usize,
    rows: usize,
    ntok: usize,
) {
    kq_matmul_many(y, x, src, k_in, rows, ntok, Q4K_BLOCK, q4k_block)
}

pub fn matmul_q5k_many(
    y: &mut [f32],
    x: &[f32],
    src: &[u8],
    k_in: usize,
    rows: usize,
    ntok: usize,
) {
    kq_matmul_many(y, x, src, k_in, rows, ntok, Q5K_BLOCK, q5k_block)
}

pub fn matmul_q6k_many(
    y: &mut [f32],
    x: &[f32],
    src: &[u8],
    k_in: usize,
    rows: usize,
    ntok: usize,
) {
    kq_matmul_many(y, x, src, k_in, rows, ntok, Q6K_BLOCK, q6k_block)
}

fn kq_matmul_many(
    y: &mut [f32],
    x: &[f32],
    src: &[u8],
    k_in: usize,
    rows: usize,
    ntok: usize,
    blk_bytes: usize,
    block: BlockFn,
) {
    assert_eq!(k_in % QK_K, 0, "k-quant rows are whole super-blocks: {k_in} % {QK_K} != 0");
    let stride = k_in / QK_K * blk_bytes;
    assert!(
        src.len() >= rows * stride,
        "batched k-quant matmul wants {rows} rows x {stride} B = {} B but the tensor holds \
         {} B (k_in {k_in}, block {blk_bytes}) -- the shape passed does not match the weight",
        rows * stride,
        src.len()
    );
    assert_eq!(x.len(), ntok * k_in, "x must be {ntok} token-major vectors of {k_in}");
    assert_eq!(y.len(), ntok * rows, "y must be {ntok} token-major vectors of {rows}");
    let n_sub = k_in / 32;
    // Hoisted per token, exactly as the single-vector path hoists it per call.
    let xs: Vec<f64> = (0..ntok).flat_map(|t| xsums(&x[t * k_in..][..k_in], n_sub)).collect();

    let mut z = vec![0f32; rows * ntok];
    let run = |z: &mut [f32], src: &[u8], base_rows: usize| {
        let mut acc = vec![0f64; ntok];
        for r in 0..base_rows {
            let row = &src[r * stride..][..stride];
            kq_row_many(row, blk_bytes, block, x, k_in, &xs, ntok, &mut acc);
            for t in 0..ntok {
                z[r * ntok + t] = acc[t] as f32;
            }
        }
    };
    if rows > crate::ops::par_min_rows() {
        let chunk = crate::ops::row_chunk_pub(rows);
        z.par_chunks_mut(chunk * ntok).enumerate().for_each(|(c, zc)| {
            let base = c * chunk;
            run(zc, &src[base * stride..], zc.len() / ntok);
        });
    } else {
        run(&mut z, src, rows);
    }
    for r in 0..rows {
        for t in 0..ntok {
            y[t * rows + r] = z[r * ntok + t];
        }
    }
}

/// How many tokens share one weight decode before the activations stop fitting in cache.
///
/// THIS IS A REAL OPTIMUM, NOT A GUESS AT ONE
/// ```text
///     Sharing a decode across `T` tokens costs roughly `256/T + 64` units per token
///     (unpack a super-block, then one 256-element dot each), so the return on `T` is steep
///     to about 8 and flat after 16. What rises with `T` is the activation footprint: the
///     row loop re-reads all `T * k_in` floats for every one of thousands of rows, so once
///     that block leaves L2 the kernel starts streaming it from L3 per row.
///
///     Measured on Qwen3.6-35B, 128 prompt tokens: untiled, a chunk of 128 took 142.2 s
///     against 72.8 s for a chunk of 8 -- the batching that WON at width 8 lost twice as
///     much at width 128. Tiling to a fixed activation budget is what makes a wide chunk
///     safe, and a wide chunk is what the routed experts need to share anything.
///
///     128 KB of activations, clamped to [4, 32]: k_in 2048 gives 16, k_in 512 gives 32.
/// ```
fn token_tile(k_in: usize, ntok: usize) -> usize {
    ((128 << 10) / (k_in * 4)).clamp(4, 32).min(ntok).max(1)
}

/// One weight row against every token in the chunk. `acc` is overwritten, not accumulated.
///
/// Tiled over tokens, which changes nothing arithmetically -- each token's accumulation is
/// still block-major over the whole row -- so the result is independent of the tile.
fn kq_row_many(
    row: &[u8],
    blk_bytes: usize,
    block: BlockFn,
    x: &[f32],
    k_in: usize,
    xs: &[f64],
    ntok: usize,
    acc: &mut [f64],
) {
    let n_sub = k_in / 32;
    acc[..ntok].fill(0.0);
    let mut w = [0f32; QK_K];
    let mut dm = [0f64; 8];
    let tile = token_tile(k_in, ntok);
    let mut t0 = 0;
    while t0 < ntok {
        let t1 = (t0 + tile).min(ntok);
        for b in 0..k_in / QK_K {
            // The decode every token in this tile is about to share.
            let d = block(&row[b * blk_bytes..][..blk_bytes], &mut w, &mut dm);
            for t in t0..t1 {
                let mut mins = 0.0f64;
                for sub in 0..8 {
                    mins += dm[sub] * xs[t * n_sub + b * 8 + sub];
                }
                acc[t] += d * dot_p8(&w, &x[t * k_in + b * QK_K..][..QK_K], QK_K) - mins;
            }
        }
        t0 = t1;
    }
}

/// Shared fan-out. One stride, unlike MXFP4's two: a k-quant carries its scales inside the
/// super-block, so a chunk of rows offsets into exactly one buffer.
fn kq_matmul(
    y: &mut [f32],
    x: &[f32],
    src: &[u8],
    k_in: usize,
    rows: usize,
    block: usize,
    row_fn: fn(&[u8], &[f32], usize, &[f64]) -> f64,
) {
    assert_eq!(k_in % QK_K, 0, "k-quant rows are whole super-blocks: {k_in} % {QK_K} != 0");
    let stride = k_in / QK_K * block;
    // Name the mismatch. Without this a wrong (k_in, rows) pair surfaces as an opaque
    // slice-index panic inside a rayon worker, with no frame naming the caller.
    assert!(
        src.len() >= rows * stride,
        "k-quant matmul wants {rows} rows x {stride} B = {} B but the tensor holds {} B \
         (k_in {k_in}, block {block}) -- the shape passed does not match the weight",
        rows * stride,
        src.len()
    );
    let xs = xsums(x, k_in / 32);
    if rows > crate::ops::par_min_rows() {
        let chunk = crate::ops::row_chunk_pub(rows);
        y[..rows].par_chunks_mut(chunk).enumerate().for_each(|(c, yc)| {
            let base = c * chunk;
            kq_serial(yc, x, &src[base * stride..], k_in, yc.len(), stride, row_fn, &xs);
        });
        return;
    }
    kq_serial(y, x, src, k_in, rows, stride, row_fn, &xs);
}

fn kq_serial(
    y: &mut [f32],
    x: &[f32],
    src: &[u8],
    k_in: usize,
    rows: usize,
    stride: usize,
    row_fn: fn(&[u8], &[f32], usize, &[f64]) -> f64,
    xs: &[f64],
) {
    for r in 0..rows {
        y[r] = row_fn(&src[r * stride..][..stride], x, k_in, xs) as f32;
    }
}

// All three row functions share one shape, and the shape is the optimisation.
//
// The naive form applies each sub-block's scale to its own dot product:
//     acc += d * sc_j * dot(q_j, x_j)      -- one horizontal reduction per 16 or 32 elements
// Q6_K's scale changes every 16 elements, so that is SIXTEEN reductions per super-block
// against only two accumulate steps each. Measured: 5.08 GMAC/s, against Q4_K's 14.15.
//
// Instead, fold the scale into the WEIGHTS and reduce once per super-block:
//     w[i] = sc_of_subblock(i) * q_i       (an exact integer: |sc*q| <= 4064 < 2^24)
//     acc += d * dot(w, x)                 -- ONE reduction per 256 elements
//
// This is exact rather than approximate, and it preserves `dot_p8`'s precondition: the
// folded weight is still f32-derived, so every product stays exact in f64 and the AVX2
// path stays bit-identical to the scalar one. The affine min term factors out entirely and
// is accumulated separately in f64, where it costs nothing.

/// Decode ONE super-block: fill `w` with the integer-scaled weights and `dm` with the
/// per-sub affine coefficients, returning the block scale `d`.
///
/// A k-quant row reconstructs as `sum_b (d_b * dot(w_b, x_b) - sum_sub dm_sub * sum(x_sub))`,
/// and everything before `x` appears in that expression depends on the WEIGHT alone. That
/// is the whole reason a chunk of tokens can share it: decode once, dot many times.
///
/// Split out rather than duplicated. The single-vector kernels below are written in terms
/// of these, so the batched path cannot drift from the serial one -- a divergence that
/// would show up as a model that is subtly wrong only when a prompt is long enough to
/// batch, which is the least debuggable shape a bug can have here.
type BlockFn = fn(&[u8], &mut [f32; QK_K], &mut [f64; 8]) -> f64;

fn q4k_block(blk: &[u8], w: &mut [f32; QK_K], dm: &mut [f64; 8]) -> f64 {
    let d = crate::st::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) as f64;
    let dmin = crate::st::f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])) as f64;
    let (scales, qs) = (&blk[4..16], &blk[16..144]);
    for p in 0..4 {
        let q = &qs[p * 32..][..32];
        for half in 0..2usize {
            let (sc, m) = scale_min_k4(p * 2 + half, scales);
            let sub = p * 2 + half;
            let dst = &mut w[sub * 32..][..32];
            for l in 0..32 {
                let v = if half == 0 { q[l] & 0xF } else { q[l] >> 4 };
                dst[l] = (sc as u32 * v as u32) as f32;
            }
            dm[sub] = dmin * m as f64;
        }
    }
    d
}

fn q4k_row(row: &[u8], x: &[f32], k_in: usize, xs: &[f64]) -> f64 {
    kq_row(row, Q4K_BLOCK, q4k_block, x, k_in, xs)
}

/// One row against one activation vector, in terms of the block decoder above.
///
/// The accumulation order is fixed here and nowhere else: block-major, and within a block
/// `d * dot - mins` with `mins` summed over subs in index order. Both the serial and the
/// batched matmul go through this shape, so they agree bit for bit by construction rather
/// than by testing.
fn kq_row(
    row: &[u8],
    blk_bytes: usize,
    block: BlockFn,
    x: &[f32],
    k_in: usize,
    xs: &[f64],
) -> f64 {
    let mut acc = 0.0f64;
    let mut w = [0f32; QK_K];
    let mut dm = [0f64; 8];
    for b in 0..k_in / QK_K {
        let d = block(&row[b * blk_bytes..][..blk_bytes], &mut w, &mut dm);
        let mut mins = 0.0f64;
        for sub in 0..8 {
            mins += dm[sub] * xs[b * 8 + sub];
        }
        acc += d * dot_p8(&w, &x[b * QK_K..][..QK_K], QK_K) - mins;
    }
    acc
}

fn q5k_block(blk: &[u8], w: &mut [f32; QK_K], dm: &mut [f64; 8]) -> f64 {
    let d = crate::st::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) as f64;
    let dmin = crate::st::f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])) as f64;
    let (scales, qh, qs) = (&blk[4..16], &blk[16..48], &blk[48..176]);
    for p in 0..4 {
        let q = &qs[p * 32..][..32];
        for half in 0..2usize {
            let (sc, m) = scale_min_k4(p * 2 + half, scales);
            let sub = p * 2 + half;
            // The high bit is selected by a mask that shifts left by two per sub-block
            // PAIR; qh is never advanced the way qs is.
            let bit = 1u8 << (2 * p + half);
            let dst = &mut w[sub * 32..][..32];
            for l in 0..32 {
                let lo = if half == 0 { q[l] & 0xF } else { q[l] >> 4 };
                let v = lo + if qh[l] & bit != 0 { 16 } else { 0 };
                dst[l] = (sc as u32 * v as u32) as f32;
            }
            dm[sub] = dmin * m as f64;
        }
    }
    d
}

fn q5k_row(row: &[u8], x: &[f32], k_in: usize, xs: &[f64]) -> f64 {
    kq_row(row, Q5K_BLOCK, q5k_block, x, k_in, xs)
}

fn q3k_row(row: &[u8], x: &[f32], k_in: usize, _xs: &[f64]) -> f64 {
    let mut acc = 0.0f64;
    let mut w = [0f32; QK_K];
    for b in 0..k_in / QK_K {
        let blk = &row[b * Q3K_BLOCK..][..Q3K_BLOCK];
        let (hm, qs) = (&blk[0..32], &blk[32..96]);
        let sc = q3k_scales(&blk[96..108]);
        let d = crate::st::f16_to_f32(u16::from_le_bytes([blk[108], blk[109]])) as f64;
        let xb = &x[b * QK_K..][..QK_K];
        // |(sc-32) * (lo-hi)| <= 32*4 = 128, so the folded weight is an exact small
        // integer and dot_p8 keeps its f32-derived precondition.
        let (mut m, mut is, mut o) = (1u8, 0usize, 0usize);
        for n in 0..2 {
            let q = &qs[n * 32..][..32];
            for j in 0..4 {
                let shift = 2 * j;
                for half in 0..2usize {
                    let s = (sc[is] - 32) as i32;
                    is += 1;
                    for l in 0..16 {
                        let i = half * 16 + l;
                        let lo = ((q[i] >> shift) & 3) as i32;
                        // Left as a branch deliberately: the "branchless" arithmetic
                        // form measured SLOWER (4.00 vs 4.90 GMAC/s), so the compiler is
                        // already selecting better code than the manual rewrite.
                        let hi = if hm[i] & m != 0 { 0 } else { 4 };
                        w[o + l] = (s * (lo - hi)) as f32;
                    }
                    o += 16;
                }
                m <<= 1;
            }
        }
        acc += d * dot_p8(&w, xb, QK_K);
    }
    acc
}

fn q6k_block(blk: &[u8], w: &mut [f32; QK_K], dm: &mut [f64; 8]) -> f64 {
    let (ql, qh, sc) = (&blk[0..128], &blk[128..192], &blk[192..208]);
    let d = crate::st::f16_to_f32(u16::from_le_bytes([blk[208], blk[209]])) as f64;
    // No min term. Zeroed rather than skipped, because `kq_row` sums `dm` unconditionally
    // and this array is reused across blocks. Subtracting an exact 0.0 changes nothing.
    *dm = [0.0; 8];
    // The four 32-element groups are INTERLEAVED at output offsets 0, 32, 64 and 96 of
    // each 128-element half, and the signed scale changes every 16.
    for n in 0..2 {
        let (yo, qlo, qho, sco) = (n * 128, n * 64, n * 32, n * 8);
        for l in 0..32 {
            let is = l / 16;
            let h = qh[qho + l];
            let (a, c) = (ql[qlo + l], ql[qlo + l + 32]);
            let q = [
                ((a & 0xF) | ((h & 3) << 4)) as i32 - 32,
                ((c & 0xF) | (((h >> 2) & 3) << 4)) as i32 - 32,
                ((a >> 4) | (((h >> 4) & 3) << 4)) as i32 - 32,
                ((c >> 4) | (((h >> 6) & 3) << 4)) as i32 - 32,
            ];
            for (g, qg) in q.iter().enumerate() {
                w[yo + l + 32 * g] = (sc[sco + is + 2 * g] as i8 as i32 * qg) as f32;
            }
        }
    }
    d
}

fn q6k_row(row: &[u8], x: &[f32], k_in: usize, xs: &[f64]) -> f64 {
    kq_row(row, Q6K_BLOCK, q6k_block, x, k_in, xs)
}

// --- Q8 activation path for Q4_K: int8 x int8 instead of dequant-to-f32 ---
//
// The forward matmul dequantises each Q4_K weight to f32 and dots it with the f32 activation.
// llama.cpp's fast path instead quantises the ACTIVATION to int8 once, keeps the weight
// nibbles as integers, and does an integer dot -- `maddubs` retires 32 int8 MACs per AVX2
// instruction where the f32 FMA does 8. The result is not bit-identical to the f32 kernel
// (the int8 activation rounds), but the error is ~100x under the Q4_K weight-quant noise, and
// it IS the computation llama.cpp ships. See `dense-speedup-levers`.

/// One activation vector quantised to int8: `q8` (per element), `d8` (one scale per 256-wide
/// super-block, Q8_K style), and `sumq8` (int8 sum per 32-wide sub-block, for Q4_K's min
/// term). Built ONCE per matmul and reused across every output row.
pub struct Q8Act {
    q8: Vec<i8>,
    d8: Vec<f32>,
    sumq8: Vec<i32>,
}

impl Q8Act {
    pub fn quantize(x: &[f32], k_in: usize) -> Q8Act {
        assert_eq!(k_in % QK_K, 0, "k-quant activation is whole super-blocks");
        let nb = k_in / QK_K;
        let mut q8 = vec![0i8; k_in];
        let mut d8 = vec![0f32; nb];
        let mut sumq8 = vec![0i32; nb * 8];
        for b in 0..nb {
            let xb = &x[b * QK_K..][..QK_K];
            let amax = xb.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let d = amax / 127.0;
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            d8[b] = d;
            for l in 0..QK_K {
                let q = (xb[l] * inv).round().clamp(-127.0, 127.0) as i32;
                q8[b * QK_K + l] = q as i8;
                sumq8[b * 8 + l / 32] += q;
            }
        }
        Q8Act { q8, d8, sumq8 }
    }
}

/// `Σ W[r]·x ≈ Σ_sub (d·d8·sc·intdot − dmin·d8·m·sumq8)`, the int8 dot for one Q4_K row.
fn q4k_row_q8_scalar(row: &[u8], act: &Q8Act, k_in: usize) -> f64 {
    let nb = k_in / QK_K;
    let mut acc = 0f64;
    for b in 0..nb {
        let blk = &row[b * Q4K_BLOCK..][..Q4K_BLOCK];
        let d = crate::st::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) as f64;
        let dmin = crate::st::f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])) as f64;
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let d8 = act.d8[b] as f64;
        for sub in 0..8 {
            let (sc, m) = scale_min_k4(sub, scales);
            let (p, half) = (sub / 2, sub & 1);
            let q8s = &act.q8[b * QK_K + sub * 32..][..32];
            let mut intdot = 0i32;
            for l in 0..32 {
                let v = ((qs[p * 32 + l] >> (4 * half)) & 0xF) as i32;
                intdot += v * q8s[l] as i32;
            }
            acc += d * d8 * sc as f64 * intdot as f64 - dmin * d8 * m as f64 * act.sumq8[b * 8 + sub] as f64;
        }
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q4k_row_q8_avx2(row: &[u8], act: &Q8Act, k_in: usize) -> f64 {
    use std::arch::x86_64::*;
    #[inline]
    unsafe fn hsum_i32(v: __m256i) -> i32 {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256(v, 1);
        let s = _mm_add_epi32(lo, hi); // 4 i32
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_00_01));
        _mm_cvtsi128_si32(s)
    }
    let nb = k_in / QK_K;
    let lomask = _mm256_set1_epi8(0x0F);
    let ones = _mm256_set1_epi16(1);
    let mut acc = 0f64;
    for b in 0..nb {
        let blk = &row[b * Q4K_BLOCK..][..Q4K_BLOCK];
        let d = crate::st::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) as f64;
        let dmin = crate::st::f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])) as f64;
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let d8 = act.d8[b] as f64;
        for sub in 0..8 {
            let (sc, m) = scale_min_k4(sub, scales);
            let (p, half) = (sub / 2, sub & 1);
            let qbytes = _mm256_loadu_si256(qs[p * 32..].as_ptr().cast());
            // low nibble of each byte, or the high nibble via >>4 (per-byte, the epi16 shift +
            // low-nibble mask isolates it exactly).
            let nib = if half == 0 {
                _mm256_and_si256(qbytes, lomask)
            } else {
                _mm256_and_si256(_mm256_srli_epi16(qbytes, 4), lomask)
            };
            let q8v = _mm256_loadu_si256(act.q8[b * QK_K + sub * 32..].as_ptr().cast());
            // maddubs: unsigned nibble x signed int8 -> i16 pairs (no saturation: 2*15*127 < 32767)
            let p16 = _mm256_maddubs_epi16(nib, q8v);
            let intdot = hsum_i32(_mm256_madd_epi16(p16, ones));
            acc += d * d8 * sc as f64 * intdot as f64 - dmin * d8 * m as f64 * act.sumq8[b * 8 + sub] as f64;
        }
    }
    acc
}

#[inline]
fn q4k_row_q8(row: &[u8], act: &Q8Act, k_in: usize) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 checked here.
            return unsafe { q4k_row_q8_avx2(row, act, k_in) };
        }
    }
    q4k_row_q8_scalar(row, act, k_in)
}

/// The Q6_K int8 dot, scalar reference. Q6_K is symmetric (`W = d·sc·(q−32)`, `q` a 6-bit
/// value in `0..64`, `sc` a signed int8 per 16 elements, no min). The integer sum
/// `Σ sc·(q−32)·q8` is accumulated in i64 (order-independent, so the AVX2 kernel matches it
/// bit-for-bit), then scaled by `d·d8` once per super-block. The reconstruction mirrors
/// [`q6k_block`] exactly.
fn q6k_row_q8_scalar(row: &[u8], act: &Q8Act, k_in: usize) -> f64 {
    let nb = k_in / QK_K;
    let mut acc = 0f64;
    for b in 0..nb {
        let blk = &row[b * Q6K_BLOCK..][..Q6K_BLOCK];
        let (ql, qh, sc) = (&blk[0..128], &blk[128..192], &blk[192..208]);
        let d = crate::st::f16_to_f32(u16::from_le_bytes([blk[208], blk[209]])) as f64;
        let d8 = act.d8[b] as f64;
        let q8 = &act.q8[b * QK_K..][..QK_K];
        let mut isum = 0i64;
        for n in 0..2 {
            let (qlo, qho, sco, yo) = (n * 64, n * 32, n * 8, n * 128);
            for l in 0..32 {
                let is = l / 16;
                let h = qh[qho + l];
                let (a, c) = (ql[qlo + l], ql[qlo + l + 32]);
                let q = [
                    (a & 0xF) | ((h & 3) << 4),
                    (c & 0xF) | (((h >> 2) & 3) << 4),
                    (a >> 4) | (((h >> 4) & 3) << 4),
                    (c >> 4) | (((h >> 6) & 3) << 4),
                ];
                for (g, &qg) in q.iter().enumerate() {
                    let elem = yo + l + 32 * g;
                    let scv = sc[sco + is + 2 * g] as i8 as i64;
                    isum += scv * (qg as i64 - 32) * q8[elem] as i64;
                }
            }
        }
        acc += d * d8 * isum as f64;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q6k_row_q8_avx2(row: &[u8], act: &Q8Act, k_in: usize) -> f64 {
    use std::arch::x86_64::*;
    // sum of 8 i16 in a 128-bit lane -> i32
    #[inline]
    unsafe fn hsum8_i16(v: __m128i) -> i32 {
        let p = _mm_madd_epi16(v, _mm_set1_epi16(1)); // 4 i32
        let s = _mm_add_epi32(p, _mm_shuffle_epi32(p, 0b01_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b00_00_00_01));
        _mm_cvtsi128_si32(s)
    }
    let nb = k_in / QK_K;
    let lomask = _mm256_set1_epi8(0x0F);
    let m3 = _mm256_set1_epi8(0x03);
    let ones = _mm256_set1_epi8(1);
    let mut acc = 0f64;
    for b in 0..nb {
        let blk = &row[b * Q6K_BLOCK..][..Q6K_BLOCK];
        let (ql, qh, sc) = (&blk[0..128], &blk[128..192], &blk[192..208]);
        let d = crate::st::f16_to_f32(u16::from_le_bytes([blk[208], blk[209]])) as f64;
        let d8 = act.d8[b] as f64;
        let mut isum = 0i64;
        for n in 0..2 {
            let (qlo, qho, sco, yo) = (n * 64, n * 32, n * 8, n * 128);
            let qh_c = _mm256_loadu_si256(qh[qho..].as_ptr().cast());
            let ql_a = _mm256_loadu_si256(ql[qlo..].as_ptr().cast());
            let ql_c = _mm256_loadu_si256(ql[qlo + 32..].as_ptr().cast());
            for g in 0..4 {
                let src = if g & 1 == 0 { ql_a } else { ql_c };
                let low = if g < 2 {
                    _mm256_and_si256(src, lomask)
                } else {
                    _mm256_and_si256(_mm256_srli_epi16(src, 4), lomask)
                };
                // high 2 bits from qh: (qh >> 2g) & 3, placed in the nibble above. The shift
                // count is runtime, so use the variable-count form.
                let hb = _mm256_and_si256(_mm256_srl_epi16(qh_c, _mm_cvtsi32_si128((2 * g) as i32)), m3);
                let hb = _mm256_slli_epi16(hb, 4);
                let q = _mm256_or_si256(low, hb); // unsigned 6-bit, 32 values
                let q8c = _mm256_loadu_si256(act.q8[b * QK_K + yo + 32 * g..].as_ptr().cast());
                let p16 = _mm256_maddubs_epi16(q, q8c); // 16 i16
                let intdot0 = hsum8_i16(_mm256_castsi256_si128(p16));
                let intdot1 = hsum8_i16(_mm256_extracti128_si256(p16, 1));
                let s16 = _mm256_maddubs_epi16(ones, q8c); // pair sums of q8
                let sumq8_0 = hsum8_i16(_mm256_castsi256_si128(s16));
                let sumq8_1 = hsum8_i16(_mm256_extracti128_si256(s16, 1));
                let sc0 = sc[sco + 2 * g] as i8 as i64;
                let sc1 = sc[sco + 1 + 2 * g] as i8 as i64;
                isum += sc0 * (intdot0 as i64 - 32 * sumq8_0 as i64)
                    + sc1 * (intdot1 as i64 - 32 * sumq8_1 as i64);
            }
        }
        acc += d * d8 * isum as f64;
    }
    acc
}

#[inline]
fn q6k_row_q8(row: &[u8], act: &Q8Act, k_in: usize) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { q6k_row_q8_avx2(row, act, k_in) };
        }
    }
    q6k_row_q8_scalar(row, act, k_in)
}

/// `y[r] = row r of the Q6_K matrix, dotted with the int8-quantised activation`. Near-bitwise
/// to [`matmul_q6k`]. Covers the `down` projection the Q4_K int8 path misses.
pub fn matmul_q6k_q8(y: &mut [f32], act: &Q8Act, src: &[u8], k_in: usize, rows: usize) {
    use rayon::prelude::*;
    let stride = k_in / QK_K * Q6K_BLOCK;
    assert!(src.len() >= rows * stride, "q6k_q8: weight shorter than rows*stride");
    let dot = |r: usize| q6k_row_q8(&src[r * stride..][..stride], act, k_in) as f32;
    if rows > crate::ops::par_min_rows() {
        let chunk = crate::ops::row_chunk_pub(rows);
        y[..rows].par_chunks_mut(chunk).enumerate().for_each(|(c, yc)| {
            let base = c * chunk;
            for (i, yi) in yc.iter_mut().enumerate() {
                *yi = dot(base + i);
            }
        });
    } else {
        for (r, yi) in y[..rows].iter_mut().enumerate() {
            *yi = dot(r);
        }
    }
}

/// `y[r] = row r of the Q4_K matrix, dotted with the int8-quantised activation`. Rayon over
/// rows, mirroring `kq_matmul`. Near-bitwise to [`matmul_q4k`]; the only difference is the
/// int8 activation rounding.
pub fn matmul_q4k_q8(y: &mut [f32], act: &Q8Act, src: &[u8], k_in: usize, rows: usize) {
    use rayon::prelude::*;
    let stride = k_in / QK_K * Q4K_BLOCK;
    assert!(src.len() >= rows * stride, "q4k_q8: weight shorter than rows*stride");
    let dot = |r: usize| q4k_row_q8(&src[r * stride..][..stride], act, k_in) as f32;
    if rows > crate::ops::par_min_rows() {
        let chunk = crate::ops::row_chunk_pub(rows);
        y[..rows].par_chunks_mut(chunk).enumerate().for_each(|(c, yc)| {
            let base = c * chunk;
            for (i, yi) in yc.iter_mut().enumerate() {
                *yi = dot(base + i);
            }
        });
    } else {
        for (r, yi) in y[..rows].iter_mut().enumerate() {
            *yi = dot(r);
        }
    }
}

/// `grad_x = Wᵀ · grad_y` -- the backward matvec through a frozen k-quant weight.
///
/// This is the one kernel a streamed backward pass needs and a streamed forward pass does
/// not. The forward matmul reads `W` row-major (row `r` = output neuron `r`) and dots each
/// row with the activation. The gradient w.r.t. the INPUT is `Wᵀ @ grad_y`, and the naive
/// reading of that -- "walk down a column of `W`" -- would stride across the whole matrix
/// per output element, which for a disk-streamed weight is death. The identity that saves
/// it is `Wᵀ @ grad_y = Σ_r grad_y[r] · W[r,:]`: visit the rows in the SAME sequential
/// order the forward pass reads them, decode each row once with the SAME [`BlockFn`], and
/// axpy it into the accumulator scaled by `grad_y[r]`. No transpose lives in memory, and
/// the dequantised weight is bit-for-bit the one `mmw` multiplies -- so this is the exact
/// adjoint of the forward matmul, which is how it is tested (`out_prod_is_the_adjoint_of_mmw`)
/// rather than against a second hand-written reference that could be wrong the same way.
///
/// `W[r,i] = d_b · w[i] − dm[sub(i)]` is the affine each decoder fills; q6k carries no min,
/// so its `dm` is zero and the same expression covers all three types. Accumulated in f64
/// and downcast once, matching the forward path's f64 reduction.
fn kq_out_prod(
    grad_x: &mut [f32],
    grad_y: &[f32],
    src: &[u8],
    blk_bytes: usize,
    block: BlockFn,
    k_in: usize,
    rows: usize,
) {
    assert_eq!(k_in % QK_K, 0, "k-quant rows are whole super-blocks: {k_in} % {QK_K} != 0");
    assert!(grad_y.len() >= rows, "grad_y shorter than the row count");
    assert!(grad_x.len() >= k_in, "grad_x shorter than the input width");
    let nb = k_in / QK_K;
    let stride = nb * blk_bytes;
    let mut acc = vec![0f64; k_in];
    let mut w = [0f32; QK_K];
    let mut dm = [0f64; 8];
    for r in 0..rows {
        let g = grad_y[r] as f64;
        let row = &src[r * stride..][..stride];
        for b in 0..nb {
            let d = block(&row[b * blk_bytes..][..blk_bytes], &mut w, &mut dm);
            let base = b * QK_K;
            for sub in 0..8 {
                let dmv = dm[sub];
                let dst = &mut acc[base + sub * 32..][..32];
                let ws = &w[sub * 32..][..32];
                for l in 0..32 {
                    // W[r, base+sub*32+l] = d * ws[l] - dmv, reconstructed in f64 exactly as
                    // the forward decoder produces it before the reduction.
                    dst[l] += g * (d * ws[l] as f64 - dmv);
                }
            }
        }
    }
    for i in 0..k_in {
        grad_x[i] = acc[i] as f32;
    }
}

/// The batched backward matvec: `ntok` gradient vectors through the same weight at once.
/// `grad_y` and `grad_x` are both TOKEN-MAJOR (`grad_y[t*rows + r]`, `grad_x[t*k_in + i]`).
///
/// The whole point is to pay the row decode ONCE for every token, exactly as `mmw_many` does
/// for the forward pass -- the frozen weight (a 248k-row vocabulary head) is the dominant cost
/// of a training step, and decoding it once per token instead of once per (token, row) is the
/// difference between a visible loss and a stalled one. Each row is dequantised into a full
/// f64 row and then axpy'd into every token's accumulator, so the per-token reduction order is
/// identical to the serial [`kq_out_prod`] and the result is BIT-for-bit the same as calling
/// it per token (`out_prod_many_matches_serial`).
fn kq_out_prod_many(
    grad_x: &mut [f32],
    grad_y: &[f32],
    src: &[u8],
    blk_bytes: usize,
    block: BlockFn,
    k_in: usize,
    rows: usize,
    ntok: usize,
) {
    assert_eq!(k_in % QK_K, 0, "k-quant rows are whole super-blocks: {k_in} % {QK_K} != 0");
    assert!(grad_y.len() >= ntok * rows, "grad_y shorter than ntok*rows");
    assert!(grad_x.len() >= ntok * k_in, "grad_x shorter than ntok*k_in");
    let nb = k_in / QK_K;
    let stride = nb * blk_bytes;
    let mut acc = vec![0f64; ntok * k_in];
    let mut w = [0f32; QK_K];
    let mut dm = [0f64; 8];
    let mut wfull = vec![0f64; k_in];
    for r in 0..rows {
        let row = &src[r * stride..][..stride];
        for b in 0..nb {
            let d = block(&row[b * blk_bytes..][..blk_bytes], &mut w, &mut dm);
            let base = b * QK_K;
            for sub in 0..8 {
                let dmv = dm[sub];
                for l in 0..32 {
                    wfull[base + sub * 32 + l] = d * w[sub * 32 + l] as f64 - dmv;
                }
            }
        }
        for t in 0..ntok {
            let g = grad_y[t * rows + r] as f64;
            if g == 0.0 {
                continue;
            }
            let a = &mut acc[t * k_in..][..k_in];
            for i in 0..k_in {
                a[i] += g * wfull[i];
            }
        }
    }
    for i in 0..ntok * k_in {
        grad_x[i] = acc[i] as f32;
    }
}

/// `grad_x = Wᵀ · grad_y` for a Q4_K weight. `k_in` is a multiple of `QK_K`; `rows` is the
/// output width (the length of `grad_y`). See [`kq_out_prod`].
pub fn out_prod_q4k(grad_x: &mut [f32], grad_y: &[f32], src: &[u8], k_in: usize, rows: usize) {
    kq_out_prod(grad_x, grad_y, src, Q4K_BLOCK, q4k_block, k_in, rows);
}

/// Batched `grad_x = Wᵀ·grad_y` for a Q4_K weight, token-major. See [`kq_out_prod_many`].
pub fn out_prod_q4k_many(grad_x: &mut [f32], grad_y: &[f32], src: &[u8], k_in: usize, rows: usize, ntok: usize) {
    kq_out_prod_many(grad_x, grad_y, src, Q4K_BLOCK, q4k_block, k_in, rows, ntok);
}

/// Batched `grad_x = Wᵀ·grad_y` for a Q5_K weight, token-major. See [`kq_out_prod_many`].
pub fn out_prod_q5k_many(grad_x: &mut [f32], grad_y: &[f32], src: &[u8], k_in: usize, rows: usize, ntok: usize) {
    kq_out_prod_many(grad_x, grad_y, src, Q5K_BLOCK, q5k_block, k_in, rows, ntok);
}

/// Batched `grad_x = Wᵀ·grad_y` for a Q6_K weight, token-major. See [`kq_out_prod_many`].
pub fn out_prod_q6k_many(grad_x: &mut [f32], grad_y: &[f32], src: &[u8], k_in: usize, rows: usize, ntok: usize) {
    kq_out_prod_many(grad_x, grad_y, src, Q6K_BLOCK, q6k_block, k_in, rows, ntok);
}

/// `grad_x = Wᵀ · grad_y` for a Q5_K weight. See [`kq_out_prod`].
pub fn out_prod_q5k(grad_x: &mut [f32], grad_y: &[f32], src: &[u8], k_in: usize, rows: usize) {
    kq_out_prod(grad_x, grad_y, src, Q5K_BLOCK, q5k_block, k_in, rows);
}

/// `grad_x = Wᵀ · grad_y` for a Q6_K weight. See [`kq_out_prod`].
pub fn out_prod_q6k(grad_x: &mut [f32], grad_y: &[f32], src: &[u8], k_in: usize, rows: usize) {
    kq_out_prod(grad_x, grad_y, src, Q6K_BLOCK, q6k_block, k_in, rows);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The six-bit scale packing, stated as the table it is. Sub-blocks 0-3 come straight
    /// out of bytes 0-7; 4-7 are split across bytes 8-11 and the top two bits of 0-7.
    #[test]
    fn scale_unpacking_matches_the_six_bit_layout() {
        // scales[0..4] = low scales, [4..8] = low mins, [8..12] = split halves.
        let mut q = [0u8; 12];
        q[0] = 0b11_000101; // sub-block 0 scale = 5, top 2 bits feed sub-block 4's scale
        q[4] = 0b10_001001; // sub-block 0 min = 9, top 2 bits feed sub-block 4's min... no:
        q[8] = 0b0110_0011; // sub-block 4: low scale nibble 3, low min nibble 6
        let (s0, m0) = scale_min_k4(0, &q);
        assert_eq!(s0, 5, "sub-block 0 scale is the low six bits of byte 0");
        assert_eq!(m0, 9, "sub-block 0 min is the low six bits of byte 4");

        let (s4, m4) = scale_min_k4(4, &q);
        // scale = low nibble of q[8] | (top two bits of q[0] << 4)
        assert_eq!(s4, 0x3 | (0b11 << 4), "sub-block 4 scale spans bytes 8 and 0");
        // min = high nibble of q[8] | (top two bits of q[4] << 4)
        assert_eq!(m4, 0x6 | (0b10 << 4), "sub-block 4 min spans bytes 8 and 4");
    }

    /// Every sub-block index must read within the twelve scale bytes. An off-by-one in the
    /// j >= 4 branch indexes byte 12 and panics only on the last super-block of a tensor.
    #[test]
    fn every_sub_block_stays_inside_the_twelve_scale_bytes() {
        let q = [0xFFu8; 12];
        for j in 0..8 {
            let (s, m) = scale_min_k4(j, &q);
            assert!(s <= 63, "a six-bit scale cannot exceed 63");
            assert!(m <= 63, "a six-bit min cannot exceed 63");
        }
    }

    /// A block whose scales and mins are all zero must dequantise to exactly zero, and one
    /// with dmin zero must reproduce d * scale * nibble -- the two ends of the formula,
    /// pinned separately so a dropped `- mn` term cannot pass.
    #[test]
    fn the_affine_form_is_scale_times_nibble_minus_min() {
        let mut blk = vec![0u8; Q4K_BLOCK];
        // d = 1.0, dmin = 0.0 in fp16.
        blk[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        blk[2..4].copy_from_slice(&0u16.to_le_bytes());
        for s in blk[4..16].iter_mut() {
            *s = 1; // every six-bit scale = 1, every min = 0
        }
        blk[16] = 0x27; // low nibble 7, high nibble 2
        let mut out = vec![0f32; QK_K];
        q4k_dequant(&mut out, &blk, 1);
        assert_eq!(out[0], 7.0, "element 0 is the LOW nibble of byte 0");
        assert_eq!(out[32], 2.0, "element 32 is the HIGH nibble of byte 0");

        // Now give it a non-zero min and check it subtracts.
        blk[2..4].copy_from_slice(&0x3C00u16.to_le_bytes()); // dmin = 1.0
        blk[8] = 3; // sub-block 0's min six bits live in scales[4] -> blk[8]
        q4k_dequant(&mut out, &blk, 1);
        assert_eq!(out[0], 7.0 - 3.0, "the min is subtracted, not added");
    }

    /// Deterministic block bytes. Every byte pattern is a legal k-quant payload except the
    /// fp16 scale fields, which are forced to a finite value so a random NaN cannot make a
    /// comparison vacuously true.
    fn synth(block: usize, nblocks: usize, seed: u64) -> Vec<u8> {
        let mut v = vec![0u8; nblocks * block];
        let mut s = seed | 1;
        for b in v.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *b = (s >> 33) as u8;
        }
        for i in 0..nblocks {
            let blk = &mut v[i * block..][..block];
            // 0x3400 = 0.25, 0x3000 = 0.125 in fp16.
            let (a, c) = if block == Q6K_BLOCK { (208, 208) } else { (0, 2) };
            blk[a..a + 2].copy_from_slice(&0x3400u16.to_le_bytes());
            blk[c..c + 2].copy_from_slice(&0x3000u16.to_le_bytes());
        }
        v
    }

    fn xs(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i as f64 * 0.017).sin() * 0.4) as f32).collect()
    }

    /// The fan-out must be a pure scheduling change. Rows write disjoint outputs and each
    /// reads its own slice, so any partition of the row space must give identical BITS --
    /// this is the same contract every other kernel in the engine holds.
    #[test]
    fn parallel_matches_serial_bitwise() {
        let (k_in, rows) = (QK_K * 3, 200usize);
        let x = xs(k_in);
        for (block, name) in
            [(Q4K_BLOCK, "q4k"), (Q5K_BLOCK, "q5k"), (Q6K_BLOCK, "q6k")]
        {
            let src = synth(block, rows * (k_in / QK_K), 12345);
            let f: fn(&[u8], &[f32], usize, &[f64]) -> f64 = match block {
                Q4K_BLOCK => q4k_row,
                Q5K_BLOCK => q5k_row,
                _ => q6k_row,
            };
            let stride = k_in / QK_K * block;
            let xsv = xsums(&x, k_in / 32);
            let mut par = vec![0f32; rows];
            kq_matmul(&mut par, &x, &src, k_in, rows, block, f);
            let mut ser = vec![0f32; rows];
            kq_serial(&mut ser, &x, &src, k_in, rows, stride, f, &xsv);
            for r in 0..rows {
                assert_eq!(par[r].to_bits(), ser[r].to_bits(), "{name} row {r}");
            }
        }
    }

    fn dotf(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(&p, &q)| p as f64 * q as f64).sum()
    }

    /// The int8-activation matmul must (a) track the f32 kernel within the int8-quant bound,
    /// and (b) match its own scalar reference BIT-for-bit on AVX2 (the integer dot is exact
    /// and the f64 scale-fold order is identical). A wiring bug in the nibble/scale/min
    /// arithmetic blows the tracking error up by orders of magnitude.
    #[test]
    fn q8_matmul_tracks_f32_and_avx2_equals_scalar() {
        let (k_in, rows) = (QK_K * 3, 200usize);
        let x = xs(k_in);
        let src = synth(Q4K_BLOCK, rows * (k_in / QK_K), 8080);
        let act = Q8Act::quantize(&x, k_in);

        let mut refy = vec![0f32; rows];
        matmul_q4k(&mut refy, &x, &src, k_in, rows);
        let mut q8y = vec![0f32; rows];
        matmul_q4k_q8(&mut q8y, &act, &src, k_in, rows);
        let scale = refy.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-6);
        let rms = (refy.iter().zip(&q8y).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum::<f64>()
            / rows as f64)
            .sqrt() as f32
            / scale;
        assert!(rms < 0.02, "int8 matmul drifted from f32 kernel: rms {rms}");

        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            let stride = k_in / QK_K * Q4K_BLOCK;
            for r in 0..rows {
                let row = &src[r * stride..][..stride];
                let s = q4k_row_q8_scalar(row, &act, k_in);
                let v = unsafe { q4k_row_q8_avx2(row, &act, k_in) };
                assert_eq!(s.to_bits(), v.to_bits(), "row {r}: avx2 {v} != scalar {s}");
            }
        }
    }

    /// The Q6_K int8 path: tracks the f32 Q6_K kernel within the int8 bound, and AVX2 equals
    /// scalar BIT-for-bit (the integer sum is order-independent, so any lane order agrees).
    #[test]
    fn q6k_q8_tracks_f32_and_avx2_equals_scalar() {
        let (k_in, rows) = (QK_K * 3, 200usize);
        let x = xs(k_in);
        let src = synth(Q6K_BLOCK, rows * (k_in / QK_K), 6161);
        let act = Q8Act::quantize(&x, k_in);

        let mut refy = vec![0f32; rows];
        matmul_q6k(&mut refy, &x, &src, k_in, rows);
        let mut q8y = vec![0f32; rows];
        matmul_q6k_q8(&mut q8y, &act, &src, k_in, rows);
        let scale = refy.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-6);
        let rms = (refy.iter().zip(&q8y).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum::<f64>()
            / rows as f64)
            .sqrt() as f32
            / scale;
        assert!(rms < 0.02, "Q6_K int8 drifted from f32 kernel: rms {rms}");

        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            let stride = k_in / QK_K * Q6K_BLOCK;
            for r in 0..rows {
                let row = &src[r * stride..][..stride];
                let s = q6k_row_q8_scalar(row, &act, k_in);
                let v = unsafe { q6k_row_q8_avx2(row, &act, k_in) };
                assert_eq!(s.to_bits(), v.to_bits(), "Q6_K row {r}: avx2 {v} != scalar {s}");
            }
        }
    }

    /// The backward matvec is the exact adjoint of the forward one:
    /// `⟨grad_y, W·x⟩ = ⟨Wᵀ·grad_y, x⟩` for every `x` and `grad_y`. The forward matmul is
    /// already diffed bit-for-bit against llama.cpp, so pinning `out_prod_q` as its adjoint
    /// inherits that verification instead of trusting a second hand-written reference that
    /// might be wrong in the same way. Both sides reduce in f64, differing only by the
    /// association order of the two reductions, so the identity holds to a tight relative
    /// tolerance rather than to the bit.
    #[test]
    fn out_prod_is_the_adjoint_of_mmw() {
        let (k_in, rows) = (QK_K * 3, 200usize);
        let x = xs(k_in);
        let grad_y = xs(rows);
        for (block, name) in [(Q4K_BLOCK, "q4k"), (Q5K_BLOCK, "q5k"), (Q6K_BLOCK, "q6k")] {
            let src = synth(block, rows * (k_in / QK_K), 4242);
            let op: fn(&mut [f32], &[f32], &[u8], usize, usize) = match block {
                Q4K_BLOCK => out_prod_q4k,
                Q5K_BLOCK => out_prod_q5k,
                _ => out_prod_q6k,
            };
            let mm: fn(&mut [f32], &[f32], &[u8], usize, usize) = match block {
                Q4K_BLOCK => matmul_q4k,
                Q5K_BLOCK => matmul_q5k,
                _ => matmul_q6k,
            };

            let mut y = vec![0f32; rows];
            mm(&mut y, &x, &src, k_in, rows);
            let lhs = dotf(&grad_y, &y); // ⟨grad_y, W·x⟩

            let mut grad_x = vec![0f32; k_in];
            op(&mut grad_x, &grad_y, &src, k_in, rows);
            let rhs = dotf(&grad_x, &x); // ⟨Wᵀ·grad_y, x⟩

            let rel = (lhs - rhs).abs() / lhs.abs().max(1e-9);
            assert!(rel < 1e-6, "{name}: adjoint identity broken, lhs={lhs} rhs={rhs} rel={rel}");

            // Teeth: the identity binds grad_x to the specific input x, not just to W.
            // Pairing grad_x with a DIFFERENT x' must break it -- ⟨Wᵀgrad_y, x'⟩ = ⟨grad_y,
            // Wx'⟩ ≠ ⟨grad_y, Wx⟩ for x' ≠ x -- so a kernel that returned some x-independent
            // vector could not pass. (A different-weight check is unreliable here: synth
            // pins the fp16 scales, so the dominant min-term component is near-identical
            // across seeds and two random W's dot too closely to discriminate.)
            let x2: Vec<f32> = (0..k_in)
                .map(|i| ((i as f64 * 0.023 + 1.0).cos() * 0.4) as f32)
                .collect();
            let rhs_wrong = dotf(&grad_x, &x2);
            assert!(
                (lhs - rhs_wrong).abs() / lhs.abs().max(1e-9) > 1e-3,
                "{name}: identity does not depend on x -- test has no teeth"
            );
        }
    }

    /// The batched backward matvec must equal calling the serial one per token in BITS -- it
    /// only reorders which token an axpy lands in, never the per-token reduction order, so any
    /// difference is a bug, not float noise. Same contract as the forward `_many` kernels.
    #[test]
    fn out_prod_many_matches_serial() {
        let (k_in, rows, ntok) = (QK_K * 3, 120usize, 7usize);
        let grad_y_tokmajor: Vec<f32> = (0..ntok * rows)
            .map(|i| ((i as f64 * 0.011).sin() * 0.3) as f32)
            .collect();
        for (block, name) in [(Q4K_BLOCK, "q4k"), (Q5K_BLOCK, "q5k"), (Q6K_BLOCK, "q6k")] {
            let src = synth(block, rows * (k_in / QK_K), 909);
            let serial: fn(&mut [f32], &[f32], &[u8], usize, usize) = match block {
                Q4K_BLOCK => out_prod_q4k,
                Q5K_BLOCK => out_prod_q5k,
                _ => out_prod_q6k,
            };
            #[allow(clippy::type_complexity)]
            let many: fn(&mut [f32], &[f32], &[u8], usize, usize, usize) = match block {
                Q4K_BLOCK => out_prod_q4k_many,
                Q5K_BLOCK => out_prod_q5k_many,
                _ => out_prod_q6k_many,
            };
            let mut batched = vec![0f32; ntok * k_in];
            many(&mut batched, &grad_y_tokmajor, &src, k_in, rows, ntok);
            for t in 0..ntok {
                let mut one = vec![0f32; k_in];
                serial(&mut one, &grad_y_tokmajor[t * rows..][..rows], &src, k_in, rows);
                for i in 0..k_in {
                    assert_eq!(
                        batched[t * k_in + i].to_bits(),
                        one[i].to_bits(),
                        "{name}: token {t} elem {i} differs"
                    );
                }
            }
        }
    }

    /// The min term is the classic silent-wrong bug: drop it and the kernel still produces
    /// a plausible gradient, just the wrong one. Reconstruct a min-dropped backward inline
    /// and confirm it FAILS the adjoint identity, so the real kernel's `- dmv` is load-bearing
    /// and could not have been left out unnoticed. Q4_K carries a real min term (unlike q6k).
    #[test]
    fn dropping_the_min_term_breaks_the_adjoint() {
        let (k_in, rows) = (QK_K * 2, 64usize);
        let x = xs(k_in);
        let grad_y = xs(rows);
        let src = synth(Q4K_BLOCK, rows * (k_in / QK_K), 7);

        let mut y = vec![0f32; rows];
        matmul_q4k(&mut y, &x, &src, k_in, rows);
        let lhs = dotf(&grad_y, &y);

        // Buggy backward: exactly kq_out_prod but with the `- dmv` subtraction removed.
        let nb = k_in / QK_K;
        let stride = nb * Q4K_BLOCK;
        let mut acc = vec![0f64; k_in];
        let mut w = [0f32; QK_K];
        let mut dm = [0f64; 8];
        for r in 0..rows {
            let g = grad_y[r] as f64;
            let row = &src[r * stride..][..stride];
            for b in 0..nb {
                let d = q4k_block(&row[b * Q4K_BLOCK..][..Q4K_BLOCK], &mut w, &mut dm);
                for i in 0..QK_K {
                    acc[b * QK_K + i] += g * (d * w[i] as f64); // <-- min term dropped
                }
            }
        }
        let grad_x_buggy: Vec<f32> = acc.iter().map(|&a| a as f32).collect();
        let rhs = dotf(&grad_x_buggy, &x);
        let rel = (lhs - rhs).abs() / lhs.abs().max(1e-9);
        assert!(rel > 1e-3, "the min term was not load-bearing? rel={rel} (lhs={lhs} rhs={rhs})");
    }

    /// The vector path must equal the scalar path in BITS, not approximately.
    ///
    /// This holds only because both operands come from f32, making every product exact in
    /// f64, so `fmadd`'s single rounding matches multiply-then-add. The quant values fed
    /// through here are small integers and the activations are f32, so the precondition is
    /// satisfied everywhere this is used -- but it is a precondition, not a law, which is
    /// why it is asserted rather than assumed.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_matches_scalar_bitwise() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let mut seed = 99u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64 / 4.2e9 - 0.5) as f32
        };
        for n in [8usize, 16, 32, 256] {
            // Quant-like weights (small integers) against ordinary activations, which is
            // exactly what the row functions feed it.
            let w: Vec<f32> = (0..n).map(|i| ((i * 37) % 64) as f32 - 32.0).collect();
            let x: Vec<f32> = (0..n).map(|_| next()).collect();
            let s = dot_p8_scalar(&w, &x, n);
            let v = unsafe { dot_p8_avx2(&w, &x, n) };
            assert_eq!(s.to_bits(), v.to_bits(), "n = {n}: {s} vs {v}");
        }
    }

    /// And the answer must not depend on how many threads rayon happens to have.
    #[test]
    fn the_result_is_independent_of_thread_count() {
        let (k_in, rows) = (QK_K * 2, 150usize);
        let x = xs(k_in);
        let src = synth(Q4K_BLOCK, rows * (k_in / QK_K), 777);
        let run = |threads: usize| {
            let pool =
                rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                let mut y = vec![0f32; rows];
                matmul_q4k(&mut y, &x, &src, k_in, rows);
                y
            })
        };
        let (a, b) = (run(1), run(4));
        for r in 0..rows {
            assert_eq!(a[r].to_bits(), b[r].to_bits(), "row {r}");
        }
    }

    /// The fused kernel must agree with dequantise-then-dot to within rounding.
    ///
    /// NOT bitwise, and deliberately so: the fused form factors the affine term per
    /// sub-block (`d*sc*sum(q*x) - dmin*m*sum(x)`) while the naive form multiplies it out
    /// per element, so the two sum in different orders. `matmul_mxfp4` carries the same
    /// disclaimer. What this pins is that the factoring is ALGEBRAICALLY right -- dropping
    /// the `- dmin*m*sum(x)` term entirely still produces finite output and fails here.
    #[test]
    fn the_fused_form_agrees_with_dequantise_then_dot() {
        let (k_in, rows) = (QK_K * 2, 8usize);
        let x = xs(k_in);
        let nb = k_in / QK_K;
        for (block, name) in
            [(Q4K_BLOCK, "q4k"), (Q5K_BLOCK, "q5k"), (Q6K_BLOCK, "q6k")]
        {
            let src = synth(block, rows * nb, 4242);
            let mut got = vec![0f32; rows];
            match block {
                Q4K_BLOCK => matmul_q4k(&mut got, &x, &src, k_in, rows),
                Q5K_BLOCK => matmul_q5k(&mut got, &x, &src, k_in, rows),
                _ => matmul_q6k(&mut got, &x, &src, k_in, rows),
            }
            let stride = nb * block;
            for r in 0..rows {
                let mut w = vec![0f32; k_in];
                match block {
                    Q4K_BLOCK => q4k_dequant(&mut w, &src[r * stride..], nb),
                    Q5K_BLOCK => q5k_dequant(&mut w, &src[r * stride..], nb),
                    _ => q6k_dequant(&mut w, &src[r * stride..], nb),
                }
                let want: f64 = (0..k_in).map(|i| w[i] as f64 * x[i] as f64).sum();
                let scale = want.abs().max(1e-3);
                assert!(
                    (got[r] as f64 - want).abs() / scale < 1e-6,
                    "{name} row {r}: fused {} vs dequantised {want}",
                    got[r]
                );
            }
        }
    }

    #[test]
    fn a_bad_magic_is_rejected() {
        let types: Vec<u32> = (0..32).collect();
        // Every type this build claims to read must have a consistent block table.
        for t in types {
            if let Some((_, elems, bytes)) = ggml_type(t) {
                assert!(elems > 0 && bytes > 0, "type {t} has a degenerate block");
                assert!(bytes * 8 >= elems, "type {t} claims under one bit per element");
            }
        }
    }
}
