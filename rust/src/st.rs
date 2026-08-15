// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use serde_json::Value;

pub const ALIGN: i64 = 4096;

const WIDEN_CHUNK: usize = 4 << 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    U8,
    Bf16,
    F16,
    F32,
    I8R,
    // Added for DeepSeek-V4, whose shards carry dtypes k3_st.c never saw.
    I8,
    I64,
    /// OCP FP8 E4M3 (fn): no infinities, 0x7F/0xFF are NaN, max finite 448.
    F8E4M3,
    /// OCP E8M0: a bare biased exponent used as a block scale. 255 is NaN.
    F8E8M0,
    // GGUF block-quantised formats. Unlike everything above, these are not element
    // types: a "value" is a super-block of 256 (Q4_K, Q6_K) or 32 (Q8_0) elements sharing
    // packed scales, so `nbytes` is not `numel * width` and the dequantiser is per block.
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    IQ4NL,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl Dtype {
    pub fn elemsize(self) -> i64 {
        match self {
            Dtype::U8 | Dtype::I8 | Dtype::F8E4M3 | Dtype::F8E8M0 => 1,
            Dtype::Bf16 | Dtype::F16 => 2,
            Dtype::F32 => 4,
            Dtype::I64 => 8,
            Dtype::I8R => 0,
            // Block-quantised: a "value" is a super-block, not an element, so there is
            // no per-element width. GGUF computes nbytes from its own block table and
            // never asks for this.
            Dtype::Q4_0 | Dtype::Q4_1 | Dtype::Q5_0 | Dtype::Q5_1 | Dtype::Q8_0
            | Dtype::IQ4NL | Dtype::Q2K | Dtype::Q3K | Dtype::Q4K | Dtype::Q5K | Dtype::Q6K => 0,
        }
    }

    fn parse(s: &str) -> Option<Dtype> {
        Some(match s {
            "U8" => Dtype::U8,
            "BF16" => Dtype::Bf16,
            "F16" => Dtype::F16,
            "F32" => Dtype::F32,
            "I8" => Dtype::I8,
            "I64" => Dtype::I64,
            "F8_E4M3" => Dtype::F8E4M3,
            "F8_E8M0" => Dtype::F8E8M0,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Dtype::U8 => "U8",
            Dtype::Bf16 => "BF16",
            Dtype::F16 => "F16",
            Dtype::F32 => "F32",
            Dtype::I8 => "I8",
            Dtype::Q4_0 => "Q4_0",
            Dtype::Q4_1 => "Q4_1",
            Dtype::Q5_0 => "Q5_0",
            Dtype::Q5_1 => "Q5_1",
            Dtype::Q8_0 => "Q8_0",
            Dtype::IQ4NL => "IQ4_NL",
            Dtype::Q2K => "Q2_K",
            Dtype::Q3K => "Q3_K",
            Dtype::Q4K => "Q4_K",
            Dtype::Q5K => "Q5_K",
            Dtype::Q6K => "Q6_K",
            Dtype::I64 => "I64",
            Dtype::F8E4M3 => "F8_E4M3",
            Dtype::F8E8M0 => "F8_E8M0",
            Dtype::I8R => "I8R",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub shard: usize,
    pub dtype: Dtype,
    pub shape: Vec<i64>,
    pub off: i64,
    pub nbytes: i64,
}

impl Tensor {
    pub fn numel(&self) -> i64 {
        if self.shape.is_empty() {
            1 // a scalar has shape [], and one element
        } else {
            self.shape.iter().product()
        }
    }
}

pub struct St {
    pub paths: Vec<PathBuf>,
    files: Vec<File>,
    direct: Vec<Option<File>>,
    pub tensors: Vec<Tensor>,
    index: HashMap<String, usize>,
    /// GGUF metadata, when the directory held `.gguf` rather than `.safetensors`. A split
    /// GGUF repeats the `general.*` keys in every shard, so first-shard-wins is correct.
    pub meta: Option<crate::gguf::Meta>,
}

#[inline]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mut man = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign // +/- zero
        } else {
            let mut sh = 0u32;
            while man & 0x400 == 0 {
                man <<= 1;
                sh += 1;
            }
            man &= 0x3FF;
            sign | ((127 - 15 - sh + 1) << 23) | (man << 13)
        }
    } else if exp == 31 {
        sign | 0x7F80_0000 | (man << 13) // inf / NaN, payload preserved
    } else {
        sign | ((exp - 15 + 127) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

// OCP FP8 E4M3FN: 1 sign, 4 exponent (bias 7), 3 mantissa. Unlike IEEE binary16 there
// are no infinities; 0x7F and 0xFF are the only NaNs, and the max finite value is 448.
/// All 256 e4m3 values, precomputed.
///
/// `f8_e4m3_to_f32` costs a `powi` per call, and the FP8 matmul calls it once per weight
/// BYTE -- a transcendental in the innermost loop of every DeepSeek-V4 projection. The
/// table is built from that same function, so it is bit-identical by construction and
/// exists purely to move the cost out of the loop.
pub fn e4m3_table() -> &'static [f32; 256] {
    use std::sync::OnceLock;
    static T: OnceLock<[f32; 256]> = OnceLock::new();
    T.get_or_init(|| std::array::from_fn(|b| f8_e4m3_to_f32(b as u8)))
}

#[inline]
pub fn f8_e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let man = (b & 0x07) as f32;
    if exp == 0 {
        sign * man * 2f32.powi(-9) // subnormal: 2^-6 * man/8
    } else if exp == 15 && man == 7.0 {
        f32::NAN
    } else {
        sign * (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    }
}

// E8M0: a bare biased exponent, no sign and no mantissa. 255 is NaN by spec and is
// mapped to zero so one bad scale byte cannot poison a block.
#[inline]
pub fn e8m0_to_f32(b: u8) -> f32 {
    if b == 255 {
        0.0
    } else {
        crate::libm::exp2i(b as i32 - 127)
    }
}

/// A page-aligned byte buffer. O_DIRECT requires the buffer ADDRESS to be aligned as
/// well as the offset and length; a plain Vec is not, and every pread against a
/// misaligned buffer fails with EINVAL, which reads as a short read.
pub struct Aligned {
    ptr: *mut u8,
    len: usize,
    align: usize,
}

// The buffer is owned outright and carries no interior references.
unsafe impl Send for Aligned {}

impl Aligned {
    pub fn new(len: usize) -> Aligned {
        Aligned::with_align(len, ALIGN as usize)
    }

    pub fn with_align(len: usize, align: usize) -> Aligned {
        let len = len.max(1).next_multiple_of(align);
        let layout = std::alloc::Layout::from_size_align(len, align).expect("layout");
        // Safety: len is non-zero and a multiple of the alignment.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Aligned { ptr, len, align }
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }
}

impl std::ops::Deref for Aligned {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl std::ops::DerefMut for Aligned {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for Aligned {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.len, self.align).unwrap();
        unsafe { std::alloc::dealloc(self.ptr, layout) };
    }
}

fn err(m: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, m)
}

impl St {
    pub fn open(dir: &Path) -> io::Result<St> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors" || e == "gguf"))
            .collect();
        if paths.is_empty() {
            return Err(err(format!(
                "k3_st: no .safetensors or .gguf files in {}",
                dir.display()
            )));
        }
        // The two formats name their tensors by different conventions, so a directory
        // holding both would bind half a model from each and report no error. Refuse.
        let gguf = |p: &PathBuf| p.extension().is_some_and(|e| e == "gguf");
        if paths.iter().any(gguf) && !paths.iter().all(gguf) {
            return Err(err(format!(
                "k3_st: {} holds both .gguf and .safetensors; their tensor naming differs, \
                 so loading them together would silently bind half a model",
                dir.display()
            )));
        }
        paths.sort();

        let mut st = St {
            files: Vec::with_capacity(paths.len()),
            direct: Vec::with_capacity(paths.len()),
            tensors: Vec::new(),
            index: HashMap::new(),
            meta: None,
            paths,
        };

        for shard in 0..st.paths.len() {
            let path = st.paths[shard].clone();
            let f = File::open(&path)?;
            if gguf(&path) {
                let (meta, ts) = crate::gguf::scan(shard, &f)?;
                st.tensors.extend(ts);
                if st.meta.is_none() {
                    st.meta = Some(meta);
                }
            } else {
                st.scan_shard(shard, &path, &f)?;
            }
            st.files.push(f);
            st.direct.push(open_direct(&path));
        }

        st.index.reserve(st.tensors.len());
        for (i, t) in st.tensors.iter().enumerate() {
            if st.index.insert(t.name.clone(), i).is_some() {
                return Err(err(format!("k3_st: duplicate tensor name {}", t.name)));
            }
        }
        Ok(st)
    }

    fn scan_shard(&mut self, shard: usize, path: &Path, f: &File) -> io::Result<usize> {
        let fsize = f.metadata()?.len() as i64;

        let mut lenbuf = [0u8; 8];
        f.read_exact_at(&mut lenbuf, 0).map_err(|_| {
            err(format!("k3_st: {} is too short for a header length", path.display()))
        })?;
        let hlen = u64::from_le_bytes(lenbuf) as i64;
        if hlen == 0 || fsize < 8 + hlen {
            return Err(err(format!(
                "k3_st: {} header length {hlen} is impossible (file {fsize} bytes)",
                path.display()
            )));
        }

        let mut json = vec![0u8; hlen as usize];
        f.read_exact_at(&mut json, 8)
            .map_err(|_| err(format!("k3_st: short read of {} header", path.display())))?;

        let base = 8 + hlen;

        let root: Value = serde_json::from_slice(&json).map_err(|e| {
            err(format!("k3_st: {} header is not valid JSON: {e}", path.display()))
        })?;
        let Some(obj) = root.as_object() else {
            return Err(err(format!("k3_st: {} header is not a JSON object", path.display())));
        };

        let mut ntensor = 0usize;
        let mut maxend = 0i64;
        for (name, v) in obj {
            if name == "__metadata__" {
                continue;
            }
            let Some(e) = v.as_object() else {
                return Err(err(format!(
                    "k3_st: {}: entry {name} is not an object",
                    path.display()
                )));
            };

            let dtype = e
                .get("dtype")
                .and_then(Value::as_str)
                .ok_or_else(|| missing(path, name))?;
            let dtype = Dtype::parse(dtype).ok_or_else(|| {
                err(format!(
                    "k3_st: {}: unsupported dtype '{dtype}' on {name}",
                    path.display()
                ))
            })?;

            let shape: Vec<i64> = e
                .get("shape")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_i64).collect())
                .unwrap_or_default();
            if shape.len() > 4 {
                return Err(err(format!("k3_st: {name} has rank > 4")));
            }

            let offs = e
                .get("data_offsets")
                .and_then(Value::as_array)
                .ok_or_else(|| missing(path, name))?;
            if offs.len() != 2 {
                return Err(err(format!("k3_st: {name} data_offsets is not a pair")));
            }
            let (o0, o1) = (
                offs[0].as_i64().ok_or_else(|| missing(path, name))?,
                offs[1].as_i64().ok_or_else(|| missing(path, name))?,
            );

            let t = Tensor {
                name: name.clone(),
                shard,
                dtype,
                shape,
                off: base + o0,
                nbytes: o1 - o0,
            };

            let want = t.numel() * t.dtype.elemsize();
            if t.nbytes != want {
                return Err(err(format!(
                    "k3_st: {}: {name} spans {} bytes but shape implies {want}",
                    path.display(),
                    t.nbytes
                )));
            }
            if base + o1 > fsize {
                return Err(err(format!("k3_st: {}: {name} ends past EOF", path.display())));
            }
            maxend = maxend.max(o1);

            self.tensors.push(t);
            ntensor += 1;
        }

        if base + maxend != fsize {
            eprintln!(
                "k3_st: note: {} has {} trailing bytes after the last tensor",
                path.display(),
                fsize - base - maxend
            );
        }
        Ok(ntensor)
    }

    pub fn find(&self, name: &str) -> Option<&Tensor> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn nshard(&self) -> usize {
        self.paths.len()
    }

    pub fn read(&self, t: &Tensor, buf: &mut [u8]) -> i64 {
        let want = (t.nbytes as usize).min(buf.len());
        let buf = &mut buf[..want];
        let mut got = 0usize;
        while got < buf.len() {
            match self.files[t.shard].read_at(&mut buf[got..], (t.off + got as i64) as u64) {
                Ok(0) | Err(_) => {
                    eprintln!("k3_st: short read on {} at +{got}", t.name);
                    return got as i64;
                }
                Ok(n) => got += n,
            }
        }
        got as i64
    }

    pub fn read_aligned(
        &self,
        shard: usize,
        off: i64,
        nbytes: i64,
        buf: &mut [u8],
    ) -> (i64, i64) {
        if shard >= self.paths.len() {
            return (0, 0);
        }
        let bufcap = buf.len() as i64;

        let Some(dfd) = self.direct[shard].as_ref() else {
            if bufcap < nbytes {
                return (0, 0);
            }
            let mut got = 0i64;
            while got < nbytes {
                let end = (nbytes - got) as usize;
                match self.files[shard].read_at(&mut buf[got as usize..][..end], (off + got) as u64)
                {
                    Ok(0) | Err(_) => return (got, 0),
                    Ok(n) => got += n as i64,
                }
            }
            return (got, 0);
        };

        let lo = off & !(ALIGN - 1);
        let hi = (off + nbytes + ALIGN - 1) & !(ALIGN - 1);
        let len = hi - lo;
        let pad = off - lo;
        if len > bufcap {
            return (0, 0);
        }

        let mut got = 0i64;
        while got < len {
            let end = (len - got) as usize;
            match dfd.read_at(&mut buf[got as usize..][..end], (lo + got) as u64) {
                Ok(0) | Err(_) => break,
                Ok(n) => got += n as i64,
            }
        }
        let avail = if got >= pad + nbytes {
            nbytes
        } else if got > pad {
            got - pad
        } else {
            0
        };
        (avail, pad)
    }

    pub fn read_f32(&self, t: &Tensor, out: &mut [f32]) -> i64 {
        let n = t.numel();
        if t.dtype == Dtype::F32 {
            let bytes: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), out.len() * 4)
            };
            return self.read(t, bytes) / 4;
        }

        let esz = t.dtype.elemsize();
        if esz <= 0 {
            return 0;
        }
        let chunk_elems = WIDEN_CHUNK as i64 / esz;
        let mut raw = vec![0u8; (chunk_elems * esz) as usize];

        let mut done = 0i64;
        while done < n {
            let take = (n - done).min(chunk_elems);
            let want = (take * esz) as usize;
            let mut got = 0usize;
            while got < want {
                let at = (t.off + done * esz + got as i64) as u64;
                match self.files[t.shard].read_at(&mut raw[got..want], at) {
                    Ok(0) | Err(_) => {
                        eprintln!("k3_st: short read widening {} at element {done}", t.name);
                        return done;
                    }
                    Ok(r) => got += r,
                }
            }

            let o = &mut out[done as usize..][..take as usize];
            match t.dtype {
                // Block quants cannot be widened a chunk at a time -- a chunk boundary
                // would fall inside a super-block whose scales live at its head. They are
                // read as raw bytes and dequantised by the kernel, exactly as MXFP4 is.
                Dtype::Q4_0 | Dtype::Q4_1 | Dtype::Q5_0 | Dtype::Q5_1 | Dtype::Q8_0
                | Dtype::IQ4NL | Dtype::Q2K | Dtype::Q3K | Dtype::Q4K | Dtype::Q5K | Dtype::Q6K => panic!(
                    "{} is {}: read it with read() and dequantise per block, not read_f32",
                    t.name,
                    t.dtype.name()
                ),
                Dtype::Bf16 => {
                    for (i, s) in o.iter_mut().enumerate() {
                        *s = bf16_to_f32(u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]));
                    }
                }
                Dtype::F16 => {
                    for (i, s) in o.iter_mut().enumerate() {
                        *s = f16_to_f32(u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]));
                    }
                }
                Dtype::U8 => {
                    for (i, s) in o.iter_mut().enumerate() {
                        *s = raw[i] as f32;
                    }
                }
                Dtype::I8 => {
                    for (i, s) in o.iter_mut().enumerate() {
                        *s = raw[i] as i8 as f32;
                    }
                }
                Dtype::F8E4M3 => {
                    for (i, s) in o.iter_mut().enumerate() {
                        *s = f8_e4m3_to_f32(raw[i]);
                    }
                }
                Dtype::F8E8M0 => {
                    for (i, s) in o.iter_mut().enumerate() {
                        *s = e8m0_to_f32(raw[i]);
                    }
                }
                Dtype::I64 => {
                    for (i, s) in o.iter_mut().enumerate() {
                        *s = i64::from_le_bytes(raw[8 * i..8 * i + 8].try_into().unwrap()) as f32;
                    }
                }
                Dtype::F32 | Dtype::I8R => unreachable!("handled above / no element size"),
            }
            done += take;
        }
        n
    }
}

fn missing(path: &Path, name: &str) -> io::Error {
    err(format!(
        "k3_st: {}: {name} is missing dtype or data_offsets",
        path.display()
    ))
}

#[cfg(target_os = "linux")]
fn open_direct(path: &Path) -> Option<File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc_o_direct())
        .open(path)
        .ok()
}

#[cfg(not(target_os = "linux"))]
fn open_direct(_path: &Path) -> Option<File> {
    None
}

#[cfg(target_os = "linux")]
const fn libc_o_direct() -> i32 {
    0o40000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_is_the_top_half_of_an_f32() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert_eq!(bf16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(bf16_to_f32(0xBF80), -1.0);
        for h in 0u16..=u16::MAX {
            assert_eq!(bf16_to_f32(h).to_bits(), (h as u32) << 16);
        }
    }

    #[test]
    fn f16_widening_is_exact_over_every_bit_pattern() {
        for h in 0u16..=u16::MAX {
            let got = f16_to_f32(h);
            let sign = if h & 0x8000 != 0 { -1.0f64 } else { 1.0 };
            let exp = ((h >> 10) & 0x1F) as i32;
            let man = (h & 0x3FF) as f64;
            let want: f32 = if exp == 0 {
                (sign * man * 2f64.powi(-24)) as f32
            } else if exp == 31 {
                if man == 0.0 {
                    if sign < 0.0 { f32::NEG_INFINITY } else { f32::INFINITY }
                } else {
                    assert!(got.is_nan(), "0x{h:04x} should widen to NaN");
                    assert_eq!(
                        got.to_bits() & 0x007F_FFFF,
                        ((h & 0x3FF) as u32) << 13,
                        "NaN payload must survive widening of 0x{h:04x}"
                    );
                    continue;
                }
            } else {
                (sign * (1.0 + man / 1024.0) * 2f64.powi(exp - 15)) as f32
            };
            assert_eq!(got.to_bits(), want.to_bits(), "f16 0x{h:04x}");
        }
    }

    #[test]
    fn a_scalar_has_one_element() {
        let t = Tensor {
            name: "s".into(),
            shard: 0,
            dtype: Dtype::F32,
            shape: vec![],
            off: 0,
            nbytes: 4,
        };
        assert_eq!(t.numel(), 1, "shape [] is a scalar, not zero elements");
    }

    #[test]
    fn elemsize_matches_the_c_table() {
        assert_eq!(Dtype::U8.elemsize(), 1);
        assert_eq!(Dtype::Bf16.elemsize(), 2);
        assert_eq!(Dtype::F16.elemsize(), 2);
        assert_eq!(Dtype::F32.elemsize(), 4);
        assert_eq!(Dtype::I8R.elemsize(), 0, "I8R rows carry a scale; no single width");
    }
}

#[cfg(test)]
mod v4_dtype_tests {
    use super::*;
    include!("../tests/data/e4m3_ref.rs");

    // Reference table generated independently by ml_dtypes 0.5.4 (numpy view of
    // float8_e4m3fn), not by restating the bit layout this implementation uses.
    #[test]
    fn f8_e4m3_matches_ml_dtypes_over_every_byte() {
        for b in 0u8..=255 {
            let got = f8_e4m3_to_f32(b);
            let want = E4M3_REF[b as usize];
            if want.is_nan() {
                assert!(got.is_nan(), "0x{b:02x} should be NaN, got {got}");
            } else {
                assert_eq!(got.to_bits(), want.to_bits(), "0x{b:02x}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn e8m0_is_a_bare_exponent_with_nan_mapped_to_zero() {
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(128), 2.0);
        assert_eq!(e8m0_to_f32(126), 0.5);
        assert_eq!(e8m0_to_f32(255), 0.0, "255 is NaN by spec; zeroing keeps one bad byte local");
    }

    #[test]
    fn deepseek_v4_dtype_names_round_trip() {
        for d in [Dtype::F8E4M3, Dtype::F8E8M0, Dtype::I8, Dtype::I64, Dtype::Bf16, Dtype::F32] {
            assert_eq!(Dtype::parse(d.name()), Some(d), "{}", d.name());
        }
        assert_eq!(Dtype::F8E4M3.elemsize(), 1);
        assert_eq!(Dtype::I64.elemsize(), 8);
    }
}
