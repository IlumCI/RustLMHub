// SPDX-License-Identifier: Apache-2.0
//
// GPU memory as a victim cache for evicted experts.
//
// WHY THIS IS A MEMORY TIER AND NOT A COMPUTE OFFLOAD
//     This workload is bandwidth-bound on storage: a token needs 3.45 GB of routed
//     experts and the device supplies ~0.5 GB/s, so compute is about 8% of a step and no
//     amount of GPU arithmetic can move the number. What the card actually has that this
//     machine does not is 3.7 GB of otherwise-idle memory attached by a link two orders
//     of magnitude faster than the storage.
//
//     The measured cost per 13.37 MB expert:
//         re-read from the USB SSD    ~27 ms
//         DMA back from VRAM          ~2-5 ms
//     So a VRAM hit is worth roughly 6x a disk read, and it costs zero system RAM --
//     which matters because the 8.29 GB resident trunk already leaves under 2 GB for the
//     expert cache on a 15 GB machine.
//
// WHY IT IS WORTH MORE THAN THE HIT RATE SUGGESTS
//     Expert access is a cyclic scan over 43 layers, and one token touches 258 distinct
//     experts. Below that many slots there is essentially nothing to reuse ACROSS tokens,
//     and simulation on a real trace shows exactly that cliff: at 119 slots LRU returns
//     0.00%, at 224 slots still 0.00%, and at 396 slots it jumps to 39.67%. The tier's
//     value is not incremental -- it is that RAM alone (119 slots) sits below the knee
//     and RAM plus VRAM (~400 slots) sits above it.
//
// EXACTNESS
//     A slot is copied to the device and back verbatim. A DMA round trip is a memcpy, so
//     an expert served from VRAM is bit-identical to one read from disk; this tier can
//     change how long a token takes and cannot change what it says.
//
// The CUDA driver API is reached through dlopen rather than a crate, so a build with no
// GPU, no driver, or no CUDA still compiles and still runs -- `Vram::new` simply fails
// and the cache carries on with RAM only.

use std::collections::HashMap;
use std::ffi::c_void;

use crate::cache::ExpertRef;

extern "C" {
    fn dlopen(filename: *const u8, flag: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const u8) -> *mut c_void;
}
const RTLD_NOW: i32 = 2;

type CuDevicePtr = u64;

/// The seven driver entry points this needs. The `_v2` suffixes are not optional: the
/// driver exports both the original and the 64-bit-clean versions, and the unsuffixed
/// ones take 32-bit sizes.
struct Fns {
    mem_alloc: unsafe extern "C" fn(*mut CuDevicePtr, usize) -> i32,
    memcpy_h2d: unsafe extern "C" fn(CuDevicePtr, *const u8, usize) -> i32,
    memcpy_d2h: unsafe extern "C" fn(*mut u8, CuDevicePtr, usize) -> i32,
    mem_free: unsafe extern "C" fn(CuDevicePtr) -> i32,
    mem_get_info: unsafe extern "C" fn(*mut usize, *mut usize) -> i32,
    host_register: Option<unsafe extern "C" fn(*mut c_void, usize, u32) -> i32>,
}

unsafe fn sym(lib: *mut c_void, name: &str) -> Result<*mut c_void, String> {
    let c = format!("{name}\0");
    let p = dlsym(lib, c.as_ptr());
    if p.is_null() {
        return Err(format!("libcuda has no {name}"));
    }
    Ok(p)
}

pub struct Vram {
    base: CuDevicePtr,
    slot_bytes: usize,
    nslot: usize,
    /// vram slot -> key, or EMPTY.
    key_of: Vec<i64>,
    slot_of: HashMap<i64, usize>,
    used_at: Vec<u64>,
    /// The host-side description of each resident slot: the per-run payload offsets and
    /// the expert's file layout. Kept in RAM because it is a few hundred bytes and
    /// round-tripping it through the device would be pure overhead.
    meta: Vec<Option<(Vec<usize>, ExpertRef)>>,
    clock: u64,
    /// (position, cycle length, experts per layer-key) of the decoder sweep.
    sweep: Option<(usize, usize, usize)>,
    f: Fns,
    pub spills: u64,
    pub fills: u64,
    pub bytes_up: u64,
    pub bytes_down: u64,
}

const EMPTY: i64 = -1;

impl Vram {
    /// Reserve `budget_bytes` of device memory, clamped to what the driver reports free
    /// less `RESERVE` so the display server is not starved out of its framebuffer.
    pub fn new(budget_bytes: i64, slot_bytes: usize) -> Result<Vram, String> {
        /// Left for whatever else owns the GPU -- a compositor on a 4 GB laptop card is
        /// not a rounding error.
        const RESERVE: usize = 640 << 20;
        unsafe {
            let lib = dlopen(c"libcuda.so.1".as_ptr().cast(), RTLD_NOW);
            if lib.is_null() {
                return Err("libcuda.so.1 not loadable (no NVIDIA driver?)".into());
            }
            let init: unsafe extern "C" fn(u32) -> i32 = std::mem::transmute::<*mut c_void, unsafe extern "C" fn(u32) -> i32>(sym(lib, "cuInit")?);
            if init(0) != 0 {
                return Err("cuInit failed".into());
            }
            let dev_get: unsafe extern "C" fn(*mut i32, i32) -> i32 =
                std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*mut i32, i32) -> i32>(sym(lib, "cuDeviceGet")?);
            let ctx_create: unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32 =
                std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32>(sym(lib, "cuCtxCreate_v2")?);
            let f = Fns {
                mem_alloc: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*mut CuDevicePtr, usize) -> i32>(sym(lib, "cuMemAlloc_v2")?),
                memcpy_h2d: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(CuDevicePtr, *const u8, usize) -> i32>(sym(lib, "cuMemcpyHtoD_v2")?),
                memcpy_d2h: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*mut u8, CuDevicePtr, usize) -> i32>(sym(lib, "cuMemcpyDtoH_v2")?),
                mem_free: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(CuDevicePtr) -> i32>(sym(lib, "cuMemFree_v2")?),
                mem_get_info: std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*mut usize, *mut usize) -> i32>(sym(lib, "cuMemGetInfo_v2")?),
                host_register: sym(lib, "cuMemHostRegister_v2")
                    .ok()
                    .map(|p| std::mem::transmute::<
                        *mut c_void,
                        unsafe extern "C" fn(*mut c_void, usize, u32) -> i32,
                    >(p)),
            };

            let mut dev = 0i32;
            if dev_get(&mut dev, 0) != 0 {
                return Err("no CUDA device 0".into());
            }
            let mut ctx: *mut c_void = std::ptr::null_mut();
            if ctx_create(&mut ctx, 0, dev) != 0 {
                return Err("cuCtxCreate failed".into());
            }

            let (mut free, mut total) = (0usize, 0usize);
            if (f.mem_get_info)(&mut free, &mut total) != 0 {
                return Err("cuMemGetInfo failed".into());
            }
            let usable = free.saturating_sub(RESERVE);
            let want = (budget_bytes.max(0) as usize).min(usable);
            let nslot = want / slot_bytes.max(1);
            if nslot == 0 {
                return Err(format!(
                    "only {:.2} GB free on the GPU after reserving {:.2} GB for display; \
                     that is under one {:.1} MB expert",
                    free as f64 / 1e9,
                    RESERVE as f64 / 1e9,
                    slot_bytes as f64 / 1e6
                ));
            }
            let mut base: CuDevicePtr = 0;
            // One allocation managed as slots, rather than nslot separate ones: the
            // driver's per-allocation overhead is real at a few hundred allocations, and
            // a single base pointer makes slot addressing trivial arithmetic.
            if (f.mem_alloc)(&mut base, nslot * slot_bytes) != 0 {
                return Err(format!("cuMemAlloc of {:.2} GB failed", (nslot * slot_bytes) as f64 / 1e9));
            }
            Ok(Vram {
                base,
                slot_bytes,
                nslot,
                key_of: vec![EMPTY; nslot],
                slot_of: HashMap::new(),
                used_at: vec![0; nslot],
                meta: vec![None; nslot],
                clock: 0,
                sweep: None,
                f,
                spills: 0,
                fills: 0,
                bytes_up: 0,
                bytes_down: 0,
            })
        }
    }

    pub fn nslot(&self) -> usize {
        self.nslot
    }

    /// Mirror the host cache's sweep position. Without this the tier evicts by LRU, and
    /// under a cyclic scan the least-recently-spilled entry is exactly the one the sweep
    /// is about to ask for -- so every spill overwrites the entry that was about to hit.
    /// Measured before this was added: 2557 spills produced 50 hits.
    pub fn at_layer(&mut self, layer: usize, cycle: usize, n_experts: usize) {
        self.sweep = (cycle > 0).then_some((layer % cycle, cycle, n_experts));
    }

    fn distance(&self, key: i64) -> usize {
        match self.sweep {
            Some((l, cycle, n_experts)) => {
                let layer = (key as usize / n_experts.max(1)) % cycle;
                (layer + cycle - l - 1) % cycle + 1
            }
            None => 0,
        }
    }

    pub fn bytes(&self) -> usize {
        self.nslot * self.slot_bytes
    }

    /// Pin the host arena so DMA does not stage through a driver bounce buffer. Purely an
    /// optimisation: failure is silent and correctness does not depend on it.
    pub fn pin(&self, host: *mut u8, len: usize) {
        if let Some(reg) = self.f.host_register {
            unsafe {
                let _ = reg(host.cast(), len, 0);
            }
        }
    }

    /// Copy an evicted expert to the device. Overwrites the VRAM-LRU victim.
    pub fn put(&mut self, key: i64, host: &[u8], pad: &[usize], r: &ExpertRef) {
        if self.slot_of.contains_key(&key) {
            return;
        }
        let slot = match self.key_of.iter().position(|&k| k == EMPTY) {
            Some(s) => s,
            None => {
                // Furthest-from-next-use first, age as the tiebreak -- the same rule the
                // host cache uses, for the same reason.
                let mut best = 0usize;
                let mut best_rank = (self.distance(self.key_of[0]), u64::MAX - self.used_at[0]);
                for i in 1..self.nslot {
                    let rank = (self.distance(self.key_of[i]), u64::MAX - self.used_at[i]);
                    if rank > best_rank {
                        best_rank = rank;
                        best = i;
                    }
                }
                if let Some(old) = self.slot_of.remove(&self.key_of[best]) {
                    debug_assert_eq!(old, best);
                }
                best
            }
        };
        let n = host.len().min(self.slot_bytes);
        let rc = unsafe {
            (self.f.memcpy_h2d)(self.base + (slot * self.slot_bytes) as u64, host.as_ptr(), n)
        };
        if rc != 0 {
            // A failed upload must not leave a slot claiming to hold the expert.
            self.key_of[slot] = EMPTY;
            self.meta[slot] = None;
            return;
        }
        self.key_of[slot] = key;
        self.slot_of.insert(key, slot);
        self.meta[slot] = Some((pad.to_vec(), r.clone()));
        self.clock += 1;
        self.used_at[slot] = self.clock;
        self.spills += 1;
        self.bytes_up += n as u64;
    }

    /// Copy an expert back into a host slot, if the device holds it.
    ///
    /// EXCLUSIVE: a hit frees the device slot. VRAM is a victim cache, so keeping a copy
    /// of something that is now resident in RAM would spend the tier's capacity on
    /// duplicates -- the one thing it cannot afford, since its whole value is pushing the
    /// combined slot count past one token's 258-expert working set.
    pub fn take(&mut self, key: i64, host: &mut [u8]) -> Option<(Vec<usize>, ExpertRef)> {
        let slot = *self.slot_of.get(&key)?;
        let n = host.len().min(self.slot_bytes);
        let rc = unsafe {
            (self.f.memcpy_d2h)(host.as_mut_ptr(), self.base + (slot * self.slot_bytes) as u64, n)
        };
        let meta = self.meta[slot].take();
        self.slot_of.remove(&key);
        self.key_of[slot] = EMPTY;
        self.used_at[slot] = 0;
        if rc != 0 {
            return None;
        }
        self.fills += 1;
        self.bytes_down += n as u64;
        meta
    }
}

impl Drop for Vram {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.f.mem_free)(self.base);
        }
    }
}

// The device pointer is an opaque handle owned outright by this struct.
unsafe impl Send for Vram {}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tier must degrade to "not present" rather than to a panic on every machine
    /// without a driver, because that is most machines the test suite runs on.
    #[test]
    fn absent_hardware_is_an_error_not_a_crash() {
        match Vram::new(1 << 30, 13_385_728) {
            Ok(v) => {
                assert!(v.nslot() > 0, "a successful tier holds at least one expert");
                assert!(v.bytes() <= 4 << 30, "cannot claim more than a laptop card holds");
            }
            Err(e) => assert!(!e.is_empty(), "a failure must say why"),
        }
    }

    /// A budget under one expert is a configuration error, not a zero-slot tier that
    /// silently accepts spills and drops them.
    #[test]
    fn a_budget_below_one_expert_is_refused() {
        if let Err(e) = Vram::new(1024, 13_385_728) {
            assert!(!e.is_empty());
        }
    }
}
