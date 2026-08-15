// SPDX-License-Identifier: Apache-2.0
//
// Optional accelerators, none of which may ever be required.
//
// THE CONTRACT
//     Every accelerator here is optional TWICE over:
//
//       compile time  a cargo feature. Absent, the crate is not in the tree at all and
//                     this file compiles to the plain path with no `dep:` reference.
//       run time      even compiled in, it must PROVE itself on this machine before it is
//                     used -- huge pages need a kernel that has them, io_uring needs a
//                     recent one, a GPU needs a driver, and a third-party kernel needs to
//                     agree with ours to the bit. Any of those failing falls back.
//
//     So `cargo build` with no features, on a machine with none of the hardware, runs
//     exactly the code it ran before any of this existed. That is not a courtesy to
//     packagers; it is what keeps the golden tests meaningful, because the fallback path
//     is the path those tests pin.
//
// WHY VERIFICATION IS NOT OPTIONAL FOR KERNELS
//     A faster matmul that disagrees with ours in the last bit is not a faster matmul, it
//     is a different model. The whole verification apparatus -- `rust-golden`, the C
//     differential suite -- rests on one reduction order. So `rage-quant` is not trusted
//     because it is published; it is trusted per-process, after being run against our own
//     kernel on real data, and dropped on any disagreement.

/// What one accelerator is doing, and why.
pub struct Status {
    pub name: &'static str,
    /// Was the cargo feature enabled for this build?
    pub compiled: bool,
    /// Did it pass its runtime probe and get used?
    pub active: bool,
    pub note: String,
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = match (self.compiled, self.active) {
            (false, _) => "not built",
            (true, false) => "built, INACTIVE",
            (true, true) => "ACTIVE",
        };
        write!(f, "{:<8} {:<16} {}", self.name, state, self.note)
    }
}

// ---------------------------------------------------------------------------
// Huge pages for the expert arena  (no feature: rustix is already a dependency)
// ---------------------------------------------------------------------------

/// A large byte region, backed by 2 MB pages when the kernel has them.
///
/// WHY IT IS WORTH DOING
/// ```text
///     The expert arena is multiple gigabytes and addressed randomly: a token touches 8
///     experts in each of 40 layers, scattered across the whole arena. At 4 KB pages a 6 GB
///     arena spans 1.5M pages against a TLB holding on the order of a thousand entries, so
///     nearly every expert access pays a page-walk on top of its cache miss. 2 MB pages
///     cover the same arena in 3000 pages.
/// ```
///
/// WHY THERE IS NO CARGO FEATURE
/// ```text
///     The first attempt used the `membase` crate. It does not compile -- version 0.2.1
///     is missing a `use std::fs::File;` in its own `huge_pages.rs`, so `cargo build`
///     fails inside the dependency. `rustix` was ALREADY a dependency with the `mm`
///     feature, and `MapFlags::HUGETLB` is the whole mechanism, so the crate bought
///     nothing but a build failure.
///
///     Since it costs no dependency, it needs no feature either: try huge pages, fall back
///     silently to a plain allocation. A machine with no hugetlb pool -- which is most
///     machines, including this one by default -- runs exactly the code it ran before.
/// ```
pub struct Arena {
    map: Option<(*mut core::ffi::c_void, usize)>,
    plain: Vec<u8>,
    pub huge_pages: bool,
}

// SAFETY: the mapping is owned exclusively by this Arena and unmapped in Drop; the raw
// pointer is never shared. The Vec fallback is Send/Sync already.
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    pub fn new(bytes: usize) -> Arena {
        // Small regions are not worth a syscall, and MADV_HUGEPAGE on anything under one
        // huge page cannot help anyway.
        if bytes < (2 << 20) {
            return Arena { map: None, plain: vec![0u8; bytes], huge_pages: false };
        }
        let prot = rustix::mm::ProtFlags::READ | rustix::mm::ProtFlags::WRITE;

        // First choice: a pool-backed hugetlb mapping. Needs `vm.nr_hugepages` reserved up
        // front, which is 0 on a default kernel, so this usually fails -- and ENOMEM here
        // means "no huge pages reserved", not "out of memory".
        // SAFETY: an anonymous mapping with a null hint asks the kernel for fresh pages;
        // no file, no aliasing, nothing existing to clobber.
        if let Ok(p) = unsafe {
            rustix::mm::mmap_anonymous(
                core::ptr::null_mut(),
                bytes,
                prot,
                rustix::mm::MapFlags::PRIVATE | rustix::mm::MapFlags::HUGETLB,
            )
        } {
            return Arena { map: Some((p, bytes)), plain: Vec::new(), huge_pages: true };
        }

        // Second choice, and the one that actually fires here: an ordinary anonymous
        // mapping plus MADV_HUGEPAGE, which asks khugepaged to back it with 2 MB pages via
        // TRANSPARENT huge pages. No pool, no privileges.
        //
        // It must be a mapping and not a Vec: MADV_HUGEPAGE requires a PAGE-ALIGNED start
        // address, and the Rust allocator hands back 8- or 16-byte alignment. Calling it
        // on a Vec fails with EINVAL every time, silently reporting "unavailable" on a
        // machine whose THP is set to `always`.
        // SAFETY: as above.
        if let Ok(p) = unsafe {
            rustix::mm::mmap_anonymous(core::ptr::null_mut(), bytes, prot, rustix::mm::MapFlags::PRIVATE)
        } {
            // SAFETY: exactly the region just mapped; madvise is advisory and cannot
            // invalidate the pointer or change the contents.
            let thp = unsafe {
                rustix::mm::madvise(p, bytes, rustix::mm::Advice::LinuxHugepage).is_ok()
            };
            return Arena { map: Some((p, bytes)), plain: Vec::new(), huge_pages: thp };
        }

        Arena { map: None, plain: vec![0u8; bytes], huge_pages: false }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        match self.map {
            // SAFETY: `len` bytes were mapped READ|WRITE and are owned by self.
            Some((p, len)) => unsafe { core::slice::from_raw_parts_mut(p.cast::<u8>(), len) },
            None => &mut self.plain,
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        match self.map {
            // SAFETY: as above; the mapping outlives the borrow.
            Some((p, len)) => unsafe { core::slice::from_raw_parts(p.cast::<u8>(), len) },
            None => &self.plain,
        }
    }

    pub fn len(&self) -> usize {
        self.map.map(|(_, n)| n).unwrap_or(self.plain.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn status(_bytes: usize) -> Status {
        // Probe with ONE huge page, never the real budget. Reporting status should not
        // reserve gigabytes -- the first version of this allocated the whole 5 GB cache
        // just to print a line about it.
        let a = Arena::new(2 << 20);
        Status {
            name: "hugepg",
            compiled: true,
            active: a.huge_pages,
            note: if a.huge_pages {
                "2 MB pages for the expert arena (MAP_HUGETLB, or THP via MADV_HUGEPAGE)"
                    .into()
            } else {
                "4 KB pages: no hugetlb pool and transparent huge pages unavailable".into()
            },
        }
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        if let Some((p, len)) = self.map.take() {
            // SAFETY: exactly the mapping this Arena created, unmapped once.
            unsafe {
                let _ = rustix::mm::munmap(p, len);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Third-party quantised kernels  (feature `rage`, crate `rage-quant`)
// ---------------------------------------------------------------------------

/// Does `rage-quant`'s Q4_K row dot agree with ours, bit for bit, on real block data?
///
/// This is the only question that matters. `rage-quant` makes the same architectural
/// argument this engine does -- never dequantise to f32, dot straight off the blocks -- so
/// it is plausible prior art. But "plausible" is how every silent wrongness in this project
/// began. The two implementations must produce the SAME f32, because `rust-golden` pins one
/// reduction order and a kernel that rounds differently is a different model.
///
/// Returns the verdict and the largest disagreement seen, so a near-miss is reported as a
/// near-miss rather than as a failure indistinguishable from garbage.
pub fn verify_rage_q4k(blocks: &[u8], x: &[f32], k_in: usize) -> (bool, f32) {
    #[cfg(feature = "rage")]
    {
        if blocks.len() < k_in / crate::gguf::QK_K * crate::gguf::Q4K_BLOCK || x.len() < k_in {
            return (false, f32::NAN);
        }
        let mine = {
            let mut y = [0f32; 1];
            crate::gguf::matmul_q4k(&mut y, x, blocks, k_in, 1);
            y[0]
        };
        let theirs = rage_quant::dot_q4_k_f32(blocks, x, k_in);
        let d = (mine - theirs).abs();
        // Bit-identical or nothing. A tolerance here would be a tolerance on the golden
        // test, which is the one thing that cannot be negotiated.
        (mine.to_bits() == theirs.to_bits(), d)
    }
    #[cfg(not(feature = "rage"))]
    {
        let (_, _, _) = (blocks, x, k_in);
        (false, f32::NAN)
    }
}

pub fn rage_status() -> Status {
    #[cfg(feature = "rage")]
    {
        // A WELL-FORMED super-block, not random bytes. A Q4_K block opens with two f16
        // scales (d, dmin); filling them with noise gives arbitrary or non-finite
        // multipliers, and any two implementations would then "disagree" for reasons that
        // say nothing about their layouts. Here d = 0.1 and dmin = 0.05, both exactly
        // representable, with pseudorandom 6-bit scales and 4-bit quants after them.
        let mut blocks = vec![0u8; crate::gguf::Q4K_BLOCK];
        blocks[0..2].copy_from_slice(&0x2E66u16.to_le_bytes()); // f16 ~= 0.1
        blocks[2..4].copy_from_slice(&0x2666u16.to_le_bytes()); // f16 ~= 0.05
        for (i, b) in blocks[4..].iter_mut().enumerate() {
            *b = (i.wrapping_mul(37).wrapping_add(11) % 251) as u8;
        }
        let x: Vec<f32> = (0..crate::gguf::QK_K).map(|i| (i as f32 * 0.01).sin()).collect();
        let (ok, d) = verify_rage_q4k(&blocks, &x, crate::gguf::QK_K);
        Status {
            name: "rage",
            compiled: true,
            active: ok,
            note: if ok {
                "rage-quant Q4_K dot is bit-identical to ours".into()
            } else {
                format!(
                    "REJECTED: rage-quant disagrees by {d:e}; using our kernels, which are \
                     the ones rust-golden pins"
                )
            },
        }
    }
    #[cfg(not(feature = "rage"))]
    {
        Status {
            name: "rage",
            compiled: false,
            active: false,
            note: "build with --features rage to bit-check rage-quant against our kernels"
                .into(),
        }
    }
}

// ---------------------------------------------------------------------------
// io_uring  (feature `uring`, crate `glommio`)  and  GPU  (feature `gpu`, cubecl)
// ---------------------------------------------------------------------------

pub fn uring_status() -> Status {
    #[cfg(feature = "uring")]
    {
        // Probe the running kernel rather than trusting the build: io_uring needs 5.8+ for
        // what expert streaming would use, and containers often mask it entirely.
        // Build a real ring. Nothing else proves io_uring is usable: the syscall can be
        // present and still be blocked by seccomp, a container policy, or
        // `kernel.io_uring_disabled`.
        let ok = io_uring::IoUring::new(8).is_ok();
        Status {
            name: "uring",
            compiled: true,
            active: false,
            note: if ok {
                "io_uring ring created; expert reads not yet routed through it".into()
            } else {
                "io_uring is compiled in but this kernel refuses a ring".into()
            },
        }
    }
    #[cfg(not(feature = "uring"))]
    {
        Status {
            name: "uring",
            compiled: false,
            active: false,
            note: "build with --features uring to give expert reads queue depth (io-uring)"
                .into(),
        }
    }
}

pub fn gpu_status() -> Status {
    #[cfg(feature = "gpu")]
    {
        Status {
            name: "gpu",
            compiled: true,
            active: false,
            note: "cubecl linked; no kernels ported yet (the GPU still only caches bytes)"
                .into(),
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        Status {
            name: "gpu",
            compiled: false,
            active: false,
            note: "build with --features gpu to compile kernels for CUDA/ROCm/WGSL (cubecl)"
                .into(),
        }
    }
}

/// Everything, for `rustlm accel`.
pub fn status(arena_bytes: usize) -> Vec<Status> {
    vec![Arena::status(arena_bytes), rage_status(), uring_status(), gpu_status()]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property the whole file exists for: with no features, nothing is compiled in,
    /// nothing is active, and allocation still works. This test runs in the DEFAULT build,
    /// which is the build every gate uses.
    #[test]
    fn every_accelerator_is_absent_and_harmless_by_default() {
        for s in status(1 << 20) {
            if !s.compiled {
                assert!(!s.active, "{} cannot be active without being built", s.name);
                assert!(
                    s.note.contains("--features"),
                    "{}: an unbuilt accelerator must say how to build it, got {:?}",
                    s.name,
                    s.note
                );
            }
        }
    }

    /// An arena must hand back the requested size whichever path it took, so no caller can
    /// tell the difference except by asking.
    #[test]
    fn the_arena_allocates_the_requested_size_on_either_path() {
        let a = Arena::new(4 << 20);
        assert_eq!(a.len(), 4 << 20);
        assert!(!a.is_empty());
        // Whichever path was taken, the bytes must be usable end to end.
        let mut a = a;
        let n = a.len();
        a.as_mut_slice()[n - 1] = 0xAB;
        a.as_mut_slice()[0] = 0xCD;
        assert_eq!(a.as_slice()[0], 0xCD);
        assert_eq!(a.as_slice()[n - 1], 0xAB, "the last byte of the mapping must be live");
    }

    #[test]
    fn a_zero_length_arena_is_empty_rather_than_a_panic() {
        assert!(Arena::new(0).is_empty());
    }

    /// Without the feature the verifier must REFUSE, never silently pass. A verification
    /// that returns "fine" when it did not run is worse than none.
    #[test]
    fn kernel_verification_refuses_rather_than_passing_when_not_built() {
        if !cfg!(feature = "rage") {
            let (ok, _) = verify_rage_q4k(&[0u8; 144], &[0f32; 256], 256);
            assert!(!ok, "an unbuilt kernel must never be reported as verified");
            assert!(!rage_status().active);
        }
    }

    /// Undersized inputs must be rejected, not read past the end.
    #[test]
    fn kernel_verification_rejects_a_short_block() {
        let (ok, _) = verify_rage_q4k(&[0u8; 8], &[0f32; 256], 256);
        assert!(!ok);
    }
}
