// SPDX-License-Identifier: Apache-2.0

use std::hint::black_box;
use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

extern "C" {
    fn expf(x: f32) -> f32;
    fn tanhf(x: f32) -> f32;
    fn sqrtf(x: f32) -> f32;
    fn sqrt(x: f64) -> f64;
    fn ldexpf(x: f32, n: i32) -> f32;
}

struct Report {
    name: &'static str,
    sites: &'static str,
    mismatches: AtomicU64,
    first: AtomicU64,
}

impl Report {
    const fn new(name: &'static str, sites: &'static str) -> Self {
        Report { name, sites, mismatches: AtomicU64::new(0), first: AtomicU64::new(0) }
    }

    #[cold]
    fn record(&self, bits: u32) {
        self.mismatches.fetch_add(1, Ordering::Relaxed);
        let _ = self.first.compare_exchange(
            0,
            bits as u64 + 1,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
    }

    fn print(&self, tested: u64) -> bool {
        let n = self.mismatches.load(Ordering::Relaxed);
        if n == 0 {
            println!("  {:<10} identical over {tested} inputs", self.name);
            true
        } else {
            let b = (self.first.load(Ordering::Relaxed) - 1) as u32;
            println!(
                "  {:<10} {n} MISMATCHES / {tested}; first at x={:e} (0x{b:08x})\n\
                 {:<12} call sites: {}",
                self.name,
                f32::from_bits(b),
                "",
                self.sites
            );
            false
        }
    }
}

static EXP: Report = Report::new("expf", "sigmoidf_, kda_decay, softmaxes, router");
static TANH: Report = Report::new("tanhf", "situ_glu");
static SQRTF: Report = Report::new("sqrtf", "mla scale, kda qscale");
static SQRT: Report = Report::new("sqrt/f64", "rmsnorm, l2norm_");

fn main() -> ExitCode {
    let nthreads = thread::available_parallelism().map_or(4, |n| n.get()) as u64;
    println!(
        "libm parity: exhaustive f32 sweep on {nthreads} threads\n\
         target {} / {}",
        std::env::consts::ARCH,
        std::env::consts::OS
    );

    let tested = AtomicU64::new(0);
    thread::scope(|s| {
        for t in 0..nthreads {
            let tested = &tested;
            s.spawn(move || {
                let mut local = 0u64;
                let mut bits = t;
                while bits <= u32::MAX as u64 {
                    let b = bits as u32;
                    bits += nthreads;
                    let x = f32::from_bits(b);
                    if x.is_nan() {
                        continue; // NaN payload propagation is not part of the contract
                    }
                    local += 1;
                    let x = black_box(x);

                    if x.exp().to_bits() != unsafe { expf(x) }.to_bits() {
                        EXP.record(b);
                    }
                    if x.tanh().to_bits() != unsafe { tanhf(x) }.to_bits() {
                        TANH.record(b);
                    }
                    if x.sqrt().to_bits() != unsafe { sqrtf(x) }.to_bits() {
                        SQRTF.record(b);
                    }
                    let d = x as f64;
                    if d.sqrt().to_bits() != unsafe { sqrt(d) }.to_bits() {
                        SQRT.record(b);
                    }
                }
                tested.fetch_add(local, Ordering::Relaxed);
            });
        }
    });

    let mut ldexp_bad = 0u32;
    for n in -127..=127 {
        if k3::libm::exp2i(n).to_bits() != unsafe { ldexpf(1.0f32, n) }.to_bits() {
            ldexp_bad += 1;
        }
    }

    let n = tested.load(Ordering::Relaxed);
    println!("\nresults:");
    let mut ok = EXP.print(n);
    ok &= TANH.print(n);
    ok &= SQRTF.print(n);
    ok &= SQRT.print(n);
    if ldexp_bad == 0 {
        println!("  {:<10} identical over n in [-127, 127]", "ldexpf");
    } else {
        println!("  {:<10} {ldexp_bad} MISMATCHES / 255", "ldexpf");
        ok = false;
    }

    if ok {
        println!("\nPASS: Rust float math is bit-identical to this target's libm.");
        println!("      The port needs no vendored libm and no FFI.");
        ExitCode::SUCCESS
    } else {
        println!(
            "\nFAIL: divergence found. The port cannot claim byte-identical\n\
             output until one implementation is pinned for both builds."
        );
        ExitCode::FAILURE
    }
}
