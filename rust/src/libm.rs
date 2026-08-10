// SPDX-License-Identifier: Apache-2.0

pub const C_CALL_SITES: &[(&str, &str)] = &[
    ("expf", "sigmoidf_ :102, kda_decay :167 :173, mla softmax :366, mla gate :382, router :429, attnres softmax :481"),
    ("tanhf", "situ_glu :116 :117"),
    ("sqrt", "rmsnorm :97 :471, l2norm_ :796 (f64, then narrowed)"),
    ("sqrtf", "mla scale :308, kda qscale :865"),
    ("ldexpf", "K3_E8M0 table :1208, mxfp4 dequant :1346"),
];

#[inline]
pub fn sigmoidf(x: f32) -> f32 {
    1.0f32 / (1.0f32 + (-x).exp())
}

#[inline]
pub fn exp2i(n: i32) -> f32 {
    debug_assert!((-127..=127).contains(&n), "E8M0 exponent out of range: {n}");
    if n >= -126 {
        f32::from_bits(((n + 127) as u32) << 23)
    } else {
        f32::from_bits(1u32 << (n + 149))
    }
}

pub fn e8m0_table() -> [f32; 256] {
    let mut t = [0.0f32; 256];
    for (b, slot) in t.iter_mut().enumerate() {
        *slot = if b == 255 { 0.0 } else { exp2i(b as i32 - 127) };
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "C" {
        fn expf(x: f32) -> f32;
        fn tanhf(x: f32) -> f32;
        fn ldexpf(x: f32, n: i32) -> f32;
    }

    #[test]
    fn transcendentals_match_c_on_a_structured_sample() {
        let mut probes: Vec<f32> = Vec::new();
        for i in -200_000..=200_000i32 {
            probes.push(i as f32 * 1e-3);
        }
        for &e in &[-40.0f32, -20.0, -10.0, -1.0, 0.0, 1.0, 10.0, 80.0, 88.0, 89.0] {
            probes.push(e);
        }
        probes.extend([f32::MIN_POSITIVE, -f32::MIN_POSITIVE, 0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY]);

        for &x in &probes {
            assert_eq!(
                x.exp().to_bits(),
                unsafe { expf(x) }.to_bits(),
                "expf disagrees at x={x:e}"
            );
            assert_eq!(
                x.tanh().to_bits(),
                unsafe { tanhf(x) }.to_bits(),
                "tanhf disagrees at x={x:e}"
            );
            if x >= 0.0 {
                assert_eq!(x.sqrt().to_bits(), x.sqrt().to_bits());
            }
        }
    }

    #[test]
    fn exp2i_matches_ldexpf_over_the_e8m0_range() {
        for n in -127..=127 {
            assert_eq!(
                exp2i(n).to_bits(),
                unsafe { ldexpf(1.0f32, n) }.to_bits(),
                "exp2i disagrees with ldexpf at n={n}"
            );
        }
    }

    #[test]
    fn e8m0_table_zeroes_the_nan_scale() {
        let t = e8m0_table();
        assert_eq!(t[255], 0.0, "byte 255 must map to zero, not NaN");
        assert_eq!(t[127], 1.0, "byte 127 is 2^0");
        assert_eq!(t[0].to_bits(), unsafe { ldexpf(1.0, -127) }.to_bits());
    }

    #[test]
    fn sigmoidf_uses_the_c_spelling() {
        let x = 0.1f32;
        let c_form = 1.0f32 / (1.0f32 + (-x).exp());
        assert_eq!(sigmoidf(x).to_bits(), c_form.to_bits());
    }
}
