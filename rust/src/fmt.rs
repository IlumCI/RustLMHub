// SPDX-License-Identifier: Apache-2.0

pub fn g(x: f64, prec: usize) -> String {
    if x.is_nan() {
        return if x.is_sign_negative() { "-nan".into() } else { "nan".into() };
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf".into() } else { "inf".into() };
    }
    let p = prec.max(1);
    let exp = if x == 0.0 {
        0
    } else {
        let e = format!("{:.*e}", p - 1, x);
        e.split('e').nth(1).and_then(|s| s.parse::<i32>().ok()).unwrap_or(0)
    };

    if exp < -4 || exp >= p as i32 {
        let s = format!("{:.*e}", p - 1, x);
        let (mant, e) = s.split_once('e').unwrap_or((s.as_str(), "0"));
        let mant = strip(mant);
        let ev: i32 = e.parse().unwrap_or(0);
        format!("{mant}e{}{:02}", if ev < 0 { '-' } else { '+' }, ev.abs())
    } else {
        let decimals = (p as i32 - 1 - exp).max(0) as usize;
        strip(&format!("{x:.decimals$}"))
    }
}

fn strip(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    let t = s.trim_end_matches('0');
    t.strip_suffix('.').unwrap_or(t).to_string()
}

#[cfg(test)]
mod tests {
    use super::g;

    #[test]
    fn matches_c_printf_percent_g() {
        let cases: &[(f64, &str)] = &[
            (-9.754315, "-9.75432"),   // rounds to 6 significant digits
            (6.734270, "6.73427"),     // trailing zero stripped
            (-0.514719, "-0.514719"),
            (0.0, "0"),                // not "0.000000"
            (1.0, "1"),
            (100.0, "100"),
            (1234567.0, "1.23457e+06"),// exponent >= precision -> %e
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),        // exponent < -4 -> %e, two exponent digits
            (-1.10336e34, "-1.10336e+34"),
            (7.84245e37, "7.84245e+37"),
            (6.59144e35, "6.59144e+35"),
        ];
        for &(x, want) in cases {
            assert_eq!(g(x, 6), want, "%.6g of {x}");
        }
    }

    #[test]
    fn non_finite_uses_the_c_spellings() {
        assert_eq!(g(f64::INFINITY, 6), "inf");
        assert_eq!(g(f64::NEG_INFINITY, 6), "-inf");
        assert_eq!(g(f64::NAN, 6), "nan");
    }
}
