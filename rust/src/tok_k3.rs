// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::path::Path;

#[path = "tokdata/unicode.rs"]
mod uni;

fn in_set(t: &[(u32, u32)], c: u32) -> bool {
    t.binary_search_by(|&(lo, hi)| {
        if c < lo {
            std::cmp::Ordering::Greater
        } else if c > hi {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Equal
        }
    })
    .is_ok()
}

fn is_l(c: u32) -> bool {
    in_set(&uni::L, c)
}
fn is_n(c: u32) -> bool {
    in_set(&uni::N, c)
}
fn is_s(c: u32) -> bool {
    in_set(&uni::S, c)
}
fn is_u(c: u32) -> bool {
    in_set(&uni::U, c)
}
fn is_x(c: u32) -> bool {
    in_set(&uni::X, c)
}

const HAN: [(u32, u32); 22] = [
    (0x2E80, 0x2E99), (0x2E9B, 0x2EF3), (0x2F00, 0x2FD5), (0x3005, 0x3005),
    (0x3007, 0x3007), (0x3021, 0x3029), (0x3038, 0x303B), (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF), (0xF900, 0xFA6D), (0xFA70, 0xFAD9), (0x16FE2, 0x16FE3),
    (0x16FF0, 0x16FF1), (0x20000, 0x2A6DF), (0x2A700, 0x2B739), (0x2B740, 0x2B81D),
    (0x2B820, 0x2CEA1), (0x2CEB0, 0x2EBE0), (0x2EBF0, 0x2EE5D), (0x2F800, 0x2FA1D),
    (0x30000, 0x3134A), (0x31350, 0x323AF),
];

fn is_han(c: u32) -> bool {
    c >= 0x2E80 && in_set(&HAN, c)
}

fn s1(c: u32) -> bool {
    (is_u(c) || is_x(c)) && !is_han(c)
}
fn s2(c: u32) -> bool {
    (is_x(c) || (is_l(c) && !is_u(c))) && !is_han(c)
}

fn b64(s: &str) -> Option<Vec<u8>> {
    const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let s = s.trim().as_bytes();
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let mut acc = 0u32;
    let mut have = 0u32;
    for &ch in s {
        if ch == b'=' {
            break;
        }
        let v = T.iter().position(|&t| t == ch)? as u32;
        acc = (acc << 6) | v;
        have += 6;
        if have >= 8 {
            have -= 8;
            out.push((acc >> have) as u8);
        }
    }
    Some(out)
}

/// GPT-2 byte-level map: every byte becomes a printable codepoint. The vocabulary is
/// keyed by these strings, not by raw bytes.
fn bytemap() -> ([String; 256], HashMap<char, u8>) {
    let mut fwd: [String; 256] = std::array::from_fn(|_| String::new());
    let mut rev = HashMap::with_capacity(256);
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable =
            (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
        let cp = if printable {
            b
        } else {
            let c = 256 + n;
            n += 1;
            c
        };
        let ch = char::from_u32(cp).expect("bytemap codepoint");
        fwd[b as usize].push(ch);
        rev.insert(ch, b as u8);
    }
    (fwd, rev)
}

pub struct TokK3 {
    vocab: HashMap<String, u32>,
    pieces: Vec<String>,
    fwd: [String; 256],
    rev: HashMap<char, u8>,
    specials: HashMap<String, u32>,
    pub eos: u32,
    pub bos: u32,
}

impl TokK3 {
    /// `dir` must hold `tiktoken.model` and, for the special tokens,
    /// `tokenizer_config.json`.
    pub fn from_dir(dir: &Path) -> Result<TokK3, String> {
        let model = std::fs::read_to_string(dir.join("tiktoken.model"))
            .map_err(|e| format!("k3_tok: tiktoken.model: {e}"))?;
        let (fwd, rev) = bytemap();

        let mut vocab = HashMap::with_capacity(model.lines().count());
        let mut pieces: Vec<String> = Vec::new();
        for (ln, line) in model.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let mut it = line.split_ascii_whitespace();
            let (Some(b), Some(r)) = (it.next(), it.next()) else {
                return Err(format!("k3_tok: tiktoken.model:{}: not 'base64 rank'", ln + 1));
            };
            let raw = b64(b).ok_or_else(|| format!("k3_tok: tiktoken.model:{}: bad base64", ln + 1))?;
            let rank: u32 = r
                .parse()
                .map_err(|_| format!("k3_tok: tiktoken.model:{}: bad rank", ln + 1))?;
            let key: String = raw.iter().map(|&x| fwd[x as usize].as_str()).collect();
            if pieces.len() <= rank as usize {
                pieces.resize(rank as usize + 1, String::new());
            }
            pieces[rank as usize] = key.clone();
            vocab.insert(key, rank);
        }
        if vocab.is_empty() {
            return Err("k3_tok: tiktoken.model is empty".into());
        }

        let mut specials = HashMap::new();
        let (mut eos, mut bos) = (0u32, 0u32);
        if let Ok(txt) = std::fs::read_to_string(dir.join("tokenizer_config.json")) {
            let v: serde_json::Value = serde_json::from_str(&txt).map_err(|e| e.to_string())?;
            if let Some(m) = v["added_tokens_decoder"].as_object() {
                for (id, e) in m {
                    let Ok(id) = id.parse::<u32>() else { continue };
                    let Some(c) = e["content"].as_str() else { continue };
                    specials.insert(c.to_string(), id);
                    if pieces.len() <= id as usize {
                        pieces.resize(id as usize + 1, String::new());
                    }
                    pieces[id as usize] = c.to_string();
                }
            }
            let named = |k: &str| v[k].as_str().and_then(|s| specials.get(s).copied());
            eos = named("eos_token").unwrap_or(0);
            bos = named("bos_token").unwrap_or(0);
        }
        Ok(TokK3 { vocab, pieces, fwd, rev, specials, eos, bos })
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    pub fn id_of(&self, s: &str) -> Option<u32> {
        self.specials.get(s).copied().or_else(|| {
            let key: String = s.bytes().map(|b| self.fwd[b as usize].as_str()).collect();
            self.vocab.get(&key).copied()
        })
    }

    /// tiktoken merge: repeatedly join the adjacent pair whose CONCATENATION has the
    /// lowest id. There is no merges list in this format.
    fn bpe(&self, piece: &[char], out: &mut Vec<u32>) {
        let s: String = piece
            .iter()
            .flat_map(|c| {
                let mut b = [0u8; 4];
                let e = c.encode_utf8(&mut b).len();
                (0..e).map(move |i| self.fwd[b[i] as usize].clone())
            })
            .collect();
        if let Some(&id) = self.vocab.get(&s) {
            out.push(id);
            return;
        }
        let mut sym: Vec<String> = s.chars().map(String::from).collect();
        loop {
            let mut best = u32::MAX;
            let mut at = None;
            for i in 0..sym.len().saturating_sub(1) {
                let mut j = sym[i].clone();
                j.push_str(&sym[i + 1]);
                if let Some(&r) = self.vocab.get(&j) {
                    if r < best {
                        best = r;
                        at = Some(i);
                    }
                }
            }
            let Some(i) = at else { break };
            let tail = sym.remove(i + 1);
            sym[i].push_str(&tail);
        }
        for t in &sym {
            if let Some(&id) = self.vocab.get(t) {
                out.push(id);
            }
        }
    }

    fn contraction(&self, cp: &[char], k: usize) -> usize {
        let low = |c: char| c.to_ascii_lowercase();
        if k < cp.len() && cp[k] == '\'' && k + 1 < cp.len() {
            let d = low(cp[k + 1]);
            if k + 2 < cp.len() {
                let e = low(cp[k + 2]);
                if (d == 'r' && e == 'e') || (d == 'v' && e == 'e') || (d == 'l' && e == 'l') {
                    return k + 3;
                }
            }
            if matches!(d, 's' | 't' | 'm' | 'd') {
                return k + 2;
            }
        }
        k
    }

    /// Branches A|B of the Kimi split regex, replaying the engine's backtracking order:
    /// greedy optional prefix, maximally-greedy S1* given back until S2+ can take one.
    fn letters(&self, cp: &[char], i: usize) -> Option<usize> {
        let n = cp.len();
        for pfx in [true, false] {
            let mut j0 = i;
            if pfx {
                let c = cp[i] as u32;
                if c == 13 || c == 10 || is_l(c) || is_n(c) || i + 1 >= n {
                    continue;
                }
                j0 = i + 1;
            }
            let mut m1 = j0;
            while m1 < n && s1(cp[m1] as u32) {
                m1 += 1;
            }
            for s in (j0..=m1).rev() {
                if s < n && s2(cp[s] as u32) {
                    let mut k = s + 1;
                    while k < n && s2(cp[k] as u32) {
                        k += 1;
                    }
                    return Some(self.contraction(cp, k));
                }
            }
        }
        for pfx in [true, false] {
            let mut j0 = i;
            if pfx {
                let c = cp[i] as u32;
                if c == 13 || c == 10 || is_l(c) || is_n(c) || i + 1 >= n {
                    continue;
                }
                j0 = i + 1;
            }
            let mut m1 = j0;
            while m1 < n && s1(cp[m1] as u32) {
                m1 += 1;
            }
            if m1 > j0 {
                let mut k = m1;
                while k < n && s2(cp[k] as u32) {
                    k += 1;
                }
                return Some(self.contraction(cp, k));
            }
        }
        None
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        // Special tokens are matched on the raw text before any splitting.
        'outer: while !rest.is_empty() {
            let mut hit: Option<(usize, &str, u32)> = None;
            for (s, &id) in &self.specials {
                if let Some(p) = rest.find(s.as_str()) {
                    if hit.as_ref().is_none_or(|h| p < h.0) {
                        hit = Some((p, s, id));
                    }
                }
            }
            match hit {
                Some((p, s, id)) => {
                    self.pretok(&rest[..p], &mut out);
                    out.push(id);
                    rest = &rest[p + s.len()..];
                }
                None => {
                    self.pretok(rest, &mut out);
                    break 'outer;
                }
            }
        }
        out
    }

    fn pretok(&self, text: &str, out: &mut Vec<u32>) {
        let cp: Vec<char> = text.chars().collect();
        let n = cp.len();
        let nl = |c: char| c == '\r' || c == '\n';
        let mut i = 0usize;
        while i < n {
            let start = i;
            let c = cp[i] as u32;

            // H: Han runs are their own chunk, and a Han codepoint can match nothing else.
            if is_han(c) {
                let mut j = i;
                while j < n && is_han(cp[j] as u32) {
                    j += 1;
                }
                i = j;
                self.bpe(&cp[start..i], out);
                continue;
            }
            if let Some(e) = self.letters(&cp, i) {
                if e > i {
                    i = e;
                    self.bpe(&cp[start..i], out);
                    continue;
                }
            }
            if is_n(c) {
                let mut j = i;
                let mut k = 0;
                while j < n && is_n(cp[j] as u32) && k < 3 {
                    j += 1;
                    k += 1;
                }
                i = j;
                self.bpe(&cp[start..i], out);
                continue;
            }
            {
                let other = |c: char| {
                    let u = c as u32;
                    !is_s(u) && !is_l(u) && !is_n(u)
                };
                let mut j = i;
                if cp[i] == ' ' && j + 1 < n && other(cp[j + 1]) {
                    j += 1;
                }
                if j < n && other(cp[j]) {
                    while j < n && other(cp[j]) {
                        j += 1;
                    }
                    while j < n && nl(cp[j]) {
                        j += 1;
                    }
                    i = j;
                    self.bpe(&cp[start..i], out);
                    continue;
                }
            }
            {
                let mut r = i;
                while r < n && is_s(cp[r] as u32) {
                    r += 1;
                }
                if r > i {
                    let last = (i..r).rfind(|&j| nl(cp[j]));
                    if let Some(l) = last {
                        i = l + 1;
                    } else {
                        let mut end = if r < n { r - 1 } else { r };
                        if end <= i {
                            end = i + 1;
                        }
                        i = end;
                    }
                    self.bpe(&cp[start..i], out);
                    continue;
                }
            }
            i += 1;
            self.bpe(&cp[start..i], out);
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(p) = self.pieces.get(id as usize) else { continue };
            if self.specials.values().any(|&v| v == id) {
                bytes.extend_from_slice(p.as_bytes());
                continue;
            }
            for ch in p.chars() {
                if let Some(&b) = self.rev.get(&ch) {
                    bytes.push(b);
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_unicode_tables_are_sorted_so_the_binary_search_is_valid() {
        for (nm, t) in [
            ("L", &uni::L[..]),
            ("N", &uni::N[..]),
            ("S", &uni::S[..]),
            ("U", &uni::U[..]),
            ("X", &uni::X[..]),
        ] {
            for w in t.windows(2) {
                assert!(w[0].1 < w[1].0, "{nm}: ranges overlap or are unsorted: {w:?}");
            }
            for r in t {
                assert!(r.0 <= r.1, "{nm}: empty range {r:?}");
            }
        }
        for w in HAN.windows(2) {
            assert!(w[0].1 < w[1].0, "HAN unsorted");
        }
    }

    #[test]
    fn the_classes_agree_with_the_c_tables_on_ascii_and_common_scripts() {
        assert!(is_l('a' as u32) && is_l('Z' as u32) && is_l('é' as u32));
        assert!(!is_l('1' as u32) && !is_l(' ' as u32));
        assert!(is_n('7' as u32) && !is_n('x' as u32));
        assert!(is_s(' ' as u32) && is_s('\n' as u32) && is_s('\t' as u32));
        assert!(is_u('Z' as u32) && !is_u('z' as u32));
        assert!(is_han('中' as u32) && !is_han('a' as u32));
        // Han is \p{Lo}, so it is a letter, and it must be masked out of both letter
        // classes or a Han run would join an adjacent latin run.
        assert!(is_l('中' as u32));
        assert!(!s1('中' as u32) && !s2('中' as u32));
    }

    #[test]
    fn the_bytemap_is_a_bijection() {
        let (fwd, rev) = bytemap();
        assert_eq!(rev.len(), 256, "two bytes collided onto one codepoint");
        for b in 0..256usize {
            let ch = fwd[b].chars().next().unwrap();
            assert_eq!(rev[&ch], b as u8);
        }
        assert_eq!(fwd[b'A' as usize], "A");
        assert_ne!(fwd[b' ' as usize], " ", "space must not stay a raw space");
    }

    #[test]
    fn base64_round_trips_the_standard_alphabet() {
        assert_eq!(b64("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(b64("IQ==").unwrap(), b"!");
        assert_eq!(b64("").unwrap(), b"");
        assert!(b64("!!!!").is_none());
    }
}
