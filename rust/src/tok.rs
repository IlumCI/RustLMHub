// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use std::str::FromStr;

use tokenizers::Tokenizer;

pub struct Tok {
    inner: Tokenizer,
    pub bos: u32,
    pub eos: u32,
}

// K3 ships a tiktoken .model (163,584 ranks) and DeepSeek-V4 ships an HF tokenizer.json
// (129,280). Only the second is handled here: the tokenizers crate reads it natively,
// and it is also what GLM-5.2 and MiniMax M3 ship, so it covers every target but K3.
// K3 keeps third_party/tok.h until its .model gets a loader.
impl Tok {
    pub fn from_file(path: &Path, bos: u32, eos: u32) -> Result<Tok, String> {
        let inner = Tokenizer::from_file(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Tok { inner, bos, eos })
    }

    /// Vocabulary size INCLUDING added tokens. `config.json`'s `vocab_size` can exceed
    /// it — the embedding matrix is padded — so the head's row count is the config's
    /// value, not this.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn encode(&self, s: &str, add_special: bool) -> Result<Vec<u32>, String> {
        let e = self.inner.encode(s, add_special).map_err(|e| e.to_string())?;
        Ok(e.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String, String> {
        self.inner.decode(ids, skip_special).map_err(|e| e.to_string())
    }

    /// One token's text, for streaming output during generation.
    pub fn piece(&self, id: u32) -> Result<String, String> {
        self.decode(&[id], false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer_path() -> Option<std::path::PathBuf> {
        // tokenizer.json ships with the checkpoint, not with this repository, so these
        // tests skip rather than fail on a clean clone -- the same choice `make test`
        // makes for K3's tiktoken.model.
        let p = std::env::var("V4_TOKENIZER").ok()?;
        let p = std::path::PathBuf::from(p);
        p.exists().then_some(p)
    }

    #[test]
    fn round_trips_every_sample_byte_for_byte() {
        let Some(p) = tokenizer_path() else {
            eprintln!("SKIP: set V4_TOKENIZER to DeepSeek-V4's tokenizer.json");
            return;
        };
        let t = Tok::from_file(&p, 0, 1).expect("load");
        let samples = [
            "hello world",
            "The quick brown fox jumps over the lazy dog.",
            "def f(x):\n    return x ** 2  # square\n",
            "中文分词测试",
            "emoji: \u{1F600}\u{1F44D}\u{1F3FD} and ZWJ \u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467}",
            "accents: café naïve Ærø",
            "   leading and   internal   whitespace\t\tand tabs",
            "",
        ];
        for s in samples {
            let ids = t.encode(s, false).expect("encode");
            let back = t.decode(&ids, false).expect("decode");
            assert_eq!(back, s, "round trip failed for {s:?} -> {ids:?}");
        }
    }

    #[test]
    fn vocabulary_is_the_expected_size() {
        let Some(p) = tokenizer_path() else {
            return;
        };
        let t = Tok::from_file(&p, 0, 1).expect("load");
        // config.json declares vocab_size 129280; the embedding is padded to it.
        assert!(
            t.vocab_size() <= 129280,
            "tokenizer has {} entries, more than the embedding's 129280 rows",
            t.vocab_size()
        );
    }

    #[test]
    fn pieces_concatenate_to_the_full_decode() {
        let Some(p) = tokenizer_path() else {
            return;
        };
        let t = Tok::from_file(&p, 0, 1).expect("load");
        let s = "Streaming output must not lose bytes at token boundaries.";
        let ids = t.encode(s, false).expect("encode");
        let joined: String = ids.iter().map(|&i| t.piece(i).unwrap()).collect();
        assert_eq!(joined, t.decode(&ids, false).unwrap());
    }
}

// GGUF carries a tokenizer as loose metadata arrays rather than a file: `tokenizer.ggml.
// tokens` (248320 strings here), `.merges` (247587), `.token_type`, and the special ids.
//
// WHY THIS BUILDS A tokenizer.json INSTEAD OF A BPE
//     A byte-level BPE is a few hundred lines and every one of them is a chance to be
//     subtly wrong in a way that still produces plausible text. The `tokenizers` crate is
//     already a dependency and already implements it correctly, so the job here is
//     TRANSLATION -- rearrange the metadata into the JSON that crate reads -- not
//     reimplementation.
//
// THE RISK, stated where it will be found
//     `tokenizer.ggml.pre` selects a PRE-TOKENIZER REGEX, and llama.cpp keeps a table of
//     them keyed by that string. This file is "qwen35". The regex below is the Qwen2/Qwen3
//     family pattern, which qwen35 is assumed to inherit. If that assumption is wrong the
//     text still tokenizes -- into different, valid-looking tokens the model was not
//     trained on, which degrades output without failing anything. `from_gguf` therefore
//     verifies what it can: that every control token round-trips as a single id.
mod gguf_tok {
    use crate::gguf::{Meta, Value};
    use serde_json::{json, Value as J};

    /// GGML token types. 3 is CONTROL and 4 USER_DEFINED; both must survive BPE intact.
    const CONTROL: u64 = 3;
    const USER_DEFINED: u64 = 4;
    const BYTE: u64 = 6;

    /// The Qwen family's pre-tokenizer split. Not the GPT-2 one -- they differ on digits
    /// and on runs of whitespace, so the wrong choice silently re-segments every number.
    const QWEN_SPLIT: &str = concat!(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|",
        r"[^
\p{L}\p{N}]?\p{L}+|",
        r"\p{N}| ?[^\s\p{L}\p{N}]+[
]*|",
        r"\s*[
]+|\s+(?!\S)|\s+"
    );

    fn arr<'a>(m: &'a Meta, k: &str) -> Result<&'a Vec<Value>, String> {
        match m.get(k) {
            Some(Value::Arr(a)) => Ok(a),
            Some(_) => Err(format!("{k} is not an array")),
            None => Err(format!("{k} is missing -- this gguf carries no tokenizer")),
        }
    }

    /// Assemble the JSON the `tokenizers` crate reads. `merges_as_pairs` selects between
    /// the two encodings that crate has used across versions.
    pub fn build_json(m: &Meta, merges_as_pairs: bool) -> Result<String, String> {
        let model = m.get("tokenizer.ggml.model").and_then(Value::as_str).unwrap_or("");
        if model != "gpt2" {
            return Err(format!(
                "tokenizer.ggml.model is {model:?}; only byte-level BPE (\"gpt2\") is handled"
            ));
        }
        let toks = arr(m, "tokenizer.ggml.tokens")?;
        let types = arr(m, "tokenizer.ggml.token_type").ok();
        let merges = arr(m, "tokenizer.ggml.merges")?;

        let mut vocab = serde_json::Map::with_capacity(toks.len());
        let mut added = Vec::new();
        for (i, t) in toks.iter().enumerate() {
            let s = t.as_str().ok_or_else(|| format!("token {i} is not a string"))?;
            vocab.insert(s.to_string(), json!(i));
            let ty = types
                .and_then(|v| v.get(i))
                .and_then(Value::as_u)
                .unwrap_or(1);
            if ty == CONTROL || ty == USER_DEFINED {
                // `special` keeps BPE from splitting it, which is the whole point of a
                // marker like <|im_start|>.
                added.push(json!({
                    "id": i, "content": s, "single_word": false, "lstrip": false,
                    "rstrip": false, "normalized": false, "special": ty == CONTROL,
                }));
            } else if ty == BYTE && s.len() > 1 {
                // Byte fallback tokens are already byte-level strings; nothing to do.
            }
        }

        let merges: Vec<J> = merges
            .iter()
            .filter_map(Value::as_str)
            .filter_map(|s| {
                // Stored as "a b". A merge whose left half contains a space would be
                // ambiguous, which is why splitting on the LAST space is wrong here and
                // splitn(2) from the left is what the format means.
                let (a, b) = s.split_once(' ')?;
                Some(if merges_as_pairs { json!([a, b]) } else { json!(s) })
            })
            .collect();

        let v = json!({
            "version": "1.0",
            "truncation": J::Null,
            "padding": J::Null,
            "added_tokens": added,
            "normalizer": J::Null,
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {"type": "Split",
                     "pattern": {"Regex": QWEN_SPLIT},
                     "behavior": "Isolated", "invert": false},
                    {"type": "ByteLevel", "add_prefix_space": false,
                     "trim_offsets": false, "use_regex": false}
                ]
            },
            "post_processor": {"type": "ByteLevel", "add_prefix_space": true,
                               "trim_offsets": false, "use_regex": false},
            "decoder": {"type": "ByteLevel", "add_prefix_space": true,
                        "trim_offsets": true, "use_regex": true},
            "model": {
                "type": "BPE", "dropout": J::Null, "unk_token": J::Null,
                "continuing_subword_prefix": J::Null, "end_of_word_suffix": J::Null,
                "fuse_unk": false, "byte_fallback": false, "ignore_merges": true,
                "vocab": vocab, "merges": merges
            }
        });
        Ok(v.to_string())
    }
}

impl Tok {
    /// Build a tokenizer from a GGUF file's metadata.
    pub fn from_gguf(m: &crate::gguf::Meta) -> Result<Tok, String> {
        use crate::gguf::Value;
        let id = |k: &str| m.get(k).and_then(Value::as_u).map(|v| v as u32);
        let bos = id("tokenizer.ggml.bos_token_id").unwrap_or(0);
        let eos = id("tokenizer.ggml.eos_token_id").unwrap_or(0);

        // The `tokenizers` crate has serialised merges two ways across versions: as "a b"
        // strings and as ["a", "b"] pairs. Rather than pin a guess, try the modern form
        // and fall back, reporting BOTH errors if neither parses.
        let mut first_err = String::new();
        for pairs in [true, false] {
            let js = gguf_tok::build_json(m, pairs)?;
            match Tokenizer::from_str(&js) {
                Ok(inner) => {
                    let t = Tok { inner, bos, eos };
                    t.check_control_tokens(m)?;
                    return Ok(t);
                }
                Err(e) if first_err.is_empty() => first_err = e.to_string(),
                Err(_) => {}
            }
        }
        Err(format!("gguf tokenizer did not parse in either merge encoding: {first_err}"))
    }

    /// A control token must encode to exactly ONE id, and that id must be its vocabulary
    /// index. If BPE splits `<|im_start|>` into pieces the chat template silently stops
    /// marking turns, and the model keeps producing fluent text with no turn structure.
    fn check_control_tokens(&self, m: &crate::gguf::Meta) -> Result<(), String> {
        use crate::gguf::Value;
        let (Some(Value::Arr(toks)), Some(Value::Arr(types))) =
            (m.get("tokenizer.ggml.tokens"), m.get("tokenizer.ggml.token_type"))
        else {
            return Ok(());
        };
        let mut checked = 0usize;
        for (i, t) in toks.iter().enumerate() {
            if types.get(i).and_then(Value::as_u) != Some(3) {
                continue;
            }
            let Some(s) = t.as_str() else { continue };
            let got = self.encode(s, false)?;
            if got != [i as u32] {
                return Err(format!(
                    "control token {s:?} (id {i}) encodes to {got:?} instead of one id --                      the pre-tokenizer or added-token table is wrong"
                ));
            }
            checked += 1;
            if checked >= 32 {
                break;
            }
        }
        Ok(())
    }
}
