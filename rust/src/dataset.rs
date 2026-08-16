//! Universal dataset layer: take a JSONL file in whatever shape it happens to be — chat
//! messages, separate system/user/assistant fields, instruction/input/output, prompt/
//! completion, or a bare text field — detect the shape, and normalise every row to one
//! [`Record`] the training and eval paths understand. This is what lets a user point the tool
//! at any dataset they find instead of reformatting it first.
//!
//! Normalisation is to `(system, prompt, response)`. SFT trains the model to produce
//! `response` given `system + prompt` (loss masked to the response, see [`crate::train`]);
//! an eval set is the same shape, with `response` used as the expected/reference answer.

use serde_json::Value;

/// The recognised source shapes. Detection guesses one from a row's keys; a caller can also
/// force one when the guess is wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fmt {
    /// `{"messages": [{"role","content"}, ...]}` — the OpenAI chat shape. The last assistant
    /// turn is the response; earlier turns are the prompt/system.
    Chat,
    /// `{"system","user","response"}` (also `assistant`/`output` for the last field).
    SysUserResp,
    /// Alpaca: `{"instruction","input"(opt),"output"}`.
    Instruction,
    /// `{"prompt","completion"}` (also `{"prompt","response"}`).
    PromptCompletion,
    /// A single field of raw text (`{"text"}`) — pure language-model completion, no prompt
    /// masking (the whole sequence is the response).
    TextOnly,
}

impl Fmt {
    pub fn label(self) -> &'static str {
        match self {
            Fmt::Chat => "chat messages",
            Fmt::SysUserResp => "system / user / response",
            Fmt::Instruction => "instruction / input / output",
            Fmt::PromptCompletion => "prompt / completion",
            Fmt::TextOnly => "text only",
        }
    }
}

/// A normalised training/eval record. `system` and `prompt` form the context; `response` is
/// what the model is trained to produce (or the reference answer for an eval). `mask_prompt`
/// is true for supervised formats (train only on the response) and false for `TextOnly`
/// (train on the whole text).
#[derive(Clone, Debug, PartialEq)]
pub struct Record {
    pub system: String,
    pub prompt: String,
    pub response: String,
    pub mask_prompt: bool,
}

fn s(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(|x| x.as_str()).map(|s| s.to_string())
}

/// Guess the format from a row's keys. Ordered most-specific first so an ambiguous row lands
/// on the richest interpretation.
pub fn detect(row: &Value) -> Option<Fmt> {
    let has = |k: &str| row.get(k).is_some();
    if row.get("messages").and_then(|m| m.as_array()).is_some() {
        return Some(Fmt::Chat);
    }
    if has("instruction") && (has("output") || has("response")) {
        return Some(Fmt::Instruction);
    }
    if (has("system") || has("user")) && (has("response") || has("assistant") || has("output")) {
        return Some(Fmt::SysUserResp);
    }
    if has("prompt") && (has("completion") || has("response")) {
        return Some(Fmt::PromptCompletion);
    }
    if has("text") {
        return Some(Fmt::TextOnly);
    }
    None
}

/// Normalise one row under a known format. Missing optional fields default to empty; a row
/// that cannot yield a non-empty response under the format is an error rather than a silently
/// dropped example (a malformed export should be visible, not shrink the set quietly).
pub fn normalise(row: &Value, fmt: Fmt) -> Result<Record, String> {
    let take = |ks: &[&str]| ks.iter().find_map(|k| s(row, k)).unwrap_or_default();
    let rec = match fmt {
        Fmt::Chat => {
            let msgs = row.get("messages").and_then(|m| m.as_array()).ok_or("no messages array")?;
            let mut system = String::new();
            let mut prompt = String::new();
            let mut response = String::new();
            for m in msgs {
                let role = s(m, "role").unwrap_or_default();
                let content = s(m, "content").unwrap_or_default();
                match role.as_str() {
                    "system" => system = content,
                    "assistant" => response = content, // last assistant turn wins
                    _ => {
                        if !prompt.is_empty() {
                            prompt.push('\n');
                        }
                        prompt.push_str(&content);
                    }
                }
            }
            Record { system, prompt, response, mask_prompt: true }
        }
        Fmt::SysUserResp => Record {
            system: take(&["system"]),
            prompt: take(&["user", "prompt", "input"]),
            response: take(&["response", "assistant", "output"]),
            mask_prompt: true,
        },
        Fmt::Instruction => {
            let instr = take(&["instruction"]);
            let input = take(&["input"]);
            let prompt = if input.is_empty() { instr } else { format!("{instr}\n\n{input}") };
            Record { system: String::new(), prompt, response: take(&["output", "response"]), mask_prompt: true }
        }
        Fmt::PromptCompletion => Record {
            system: String::new(),
            prompt: take(&["prompt"]),
            response: take(&["completion", "response"]),
            mask_prompt: true,
        },
        Fmt::TextOnly => Record {
            system: String::new(),
            prompt: String::new(),
            response: take(&["text"]),
            mask_prompt: false,
        },
    };
    if rec.response.is_empty() {
        return Err(format!("{}: row has no response content", fmt.label()));
    }
    Ok(rec)
}

/// Load a JSONL file, detect the format from the first non-empty row (or use `force`), and
/// normalise every row. Returns the format used and the records. A row that fails to
/// normalise is reported with its line number, not skipped.
pub fn load(path: &str, force: Option<Fmt>) -> Result<(Fmt, Vec<Record>), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut fmt = force;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).map_err(|e| format!("{path}:{}: {e}", i + 1))?;
        let f = match fmt {
            Some(f) => f,
            None => {
                let d = detect(&v).ok_or_else(|| format!("{path}:{}: unrecognised format", i + 1))?;
                fmt = Some(d);
                d
            }
        };
        out.push(normalise(&v, f).map_err(|e| format!("{path}:{}: {e}", i + 1))?);
    }
    let fmt = fmt.ok_or_else(|| format!("{path}: no rows"))?;
    Ok((fmt, out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn detects_and_normalises_each_shape() {
        // chat
        let chat = json!({"messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]});
        assert_eq!(detect(&chat), Some(Fmt::Chat));
        let r = normalise(&chat, Fmt::Chat).unwrap();
        assert_eq!((r.system.as_str(), r.prompt.as_str(), r.response.as_str()), ("be brief", "hi", "hello"));
        assert!(r.mask_prompt);

        // system/user/response
        let sur = json!({"system":"s","user":"u","response":"r"});
        assert_eq!(detect(&sur), Some(Fmt::SysUserResp));
        assert_eq!(normalise(&sur, Fmt::SysUserResp).unwrap().response, "r");

        // instruction/input/output (alpaca)
        let alp = json!({"instruction":"translate","input":"hola","output":"hello"});
        assert_eq!(detect(&alp), Some(Fmt::Instruction));
        let r = normalise(&alp, Fmt::Instruction).unwrap();
        assert_eq!(r.prompt, "translate\n\nhola");
        assert_eq!(r.response, "hello");

        // prompt/completion
        let pc = json!({"prompt":"2+2=","completion":"4"});
        assert_eq!(detect(&pc), Some(Fmt::PromptCompletion));
        assert_eq!(normalise(&pc, Fmt::PromptCompletion).unwrap().response, "4");

        // text only (no prompt masking)
        let t = json!({"text":"the quick brown fox"});
        assert_eq!(detect(&t), Some(Fmt::TextOnly));
        let r = normalise(&t, Fmt::TextOnly).unwrap();
        assert_eq!(r.response, "the quick brown fox");
        assert!(!r.mask_prompt);
    }

    #[test]
    fn an_empty_response_is_an_error_not_a_silent_drop() {
        let bad = json!({"system":"s","user":"u"}); // no response
        assert!(normalise(&bad, Fmt::SysUserResp).is_err());
    }

    #[test]
    fn the_redteam_shape_is_recognised_as_chat_or_sysuser() {
        // redteaming-man rows carry BOTH messages and system/user/response; messages wins.
        let row = json!({
            "messages":[{"role":"system","content":"educator"},{"role":"user","content":"explain X"},{"role":"assistant","content":"X is..."}],
            "system":"educator","user":"explain X","response":"X is..."
        });
        assert_eq!(detect(&row), Some(Fmt::Chat));
        assert_eq!(normalise(&row, Fmt::Chat).unwrap().response, "X is...");
        // forcing the flat interpretation also works
        assert_eq!(normalise(&row, Fmt::SysUserResp).unwrap().response, "X is...");
    }

    #[test]
    fn load_detects_from_first_row_and_reports_line_errors() {
        let dir = std::env::temp_dir();
        let p = dir.join("k3_ds.jsonl");
        std::fs::write(&p, "{\"prompt\":\"a\",\"completion\":\"b\"}\n{\"prompt\":\"c\",\"completion\":\"d\"}\n").unwrap();
        let (fmt, recs) = load(p.to_str().unwrap(), None).unwrap();
        assert_eq!(fmt, Fmt::PromptCompletion);
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[1].response, "d");
        let _ = std::fs::remove_file(&p);
    }
}
