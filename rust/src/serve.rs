// SPDX-License-Identifier: Apache-2.0
//
// An OpenAI-compatible HTTP server, so something other than a shell command can use this.
//
// WHY THIS SHAPE
//     The engine could generate correct tokens for months without anything consuming them.
//     What was missing was not capability but a socket. `/v1/chat/completions` is the one
//     interface that every local-model client already speaks -- Open WebUI, Continue,
//     Aider, Zed, the OpenAI SDKs, and claurst, whose `ollama` and `custom-openai`
//     providers both default to port 11434. Implementing it once reaches all of them;
//     writing a bespoke protocol would have reached none.
//
// ONE REQUEST AT A TIME, ON PURPOSE
//     `Cache`, the VRAM tier and the sweep position are single-instance and stateful, and
//     the machine is already streaming 3.45 GB of experts per token off one USB device.
//     Two concurrent generations would not be twice as fast; they would evict each other's
//     experts from a cache sized to hold one working set, and both would run slower than
//     either alone. So requests queue, which is also why a synchronous server is the right
//     tool and an async runtime would have bought nothing.
//
// WHERE THE TIME ACTUALLY GOES
//     Prefill, and it is not close. A `rustlm code` turn arrives with ~7600 prompt tokens
//     against a decode budget of a few hundred, so what a user waits for is almost entirely
//     the prompt.
//
//     Two things attack that, and only one of them was obvious. Prefix caching (`prefix.rs`)
//     removes the tokens a later turn resends -- turn 2 of a conversation pays only for what
//     it appended. That leaves turn 1, which no cache can help, and turn 1 turns out to be
//     77% COMPUTE rather than expert I/O: measured at three chunk widths on Qwen3.6-35B,
//     time barely moved while bytes read swung 35%. The lever there is batching the weight
//     DECODE across a prompt chunk (`ops::mmw_many`), not fetching fewer bytes.

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use tiny_http::{Header, Method, Request, Response, Server};

use crate::arch::{self, Family};
use crate::registry::Registry;
use crate::sample::SampleParams;
use crate::st::St;
use crate::v4run::{self, Engine, Params, Session};

pub struct Cfg {
    /// Bind address. Default port 11434 is Ollama's, which is what makes an unconfigured
    /// client find us.
    pub addr: String,
    /// A registered name or a path.
    pub model: String,
    /// Sizes the rope tables once, for the life of the process. They are lookup tables, so
    /// a longer one gives identical values at every position.
    pub max_ctx: usize,
    /// Per-request defaults; each request clones and overrides.
    pub params: Params,
    /// Budget for cached conversation state, in GB. This is the cache that makes an
    /// agent's turn 2 cheap; it competes with the expert cache for the same RAM.
    pub conv_gb: f64,
    /// Prompt-chunk width, or 0 to derive it (see `v4run::prefill_width`).
    ///
    /// Exposed because the derivation optimises BYTES READ, and prefill on this engine is
    /// mostly compute -- so the best width is a property of the machine's cache hierarchy
    /// as much as of the model, and the only way to know it for a new one is to try.
    pub prefill_width: usize,
}

impl Default for Cfg {
    fn default() -> Self {
        Cfg {
            addr: "127.0.0.1:11434".into(),
            model: String::new(),
            max_ctx: 8192,
            params: Params::default(),
            conv_gb: 2.0,
            prefill_width: 0,
        }
    }
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

static SEQ: AtomicU64 = AtomicU64::new(0);

/// Request ids are `<seconds>-<counter>`: unique within a process without an RNG, and
/// legible in a log next to the timestamp of the request that produced them.
fn new_id(prefix: &str) -> String {
    format!("{prefix}-{}{:04}", now(), SEQ.fetch_add(1, Ordering::Relaxed) % 10_000)
}

fn hdr(k: &str, v: &str) -> Header {
    // Both sides are ASCII literals from this file, so the parse cannot fail.
    Header::from_bytes(k.as_bytes(), v.as_bytes()).expect("static header")
}

fn json_response(code: u16, v: &Value) -> Response<std::io::Cursor<Vec<u8>>> {
    Response::from_string(v.to_string())
        .with_status_code(code)
        .with_header(hdr("Content-Type", "application/json"))
        .with_header(hdr("Access-Control-Allow-Origin", "*"))
}

fn api_error(code: u16, msg: &str, kind: &str) -> Response<std::io::Cursor<Vec<u8>>> {
    // OpenAI's error envelope, because clients parse `error.message` to show the user
    // something better than "request failed".
    json_response(code, &json!({"error": {"message": msg, "type": kind, "code": Value::Null}}))
}

// ---------------------------------------------------------------------------
// Prompt assembly
// ---------------------------------------------------------------------------

/// How a chat turns into one string, and what marks the end of the model's turn.
///
/// DeepSeek-V4-Flash ships **no** `chat_template` in `tokenizer_config.json` -- it is a
/// base model, and checking rather than assuming is what kept a Jinja engine out of this
/// build entirely. So the format below is ours, chosen to be legible and unambiguous, not
/// recovered from the checkpoint.
///
/// The consequence to be honest about: a base model was never trained on these markers. It
/// will follow the pattern because the pattern is obvious in text, not because it was
/// tuned to. The turn delimiters are therefore registered as stop sequences -- without
/// them the model happily writes the user's next message too, which is the single most
/// visible failure of serving a base model as a chat model.
pub struct Template {
    pub system: &'static str,
    pub user: &'static str,
    pub assistant: &'static str,
    /// Emitted last, to hand the turn to the model.
    pub open: &'static str,
}

pub const PLAIN: Template = Template {
    system: "### System\n",
    user: "### User\n",
    assistant: "### Assistant\n",
    open: "### Assistant\n",
};

impl Template {
    pub fn stops(&self) -> Vec<String> {
        vec![self.user.trim_end().to_string(), self.system.trim_end().to_string()]
    }
}

/// OpenAI allows `content` to be a bare string or an array of typed parts. A client that
/// sends the array form (any of them that support images) would otherwise be silently read
/// as an empty message -- so both are accepted and non-text parts are named, not dropped
/// in silence.
pub fn content_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .map(|p| match p.get("type").and_then(Value::as_str) {
                Some("text") => p.get("text").and_then(Value::as_str).unwrap_or("").to_string(),
                Some(other) => format!("[unsupported content part: {other}]"),
                None => String::new(),
            })
            .collect::<Vec<_>>()
            .join(""),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Render the tool list into the prompt.
///
/// There is no trained tool-call format here to target, so this states the contract in
/// the prompt and parses what comes back. The `<tool_call>` convention is borrowed from
/// Qwen/Hermes because it is the one most widely emitted in the wild.
fn render_tools(tools: &[Value]) -> String {
    let mut s = String::from(
        "You can call functions. To call one, reply with exactly:\n\
         <tool_call>{\"name\": \"FUNCTION_NAME\", \"arguments\": {...}}</tool_call>\n\
         Call a function only when it is needed. Available functions:\n",
    );
    for t in tools {
        let f = t.get("function").unwrap_or(t);
        let name = f.get("name").and_then(Value::as_str).unwrap_or("?");
        let desc = f.get("description").and_then(Value::as_str).unwrap_or("");
        let params = f.get("parameters").map(|p| p.to_string()).unwrap_or_else(|| "{}".into());
        s.push_str(&format!("- {name}: {desc}\n  parameters: {params}\n"));
    }
    s
}

fn build_prompt(t: &Template, messages: &[Value], tools: &[Value]) -> String {
    let mut out = String::new();
    let mut tools_rendered = tools.is_empty();
    for m in messages {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
        let text = content_text(m.get("content").unwrap_or(&Value::Null));
        match role {
            "system" => {
                out.push_str(t.system);
                out.push_str(&text);
                // Tools ride with the first system message so they stay above the
                // conversation rather than interrupting it.
                if !tools_rendered {
                    out.push('\n');
                    out.push_str(&render_tools(tools));
                    tools_rendered = true;
                }
                out.push_str("\n\n");
            }
            "assistant" => {
                out.push_str(t.assistant);
                out.push_str(&text);
                out.push_str("\n\n");
            }
            "tool" => {
                // A tool result is context, not a turn: it is the answer to a call the
                // assistant already made.
                out.push_str(t.user);
                let name = m.get("name").and_then(Value::as_str).unwrap_or("tool");
                out.push_str(&format!("Result of {name}: {text}\n\n"));
            }
            _ => {
                out.push_str(t.user);
                out.push_str(&text);
                out.push_str("\n\n");
            }
        }
    }
    if !tools_rendered {
        // No system message was sent, so the tool contract needs its own.
        out.insert_str(0, &format!("{}{}\n\n", t.system, render_tools(tools)));
    }
    out.push_str(t.open);
    out
}

/// Pull `<tool_call>{...}</tool_call>` blocks out of a completion.
///
/// Returns the calls and the text with those blocks removed, because a model that emits
/// both prose and a call should not have the prose thrown away.
fn parse_tool_calls(text: &str) -> (Vec<Value>, String) {
    let (mut calls, mut clean, mut rest) = (Vec::new(), String::new(), text);
    while let Some(a) = rest.find("<tool_call>") {
        let after = &rest[a + "<tool_call>".len()..];
        let Some(b) = after.find("</tool_call>") else { break };
        clean.push_str(&rest[..a]);
        if let Ok(v) = serde_json::from_str::<Value>(after[..b].trim()) {
            if let Some(name) = v.get("name").and_then(Value::as_str) {
                calls.push(json!({
                    "id": new_id("call"),
                    "type": "function",
                    "function": {
                        "name": name,
                        // OpenAI carries arguments as a JSON *string*, not an object.
                        "arguments": v.get("arguments").unwrap_or(&json!({})).to_string(),
                    }
                }));
            }
        }
        rest = &after[b + "</tool_call>".len()..];
    }
    clean.push_str(rest);
    (calls, clean)
}

// ---------------------------------------------------------------------------
// Stop sequences
// ---------------------------------------------------------------------------

/// Streams text while holding back just enough of the tail to recognise a stop sequence
/// before any of it is emitted.
///
/// The subtlety this exists for: a stop sequence arrives split across tokens. Checking
/// only the newest piece misses it; emitting eagerly and stopping afterwards means the
/// client has already been shown "### User". So everything within `max_stop - 1` bytes of
/// the end is withheld until it is known not to be the start of one.
struct StopScan {
    stops: Vec<String>,
    acc: String,
    sent: usize,
    hold: usize,
}

impl StopScan {
    fn new(stops: Vec<String>) -> StopScan {
        let hold = stops.iter().map(String::len).max().unwrap_or(1).saturating_sub(1);
        StopScan { stops, acc: String::new(), sent: 0, hold }
    }

    /// Append a piece. Returns (text safe to emit now, the stop sequence that just fired).
    fn push(&mut self, piece: &str) -> (String, Option<String>) {
        self.acc.push_str(piece);
        for s in &self.stops {
            // Search only from where a stop could newly complete, so an earlier literal
            // occurrence inside already-emitted text cannot re-fire.
            let from = self.sent.saturating_sub(s.len());
            if let Some(i) = self.acc[from..].find(s.as_str()).map(|i| i + from) {
                if i >= self.sent {
                    let out = self.acc[self.sent..i].to_string();
                    self.sent = i;
                    return (out, Some(s.clone()));
                }
            }
        }
        // Withhold the tail, backing up to a char boundary: slicing a String mid-codepoint
        // panics, and a multi-byte character straddling the hold-back window is ordinary.
        let mut safe = self.acc.len().saturating_sub(self.hold);
        if safe <= self.sent {
            return (String::new(), None);
        }
        while safe > self.sent && !self.acc.is_char_boundary(safe) {
            safe -= 1;
        }
        let out = self.acc[self.sent..safe].to_string();
        self.sent = safe;
        (out, None)
    }

    /// Whatever is still held back when generation ends for another reason.
    fn flush(&mut self) -> String {
        let out = self.acc[self.sent..].to_string();
        self.sent = self.acc.len();
        out
    }
}

// ---------------------------------------------------------------------------
// Request handling
// ---------------------------------------------------------------------------

struct ChatReq {
    messages: Vec<Value>,
    tools: Vec<Value>,
    stream: bool,
    max_tokens: usize,
    stops: Vec<String>,
    sample: Option<SampleParams>,
    seed: u64,
}

fn parse_chat(body: &Value, defaults: &Params, base_stops: &[String]) -> Result<ChatReq, String> {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or("`messages` is required and must be an array")?
        .clone();
    if messages.is_empty() {
        return Err("`messages` is empty".into());
    }
    let num = |k: &str| body.get(k).and_then(Value::as_f64);
    let temperature = num("temperature").unwrap_or(0.0) as f32;

    // A seed is always chosen and always reported, so any output can be reproduced. When
    // the client does not supply one the clock provides it -- that is the ONLY place
    // nondeterminism enters, and it is recorded in the response.
    let seed = body
        .get("seed")
        .and_then(Value::as_u64)
        .unwrap_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.subsec_nanos() as u64 ^ d.as_secs()).unwrap_or(0));

    let sample = if temperature > 0.0 {
        Some(SampleParams {
            temperature,
            top_p: num("top_p").unwrap_or(1.0) as f32,
            top_k: num("top_k").unwrap_or(0.0) as usize,
            min_p: num("min_p").unwrap_or(0.0) as f32,
            repeat_penalty: num("repetition_penalty")
                .or_else(|| num("repeat_penalty"))
                .unwrap_or(1.0) as f32,
            repeat_last_n: num("repeat_last_n").unwrap_or(64.0) as usize,
            seed,
        })
    } else {
        None
    };

    // The model's own stops, NOT the base-model markers unconditionally. A model with a
    // real chat template ends its turn on EOS; adding "### User" there would never match,
    // and would truncate any answer that legitimately contained that text.
    let mut stops: Vec<String> = base_stops.to_vec();
    match body.get("stop") {
        Some(Value::String(s)) => stops.push(s.clone()),
        Some(Value::Array(a)) => {
            stops.extend(a.iter().filter_map(Value::as_str).map(str::to_string))
        }
        _ => {}
    }
    stops.retain(|s| !s.is_empty());

    Ok(ChatReq {
        messages,
        tools: body.get("tools").and_then(Value::as_array).cloned().unwrap_or_default(),
        stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
        max_tokens: body
            .get("max_tokens")
            .or_else(|| body.get("max_completion_tokens"))
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(defaults.max_tokens),
        stops,
        sample,
        seed,
    })
}

fn sse_headers(w: &mut dyn Write) -> std::io::Result<()> {
    // Terminated by connection close rather than chunked encoding: valid HTTP/1.1, and it
    // keeps the framing of a long-lived stream down to something readable in a `curl -N`.
    w.write_all(
        b"HTTP/1.1 200 OK\r\n\
          Content-Type: text/event-stream\r\n\
          Cache-Control: no-cache\r\n\
          Connection: close\r\n\
          Access-Control-Allow-Origin: *\r\n\r\n",
    )?;
    w.flush()
}

fn sse_frame(w: &mut dyn Write, v: &Value) -> std::io::Result<()> {
    write!(w, "data: {v}\n\n")?;
    w.flush()
}

fn chunk(id: &str, model: &str, delta: Value, finish: Option<&str>) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": now(),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    })
}

/// The loaded model, whichever stack it came from.
///
/// Everything a request needs -- tokenizer, chat template, eos, generation -- is reached
/// through here, so `handle_chat` mentions no architecture at all. Only ONE place still
/// branches, and it is named and explained below.
enum Backend<'a> {
    /// Anything implementing `model::Model`. `qwen35moe` is here.
    Trait(Box<dyn crate::model::Model + 'a>),
    /// DeepSeek-V4, still on `v4run::generate_on`.
    ///
    /// Not yet behind the trait because its forward pass is entangled with speculative
    /// verification and batched prefill INSIDE that loop -- `feed(ids) -> logits` would
    /// have to be cut out of the middle of it. Doing that badly would silently disable
    /// DSpark and chunked prefill, so it is left as deliberate follow-up rather than a
    /// rushed extraction. The cost of waiting is one `match`, not fifteen call sites.
    V4 {
        eng: Box<Engine<'a>>,
        spec: &'a arch::Spec,
        params: Params,
        /// V4 gets the prefix cache too, WITHOUT extracting `feed` from `generate_on`.
        ///
        /// `generate_on` already resumes a partially-filled session: it prefills from
        /// `sess.pos` up to `sess.prompt_len` in batches. So restoring a snapshot and
        /// setting `pos` to what was restored makes the appended tokens take the
        /// batch-prefill path, which is exactly what was wanted -- speculation and chunked
        /// prefill are untouched because that loop was never cut open.
        prefix: Box<crate::prefix::PrefixCache<crate::prefix::V4State>>,
        fp: crate::prefix::Fingerprint,
    },
}

impl Backend<'_> {
    /// (expert GB, conversation GB, moves) when the backend has an adaptive split.
    fn split(&self) -> Option<(f64, f64, u64)> {
        match self {
            Backend::Trait(m) => m.split_report(),
            Backend::V4 { .. } => None,
        }
    }

    fn tok(&self) -> Option<&crate::tok::Tok> {
        match self {
            Backend::Trait(m) => m.tok(),
            Backend::V4 { eng, .. } => eng.tok.as_ref(),
        }
    }

    /// The model's own chat template, or `Plain` for a base model like DeepSeek-V4.
    fn template(&self) -> &crate::chat::Template {
        match self {
            Backend::Trait(m) => m.template(),
            Backend::V4 { .. } => &crate::chat::Template::Plain,
        }
    }

    /// Stop strings. A model with a real template ends its turn on EOS, so the invented
    /// turn markers must NOT be added -- they would never match, and worse, a template
    /// that legitimately emits "### User" inside content would truncate the answer.
    fn stops(&self) -> Vec<String> {
        if self.template().is_jinja() {
            Vec::new()
        } else {
            PLAIN.stops()
        }
    }

    fn generate(
        &mut self,
        ids: Vec<u32>,
        max_tokens: usize,
        sample: Option<SampleParams>,
        sink: &mut dyn FnMut(u32, &str) -> bool,
    ) -> Result<crate::model::Finish, String> {
        match self {
            Backend::Trait(m) => crate::model::generate(
                m.as_mut(),
                &ids,
                &crate::model::GenParams { max_tokens, sample },
                sink,
            ),
            Backend::V4 { eng, spec, params, prefix, fp } => {
                let mut p = params.clone();
                p.max_tokens = max_tokens;
                p.sample = sample.clone();
                // Sampling and speculation cannot coexist: the accept test is an equality
                // against the greedy argmax. `generate_on` REFUSES the combination, so
                // clearing it here is what lets a temperature request be served instead of
                // rejected. The reason is recorded in both places on purpose.
                if sample.is_some() {
                    p.spec_k = 0;
                    p.dspark = false;
                }
                let mut sess = Session::new(spec.n_layers, ids.clone());
                if let Some(i) = prefix.find(&ids, *fp) {
                    let snap = prefix.get(i);
                    let n = snap.ids.len();
                    eprintln!(
                        "  [prefix] {n}/{} tokens restored, {} to prefill",
                        ids.len(),
                        ids.len() - n
                    );
                    snap.state.restore(&mut sess, ids.clone());
                }
                let r = v4run::generate_on(eng, &mut sess, &p, None, None, sink);
                // Snapshot the WHOLE sequence, prompt plus what was generated: an agent's
                // next turn begins with exactly that, so it is the prefix a later request
                // can match. Only on success -- a failed run leaves the session mid-token.
                if r.is_ok() {
                    let st = crate::prefix::V4State::take(&sess);
                    prefix.insert(crate::prefix::Snapshot::new(st, &sess.ids, *fp));
                }
                r
                    // `generate_on` does not report WHY it stopped, so the caller's own
                    // stop-sequence scan is the authority; "length" is the honest default.
                    .map(|()| crate::model::Finish::Length)
            }
        }
    }
}

/// Everything the server holds for the life of the process.
struct Ctx<'a> {
    backend: Backend<'a>,
    model_name: String,
    /// Only `max_tokens` and `cache_gb` are read from here now; the rest is per-request.
    defaults: Params,
}

impl Ctx<'_> {
    fn handle_chat(&mut self, req: Request, body: Value) {
        let base_stops = self.backend.stops();
        let r = match parse_chat(&body, &self.defaults, &base_stops) {
            Ok(r) => r,
            Err(e) => {
                let _ = req.respond(api_error(400, &e, "invalid_request_error"));
                return;
            }
        };
        // The model's OWN template when it ships one, the base-model convention when it
        // does not. Rendering Qwen through `PLAIN` produced a prompt it was never trained
        // on and stop strings that never fire.
        let prompt = if self.backend.template().is_jinja() {
            let msgs = crate::chat::normalise(&r.messages);
            match self.backend.template().render(&msgs, &r.tools, true) {
                Ok(p) => p,
                Err(e) => {
                    let _ = req.respond(api_error(400, &e, "invalid_request_error"));
                    return;
                }
            }
        } else {
            build_prompt(&PLAIN, &r.messages, &r.tools)
        };
        let Some(tok) = self.backend.tok() else {
            let _ = req.respond(api_error(
                500,
                "this model has no tokenizer, so it can only be driven by raw token ids",
                "server_error",
            ));
            return;
        };
        let ids = match tok.encode(&prompt, false) {
            Ok(v) => v,
            Err(e) => {
                let _ = req.respond(api_error(400, &e.to_string(), "invalid_request_error"));
                return;
            }
        };
        let prompt_tokens = ids.len();

        let id = new_id("chatcmpl");
        eprintln!(
            "[{id}] {} msg, {prompt_tokens} prompt tokens, max {} , temp {:.2}, seed {}",
            r.messages.len(),
            r.max_tokens,
            r.sample.as_ref().map(|s| s.temperature).unwrap_or(0.0),
            r.seed
        );

        let mut scan = StopScan::new(r.stops.clone());
        let mut n_out = 0usize;
        let mut finish = "length";
        let mut full = String::new();

        if r.stream {
            let mut w = req.into_writer();
            if sse_headers(w.as_mut()).is_err() {
                return;
            }
            let _ = sse_frame(
                w.as_mut(),
                &chunk(&id, &self.model_name, json!({"role": "assistant", "content": ""}), None),
            );
            let mut dead = false;
            {
                let (w, scan, full, n_out, finish, dead) =
                    (&mut w, &mut scan, &mut full, &mut n_out, &mut finish, &mut dead);
                let model_name = self.model_name.clone();
                let id2 = id.clone();
                let mut sink = |_tid: u32, piece: &str| -> bool {
                    *n_out += 1;
                    let (out, hit) = scan.push(piece);
                    if !out.is_empty() {
                        full.push_str(&out);
                        if sse_frame(
                            w.as_mut(),
                            &chunk(&id2, &model_name, json!({"content": out}), None),
                        )
                        .is_err()
                        {
                            // The client hung up. At seconds per token, continuing would
                            // burn minutes of expert streaming for nobody.
                            *dead = true;
                            return false;
                        }
                    }
                    if hit.is_some() {
                        *finish = "stop";
                        return false;
                    }
                    true
                };
                let res = self.backend.generate(ids, r.max_tokens, r.sample.clone(), &mut sink);
                if let Err(e) = res {
                    eprintln!("[{id2}] generation failed: {e}");
                    let _ = sse_frame(
                        w.as_mut(),
                        &chunk(&id2, &model_name, json!({"content": ""}), Some("error")),
                    );
                }
            }
            if dead {
                eprintln!("[{id}] client disconnected after {n_out} tokens");
                return;
            }
            if finish == "length" {
                // Reaching the end of generation without a stop sequence: whatever the
                // scanner was still holding back is real output and must be emitted.
                let tail = scan.flush();
                if !tail.is_empty() {
                    full.push_str(&tail);
                    let _ = sse_frame(
                        w.as_mut(),
                        &chunk(&id, &self.model_name, json!({"content": tail}), None),
                    );
                }
            }
            let (calls, _clean) = parse_tool_calls(&full);
            if !calls.is_empty() {
                finish = "tool_calls";
                // Streamed as one complete delta rather than incrementally. OpenAI streams
                // tool calls in fragments; every client that consumes them also accepts a
                // whole one, and reassembling fragments is complexity with no reader.
                let _ = sse_frame(
                    w.as_mut(),
                    &chunk(&id, &self.model_name, json!({"tool_calls": calls}), None),
                );
            }
            let _ = sse_frame(w.as_mut(), &chunk(&id, &self.model_name, json!({}), Some(finish)));
            let _ = w.write_all(b"data: [DONE]\n\n");
            let _ = w.flush();
            eprintln!("[{id}] {n_out} tokens, finish {finish}");
            return;
        }

        {
            let (scan, full, n_out, finish) = (&mut scan, &mut full, &mut n_out, &mut finish);
            let mut sink = |_tid: u32, piece: &str| -> bool {
                *n_out += 1;
                let (out, hit) = scan.push(piece);
                full.push_str(&out);
                if hit.is_some() {
                    *finish = "stop";
                    return false;
                }
                true
            };
            if let Err(e) = self.backend.generate(ids, r.max_tokens, r.sample.clone(), &mut sink) {
                eprintln!("[{id}] generation failed: {e}");
                let _ = req.respond(api_error(500, &e, "server_error"));
                return;
            }
        }
        if finish == "length" {
            full.push_str(&scan.flush());
        }
        let (calls, clean) = parse_tool_calls(&full);
        let mut msg = json!({"role": "assistant", "content": clean});
        if !calls.is_empty() {
            finish = "tool_calls";
            msg["tool_calls"] = Value::Array(calls);
        }
        eprintln!("[{id}] {n_out} tokens, finish {finish}");
        let _ = req.respond(json_response(
            200,
            &json!({
                "id": id,
                "object": "chat.completion",
                "created": now(),
                "model": self.model_name,
                // Not an OpenAI field. It is here because reproducing an answer requires
                // it, and a seed the caller cannot see is a seed that does not exist.
                "seed": r.seed,
                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": n_out,
                    "total_tokens": prompt_tokens + n_out,
                },
            }),
        ));
    }
}

fn models_json(reg: &Registry, served: &str) -> Value {
    let mut out = Vec::new();
    for e in &reg.entries {
        // Only runnable models are advertised. A client that offers a model it cannot use
        // turns a clear registry blocker into a mid-conversation failure.
        if !e.runnable() && e.name != served {
            continue;
        }
        out.push(json!({
            "id": e.name,
            "object": "model",
            "created": 0,
            "owned_by": "rustlm",
        }));
    }
    if out.is_empty() {
        out.push(json!({"id": served, "object": "model", "created": 0, "owned_by": "rustlm"}));
    }
    json!({"object": "list", "data": out})
}

fn tags_json(reg: &Registry, served: &str) -> Value {
    // Ollama's native listing. claurst's ollama provider prefers /api/tags over
    // /v1/models, and its live list REPLACES the embedded catalogue for that provider --
    // so this endpoint is what makes the model appear in its picker at all.
    let mut out = Vec::new();
    for e in &reg.entries {
        if !e.runnable() && e.name != served {
            continue;
        }
        out.push(json!({
            "name": format!("{}:latest", e.name),
            "model": format!("{}:latest", e.name),
            "modified_at": "1970-01-01T00:00:00Z",
            "size": e.bytes,
            "digest": "",
            "details": {"family": e.arch, "format": e.format, "parameter_size": "", "quantization_level": ""},
        }));
    }
    json!({"models": out})
}

/// Say plainly that the port is taken, and by whom if it is guessable.
fn bind_advice(addr: &str, e: &str) -> String {
    let ollama = addr.ends_with(":11434");
    format!(
        "cannot bind {addr}: {e}{}",
        if ollama {
            "\n  Port 11434 is Ollama's, which is the point -- an unconfigured client finds \
             us there. If Ollama is running, stop it (`systemctl --user stop ollama`) or \
             serve elsewhere with --addr 127.0.0.1:11435 and point the client at it."
        } else {
            ""
        }
    )
}

pub fn run(cfg: &Cfg) -> Result<(), String> {
    let reg = Registry::open();
    let path: PathBuf = reg
        .resolve(&cfg.model)
        .ok_or_else(|| format!("no model {:?} -- `rustlm list` shows what is registered", cfg.model))?;

    // Which stack loads this checkpoint is decided by what is IN it, not by a flag.
    // GGUF carries its architecture in metadata; safetensors carries it in config.json.
    let st = St::open(&path).map_err(|e| e.to_string())?;
    let gguf_arch = st
        .meta
        .as_ref()
        .and_then(|m| m.get("general.architecture"))
        .and_then(crate::gguf::Value::as_str)
        .unwrap_or("")
        .to_string();

    // Bind BEFORE loading, so a port clash costs a second rather than the minutes it takes
    // to bring gigabytes of trunk into memory.
    let server = Server::http(&cfg.addr).map_err(|e| bind_advice(&cfg.addr, &e.to_string()))?;

    let model_name = reg
        .entries
        .iter()
        .find(|e| e.path == path)
        .map(|e| e.name.clone())
        .unwrap_or_else(|| cfg.model.clone());

    println!("loading {} ...", path.display());
    let t0 = std::time::Instant::now();

    // `spec` must outlive the borrow inside Backend::V4, so it is declared here and only
    // filled on that path.
    let spec_slot;
    // Both qwen35 families take the same backend: one `Cfg`, one trunk, one block set.
    // The dense one differs only in that its feed-forward streams as a single unit per
    // layer rather than as routed experts -- decided inside `Qwen35::load`, not here.
    let backend = if gguf_arch == "qwen35moe" || gguf_arch == "qwen35" {
        let mut m =
            crate::model::Qwen35::load(&st, cfg.params.cache_gb, cfg.conv_gb, cfg.max_ctx)?;
        m.set_width(cfg.prefill_width);
        let c = m.cfg();
        println!(
            "{gguf_arch} | {} blocks, hidden {}, vocab {} | {} | {:.2} GB resident",
            c.n_layers,
            c.hidden,
            c.vocab,
            if c.is_dense() {
                format!("dense ffn {}", c.dense_inter)
            } else {
                format!("{} experts top-{}", c.n_experts, c.topk)
            },
            m.bytes() as f64 / 1e9
        );
        println!("prefill   | chunks of {} tokens", m.width());
        Backend::Trait(Box::new(m))
    } else {
        let spec = arch::spec_file(&path.join("config.json"))?;
        if spec.family != Family::V4 {
            return Err(format!(
                "{:?} is {} ({gguf_arch:?} in gguf), which has no forward pass here. \
                 `rustlm inspect` lists what is missing.",
                cfg.model,
                spec.family.as_str()
            ));
        }
        println!("{}", spec.summary(&path.display().to_string()));
        let tok = v4run::open_tokenizer(&path, None)?;
        if tok.is_none() {
            return Err(format!(
                "{} has no tokenizer.json, so text requests cannot be encoded",
                path.display()
            ));
        }
        spec_slot = spec;
        let eng = Engine::load(&st, &spec_slot, &path, tok, &cfg.params, cfg.max_ctx, false)?;
        Backend::V4 {
            eng: Box::new(eng),
            spec: &spec_slot,
            params: cfg.params.clone(),
            prefix: Box::new(crate::prefix::PrefixCache::new((cfg.conv_gb * 1e9) as usize)),
            fp: crate::prefix::Fingerprint::of_spec(&spec_slot),
        }
    };
    println!("ready in {:.1} s", t0.elapsed().as_secs_f64());

    println!(
        "rustlm serving {model_name:?} on http://{}\n  \
         POST /v1/chat/completions   GET /v1/models   GET /api/tags   GET /health\n  \
         one request at a time; the cache and the sweep position are single-instance",
        cfg.addr
    );

    let mut ctx = Ctx { backend, model_name, defaults: cfg.params.clone() };

    for mut req in server.incoming_requests() {
        let url = req.url().split('?').next().unwrap_or("").to_string();
        let method = req.method().clone();
        match (&method, url.as_str()) {
            (Method::Options, _) => {
                let _ = req.respond(
                    Response::empty(204)
                        .with_header(hdr("Access-Control-Allow-Origin", "*"))
                        .with_header(hdr("Access-Control-Allow-Methods", "GET, POST, OPTIONS"))
                        .with_header(hdr("Access-Control-Allow-Headers", "*")),
                );
            }
            (Method::Get, "/health" | "/") => {
                let mut h = json!({"status": "ok", "model": ctx.model_name});
                if let Some((e, c, moves)) = ctx.backend.split() {
                    // Whoever is wondering why a request was slow should not have to guess
                    // how memory is currently divided.
                    h["cache_gb"] = json!({"experts": e, "conversations": c, "moves": moves});
                }
                let _ = req.respond(json_response(200, &h));
            }
            (Method::Get, "/v1/models" | "/models") => {
                let v = models_json(&Registry::open(), &ctx.model_name);
                let _ = req.respond(json_response(200, &v));
            }
            (Method::Get, "/api/tags") => {
                let v = tags_json(&Registry::open(), &ctx.model_name);
                let _ = req.respond(json_response(200, &v));
            }
            (Method::Post, "/v1/chat/completions" | "/chat/completions") => {
                let mut s = String::new();
                if req.as_reader().read_to_string(&mut s).is_err() {
                    let _ = req.respond(api_error(400, "body is not UTF-8", "invalid_request_error"));
                    continue;
                }
                match serde_json::from_str::<Value>(&s) {
                    Ok(v) => ctx.handle_chat(req, v),
                    Err(e) => {
                        let _ = req.respond(api_error(
                            400,
                            &format!("body is not JSON: {e}"),
                            "invalid_request_error",
                        ));
                    }
                }
            }
            _ => {
                let _ = req.respond(api_error(
                    404,
                    &format!("no route for {method} {url}"),
                    "invalid_request_error",
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stop sequence that arrives split across tokens must still be caught, and none of
    /// it may reach the client. This is the failure the hold-back exists for: check only
    /// the newest piece and "### User" is missed; emit eagerly and the user sees it.
    #[test]
    fn a_stop_sequence_split_across_tokens_is_caught_before_it_is_emitted() {
        let mut s = StopScan::new(vec!["### User".to_string()]);
        let mut seen = String::new();
        let mut fired = None;
        for piece in ["Hello", " there", ".\n\n#", "##", " Us", "er", "\nnext"] {
            let (out, hit) = s.push(piece);
            seen.push_str(&out);
            if hit.is_some() {
                fired = hit;
                break;
            }
        }
        assert_eq!(fired.as_deref(), Some("### User"), "the split stop must fire");
        assert!(!seen.contains('#'), "no part of the stop may be emitted: {seen:?}");
        assert_eq!(seen, "Hello there.\n\n");
    }

    /// Text held back for stop-detection is real output when generation ends for another
    /// reason. Dropping it silently truncates every answer by a few characters.
    #[test]
    fn held_back_text_is_flushed_when_no_stop_fires() {
        let mut s = StopScan::new(vec!["### User".to_string()]);
        let (a, hit) = s.push("The answer is 42");
        assert!(hit.is_none());
        let b = s.flush();
        assert_eq!(format!("{a}{b}"), "The answer is 42", "nothing may be lost");
    }

    /// Multi-byte characters straddle the hold-back window routinely, and slicing a String
    /// off a char boundary panics.
    #[test]
    fn the_hold_back_window_never_splits_a_character() {
        let mut s = StopScan::new(vec!["STOPSEQUENCE".to_string()]);
        let mut seen = String::new();
        for piece in ["héllo ", "wörld ", "日本語のテキスト", " ✓"] {
            let (out, _) = s.push(piece);
            seen.push_str(&out);
        }
        seen.push_str(&s.flush());
        assert_eq!(seen, "héllo wörld 日本語のテキスト ✓");
    }

    /// OpenAI's `content` is a string OR an array of parts. Reading only the string form
    /// turns a vision-capable client's message into an empty one, silently.
    #[test]
    fn both_content_encodings_are_read() {
        assert_eq!(content_text(&json!("plain")), "plain");
        assert_eq!(
            content_text(&json!([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])),
            "ab"
        );
        // An unsupported part is named rather than dropped in silence.
        let mixed = content_text(&json!([{"type": "text", "text": "look: "},
                                         {"type": "image_url", "image_url": {"url": "x"}}]));
        assert!(mixed.starts_with("look: ") && mixed.contains("image_url"), "{mixed}");
    }

    #[test]
    fn a_tool_call_is_parsed_out_and_the_prose_around_it_survives() {
        let (calls, clean) = parse_tool_calls(
            "Let me check. <tool_call>{\"name\": \"get_weather\", \
             \"arguments\": {\"city\": \"Paris\"}}</tool_call> One moment.",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        // OpenAI carries arguments as a JSON *string*; an object here breaks every SDK.
        let args = calls[0]["function"]["arguments"].as_str().expect("arguments must be a string");
        assert_eq!(serde_json::from_str::<Value>(args).unwrap()["city"], "Paris");
        assert_eq!(clean.trim(), "Let me check.  One moment.".trim());
    }

    #[test]
    fn text_with_no_tool_call_is_returned_unchanged() {
        let (calls, clean) = parse_tool_calls("just an answer");
        assert!(calls.is_empty());
        assert_eq!(clean, "just an answer");
    }

    /// A malformed call must not swallow the rest of the reply.
    #[test]
    fn a_broken_tool_call_does_not_eat_the_response() {
        let (calls, clean) = parse_tool_calls("a <tool_call>{not json}</tool_call> b");
        assert!(calls.is_empty(), "unparseable arguments yield no call");
        assert_eq!(clean, "a  b");
    }

    /// The turn delimiters must be stop sequences, or a base model writes the user's next
    /// message as well -- the single most visible failure of serving a base model as chat.
    #[test]
    fn the_prompt_ends_on_the_assistant_marker_and_the_user_marker_stops_it() {
        let p = build_prompt(&PLAIN, &[json!({"role": "user", "content": "hi"})], &[]);
        assert!(p.ends_with(PLAIN.open), "the model must be handed an open turn: {p:?}");
        assert!(p.contains("### User\nhi"));
        assert!(PLAIN.stops().contains(&"### User".to_string()));
    }

    #[test]
    fn tools_reach_the_prompt_even_with_no_system_message() {
        let tools = vec![json!({"type": "function", "function": {
            "name": "get_weather", "description": "look up weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}})];
        let p = build_prompt(&PLAIN, &[json!({"role": "user", "content": "weather?"})], &tools);
        assert!(p.contains("get_weather"), "the tool must be described: {p}");
        assert!(p.contains("<tool_call>"), "the call format must be stated");
        assert!(p.find("get_weather") < p.find("weather?"), "tools go above the conversation");
    }

    /// Temperature 0 must produce no sampler at all, so the server's default path is the
    /// engine's historical greedy path exactly.
    #[test]
    fn the_default_request_is_greedy_and_carries_a_reported_seed() {
        let d = Params::default();
        let r = parse_chat(&json!({"messages": [{"role": "user", "content": "hi"}]}), &d, &PLAIN.stops()).unwrap();
        assert!(r.sample.is_none(), "no temperature means no sampling");
        assert!(!r.stream);
        let r2 = parse_chat(
            &json!({"messages": [{"role": "user", "content": "hi"}],
                    "temperature": 0.8, "top_p": 0.9, "seed": 7}),
            &d,
            &PLAIN.stops(),
        )
        .unwrap();
        let s = r2.sample.expect("temperature must produce a sampler");
        assert_eq!(s.seed, 7, "a supplied seed must be honoured");
        assert_eq!(r2.seed, 7, "and reported back");
        assert!((s.top_p - 0.9).abs() < 1e-6);
    }

    #[test]
    fn client_stop_strings_are_added_to_the_template_ones() {
        let d = Params::default();
        let r = parse_chat(
            &json!({"messages": [{"role": "user", "content": "hi"}], "stop": ["END", ""]}),
            &d,
            &PLAIN.stops(),
        )
        .unwrap();
        assert!(r.stops.contains(&"END".to_string()));
        assert!(r.stops.contains(&"### User".to_string()), "template stops are kept");
        assert!(!r.stops.iter().any(String::is_empty), "an empty stop would fire immediately");
    }

    #[test]
    fn a_request_without_messages_is_refused_rather_than_guessed_at() {
        let d = Params::default();
        assert!(parse_chat(&json!({}), &d, &PLAIN.stops()).is_err());
        assert!(parse_chat(&json!({"messages": []}), &d, &PLAIN.stops()).is_err());
    }
}
