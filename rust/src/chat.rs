// SPDX-License-Identifier: Apache-2.0
//
// Turning a list of chat messages into the exact string a given model was trained on.
//
// WHY A TEMPLATE ENGINE, HAVING ARGUED AGAINST ONE
//     `serve.rs` renders `### User` / `### Assistant` because DeepSeek-V4-Flash is a BASE
//     model: its `tokenizer_config.json` carries no `chat_template` at all, so there is
//     nothing to render and a legible convention is as good as any.
//
//     Qwen3.6 is not that. It ships a real Jinja `chat_template` in its GGUF metadata,
//     using `<|im_start|>` / `<|im_end|>` markers, and `<|im_end|>` IS its EOS token.
//     Serving it through the base-model convention produces a prompt it was never trained
//     on and stop strings that never fire -- fluent, structureless output with no error.
//     So the engine either renders the model's own template or it is lying about which
//     model it is running.
//
// WHAT IS DELIBERATELY NOT DONE
//     No attempt to reimplement Jinja, and no attempt to "clean up" a template before
//     rendering. The template is a property of the checkpoint; the only correct action is
//     to run it as written and let a failure be a failure.

use minijinja::{context, Environment};
use serde_json::{json, Value};

/// How one model wants its conversation rendered.
pub enum Template {
    /// The model's own Jinja template, from `tokenizer.chat_template`.
    Jinja(Box<String>),
    /// A base model with no template. See `serve::PLAIN` for the convention.
    Plain,
}

impl Template {
    /// Read the template out of GGUF metadata, if the checkpoint has one.
    pub fn from_gguf(m: &crate::gguf::Meta) -> Template {
        match m.get("tokenizer.chat_template").and_then(crate::gguf::Value::as_str) {
            Some(s) if !s.trim().is_empty() => Template::Jinja(Box::new(s.to_string())),
            _ => Template::Plain,
        }
    }

    pub fn is_jinja(&self) -> bool {
        matches!(self, Template::Jinja(_))
    }

    /// Render `messages` (OpenAI shape) into the model's prompt string.
    ///
    /// `add_generation_prompt` appends the assistant-turn opener, which is what makes the
    /// model continue rather than predict another user turn. Every template in this family
    /// keys off it, and omitting it is the single most common way to get a model that
    /// answers its own questions.
    pub fn render(
        &self,
        messages: &[Value],
        tools: &[Value],
        add_generation_prompt: bool,
    ) -> Result<String, String> {
        let src = match self {
            Template::Plain => return Err("this model has no chat template".into()),
            Template::Jinja(s) => s.as_str(),
        };
        let mut env = Environment::new();
        // Templates in the wild call `raise_exception` to reject malformed conversations
        // (Qwen's rejects images in a system message, and a message list with no user
        // turn). Without it those templates fail to compile at all, so the function has to
        // exist -- and it must actually raise, not return, or an invalid conversation
        // would render to something plausible.
        fn raise_exception(msg: String) -> Result<minijinja::Value, minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
        }
        env.add_function("raise_exception", raise_exception);
        // Chat templates are written for PYTHON Jinja2 and freely use Python string
        // methods -- Qwen3.6's calls `.startswith()`. Without this shim the template does
        // not render at all.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template("chat", src).map_err(|e| format!("chat template does not parse: {e}"))?;
        let t = env.get_template("chat").map_err(|e| e.to_string())?;

        // `tools` must be undefined rather than an empty list when there are none: every
        // template in this family branches on `if tools`, and an empty array is falsy in
        // Jinja but a *present* variable in some renderers. Passing none keeps the branch
        // unambiguous.
        let out = if tools.is_empty() {
            t.render(context! {
                messages => messages,
                add_generation_prompt => add_generation_prompt,
            })
        } else {
            t.render(context! {
                messages => messages,
                tools => tools,
                add_generation_prompt => add_generation_prompt,
            })
        };
        out.map_err(|e| {
            // minijinja nests the real cause; a bare "invalid operation" is useless when
            // the template raised its own message.
            let mut s = e.to_string();
            let mut src: &dyn std::error::Error = &e;
            while let Some(next) = src.source() {
                s.push_str(&format!(": {next}"));
                src = next;
            }
            format!("chat template failed: {s}")
        })
    }
}

/// Normalise an OpenAI message list into what these templates expect.
///
/// The templates index `message.role` and `message.content` directly, and OpenAI allows
/// `content` to be an array of typed parts. Left as an array, a template either renders
/// its debug form or raises. Flattening to text here keeps the template unmodified.
pub fn normalise(messages: &[Value]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("user");
            let text = crate::serve::content_text(m.get("content").unwrap_or(&Value::Null));
            let mut out = json!({ "role": role, "content": text });
            // Carry the fields tool-calling templates read. Dropping them would silently
            // turn an assistant's tool call into an empty turn.
            for k in ["tool_calls", "name", "tool_call_id", "reasoning_content"] {
                if let Some(v) = m.get(k) {
                    out[k] = v.clone();
                }
            }
            out
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape of every template in this family, small enough to reason about.
    const QWEN_ISH: &str = concat!(
        "{%- for message in messages %}",
        "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}",
        "{%- endfor %}",
        "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
    );

    fn meta_with(t: Option<&str>) -> crate::gguf::Meta {
        let mut m = crate::gguf::Meta::new();
        if let Some(t) = t {
            m.insert("tokenizer.chat_template".into(), crate::gguf::Value::Str(t.into()));
        }
        m
    }

    /// A model with a template must use it; a base model with none must fall back rather
    /// than inventing one.
    #[test]
    fn a_template_is_used_when_present_and_absent_means_plain() {
        assert!(Template::from_gguf(&meta_with(Some(QWEN_ISH))).is_jinja());
        assert!(!Template::from_gguf(&meta_with(None)).is_jinja());
        // A present-but-blank template is not a template.
        assert!(!Template::from_gguf(&meta_with(Some("   "))).is_jinja());
    }

    #[test]
    fn messages_render_with_the_models_own_markers() {
        let t = Template::from_gguf(&meta_with(Some(QWEN_ISH)));
        let msgs = normalise(&[
            json!({"role": "system", "content": "be brief"}),
            json!({"role": "user", "content": "hi"}),
        ]);
        let s = t.render(&msgs, &[], true).unwrap();
        assert_eq!(
            s,
            "<|im_start|>system\nbe brief<|im_end|>\n\
             <|im_start|>user\nhi<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    /// Without the generation prompt the model predicts another USER turn instead of
    /// answering. This is the most common way to get a model that talks to itself.
    #[test]
    fn the_generation_prompt_is_what_hands_the_turn_over() {
        let t = Template::from_gguf(&meta_with(Some(QWEN_ISH)));
        let msgs = normalise(&[json!({"role": "user", "content": "hi"})]);
        let with = t.render(&msgs, &[], true).unwrap();
        let without = t.render(&msgs, &[], false).unwrap();
        assert!(with.ends_with("<|im_start|>assistant\n"));
        assert!(!without.ends_with("<|im_start|>assistant\n"));
        assert!(with.starts_with(&without), "it must be an append, not a re-render");
    }

    /// Appending a turn must extend the previous render, or prefix caching has nothing to
    /// match on. Checked here because it is a property of the TEMPLATE, not of the cache.
    #[test]
    fn appending_a_turn_extends_the_previous_render() {
        let t = Template::from_gguf(&meta_with(Some(QWEN_ISH)));
        let one = normalise(&[json!({"role": "user", "content": "first"})]);
        let two = normalise(&[
            json!({"role": "user", "content": "first"}),
            json!({"role": "assistant", "content": "reply"}),
            json!({"role": "user", "content": "second"}),
        ]);
        // Compare WITHOUT the generation prompt: the opener is re-emitted at the end of
        // every render, so it is the one part that is not a prefix.
        let a = t.render(&one, &[], false).unwrap();
        let b = t.render(&two, &[], false).unwrap();
        assert!(b.starts_with(&a), "turn 2 must extend turn 1:\n{a:?}\n{b:?}");
    }

    /// OpenAI's array-form content must reach the template as text.
    #[test]
    fn array_content_is_flattened_before_rendering() {
        let t = Template::from_gguf(&meta_with(Some(QWEN_ISH)));
        let msgs = normalise(&[json!({"role": "user",
            "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]})]);
        assert!(t.render(&msgs, &[], false).unwrap().contains("\nab<|im_end|>"));
    }

    /// A template that raises must produce an error carrying its message, not render
    /// something plausible.
    #[test]
    fn a_template_that_raises_reports_why() {
        let src = "{{- raise_exception('No user query found in messages.') }}";
        let t = Template::from_gguf(&meta_with(Some(src)));
        let e = t.render(&[], &[], false).unwrap_err();
        assert!(e.contains("No user query found"), "the reason must survive: {e}");
    }

    #[test]
    fn a_malformed_template_fails_to_parse_rather_than_rendering_partially() {
        let t = Template::from_gguf(&meta_with(Some("{%- for x in %}")));
        assert!(t.render(&[], &[], false).unwrap_err().contains("does not parse"));
    }

    /// Fields a tool-calling template reads must survive normalisation.
    #[test]
    fn tool_call_fields_are_carried_through() {
        let msgs = normalise(&[json!({
            "role": "assistant", "content": null,
            "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}]
        })]);
        assert!(msgs[0].get("tool_calls").is_some(), "an assistant's call must not vanish");
        assert_eq!(msgs[0]["content"], "");
    }
}
