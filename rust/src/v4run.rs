// SPDX-License-Identifier: Apache-2.0
//
// DeepSeek-V4-Flash generation with a RESIDENT trunk.
//
// The trunk is loaded once and kept in its stored FP8/bf16 form. Dequantising it to f32
// would turn 8.29 GB into roughly 33 GB and put it out of reach of the 16 GB machine this
// arrangement exists to serve; re-reading it per token, which is what the first version of
// this loop did, moves all 8.29 GB off the device on every step.

use std::path::Path;

use crate::arch::Spec;
use crate::cache::{v4_expert_names, Cache};
use crate::ops::{HcLayer, HyperConnResidual, W};
use crate::st::St;
use crate::v4::*;

pub(crate) const HC: usize = 4;
pub(crate) const BLOCK: usize = 128;

pub(crate) struct Fp8 {
    w: Vec<u8>,
    s: Vec<u8>,
}

impl Fp8 {
    pub(crate) fn w(&self) -> W<'_> {
        W::F8Block { w: &self.w, scale: &self.s, block: BLOCK }
    }
    pub(crate) fn bytes(&self) -> usize {
        self.w.len() + self.s.len()
    }
}

struct Comp {
    ape: Vec<f32>,
    wkv: Vec<f32>,
    wgate: Vec<f32>,
    norm: Vec<f32>,
}

struct Idx {
    c: Comp,
    wq_b: Fp8,
    weights_proj: Vec<f32>,
}

pub(crate) struct Layer {
    pub(crate) ratio: usize,
    pub(crate) wq_a: Fp8,
    pub(crate) wq_b: Fp8,
    pub(crate) wkv: Fp8,
    pub(crate) wo_a: Fp8,
    pub(crate) wo_b: Fp8,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) kv_norm: Vec<f32>,
    pub(crate) sink: Vec<f32>,
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) ffn_norm: Vec<f32>,
    comp: Option<Comp>,
    idx: Option<Idx>,
    pub(crate) hc_attn: [Vec<f32>; 3],
    pub(crate) hc_ffn: [Vec<f32>; 3],
    pub(crate) gate: Vec<f32>,
    pub(crate) bias: Option<Vec<f32>>,
    pub(crate) tid2eid: Option<Vec<i32>>,
    pub(crate) sh1: Fp8,
    pub(crate) sh3: Fp8,
    pub(crate) sh2: Fp8,
}

impl Layer {
    /// Load one block's non-expert weights from `<prefix>.*`. Shared by the 43 decoder
    /// layers (`layers.N`) and the three DSpark stages (`mtp.N`), which are the same
    /// block type -- the stages differ only by `ratio == 0`, so no Compressor or Indexer,
    /// and by the extra glue tensors DSpark loads alongside.
    pub(crate) fn load(
        st: &St,
        s: &Spec,
        prefix: &str,
        ratio: usize,
        hash_layer: Option<usize>,
    ) -> Result<Layer, String> {
        let p = |n: &str| format!("{prefix}.{n}");
        let is_hash = matches!(hash_layer, Some(l) if l < s.n_hash_layers);
        let (comp, idx) = if ratio > 0 {
            let c = comp_of(st, &p("attn.compressor"))?;
            let i = if ratio == 4 {
                Some(Idx {
                    c: comp_of(st, &p("attn.indexer.compressor"))?,
                    wq_b: fp8(st, &p("attn.indexer.wq_b"))?,
                    weights_proj: f32s(st, &p("attn.indexer.weights_proj.weight"))?,
                })
            } else {
                None
            };
            (Some(c), i)
        } else {
            (None, None)
        };
        Ok(Layer {
            ratio,
            wq_a: fp8(st, &p("attn.wq_a"))?,
            wq_b: fp8(st, &p("attn.wq_b"))?,
            wkv: fp8(st, &p("attn.wkv"))?,
            wo_a: fp8(st, &p("attn.wo_a"))?,
            wo_b: fp8(st, &p("attn.wo_b"))?,
            q_norm: f32s(st, &p("attn.q_norm.weight"))?,
            kv_norm: f32s(st, &p("attn.kv_norm.weight"))?,
            sink: f32s(st, &p("attn.attn_sink"))?,
            attn_norm: f32s(st, &p("attn_norm.weight"))?,
            ffn_norm: f32s(st, &p("ffn_norm.weight"))?,
            comp,
            idx,
            hc_attn: [
                f32s(st, &p("hc_attn_fn"))?,
                f32s(st, &p("hc_attn_base"))?,
                f32s(st, &p("hc_attn_scale"))?,
            ],
            hc_ffn: [
                f32s(st, &p("hc_ffn_fn"))?,
                f32s(st, &p("hc_ffn_base"))?,
                f32s(st, &p("hc_ffn_scale"))?,
            ],
            gate: f32s(st, &p("ffn.gate.weight"))?,
            // DeepSeek-V4 stores this as `ffn.gate.bias`; `e_score_correction_bias` is
            // the DeepSeek-V3 / HF spelling. Asking only for the V3 name and swallowing
            // the miss with `.ok()` silently routed every V4 token WITHOUT the bias --
            // which shifts expert SELECTION only, so the model kept producing fluent
            // text from the wrong six experts. A hash layer legitimately has no bias
            // (Gate.__init__ sets it to None), so absence is an error only past those.
            bias: match f32s(st, &p("ffn.gate.bias"))
                .or_else(|_| f32s(st, &p("ffn.gate.e_score_correction_bias")))
            {
                Ok(b) => Some(b),
                Err(_) if is_hash => None,
                Err(e) => return Err(format!("{prefix}: {e}")),
            },
            tid2eid: if is_hash {
                Some(f32s(st, &p("ffn.gate.tid2eid"))?.into_iter().map(|x| x as i32).collect())
            } else {
                None
            },
            sh1: fp8(st, &p("ffn.shared_experts.w1"))?,
            sh3: fp8(st, &p("ffn.shared_experts.w3"))?,
            sh2: fp8(st, &p("ffn.shared_experts.w2"))?,
        })
    }

    pub(crate) fn bytes(&self) -> usize {
        self.wq_a.bytes()
            + self.wq_b.bytes()
            + self.wkv.bytes()
            + self.wo_a.bytes()
            + self.wo_b.bytes()
            + self.sh1.bytes()
            + self.sh3.bytes()
            + self.sh2.bytes()
            + self.gate.len() * 4
    }
}

pub struct Trunk {
    layers: Vec<Layer>,
    embed: Vec<u8>,
    embed_bf16: bool,
    final_norm: Vec<f32>,
    head: Vec<u8>,
    head_scale: Vec<u8>,
    /// The learned final Hyper-Connections reduce: fn [hc][hc*d], base [hc], scale.
    hc_head_fn: Vec<f32>,
    hc_head_base: Vec<f32>,
    hc_head_scale: f32,
    bytes: usize,
}

pub(crate) fn raw(st: &St, n: &str) -> Result<Vec<u8>, String> {
    let t = st.find(n).ok_or_else(|| format!("missing {n}"))?;
    let mut b = vec![0u8; t.nbytes as usize];
    st.read(t, &mut b);
    Ok(b)
}

fn opt_raw(st: &St, n: &str) -> Vec<u8> {
    raw(st, n).unwrap_or_default()
}

pub(crate) fn f32s(st: &St, n: &str) -> Result<Vec<f32>, String> {
    let t = st.find(n).ok_or_else(|| format!("missing {n}"))?;
    let mut v = vec![0f32; t.numel() as usize];
    st.read_f32(t, &mut v);
    Ok(v)
}

pub(crate) fn fp8(st: &St, base: &str) -> Result<Fp8, String> {
    Ok(Fp8 { w: raw(st, &format!("{base}.weight"))?, s: raw(st, &format!("{base}.scale"))? })
}

fn comp_of(st: &St, p: &str) -> Result<Comp, String> {
    Ok(Comp {
        ape: f32s(st, &format!("{p}.ape"))?,
        wkv: f32s(st, &format!("{p}.wkv.weight"))?,
        wgate: f32s(st, &format!("{p}.wgate.weight"))?,
        norm: f32s(st, &format!("{p}.norm.weight"))?,
    })
}

impl Trunk {
    /// Everything outside the decoder stack: the embedding, the vocabulary head and the
    /// final norms. DSpark shares all of these with the main model and touches none of the
    /// 43 layers, so a harness that only exercises the drafter can skip 8.29 GB of loads.
    pub fn load_io(st: &St, _s: &Spec) -> Result<Trunk, String> {
        let et = st.find("embed.weight").ok_or("missing embed.weight")?;
        let embed_bf16 = et.dtype == crate::st::Dtype::Bf16;
        let embed = raw(st, "embed.weight")?;
        let head = raw(st, "head.weight")?;
        let head_scale = opt_raw(st, "head.scale");
        let bytes = embed.len() + head.len() + head_scale.len();
        Ok(Trunk {
            layers: Vec::new(),
            embed,
            embed_bf16,
            final_norm: f32s(st, "norm.weight")?,
            head,
            head_scale,
            hc_head_fn: f32s(st, "hc_head_fn")?,
            hc_head_base: f32s(st, "hc_head_base")?,
            hc_head_scale: f32s(st, "hc_head_scale")?[0],
            bytes,
        })
    }

    pub fn load(st: &St, s: &Spec, n_layers: usize) -> Result<Trunk, String> {
        let et = st.find("embed.weight").ok_or("missing embed.weight")?;
        let embed_bf16 = et.dtype == crate::st::Dtype::Bf16;
        let mut bytes = 0usize;
        let embed = raw(st, "embed.weight")?;
        bytes += embed.len();
        let final_norm = f32s(st, "norm.weight")?;
        let head = raw(st, "head.weight")?;
        let head_scale = opt_raw(st, "head.scale");
        bytes += head.len() + head_scale.len();
        let hc_head_fn = f32s(st, "hc_head_fn")?;
        let hc_head_base = f32s(st, "hc_head_base")?;
        let hc_head_scale = f32s(st, "hc_head_scale")?[0];

        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let lay = Layer::load(st, s, &format!("layers.{l}"), compress_ratio(l), Some(l))?;
            bytes += lay.bytes();
            layers.push(lay);
            if l % 8 == 0 || l + 1 == n_layers {
                eprint!("\r  trunk: layer {}/{n_layers}, {:.2} GB", l + 1, bytes as f64 / 1e9);
            }
        }
        eprintln!();
        Ok(Trunk {
            layers,
            embed,
            embed_bf16,
            final_norm,
            head,
            head_scale,
            hc_head_fn,
            hc_head_base,
            hc_head_scale,
            bytes,
        })
    }

    /// The vocabulary head, in whichever form the checkpoint stores it. DSpark shares it
    /// with the main model -- `convert.py` skips `mtp.*` embed/head tensors precisely
    /// because they are tied, so there is no separate copy to load.
    pub fn head_w(&self) -> W<'_> {
        if self.head_scale.is_empty() {
            W::Bf16(unsafe {
                std::slice::from_raw_parts(self.head.as_ptr().cast(), self.head.len() / 2)
            })
        } else {
            W::F8Block { w: &self.head, scale: &self.head_scale, block: BLOCK }
        }
    }

    pub fn embed_row(&self, id: u32, hidden: usize, dst: &mut [f32]) {
        let base = id as usize * hidden;
        if self.embed_bf16 {
            let p: &[u16] = unsafe {
                std::slice::from_raw_parts(self.embed.as_ptr().cast(), self.embed.len() / 2)
            };
            for i in 0..hidden {
                dst[i] = crate::st::bf16_to_f32(p[base + i]);
            }
        } else {
            let p: &[f32] = unsafe {
                std::slice::from_raw_parts(self.embed.as_ptr().cast(), self.embed.len() / 4)
            };
            dst[..hidden].copy_from_slice(&p[base..][..hidden]);
        }
    }
}

/// Per-request generation settings.
///
/// These were process-global environment variables, which is fine for one run from a shell
/// and impossible for a server: two concurrent requests in one process cannot disagree
/// about `max_tokens` if the knob is a `std::env::var`. `from_env` reproduces the previous
/// behaviour exactly, so the CLI is unchanged and every existing measurement still
/// reproduces.
#[derive(Clone, Debug)]
pub struct Params {
    pub max_tokens: usize,
    pub cache_gb: f64,
    /// DSpark block drafting, and how many of its `block` tokens to actually submit for
    /// verification. See the note on `dspark_k`'s default below.
    pub dspark: bool,
    pub dspark_k: usize,
    pub dspark_conf: Option<f32>,
    /// n-gram speculation width; 0 disables it.
    pub spec_k: usize,
    /// `None` means "derive it from the cache size" -- see `prefill_width`.
    pub prefill_chunk: Option<usize>,
    pub vram_gb: Option<f64>,
    /// (bits, cold-threshold) for the quantisation sweep.
    pub qdq: Option<(u32, usize)>,
    pub prefetch: bool,
    /// `None` is greedy argmax -- byte for byte the pre-sampling engine, which is what
    /// every existing gate and measurement reproduces against.
    pub sample: Option<crate::sample::SampleParams>,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            max_tokens: 8,
            cache_gb: 5.0,
            dspark: false,
            // 2, not the architecture's block of 5, and the reason is measured rather than
            // assumed. DeepSeek-V4-Flash, 1.6 GB cache, 8 tokens, all token-identical to
            // serial decode: no dspark 94.8 s; k=2 103.2 s (67% of 6 accepted); k=5
            // 168.4 s (27% of 15 accepted). Every drafted token widens the verification
            // batch, and width is paid in expert bytes on a saturated device.
            dspark_k: 2,
            dspark_conf: None,
            spec_k: 0,
            prefill_chunk: None,
            vram_gb: None,
            qdq: None,
            prefetch: false,
            sample: None,
        }
    }
}

impl Params {
    /// Exactly the environment the CLI used before this struct existed.
    pub fn from_env(max_tokens: usize, cache_gb: f64) -> Params {
        // A generic fn, not a closure: a closure cannot be generic over its return type,
        // and these knobs are usize, f32 and f64.
        fn num<T: std::str::FromStr>(k: &str) -> Option<T> {
            std::env::var(k).ok().and_then(|v| v.parse().ok())
        }
        Params {
            max_tokens,
            cache_gb,
            dspark: std::env::var_os("K3_DSPARK").is_some(),
            dspark_k: num::<usize>("K3_DSPARK_K").unwrap_or(2),
            dspark_conf: num("K3_DSPARK_CONF"),
            spec_k: num::<usize>("K3_SPEC").unwrap_or(0),
            prefill_chunk: num("K3_PREFILL_CHUNK"),
            vram_gb: num("K3_VRAM_GB"),
            qdq: num::<u32>("K3_QDQ").map(|b| (b, num::<usize>("K3_QDQ_COLD").unwrap_or(0))),
            prefetch: std::env::var_os("K3_PREFETCH").is_some(),
            // The CLI stays greedy unless asked. Sampling is a server concern, and making
            // it opt-in is what keeps every historical measurement reproducible.
            sample: num::<f32>("K3_TEMP").map(|t| crate::sample::SampleParams {
                temperature: t,
                top_p: num("K3_TOP_P").unwrap_or(1.0),
                top_k: num("K3_TOP_K").unwrap_or(0),
                min_p: num("K3_MIN_P").unwrap_or(0.0),
                repeat_penalty: num("K3_REPEAT_PENALTY").unwrap_or(1.0),
                repeat_last_n: num("K3_REPEAT_LAST_N").unwrap_or(64),
                seed: num("K3_SEED").unwrap_or(0),
            }),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn generate(
    st: &St,
    spec: &Spec,
    model: &Path,
    tokenizer: Option<&Path>,
    prompt: Option<&str>,
    ids_in: Option<&[u32]>,
    params: &Params,
    out: Option<&Path>,
    trace: Option<&Path>,
    // Called once per emitted token, with its id and decoded piece. The CLI prints and
    // flushes; a server writes an SSE frame. Nothing in the loop writes to stdout itself,
    // which is what makes streaming possible at all.
    //
    // Returning false stops generation. That is not a convenience: a stop sequence or a
    // disconnected client at 8 s/token would otherwise cost minutes of expert streaming
    // for output nobody will read.
    sink: &mut dyn FnMut(u32, &str) -> bool,
) -> Result<(), String> {
    let tok = open_tokenizer(model, tokenizer)?;
    let ids: Vec<u32> = match (ids_in, prompt) {
        (Some(v), _) => v.to_vec(),
        (None, Some(p)) => tok
            .as_ref()
            .ok_or("a prompt needs a tokenizer: pass --tokenizer or put tokenizer.json in the model dir")?
            .encode(p, false)
            .map_err(|e| e.to_string())?,
        _ => return Err("one of --prompt or --ids is required".into()),
    };
    println!("prompt: {} tokens {:?}", ids.len(), &ids[..ids.len().min(12)]);
    let cap = ids.len() + params.max_tokens + 2;
    let mut eng = Engine::load(st, spec, model, tok, params, cap, trace.is_some())?;
    let mut sess = Session::new(spec.n_layers, ids);
    generate_on(&mut eng, &mut sess, params, out, trace, sink)
}

/// Open the tokenizer a model directory ships, if it has one.
pub fn open_tokenizer(
    model: &Path,
    tokenizer: Option<&Path>,
) -> Result<Option<crate::tok::Tok>, String> {
    match tokenizer.map(std::path::Path::to_path_buf).or_else(|| {
        let p = model.join("tokenizer.json");
        p.exists().then_some(p)
    }) {
        // TODO: bos/eos are hardcoded here and at every other call site. Correct for
        // DeepSeek-V4, wrong for the next model -- they should come from the tokenizer.
        Some(p) => Ok(Some(crate::tok::Tok::from_file(&p, 0, 1).map_err(|e| e.to_string())?)),
        None => Ok(None),
    }
}

/// Everything expensive enough that a server must not redo it per request: the resident
/// trunk (seconds and GB), the expert cache and its GPU tier, the DSpark stages, and the
/// rope tables. Borrows the checkpoint rather than owning it, so one `St` can back several
/// engines.
pub struct Engine<'a> {
    pub st: &'a St,
    pub spec: &'a Spec,
    pub tok: Option<crate::tok::Tok>,
    trunk: Trunk,
    cache: Cache,
    dspark: Option<crate::dspark::DSpark>,
    md: MoeDimsV4,
    ropes: Vec<Rope>,
    ds_rope: Rope,
    n_ds: usize,
    use_dspark: bool,
}

impl<'a> Engine<'a> {
    /// `max_ctx` sizes the rope tables. They are pure lookup tables, so a longer one gives
    /// identical values at every position -- sizing generously is behaviour-preserving and
    /// is what lets the engine outlive a single request.
    pub fn load(
        st: &'a St,
        spec: &'a Spec,
        model: &Path,
        tok: Option<crate::tok::Tok>,
        params: &Params,
        max_ctx: usize,
        trace: bool,
    ) -> Result<Engine<'a>, String> {
        let n_ds = crate::dspark::n_stages(st);
        let use_dspark = params.dspark && n_ds > 0;
    // Refuse a budget that does not fit, rather than letting the kernel discover it.
    //
    // Over-committing here does NOT produce a clean OOM kill. Measured on a 15 GB machine
    // with 8.4 GB of swap on the same USB device the model streams from: the kernel pages
    // out instead of failing the allocation, every page-in then competes with the expert
    // reads on an already-saturated disk, and the machine becomes unresponsive without
    // anything ever being killed. Checking is cheap; recovering is a power cycle.
    // `cache_gb <= 0` (the serve default) means AUTO: size the arena to the largest swap-safe
    // slice of RAM, computed from MemAvailable below. Falls back to 5 GB if it can't be read.
    let mut cache_gb = params.cache_gb;
    let auto = cache_gb <= 0.0;
    let margin_gb = std::env::var("RUSTLM_MEM_MARGIN_GB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(2.5);
    let avail = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|m| {
            m.lines()
                .find(|l| l.starts_with("MemAvailable:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<u64>().ok())
        })
        .map(|kb| kb as f64 * 1024.0);
    if let Some(avail) = avail {
        // The trunk is the checkpoint's non-expert bytes; 1 GB covers activations, the
        // KV cache and the allocator's slack.
        // Only the layers that actually load. Counting the extra MTP blocks past the
        // decoder stack over-estimates by ~1.4 GB and would refuse a configuration that
        // demonstrably runs.
        let trunk_est = st
            .tensors
            .iter()
            .filter(|t| !t.name.contains(".experts."))
            // `mtp.0/1/2` are the DSpark drafter stages -- the same three that make
            // compress_ratios 46 long for 43 decoder layers. Without --dspark they are
            // never loaded, and counting them over-estimates the trunk by ~1.4 GB.
            .filter(|t| use_dspark || !t.name.starts_with("mtp."))
            .map(|t| t.nbytes as f64)
            .sum::<f64>();
        if auto {
            // Largest swap-safe cache: everything free beyond the trunk and a margin.
            cache_gb = (((avail - trunk_est - margin_gb * 1e9) / 1e9).max(0.5) * 10.0).floor() / 10.0;
            println!(
                "memory      : {:.1} GB available -> auto cache {:.1} GB (trunk {:.1} + {:.1} GB margin)",
                avail / 1e9, cache_gb, trunk_est / 1e9, margin_gb
            );
        } else {
            let need = trunk_est + cache_gb * 1e9 + 1e9;
            if need > avail {
                // The suggestion has to be STRICTLY smaller than what was just refused, and
                // rounded DOWN. Printing it with `{:.0}` recommended "--cache-gb 2" to someone
                // who had passed exactly that -- advice that fails identically when followed.
                let room = (avail - trunk_est - 1e9) / 1e9;
                let advice = if room < 0.5 {
                    // No cache size helps: the resident trunk alone does not fit, so the fix
                    // is elsewhere entirely.
                    "\n  No cache size fixes this -- the resident trunk alone exceeds what is \
                     free. Close something, or run when the machine is quieter, or pass \
                     --cache-gb auto."
                        .to_string()
                } else {
                    format!("\n  Try --cache-gb {:.1} (or --cache-gb auto)", (room * 10.0).floor() / 10.0)
                };
                return Err(format!(
                    "this configuration needs about {:.1} GB (trunk {:.1} + cache {:.1} + ~1 \
                     for working memory) but only {:.1} GB is available.\n  \
                     Refusing to start: with swap on the same device the model streams from, \
                     over-committing does not fail cleanly -- it thrashes until the machine \
                     is unusable.{advice}",
                    need / 1e9,
                    trunk_est / 1e9,
                    cache_gb,
                    avail / 1e9,
                ));
            }
            println!(
                "memory      : {:.1} GB available, plan needs ~{:.1} GB",
                avail / 1e9,
                need / 1e9
            );
        }
    }

    let slot = {
        let r = crate::cache::locate(st, &v4_expert_names(2, 0)).ok_or("cannot locate an expert")?;
        crate::cache::slot_need(&r)
    };
    if auto && cache_gb <= 0.0 {
        cache_gb = 5.0; // MemAvailable was unreadable; a safe default rather than a 0 arena.
    }
    let mut cache = Cache::new((cache_gb * 1e9) as i64, slot, spec.n_experts, spec.topk)?;
    println!("expert cache: {} slots of {:.1} MB", cache.nslot(), slot as f64 / 1e6);
    if trace {
        cache.trace_on();
    }
    if let Some((bits, cold)) = params.qdq {
        {
            cache.qdq_on(bits, cold);
            println!(
                "QDQ sweep   : experts degraded to {bits} bits{}",
                if cold > 0 { format!(", cold only (<= {cold} requests)") } else { ", ALL".into() }
            );
        }
    }
    // Off by default, and the reason is measured rather than assumed. Prefetching hides
    // LATENCY; this workload is bandwidth-bound. At 604 MB/s and ~3.6 GB of experts per
    // token the device is busy ~6 s of every ~7 s step, so there is no idle window to
    // overlap into and the reader thread only competes with the real reads. Measured on
    // DeepSeek-V4-Flash, 5 GB cache: 7.0 s/token off, 8.3 s/token on -- despite the
    // prefetch IMPROVING hit rate (19.1% -> 21.1%) and bytes read (3.91 -> 3.57 GB/token).
    //
    // It should win on a device with headroom. K3_PREFETCH=1 turns it on.
    // GPU memory as a victim cache. Costs zero system RAM, which is the point: the
    // resident trunk already leaves under 2 GB for experts, and simulation on a real
    // trace puts the useful knee at ~258 slots -- one token's working set. RAM alone is
    // below it; RAM plus VRAM is above it.
    if let Some(gb) = params.vram_gb {
        match crate::vram::Vram::new((gb * 1e9) as i64, slot) {
            Ok(v) => {
                println!(
                    "vram tier   : {} slots ({:.2} GB) on the GPU, zero system RAM",
                    v.nslot(),
                    v.bytes() as f64 / 1e9
                );
                cache.attach_vram(v);
            }
            Err(e) => eprintln!("vram tier   : off ({e})"),
        }
    }
    if params.prefetch {
        match crate::cache::Prefetcher::new(model, slot) {
            Ok(p) => {
                cache.attach_prefetcher(p);
                println!("prefetch    : ON (K3_PREFETCH set), layer L+1 predicted from L");
            }
            Err(e) => eprintln!("prefetch    : off ({e})"),
        }
    }

    let t0 = std::time::Instant::now();
    let trunk = Trunk::load(st, spec, spec.n_layers)?;
    println!(
        "trunk resident: {:.2} GB in {:.1} s (loaded ONCE, not per token)",
        trunk.bytes as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );

    let dspark = if use_dspark {
        let t = std::time::Instant::now();
        let d = crate::dspark::DSpark::load(st, spec, n_ds, 5, 128799)?;
        println!(
            "dspark      : {} stages, block {}, {:.2} GB resident in {:.1} s",
            n_ds,
            d.block,
            d.bytes as f64 / 1e9,
            t.elapsed().as_secs_f64()
        );
        Some(d)
    } else {
        None
    };

    let e = spec.hidden;
    let md = MoeDimsV4 {
        hidden: e,
        moe_inter: spec.moe_inter,
        n_experts: spec.n_experts,
        topk: spec.topk,
        route_scale: spec.routed_scale,
        swiglu_limit: match spec.glu {
            crate::ops::Glu::SwigluClamped { limit } => limit,
            _ => 10.0,
        },
        n_hash_layers: spec.n_hash_layers,
    };

    let cap = max_ctx;
    let ropes: Vec<Rope> = (0..spec.n_layers)
        .map(|l| {
            if trunk.layers[l].ratio == 0 {
                precompute_rope(64, cap, 0, spec.rope_theta, 16.0, 32.0, 1.0)
            } else {
                precompute_rope(64, cap, 65536, 160000.0, 16.0, 32.0, 1.0)
            }
        })
        .collect();

    // DSpark's attention asserts compress_ratio == 0, so it shares the ratio-0 rope table
    // -- but its queries sit up to `block` positions past anything the main model has
    // reached, so the table has to be longer.
    let ds_rope = precompute_rope(64, cap + 8, 0, spec.rope_theta, 16.0, 32.0, 1.0);


        Ok(Engine { st, spec, tok, trunk, cache, dspark, md, ropes, ds_rope, n_ds, use_dspark })
    }

    pub fn cache_mut(&mut self) -> &mut Cache {
        &mut self.cache
    }
}

/// One conversation's decode state. Kept separate from `Engine` because this is what a
/// server has to hold per session and evict under pressure -- roughly 1 MB per token, so
/// it, not the code, is what limits how many chats stay warm.
pub struct Session {
    pub state: Vec<LayerState>,
    pub hin: Vec<Vec<f32>>,
    pub ids: Vec<u32>,
    pub pos: usize,
    pub prompt_len: usize,
}

impl Session {
    pub fn new(n_layers: usize, ids: Vec<u32>) -> Session {
        Session {
            state: (0..n_layers).map(|_| LayerState::default()).collect(),
            hin: vec![Vec::new(); n_layers],
            prompt_len: ids.len(),
            ids,
            pos: 0,
        }
    }
}

/// Run one request against a loaded engine and a session.
#[allow(clippy::too_many_arguments)]
pub fn generate_on(
    eng: &mut Engine,
    sess: &mut Session,
    params: &Params,
    out: Option<&Path>,
    trace: Option<&Path>,
    sink: &mut dyn FnMut(u32, &str) -> bool,
) -> Result<(), String> {
    let (n, _cache_gb) = (params.max_tokens, params.cache_gb);
    // Destructured into the names the loop body already uses, so the body below is moved
    // verbatim rather than rewritten -- the end-to-end output check is what proves this
    // refactor preserved behaviour, and a verbatim move is what makes that check meaningful.
    let Engine { st, spec, tok, trunk, cache, dspark, md, ropes, ds_rope, n_ds, use_dspark } = eng;
    let (st, spec, n_ds, use_dspark) = (*st, *spec, *n_ds, *use_dspark);
    let (e, eps) = (spec.hidden, spec.rms_eps);
    let ntgt = crate::dspark::TARGET_LAYERS.len();
    // `mem::take` rather than borrowing: the loop mutates these as owned Vecs, and taking
    // them costs a pointer swap. They go back at the end.
    let mut state = std::mem::take(&mut sess.state);
    let mut hin = std::mem::take(&mut sess.hin);
    let mut ids = std::mem::take(&mut sess.ids);
    let mut pos = sess.pos;
    let prompt_len = sess.prompt_len;
    let mut text = String::new();
    let mut row = vec![0f32; e];
    let mut logits = vec![0f32; spec.vocab];
    let t_all = std::time::Instant::now();

    // !! THE INTERACTION THAT WOULD OTHERWISE BE SILENT !!
    //
    // Speculation accepts a draft when it equals `preds[t]`, the GREEDY argmax -- that
    // equality is the entire reason speculative decode is exact rather than approximate.
    // Sample the verifier independently and the accept test starts comparing a draft
    // against a token the sampler was never going to choose: the output distribution
    // shifts, and NOTHING catches it, because the text still reads perfectly.
    //
    // So this refuses the combination rather than quietly disabling half of it. A caller
    // that wants both must be told, not silently given one.
    let mut sampler = params.sample.clone().map(crate::sample::Sampler::new);
    if let Some(s) = &sampler {
        if !s.p.speculation_safe() && (params.spec_k > 0 || use_dspark) {
            return Err(format!(
                "temperature {:.2} cannot be combined with speculative decoding: a draft is \
                 accepted by equality against the greedy argmax, which sampling replaces. \
                 Set temperature 0, or turn off K3_SPEC/K3_DSPARK.",
                s.p.temperature
            ));
        }
    }

    let spec_k = params.spec_k;
    if spec_k > 0 {
        println!("spec decode : drafting up to {spec_k} tokens per step (n-gram)");
    }
    // 2, not the architecture's block of 5, and the reason is measured rather than
    // assumed. DeepSeek-V4-Flash, 1.6 GB cache, 8 tokens, "The capital of France is",
    // all three token-identical to serial decode:
    //
    //   no dspark        94.8 s   36.48 GB read
    //   dspark, k = 2   103.2 s   37.71 GB   67% of 6 drafted accepted
    //   dspark, k = 5   168.4 s   51.26 GB   27% of 15 drafted accepted
    //
    // Every drafted token widens the verification batch, and width is paid in expert
    // bytes on a device that is already saturated. k = 5 accepts a smaller FRACTION and
    // pays for all of it. Even at k = 2 this is still 9% behind serial here, because the
    // three DSpark stages add ~1.2 GB of their own expert traffic to a 119-slot cache
    // that cannot hold them -- so they are re-read every step. On a machine with room to
    // keep them resident that term goes away and the arithmetic changes; on this one it
    // does not, which is why K3_DSPARK stays opt-in.
    let dspark_k = params.dspark_k;
    let dspark_conf = params.dspark_conf;
    let mut drafted = 0u64;
    let mut accepted = 0u64;
    // The confidence head's output has no calibrated scale -- it is only ever meaningful
    // against itself, so report it rather than thresholding on a number picked here.
    let mut conf_sum = 0.0f64;
    let mut conf_n = 0u64;
    // Clamped to 8 here and NOT in `prefill_width`, because 8 is a fact about this stack
    // rather than about the arithmetic: it is the width DeepSeek-V4 was validated at end to
    // end, and V4's own `layer_forward_spec` batches speculation into the same chunk, so a
    // wider prompt chunk changes more than the prompt path. qwen35moe has no such
    // entanglement and takes the derived width. Lifting this needs a measurement on the
    // 167 GB checkpoint, not an edit here.
    let prefill_chunk = params
        .prefill_chunk
        .unwrap_or_else(|| {
            prefill_width(cache.nslot(), spec.n_experts, spec.topk, spec.hidden, spec.n_layers,
                          spec.moe_inter)
            .min(8)
        })
        .max(1);
    if prefill_chunk != 8 {
        println!(
            "prefill     : chunks of {prefill_chunk} (from {} cache slots)",
            cache.nslot()
        );
    }

    loop {
        let step0 = std::time::Instant::now();
        // Verify the next token plus, when past the prompt, a cheap n-gram draft. The
        // batch shares one pass over 43 layers, so k tokens cost the UNION of their
        // experts rather than 6k reads.
        // Prompt tokens are KNOWN, so a chunk of them needs no speculation at all: route
        // the whole chunk, fetch the union of its experts once, and the per-token cost
        // falls by whatever the routing overlaps. Measured on a real trace, a chunk of 8
        // needs 1062 expert reads where 8 separate tokens need 2064 -- a 48.6% saving,
        // which is the same effect k3_moe_prefill was written for ("about half the expert
        // bytes on a prompt").
        //
        // The chunk is deliberately small. K3 uses 64, but it has 896 experts and a cache
        // sized for them; here 64 tokens would ask for ~200 unique experts per layer
        // against a 373-slot cache and thrash it. Eight keeps the per-layer union at ~25.
        // "Still consuming the prompt" -- true whenever there is more than one prompt
        // token left to feed, so the chunk is made of tokens we already know.
        // Against `prompt_len`, NOT `ids.len()`: every emitted token is appended to `ids`,
        // so `pos < ids.len()` stayed true forever and the loop kept consuming its own
        // output one token at a time as though it were still prompt. The output was
        // correct -- that path is plain serial decode -- but `k` was always 1 and the
        // speculative branch below was unreachable, n-gram and DSpark alike.
        let prompt_phase = pos < prompt_len;
        let batch: Vec<u32> = if prompt_phase {
            let take = (prompt_len - pos).min(prefill_chunk);
            ids[pos..pos + take].to_vec()
        } else if let Some(ds) = dspark.as_mut() {
            // `ids[pos]` is the token just emitted, at main-model position `pos`; the KV
            // rows the drafter needs cover positions 0..pos-1, which is what has been
            // pushed. It drafts the `block` tokens that follow.
            let d = ds.draft(trunk, spec, md, ids[pos], pos - 1, ds_rope, st, cache, eps)?;
            conf_sum += d.conf.iter().map(|&c| c as f64).sum::<f64>();
            conf_n += d.conf.len() as u64;
            // Every drafted token widens the VERIFICATION batch, and on a bandwidth-bound
            // device that width is paid in expert bytes whether the token is accepted or
            // not. Drafting all 5 is only worth it if most survive; below that, taking a
            // shorter prefix costs less to check. `dspark_k` caps it outright,
            // `dspark_conf` cuts at the first position the confidence head doubts -- which
            // is the signal the checkpoint ships that head for.
            let mut keep = d.ids.len().min(dspark_k);
            if let Some(th) = dspark_conf {
                keep = keep.min(d.conf.iter().take_while(|&&c| c >= th).count());
            }
            drafted += keep as u64;
            let mut b = vec![ids[pos]];
            b.extend_from_slice(&d.ids[..keep]);
            b
        } else {
            let mut b = vec![ids[pos]];
            if spec_k > 0 {
                let d = draft_ngram(&ids, spec_k, 3);
                drafted += d.len() as u64;
                b.extend_from_slice(&d);
            }
            b
        };
        let k = batch.len();
        let mut main_h = vec![0f32; if dspark.is_some() { k * ntgt * e } else { 0 }];
        let mut x0 = vec![0f32; k * HC * e];
        for (t, &bid) in batch.iter().enumerate() {
            trunk.embed_row(bid, e, &mut row);
            for c in 0..HC {
                x0[(t * HC + c) * e..][..e].copy_from_slice(&row);
            }
        }
        let mut res = HyperConnResidual::new(&x0, k, e, HC, eps, eps, 20);

        // The sweep is a cycle: DSpark's stages (keyed as layers n_layers..) run first,
        // then the 43 decoder layers, then the next token starts over. Reporting the
        // position lets the cache evict by distance-to-next-use instead of by age.
        let cycle = spec.n_layers + if use_dspark { n_ds } else { 0 };
        for (l, lay) in trunk.layers.iter().enumerate() {
            cache.at_layer(l, cycle);
            let ad = AttnDims {
                hidden: e,
                n_heads: spec.n_heads,
                head_dim: spec.head_dim,
                rope_head_dim: 64,
                q_lora_rank: 1024,
                o_lora_rank: 1024,
                o_groups: 8,
                window: BLOCK,
                compress_ratio: lay.ratio,
                eps,
            };
            let attn = AttnW {
                wq_a: lay.wq_a.w(),
                q_norm: &lay.q_norm,
                wq_b: lay.wq_b.w(),
                wkv: lay.wkv.w(),
                kv_norm: &lay.kv_norm,
                wo_a: lay.wo_a.w(),
                wo_b: lay.wo_b.w(),
                attn_sink: &lay.sink,
            };
            // These must outlive `layer`, which borrows them.
            let (cw, cd, iw, idm, icw, icd);
            let compressed = match &lay.comp {
                Some(c) => {
                    cw = CompressorW {
                        ape: &c.ape,
                        wkv: W::F32(&c.wkv),
                        wgate: W::F32(&c.wgate),
                        norm: &c.norm,
                    };
                    cd = CompressorDims {
                        hidden: e,
                        head_dim: spec.head_dim,
                        rope_head_dim: 64,
                        ratio: lay.ratio,
                        rotate: false,
                        eps,
                    };
                    let indexer = match &lay.idx {
                        Some(ix) => {
                            icw = CompressorW {
                                ape: &ix.c.ape,
                                wkv: W::F32(&ix.c.wkv),
                                wgate: W::F32(&ix.c.wgate),
                                norm: &ix.c.norm,
                            };
                            icd = CompressorDims {
                                hidden: e,
                                head_dim: 128,
                                rope_head_dim: 64,
                                ratio: 4,
                                rotate: true,
                                eps,
                            };
                            iw = IndexerW {
                                wq_b: ix.wq_b.w(),
                                weights_proj: W::F32(&ix.weights_proj),
                            };
                            idm = IndexerDims {
                                hidden: e,
                                n_heads: spec.n_heads,
                                head_dim: 128,
                                rope_head_dim: 64,
                                q_lora_rank: 1024,
                                index_topk: 512,
                                ratio: 4,
                            };
                            Some((&iw, &idm, &icw, &icd))
                        }
                        None => None,
                    };
                    Some(Compressed { w: &cw, d: &cd, indexer })
                }
                None => None,
            };
            let hc = HcLayer {
                attn_fn: &lay.hc_attn[0],
                attn_base: &lay.hc_attn[1],
                attn_scale: [lay.hc_attn[2][0], lay.hc_attn[2][1], lay.hc_attn[2][2]],
                ffn_fn: &lay.hc_ffn[0],
                ffn_base: &lay.hc_ffn[1],
                ffn_scale: [lay.hc_ffn[2][0], lay.hc_ffn[2][1], lay.hc_ffn[2][2]],
            };
            let moe = MoeWV4 {
                gate: &lay.gate,
                bias: lay.bias.as_deref(),
                tid2eid: lay.tid2eid.as_deref(),
                sh1: lay.sh1.w(),
                sh3: lay.sh3.w(),
                sh2: lay.sh2.w(),
            };
            let layer = LayerV4 {
                attn,
                attn_dims: &ad,
                compressed,
                hc,
                attn_norm: &lay.attn_norm,
                ffn_norm: &lay.ffn_norm,
                moe,
            };
            res.begin_layer(&layer.hc);
            layer_forward_spec(&mut res, &layer, md, l, &batch, &ropes[l], st, cache,
                               v4_expert_names, eps, &mut state[l], &mut hin[l])?;

            // DSpark reads the hc-MEAN of layers 40/41/42's outputs -- a plain mean, not
            // the learned hc_head reduce the logits path uses. Transformer.forward:
            // `if i in self.target_layer_ids: main_hiddens.append(h.mean(dim=2))`.
            if !main_h.is_empty() {
                if let Some(j) = crate::dspark::TARGET_LAYERS.iter().position(|&x| x == l) {
                    let s8 = res.state();
                    for t in 0..k {
                        let dst = &mut main_h[(t * ntgt + j) * e..][..e];
                        for c in 0..HC {
                            let src = &s8[(t * HC + c) * e..][..e];
                            for i in 0..e {
                                dst[i] += src[i] / HC as f32;
                            }
                        }
                    }
                }
            }

            // Predict what layer l+1 will route to and start those reads now, so they
            // overlap l+1's attention rather than stalling its MoE. The gate is ~1M MACs
            // against the 151M the experts cost, and a wrong guess wastes a read, never
            // an answer.
            if l + 1 < trunk.layers.len() {
                let nxt = &trunk.layers[l + 1];
                let st8 = res.state();
                let mut mix = vec![0f32; e];
                for k in 0..HC {
                    for i in 0..e {
                        mix[i] += st8[k * e + i] / HC as f32;
                    }
                }
                let mut nrm = vec![0f32; e];
                crate::ops::rmsnorm_acc(&mut nrm, &mix, &nxt.ffn_norm, e, eps, spec.rms_acc);
                let want = predict_experts(&nrm, nxt, md, l + 1, batch[0], PREFETCH_M);
                cache.prefetch_hint(l + 1, &want, v4_expert_names);
            }
        }

        let sstate = res.state();
        let hw0 = trunk.head_scale.is_empty();
        let mut preds = Vec::with_capacity(k);
        for t in 0..k {
            let mut last = vec![0f32; e];
            crate::ops::hc_head(
                &mut last,
                &sstate[t * HC * e..][..HC * e],
                &trunk.hc_head_fn,
                &trunk.hc_head_base,
                trunk.hc_head_scale,
                e,
                HC,
                eps,
                eps,
            );
            let mut normed = vec![0f32; e];
            crate::ops::rmsnorm_acc(&mut normed, &last, &trunk.final_norm, e, eps, spec.rms_acc);
            let hw = if hw0 {
                W::Bf16(unsafe {
                    std::slice::from_raw_parts(trunk.head.as_ptr().cast(), trunk.head.len() / 2)
                })
            } else {
                W::F8Block { w: &trunk.head, scale: &trunk.head_scale, block: BLOCK }
            };
            crate::ops::mmw(&mut logits, &normed, hw, e, spec.vocab);
            preds.push(
                logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32)
                    .ok_or("empty logits")?,
            );
        }
        // Sampling replaces exactly the position that becomes a real token, and nothing
        // else. `preds` stays greedy because it is ALSO the verification oracle: a drafted
        // token is accepted by comparing against it, so sampling into it would corrupt the
        // accept test rather than the output.
        //
        // The emitted position is `k - 1` in both phases, which is why one line covers
        // both: a prompt chunk emits `preds[take - 1]` with `take == k`, and decode with
        // sampling runs `k == 1` because the guard above refuses speculation. `logits`
        // holds position `k - 1` here -- the loop overwrites it each iteration and the
        // last write wins.
        if let Some(s) = sampler.as_mut() {
            let last = k - 1;
            preds[last] = s.pick(&mut logits, &ids);
        }
        let dt = step0.elapsed().as_secs_f64();

        // preds[t] is what the model says follows batch[..=t]. preds[0] always holds:
        // batch[0] is a real token, so its prediction is the true next token. A drafted
        // batch[t+1] is CORRECT exactly when it equals preds[t]; the first place that
        // fails, preds at that position is still right, so one token past the last match
        // is always accepted. That is what makes this identical to serial greedy decode
        // rather than an approximation of it.
        // Prompt tokens are given, not guessed, so the whole chunk is accepted. Only a
        // speculative batch has to be verified against what the model actually predicts.
        let mut take = 1usize;
        if prompt_phase {
            take = k;
        } else {
            while take < k && batch[take] == preds[take - 1] {
                take += 1;
            }
        }
        // A prompt chunk predicts tokens we already have, except for its LAST position,
        // which predicts the first genuinely new one. A speculative batch emits every
        // accepted position.
        let emit: Vec<u32> = if prompt_phase {
            if pos + take < prompt_len { Vec::new() } else { vec![preds[take - 1]] }
        } else {
            accepted += (take - 1) as u64;
            (0..take).map(|t| preds[t]).collect()
        };

        // Everything past the accepted prefix was written into the KV cache during
        // verification and must be rolled back, or later tokens would attend to a future
        // that was rejected.
        let keep = pos + take;
        if keep < pos + k {
            for l in 0..spec.n_layers {
                state[l].truncate(keep, spec.head_dim, 1024);
                hin[l].truncate(keep * e);
            }
        }
        // Only ACCEPTED positions become DSpark history. Rejected ones are never pushed,
        // so there is nothing here to roll back -- unlike the decoder KV above, which is
        // written during verification and has to be truncated.
        if let Some(ds) = dspark.as_mut() {
            for t in 0..take {
                ds.push(&main_h[t * ntgt * e..][..ntgt * e], pos + t, ds_rope, e,
                        spec.head_dim, 64, eps);
            }
        }
        pos += take;

        if emit.is_empty() {
            eprintln!("  [prompt {}/{prompt_len}, {k} in one pass] {dt:.1} s", pos.min(prompt_len));
            continue;
        }
        let mut stop = false;
        for &next in &emit {
            let mut want_more = true;
            if let Some(t) = tok {
                if next == t.eos {
                    stop = true;
                    break;
                }
                let piece = t.piece(next)?;
                want_more = sink(next, &piece);
                text.push_str(&piece);
            }
            ids.push(next);
            // `ids` is pushed exactly once per emitted token, outside the tokenizer arm,
            // because generation by raw ids has no tokenizer and must still advance.
            if !want_more || ids.len() - prompt_len >= n {
                stop = true;
                break;
            }
        }
        eprintln!(
            "  [+{} token{} in one pass, {} verified] {dt:.1} s",
            emit.len(),
            if emit.len() == 1 { "" } else { "s" },
            k
        );
        if stop {
            break;
        }
    }
    if drafted > 0 {
        eprintln!(
            "  speculative: {drafted} drafted, {accepted} accepted ({:.0}%){}",
            100.0 * accepted as f64 / drafted as f64,
            if conf_n > 0 {
                format!(", mean confidence {:.3}", conf_sum / conf_n as f64)
            } else {
                String::new()
            }
        );
    }
    eprintln!(
        "  {} tokens in {:.1} s",
        ids.len() - prompt_len,
        t_all.elapsed().as_secs_f64()
    );
    println!();
    if let Some(p) = out {
        let s = if text.is_empty() {
            ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
        } else {
            text.clone()
        };
        std::fs::write(p, s).map_err(|e| e.to_string())?;
    }
    cache.report("final");
    if let Some((slots, spills, fills, down)) = cache.vram_report() {
        eprintln!(
            "  vram tier    : {slots} slots, {spills} spilled, {fills} served from GPU \
             ({:.2} GB DMA'd back instead of re-read)",
            down as f64 / 1e9
        );
    }
    if let Some(p) = trace {
        let b = cache.trace_bytes();
        std::fs::write(p, &b).map_err(|e| e.to_string())?;
        println!("cache trace: {} requests -> {}", b.len() / 8, p.display());
    }
    sess.state = state;
    sess.hin = hin;
    // `prompt_len` must be written back too. Without it a REUSED session keeps the
    // previous turn's value, so `prompt_phase = pos < prompt_len` is false immediately and
    // the newly appended prompt tokens are decoded ONE AT A TIME instead of batched. A
    // prefix cache built on top would appear to work while delivering none of the win --
    // right answers, no speedup, nothing failing.
    sess.prompt_len = prompt_len.max(ids.len());
    sess.ids = ids;
    sess.pos = pos;
    Ok(())
}

/// Prompt-chunk width, derived from the cache size instead of fixed at 8.
///
/// A chunk of `k` tokens draws `k * topk` experts per layer, and `prefetch_many` installs
/// that whole union BEFORE any of it is consumed. If the union approaches the slot count
/// it evicts itself -- the first expert is gone by the time the first token's MoE asks for
/// it -- and the layer gets read twice. The width therefore has to be a function of the
/// cache, which it was not: the reasoning in the decode loop's comment sizes 8 against a
/// 373-slot cache, and a 1.5 GB cache holds 119.
///
/// Expected distinct draws is the occupancy formula
///
/// ```text
/// U(k) = n * (1 - (1 - 1/n)^(k*topk))
/// ```
///
/// which is an upper bound in practice, because adjacent tokens route similarly -- a
/// repetitive prompt measured 1062 distinct where this predicts 1883. Using the bound is
/// deliberate: it is the diverse-prompt case that thrashes, and that is the one to be safe
/// against.
///
/// Shared with qwen35moe: the occupancy bound is a property of top-k routing over an
/// LRU-ish cache, not of any one architecture.
///
/// THE BOUND IS THE SWEEP, NOT THE LAYER -- MEASURED 2026-08-15
/// ```text
///     This used to search `(1..=8)` against `U(k) <= nslot/4`, with 8 described as "the
///     width validated end to end on DeepSeek-V4". Two things were wrong with that. The
///     ceiling was a constant, and the budget was a SINGLE layer's union when the quantity
///     that actually has to fit is a whole sweep's.
///
///     A chunk walks every layer once, installing `U(k)` experts at each. So its footprint
///     is `U(k) * n_layers`, and the next chunk hits the cache only if that footprint
///     still fits. Widen past the point where it does and each chunk evicts its own layer
///     0 before the next chunk gets back to it -- more bytes read, not fewer.
///
///     Measured on Qwen3.6-35B, 128 prompt tokens, 5 GB cache (2422 slots), 40 layers:
///
///         width      time     GB read     sharing within a chunk
///             8    89.7 s       14.55                     52.7%
///            32    99.1 s       19.62                     74.6%   <- more sharing,
///           128    91.3 s       14.12                     87.8%      MORE bytes read
///
///     Width 32 deduplicates far better inside each chunk and still reads 35% more,
///     because its footprint (162 experts x 40 layers = 6480) is nearly three times the
///     cache. The sweep rule predicts the turning point exactly: `U(k) * 40 <= 2422` gives
///     `U(k) <= 60.5`, hence `k <= 8`. The old constant was right about this model by
///     luck, and it is the reasoning that was missing.
///
///     Logits were bit-identical at all three widths, so none of this is a correctness
///     knob -- only bytes.
/// ```
///
/// AND WIDTH IS NOT WHERE PREFILL TIME GOES
/// ```text
///     The same run says so. Width 32 reads 35% more than width 8 and costs 10% more
///     time; differencing the two gives ~700 MB/s of marginal disk, so 14.55 GB is ~21 s
///     of an 89.7 s prefill. The other 77% is compute, and no chunking changes it.
///
///     That is where the win actually was. Sharing each weight's DECODE across a chunk
///     (`ops::mmw_many`, plus expert-major grouping in `moe_many`) took the same 128
///     tokens at the same width from 89.7 s to 64.3 s, bit-identical, with bytes read
///     unchanged at 14.5 GB:
///
///         width      before      after      GB read
///             8      89.7 s     64.3 s        14.7      <- 1.39x, and the default
///            32      99.1 s    139.9 s        21.5
///           128      91.3 s    189.2 s        13.0
///
///     Note what widening does AFTER that change: it gets worse, not better. Batched
///     decode is a cache-residency trick, and a wide chunk defeats it -- the activation
///     block stops fitting, and `token_tile` can only bound the damage, not remove it. So
///     the sweep rule and the compute both point at the same modest width, for unrelated
///     reasons. Do not widen this without re-running the measurement.
/// ```
pub fn prefill_width(
    nslot: usize,
    n_experts: usize,
    topk: usize,
    hidden: usize,
    n_layers: usize,
    ffn_width: usize,
) -> usize {
    if n_experts == 0 || topk == 0 || n_layers == 0 {
        return 1;
    }
    // One sweep's worth of experts must survive until the next chunk asks for them again.
    let budget = (nslot / n_layers).max(1) as f64;
    let n = n_experts as f64;
    // Activations are linear in k, so this bound is a division rather than a search.
    // Per token of chunk: 16 bytes per hidden element for the block activations, 4 per
    // routed expert slot (`moe_many` parks each token's per-expert contribution so it can
    // sum them back in route order), and 20 per FFN element -- `expert_fwd_many` holds
    // gate, up, act and the re-paired gu across the chunk.
    //
    // The FFN term is not a rounding detail on a DENSE model. Qwen3.8-27B's width is 17408
    // against a hidden of 5120, so the intermediates are 3.4x the block activations and
    // dominate the budget; ignoring them sized a chunk at 2621 tokens that would have asked
    // for roughly 1.2 GB of scratch on an 11 GB machine.
    let per_tok = (16 + 4 * topk) * hidden + 20 * ffn_width;
    let act = ACT_BUDGET.checked_div(per_tok).unwrap_or(MAX_WIDTH);
    let hi = act.clamp(1, MAX_WIDTH);
    (1..=hi)
        .rev()
        .find(|&k| n * (1.0 - (1.0 - 1.0 / n).powi((k * topk) as i32)) <= budget)
        .unwrap_or(1)
}

/// Activation bytes a prompt chunk may hold. `step_many` holds `xs` and `branch` at
/// `k * hidden` f32 and `moe_many` holds `hs` at the same, so roughly `16 * k * hidden`
/// bytes live for the whole pass. This only binds when the cache is too small for the
/// sweep rule to bind first, which is the regime where wider is unambiguously better.
const ACT_BUDGET: usize = 256 << 20;

/// A ceiling no arithmetic should be trusted past. Not a tuning knob -- a backstop against
/// a zero or absurd `hidden` turning into a multi-gigabyte allocation.
const MAX_WIDTH: usize = 8192;

/// How many candidates to queue per layer. SpecPrefetch (arxiv 2607.24787) found the
/// optimum near "about two transfers past the native top-k" for K=6, which is where 8
/// comes from; more than that and the reads stop fitting inside the overlap window and
/// start evicting things the current layer still needs.
const PREFETCH_M: usize = 8;

/// The top-`m` experts layer `l` would route to, given its normalised input. Uses the
/// layer's real gate, so it is exact except that `x` is the PREVIOUS layer's residual
/// rather than this layer's post-attention one.
fn predict_experts(
    x: &[f32],
    lay: &Layer,
    d: &MoeDimsV4,
    l: usize,
    id: u32,
    m: usize,
) -> Vec<usize> {
    // Hash-routed layers are not a prediction at all: the experts follow from the token
    // id, so they are known exactly and for free.
    if l < d.n_hash_layers {
        if let Some(t2e) = lay.tid2eid.as_deref() {
            let base = id as usize * d.topk;
            if base + d.topk <= t2e.len() {
                return t2e[base..base + d.topk].iter().map(|&v| v as usize).collect();
            }
        }
    }
    let m = m.min(d.n_experts);
    let mut score = vec![0f32; d.n_experts];
    for ex in 0..d.n_experts {
        let row = &lay.gate[ex * d.hidden..][..d.hidden];
        let mut acc = 0.0f64;
        for i in 0..d.hidden {
            acc += row[i] as f64 * x[i] as f64;
        }
        score[ex] = acc as f32 + lay.bias.as_ref().map_or(0.0, |b| b[ex]);
    }
    let mut idx: Vec<usize> = (0..d.n_experts).collect();
    idx.select_nth_unstable_by(m - 1, |&a, &b| {
        score[b].partial_cmp(&score[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(m);
    idx
}

/// n-gram draft: find the most recent earlier occurrence of the last `n` tokens and
/// propose whatever followed it.
///
/// Costs nothing -- no model, no I/O -- which is what makes it the right drafter here.
/// A drafted token that is wrong costs only its share of one batched verification, and
/// the verification is dominated by expert reads that the batch amortises anyway.
fn draft_ngram(ids: &[u32], k: usize, n: usize) -> Vec<u32> {
    if ids.len() < n + 1 || k == 0 {
        return Vec::new();
    }
    let tail = &ids[ids.len() - n..];
    // Scan backwards: the most recent match is the best predictor of what comes next.
    for start in (0..ids.len() - n).rev() {
        if &ids[start..start + n] == tail {
            let from = start + n;
            let take = k.min(ids.len() - from);
            if take > 0 {
                return ids[from..from + take].to_vec();
            }
        }
    }
    Vec::new()
}

#[cfg(test)]
mod draft_tests {
    use super::{draft_ngram, prefill_width};

    /// The property that matters: one chunk's per-layer union must stay well inside the
    /// cache. Asserted against the occupancy formula directly rather than against
    /// hand-picked answers, so the test does not just restate the implementation.
    #[test]
    fn a_chunks_whole_sweep_has_to_fit_in_the_cache() {
        for &nslot in &[8usize, 37, 119, 224, 373, 1000, 2422] {
            let k = prefill_width(nslot, 256, 6, 7168, 43, 2048);
            assert!(k >= 1, "{nslot} slots gave chunk {k}");
            let n = 256.0f64;
            let union = n * (1.0 - (1.0 - 1.0 / n).powi((k * 6) as i32));
            assert!(
                k == 1 || union * 43.0 <= nslot as f64,
                "{nslot} slots: chunk {k} sweeps {:.0} experts, over budget",
                union * 43.0
            );
        }
    }

    /// A bigger cache must never give a SMALLER chunk.
    #[test]
    fn the_width_is_monotonic_in_cache_size() {
        let mut prev = 0;
        for &nslot in &[8usize, 37, 60, 119, 224, 373, 4096, 40_000] {
            let k = prefill_width(nslot, 256, 6, 7168, 43, 2048);
            assert!(k >= prev, "{nslot} slots gave {k} after {prev}");
            prev = k;
        }
        assert_eq!(prefill_width(1, 256, 6, 7168, 43, 2048), 1, "a tiny cache falls back to one token");
    }

    /// The measurement, as an assertion.
    ///
    /// Qwen3.6-35B with a 5 GB cache: 2422 slots, 40 layers, 256 experts, top-8. Timed at
    /// widths 8 / 32 / 128, width 8 read the fewest bytes and 32 read 35% MORE despite
    /// deduplicating better inside each chunk. The sweep rule has to reproduce that, or it
    /// is not the rule that explains the data.
    #[test]
    fn the_measured_optimum_on_qwen35moe_is_reproduced() {
        assert_eq!(prefill_width(2422, 256, 8, 2048, 40, 512), 8, "the width that measured best");
        // And for the reason claimed, not by coincidence: one more token overflows a sweep.
        let sweep = |k: usize| 256.0 * (1.0 - (1.0f64 - 1.0 / 256.0).powi((k * 8) as i32)) * 40.0;
        assert!(sweep(8) <= 2422.0, "a chunk of 8 sweeps {:.0} of 2422 slots", sweep(8));
        assert!(sweep(9) > 2422.0, "a chunk of 9 sweeps {:.0}, over", sweep(9));
    }

    /// A DENSE model is one expert per layer that every token takes, so `U(k) = 1` for any
    /// width and the sweep bound never binds. Width is then the activation budget alone --
    /// and on Qwen3.8-27B the FFN intermediates, not the block activations, are what fills
    /// it: 17408 wide against a hidden of 5120.
    #[test]
    fn a_dense_model_is_bounded_by_its_ffn_intermediates() {
        // 64 layers, 3.1 GB of cache over 165 MB slots is ~19 slots.
        let k = prefill_width(19, 1, 1, 5120, 64, 17408);
        assert!(k > 1, "a dense chunk must batch, or prefill loses `mmw_many` entirely");
        // The FFN term dominates: 20*17408 is over four times (16+4)*5120.
        let per_tok = (16 + 4) * 5120 + 20 * 17408;
        assert_eq!(k, (256 << 20) / per_tok);
        // 20*17408 = 348160 of intermediates against 20*5120 = 102400 of activations: the
        // FFN is 3.4x the rest, which is why leaving it out of the budget mattered.
        // And the chunk's scratch stays inside the budget it was sized against.
        assert!(k * per_tok <= (256 << 20), "width {k} would ask for more than 256 MB");
        // Ignoring the FFN term is the bug this guards: it would have allowed 2621 tokens.
        assert_eq!(k, 595, "the measured budget for this geometry");
        assert!(k < 2621, "and far under the 2621 an FFN-blind budget would have allowed");
    }

    /// When the cache cannot hold even a one-token sweep there is no cross-chunk reuse to
    /// protect, so the activation budget is what binds -- and wider is then free.
    #[test]
    fn a_cache_too_small_for_any_sweep_falls_back_to_the_activation_budget() {
        let k = prefill_width(usize::MAX / 64, 256, 8, 2048, 40, 512);
        assert_eq!(k, (256 << 20) / ((16 + 4 * 8) * 2048 + 20 * 512), "activation-bounded");
        for &h in &[2048usize, 5120, 7168, 8192] {
            for &topk in &[2usize, 8, 10] {
                let k = prefill_width(usize::MAX / 64, 256, topk, h, 40, 512);
                assert!(
                    ((16 + 4 * topk) * h + 20 * 512) * k <= (256 << 20),
                    "hidden {h} top-{topk} gave width {k}, over budget"
                );
            }
        }
        // A degenerate hidden must not turn into an unbounded allocation.
        assert_eq!(prefill_width(usize::MAX / 64, 256, 8, 0, 40, 0), 8192, "backstopped");
    }

    #[test]
    fn it_proposes_the_continuation_of_the_most_recent_match() {
        // "a b c" appeared once, followed by "d e".
        let ids = vec![1, 2, 3, 4, 5, 9, 9, 1, 2, 3];
        assert_eq!(draft_ngram(&ids, 2, 3), vec![4, 5]);
    }

    #[test]
    fn it_prefers_the_most_recent_occurrence() {
        // "7 8" is followed by 100 early on and by 200 later; the later one wins.
        let ids = vec![7, 8, 100, 0, 7, 8, 200, 0, 7, 8];
        assert_eq!(draft_ngram(&ids, 1, 2), vec![200]);
    }

    #[test]
    fn no_match_drafts_nothing_rather_than_guessing() {
        let ids = vec![1, 2, 3, 4, 5];
        assert!(draft_ngram(&ids, 4, 3).is_empty());
    }

    #[test]
    fn a_short_history_drafts_nothing() {
        assert!(draft_ngram(&[1, 2], 4, 3).is_empty());
    }

    // The drafter must never look at tokens that do not exist yet.
    #[test]
    fn it_never_proposes_past_the_end_of_history() {
        let ids = vec![5, 6, 7, 5, 6];
        let d = draft_ngram(&ids, 8, 2);
        assert!(d.len() <= ids.len(), "drafted {} from {} tokens", d.len(), ids.len());
    }
}
