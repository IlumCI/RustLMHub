#!/usr/bin/env python
"""
k3_ref.py - a pure-PyTorch reference implementation of Kimi K3.

WHY THIS EXISTS
    The released reference, modeling_kimi_linear.py, imports its KDA recurrence,
    its short convolution and its gated RMSNorm from `fla` (flash-linear-attention),
    which is Triton-backed and therefore needs a CUDA GPU. That makes it unusable as
    a CPU oracle and unreadable as a porting target.

    This file reimplements the same mathematics in plain PyTorch, runs on CPU, and
    is written to be read line by line while writing C. It is deliberately naive:
    no fusion, no chunking tricks, no in-place cleverness. Correctness first.

    Two-tier validation strategy:
        tier 1  this file  vs  the real modeling_kimi_linear.py on a GPU box
        tier 2  the C engine  vs  this file, on any box

PROVENANCE OF EVERY DESIGN DECISION
    Facts below were verified against the released code, the released checkpoint
    headers, and the fla kernels. Citations are to files under the project root.

    kimi_k3_hf/files/modeling_kimi_linear.py
        :64-82    SituAndMul. situ_a = beta*tanh(gate/beta)*sigmoid(gate); the
                  sigmoid takes the UNCAPPED gate. up is separately capped by
                  linear_beta*tanh(up/linear_beta). Bound is beta*linear_beta = 100.
        :226-239  KimiRMSNorm, eps default 1e-6, but the model passes rms_norm_eps.
        :357,359  MLA q_head_dim = qk_nope + qk_rope = 192, softmax scale is
                  q_head_dim ** -0.5, NOT qk_nope ** -0.5.
        :396,403  MLA asserts use_nope and sets rotary_emb = None: no rotation is
                  ever applied, yet the 64 "rope" dims still exist and are still
                  stored in the KV cache. Deleting them is a silent, fatal bug.
        :435-437  k_rot is computed for ONE head and broadcast to all heads.
        :504-518  ShortConvolution(kernel_size=4, activation='silu'): the SiLU is
                  FUSED INSIDE the convolution, so the order is Linear -> Conv -> SiLU.
        :520-521  A_log is declared with num_heads elements: PER HEAD.
        :523-524  f_a_proj/f_b_proj form ONE shared low-rank pair (rank = head_dim)
                  that feeds all heads. No per-head bottleneck, no activation between.
        :529      b_proj maps hidden -> num_heads, so beta is a PER-HEAD SCALAR.
        :601-607  g = f_b_proj(f_a_proj(x)) then rearranged head-major, channel-minor.
        :666-876  MoE: router reads FULL width before the latent down-projection;
                  the frozen bias affects SELECTION ONLY; combining weights are
                  gathered from the UNBIASED sigmoid scores, renormalised, then
                  scaled by routed_scaling_factor.
        :816-837  Latent path: down_proj -> experts -> RMSNorm on the AGGREGATE ->
                  up_proj, then shared experts on the ORIGINAL input, added UNWEIGHTED.
        :995-998  AttnRes boundary: fires when layer_idx % attn_res_block_size == 0,
                  appends prefix_sum to the stack and then sets prefix_sum = None.
        :1023-1046 The residual wiring reproduced exactly in DecoderLayer.forward below.
        :1075-1088 _apply_attn_res: keys are RMSNormed, VALUES ARE RAW, one scalar
                  logit per source, softmax, weighted average. The norm gain and the
                  projection collapse into a single vector, which C should fold at load.

    .venv/Lib/site-packages/fla/ops/kda/gate.py
        :60-69    With a lower bound the decay is
                      g = lower_bound * sigmoid(exp(A_log[h]) * (z + dt_bias))
                  and A_log is indexed BY HEAD (A_log.view(H,1), and A_log + i_h in
                  the Triton path). Without a lower bound the same kernel computes
                  -exp(A)*softplus(z), which is the Kimi Linear / GDN form K3 replaced.

    .venv/Lib/site-packages/fla/ops/kda/naive.py
        :52       q is pre-scaled by K ** -0.5.
        :59-63    The recurrence, and the ordering is load bearing:
                      S = S * exp(g)                      decay first
                      S = S + outer(beta*k, v - S^T k)    then the delta write
                      o = S^T q                           then read the UPDATED state

    Checkpoint headers (read by HTTP range read of the safetensors shards):
        A_log ships as 128 F32 values but elements 96..127 are EXACTLY ZERO. It is
        semantically per-head, zero-padded to head_dim, presumably for aligned loads.
        This file stores it at num_heads and the loader must slice.

NOT IMPLEMENTED, deliberately
    Vision (MoonViT-V2), chunked prefill, MXFP4 dequantisation, speculative decoding.
    The released checkpoint has num_nextn_predict_layers = 0, so no draft head exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- config
@dataclass
class K3Config:
    """Defaults are the REAL Kimi K3 values from kimi_k3_hf/files/config.json."""
    hidden_size: int = 7168
    num_hidden_layers: int = 93
    vocab_size: int = 163840
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False

    # ---- KDA (linear_attn_config) ----
    kda_num_heads: int = 96
    kda_head_dim: int = 128
    short_conv_kernel_size: int = 4
    gate_lower_bound: float = -5.0

    # ---- Gated MLA ----
    num_attention_heads: int = 96
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64          # exists but is NEVER rotated (NoPE)
    v_head_dim: int = 128
    mla_use_output_gate: bool = True

    # ---- MoE ----
    num_experts: int = 896
    num_experts_per_token: int = 16
    num_shared_experts: int = 2
    routed_expert_hidden_size: int = 3584     # the latent width
    moe_intermediate_size: int = 3072
    routed_scaling_factor: float = 1.0
    moe_renormalize: bool = True
    latent_moe_use_norm: bool = True

    # ---- dense layer ----
    first_k_dense_replace: int = 1
    intermediate_size: int = 33792

    # ---- AttnRes ----
    attn_res_block_size: int = 12

    # ---- SiTU-GLU ----
    situ_beta: float = 4.0
    situ_linear_beta: float = 25.0

    # ---- layer map. ONE-BASED in the released config, mirrored here. ----
    # configuration_kimi_k3.py:152-156 tests (layer_idx + 1) in kda_layers.
    full_attn_layers: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.full_attn_layers:
            # 3 KDA then 1 MLA, repeating, plus a trailing MLA at the very end.
            fa = [i for i in range(4, self.num_hidden_layers + 1) if i % 4 == 0]
            if self.num_hidden_layers not in fa:
                fa.append(self.num_hidden_layers)
            self.full_attn_layers = sorted(fa)

    def is_mla(self, layer_idx: int) -> bool:
        """layer_idx is ZERO-BASED here; the config list is ONE-BASED."""
        return (layer_idx + 1) in self.full_attn_layers

    def is_kda(self, layer_idx: int) -> bool:
        return not self.is_mla(layer_idx)

    def is_dense(self, layer_idx: int) -> bool:
        return layer_idx < self.first_k_dense_replace


# --------------------------------------------------------------------------- pieces
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        d = x.float()
        d = d * torch.rsqrt(d.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * d).to(x.dtype)


def situ_glu(x: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
    """modeling_kimi_linear.py:75-82. Split in half: gate then up.

    The sigmoid receives the UNCAPPED gate. Feeding it the capped value instead
    still produces a bounded, plausible-looking function and is WRONG.
    """
    d = x.shape[-1] // 2
    gate = x[..., :d].float()
    up = x[..., d:].float()
    a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (a * up).to(x.dtype)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.float().pow(2).sum(-1, keepdim=True) + eps).to(x.dtype)


class ShortConv(nn.Module):
    """Causal depthwise conv1d with a fused SiLU. modeling_kimi_linear.py:504-518.

    Checkpoint shape is [channels, 1, kernel], the middle 1 marking depthwise.
    During decode the engine must carry (kernel - 1) past inputs per channel; that
    ring buffer is a second piece of per-sequence state beyond the recurrent matrix
    and is easy to forget.
    """

    def __init__(self, channels: int, kernel: int):
        super().__init__()
        self.channels, self.kernel = channels, kernel
        self.weight = nn.Parameter(torch.zeros(channels, 1, kernel))

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None):
        """x is [B, T, C]. state is [B, C, kernel-1] of previous inputs."""
        B, _T, C = x.shape
        xt = x.transpose(1, 2)                                   # [B, C, T]
        pad = state if state is not None else xt.new_zeros(B, C, self.kernel - 1)
        xp = torch.cat([pad, xt], dim=-1)                        # left pad = causal
        y = F.conv1d(xp, self.weight, groups=C)                  # [B, C, T]
        new_state = xp[..., -(self.kernel - 1):] if self.kernel > 1 else None
        return F.silu(y.transpose(1, 2)), new_state              # SiLU is fused here


# --------------------------------------------------------------------------- KDA
class KimiDeltaAttention(nn.Module):
    """69 of the 93 layers. Fixed-size recurrent state, no growing KV cache."""

    def __init__(self, c: K3Config):
        super().__init__()
        self.c = c
        H, D, E = c.kda_num_heads, c.kda_head_dim, c.hidden_size
        P = H * D
        self.H, self.D, self.P = H, D, P

        self.q_proj = nn.Linear(E, P, bias=False)
        self.k_proj = nn.Linear(E, P, bias=False)
        self.v_proj = nn.Linear(E, P, bias=False)
        self.q_conv1d = ShortConv(P, c.short_conv_kernel_size)
        self.k_conv1d = ShortConv(P, c.short_conv_kernel_size)
        self.v_conv1d = ShortConv(P, c.short_conv_kernel_size)

        # ONE shared low-rank pair of rank head_dim feeds every head. :523-524
        self.f_a_proj = nn.Linear(E, D, bias=False)
        self.f_b_proj = nn.Linear(D, P, bias=False)
        self.A_log = nn.Parameter(torch.zeros(H))        # PER HEAD. :520-521
        self.dt_bias = nn.Parameter(torch.zeros(P))      # per (head, channel)
        self.b_proj = nn.Linear(E, H, bias=False)        # beta: per-head SCALAR. :529
        self.g_proj = nn.Linear(E, P, bias=False)        # full-rank output gate
        self.o_norm = RMSNorm(D, c.rms_norm_eps)         # head-wise
        self.o_proj = nn.Linear(P, E, bias=False)

    def decay(self, x):
        """fla/ops/kda/gate.py:60-69. Returns log-decay g in (lower_bound, 0)."""
        H, D = self.H, self.D
        z = self.f_b_proj(self.f_a_proj(x)).float()                  # [B,T,P]
        z = z + self.dt_bias.float()                                 # per (h,d)
        z = z.view(*z.shape[:-1], H, D)
        a = self.A_log.float().exp().view(H, 1)                      # per HEAD
        return self.c.gate_lower_bound * torch.sigmoid(a * z)        # [B,T,H,D]

    def forward(self, x, state=None):
        """state = (S [B,H,D,D], conv_q, conv_k, conv_v) or None."""
        B, T, _ = x.shape
        H, D = self.H, self.D
        S = state[0] if state else x.new_zeros(B, H, D, D, dtype=torch.float32)
        cq, ck, cv = (state[1], state[2], state[3]) if state else (None, None, None)

        # Linear -> Conv (SiLU fused) -> L2Norm on q,k only. Eq 2.
        q, cq = self.q_conv1d(self.q_proj(x), cq)
        k, ck = self.k_conv1d(self.k_proj(x), ck)
        v, cv = self.v_conv1d(self.v_proj(x), cv)
        q = l2norm(q.view(B, T, H, D))
        k = l2norm(k.view(B, T, H, D))
        v = v.view(B, T, H, D)

        beta = torch.sigmoid(self.b_proj(x).float())                 # [B,T,H]
        g = self.decay(x)                                            # [B,T,H,D]
        alpha = g.exp()

        q = q.float() * (D ** -0.5)                                  # naive.py:52
        k, v = k.float(), v.float()

        o = x.new_zeros(B, T, H, D, dtype=torch.float32)
        for t in range(T):                                           # naive.py:59-63
            S = S * alpha[:, t].unsqueeze(-1)                        # decay rows
            u = torch.einsum('bhk,bhkv->bhv', k[:, t], S)            # S^T k
            w = beta[:, t].unsqueeze(-1) * (v[:, t] - u)             # delta error
            S = S + torch.einsum('bhk,bhv->bhkv', k[:, t], w)
            o[:, t] = torch.einsum('bhk,bhkv->bhv', q[:, t], S)      # updated state

        y = self.o_norm(o).view(B, T, self.P)                        # head-wise norm
        y = y * torch.sigmoid(self.g_proj(x))                        # then gate. Eq 6
        return self.o_proj(y.to(x.dtype)), (S, cq, ck, cv)


# --------------------------------------------------------------------------- MLA
class GatedMLA(nn.Module):
    """24 of the 93 layers. NoPE: the 64 rope dims exist but are never rotated."""

    def __init__(self, c: K3Config):
        super().__init__()
        self.c = c
        E, Hn = c.hidden_size, c.num_attention_heads
        self.H = Hn
        self.qk_nope, self.qk_rope, self.vh = c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim
        self.qh = self.qk_nope + self.qk_rope                        # 192
        self.scale = self.qh ** -0.5                                 # :359

        self.q_a_proj = nn.Linear(E, c.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(c.q_lora_rank, c.rms_norm_eps)
        self.q_b_proj = nn.Linear(c.q_lora_rank, Hn * self.qh, bias=False)
        # one projection emits the latent AND the shared rope-slot key
        self.kv_a_proj_with_mqa = nn.Linear(E, c.kv_lora_rank + self.qk_rope, bias=False)
        self.kv_a_layernorm = RMSNorm(c.kv_lora_rank, c.rms_norm_eps)
        self.kv_b_proj = nn.Linear(c.kv_lora_rank, Hn * (self.qk_nope + self.vh), bias=False)
        self.o_proj = nn.Linear(Hn * self.vh, E, bias=False)
        self.g_proj = nn.Linear(E, Hn * self.vh, bias=False) if c.mla_use_output_gate else None

    def forward(self, x, cache=None):
        B, T, _ = x.shape
        H = self.H
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x))).view(B, T, H, self.qh)
        q_nope, q_rope = q.split([self.qk_nope, self.qk_rope], dim=-1)

        c_kv = self.kv_a_proj_with_mqa(x)
        latent, k_rope = c_kv.split([self.c.kv_lora_rank, self.qk_rope], dim=-1)
        latent = self.kv_a_layernorm(latent)
        # NO ROTATION APPLIED ANYWHERE. rotary_emb is None. :403

        if cache is not None and cache[0] is not None:
            latent = torch.cat([cache[0], latent], dim=1)
            k_rope = torch.cat([cache[1], k_rope], dim=1)
        new_cache = (latent, k_rope)

        kv = self.kv_b_proj(latent).view(B, -1, H, self.qk_nope + self.vh)
        k_nope, v = kv.split([self.qk_nope, self.vh], dim=-1)
        k_rope_b = k_rope.unsqueeze(2).expand(-1, -1, H, -1)          # one head -> all. :437

        qs = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2).float()
        ks = torch.cat([k_nope, k_rope_b], dim=-1).transpose(1, 2).float()
        vs = v.transpose(1, 2).float()

        att = torch.matmul(qs, ks.transpose(-1, -2)) * self.scale
        Tq, Tk = att.shape[-2], att.shape[-1]
        causal = torch.ones(Tq, Tk, dtype=torch.bool, device=x.device).tril(Tk - Tq)
        att = att.masked_fill(~causal, float('-inf')).softmax(-1)
        y = torch.matmul(att, vs).transpose(1, 2).reshape(B, T, H * self.vh)

        if self.g_proj is not None:
            y = y * torch.sigmoid(self.g_proj(x)).float()             # gate, no norm
        return self.o_proj(y.to(x.dtype)), new_cache


# --------------------------------------------------------------------------- MoE
class Expert(nn.Module):
    """w1 = gate, w3 = up, w2 = down. Verified shapes: w1/w3 [inter, latent],
    w2 [latent, inter]."""

    def __init__(self, din: int, dinter: int):
        super().__init__()
        self.w1 = nn.Linear(din, dinter, bias=False)
        self.w3 = nn.Linear(din, dinter, bias=False)
        self.w2 = nn.Linear(dinter, din, bias=False)

    def forward(self, x, beta, linear_beta):
        return self.w2(situ_glu(torch.cat([self.w1(x), self.w3(x)], -1), beta, linear_beta))


class LatentMoE(nn.Module):
    def __init__(self, c: K3Config):
        super().__init__()
        self.c = c
        E, L, I = c.hidden_size, c.routed_expert_hidden_size, c.moe_intermediate_size
        self.gate = nn.Linear(E, c.num_experts, bias=False)          # reads FULL width
        self.e_score_correction_bias = nn.Parameter(torch.zeros(c.num_experts))
        self.down = nn.Linear(E, L, bias=False)
        self.up = nn.Linear(L, E, bias=False)
        self.norm = RMSNorm(L, c.rms_norm_eps) if c.latent_moe_use_norm else None
        self.experts = nn.ModuleList([Expert(L, I) for _ in range(c.num_experts)])
        self.shared = Expert(E, I * c.num_shared_experts)

    def route(self, x):
        """:707-757. Bias steers SELECTION; weights come from UNBIASED scores."""
        logits = F.linear(x.float(), self.gate.weight.float())
        scores = logits.sigmoid()
        biased = scores + self.e_score_correction_bias.float()
        idx = torch.topk(biased, self.c.num_experts_per_token, dim=-1, sorted=False).indices
        w = scores.gather(1, idx)
        if self.c.moe_renormalize:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        return idx, w * self.c.routed_scaling_factor

    def forward(self, x):
        B, T, E = x.shape
        identity = x                                                  # ORIGINAL width
        flat = x.view(-1, E)
        idx, w = self.route(flat)
        z = self.down(flat)                                           # -> latent
        out = torch.zeros_like(z)
        for n in range(flat.shape[0]):                                # naive on purpose
            for j in range(idx.shape[1]):
                out[n] += w[n, j].to(z.dtype) * self.experts[idx[n, j]](
                    z[n:n + 1], self.c.situ_beta, self.c.situ_linear_beta)[0]
        if self.norm is not None:
            out = self.norm(out)                                      # on the AGGREGATE
        y = self.up(out).view(B, T, E)
        return y + self.shared(identity, self.c.situ_beta, self.c.situ_linear_beta)


class DenseMLP(nn.Module):
    def __init__(self, c: K3Config):
        super().__init__()
        self.c = c
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        g = torch.cat([self.gate_proj(x), self.up_proj(x)], -1)
        return self.down_proj(situ_glu(g, self.c.situ_beta, self.c.situ_linear_beta))


# --------------------------------------------------------------------- AttnRes
def apply_attn_res(prefix_sum, block_residual, proj_w, norm_w, eps):
    """modeling_kimi_linear.py:1075-1088.

    Keys are RMSNormed; the VALUES averaged at the end are the RAW sources.
    score_weight = norm.weight * proj.weight is a single vector: fold it at load.
    """
    v = torch.cat([block_residual, prefix_sum.unsqueeze(1)], dim=1).float()
    k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    scores = (k * (norm_w.float() * proj_w.float())).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probs, v).squeeze(1).to(prefix_sum.dtype)


class DecoderLayer(nn.Module):
    def __init__(self, c: K3Config, idx: int):
        super().__init__()
        self.c, self.idx = c, idx
        self.is_mla = c.is_mla(idx)
        self.self_attn = GatedMLA(c) if self.is_mla else KimiDeltaAttention(c)
        self.mlp = DenseMLP(c) if c.is_dense(idx) else LatentMoE(c)
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.self_attention_res_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.self_attention_res_proj = nn.Linear(c.hidden_size, 1, bias=False)
        self.mlp_res_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.mlp_res_proj = nn.Linear(c.hidden_size, 1, bias=False)

    def forward(self, h, block_residual, state=None):
        """Mirrors _forward_attn_residual, :984-1046, statement for statement."""
        B, T, E = h.shape
        eps = self.c.rms_norm_eps
        prefix_sum = h

        if block_residual is not None and block_residual.shape[1] > 0:
            h = apply_attn_res(prefix_sum.view(-1, E), block_residual,
                               self.self_attention_res_proj.weight.squeeze(0),
                               self.self_attention_res_norm.weight, eps).view(B, T, E)

        if self.idx % self.c.attn_res_block_size == 0:
            block_residual = torch.cat(
                [block_residual, prefix_sum.view(-1, E).unsqueeze(1)], dim=1)
            prefix_sum = None                                        # <- reset. :998

        h = self.input_layernorm(h)
        h, state = self.self_attn(h, state)
        prefix_sum = h if prefix_sum is None else prefix_sum + h

        h = apply_attn_res(prefix_sum.view(-1, E), block_residual,
                           self.mlp_res_proj.weight.squeeze(0),
                           self.mlp_res_norm.weight, eps).view(B, T, E)
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        prefix_sum = h if prefix_sum is None else prefix_sum + h
        return prefix_sum, block_residual, state


class K3Model(nn.Module):
    def __init__(self, c: K3Config):
        super().__init__()
        self.c = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(c, i) for i in range(c.num_hidden_layers)])
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.output_attn_res_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.output_attn_res_proj = nn.Linear(c.hidden_size, 1, bias=False)

    def forward(self, ids, states=None):
        B, T = ids.shape
        h = self.embed_tokens(ids)
        block_residual = h.new_zeros(B * T, 0, self.c.hidden_size)
        states = states or [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            h, block_residual, states[i] = layer(h, block_residual, states[i])
        h = apply_attn_res(h.view(-1, self.c.hidden_size), block_residual,
                           self.output_attn_res_proj.weight.squeeze(0),
                           self.output_attn_res_norm.weight,
                           self.c.rms_norm_eps).view(B, T, self.c.hidden_size)
        return self.lm_head(self.norm(h)), states

    @torch.no_grad()
    def greedy(self, ids, n_new: int):
        """Full re-forward each step: slow, but it exercises the same code path as
        prefill and cannot drift from it. Correctness over speed, by design."""
        out = ids
        for _ in range(n_new):
            logits, _ = self.forward(out)
            out = torch.cat([out, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        return out


def tiny_config(**kw) -> K3Config:
    """A small model that still exposes every mechanism.

    13 layers with attn_res_block_size 3 gives boundaries at 0,3,6,9,12, so five
    snapshots and four COMPLETE blocks. A 5-layer model cannot surface AttnRes bugs
    at all, which is why the tiny fixture must be this deep.
    """
    c = dict(hidden_size=128, num_hidden_layers=13, vocab_size=256, rms_norm_eps=1e-5,
             kda_num_heads=4, kda_head_dim=16, short_conv_kernel_size=4,
             num_attention_heads=4, q_lora_rank=64, kv_lora_rank=32,
             qk_nope_head_dim=24, qk_rope_head_dim=8, v_head_dim=16,
             num_experts=8, num_experts_per_token=2, num_shared_experts=2,
             routed_expert_hidden_size=64, moe_intermediate_size=48,
             first_k_dense_replace=1, intermediate_size=96,
             attn_res_block_size=3, full_attn_layers=[4, 8, 12, 13])
    c.update(kw)
    return K3Config(**c)
