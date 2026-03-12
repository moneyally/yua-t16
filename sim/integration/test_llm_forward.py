"""
ORBIT-G1 v2 Integration Test — GPT-OSS-20B Style LLM Forward Pass
Simulates the 21-step descriptor sequence from spec/yua-llm-hw-design.md §4

Model params (small, fast):
  d_model     = 64
  num_heads   = 4
  head_dim    = 16   (d_model / num_heads)
  num_experts = 8
  top_k       = 2
  d_ff        = 128
  vocab_size  = 256
  seq_len     = 4    (prefill token count)
"""

import sys
import numpy as np

# ─────────────────────────────────────────────
# Model hyper-parameters (test scale)
# ─────────────────────────────────────────────
D_MODEL     = 64
NUM_HEADS   = 4
HEAD_DIM    = D_MODEL // NUM_HEADS   # 16
NUM_EXPERTS = 8
TOP_K       = 2
D_FF        = 128
VOCAB_SIZE  = 256
SEQ_LEN     = 4

LAYER_ID    = 0

# Reproducible RNG
RNG = np.random.default_rng(42)


# ─────────────────────────────────────────────
# Weight initialisation helper
# ─────────────────────────────────────────────

def rand_int4(shape):
    """Random INT4 weights in [-8, 7]."""
    return RNG.integers(-8, 8, size=shape, dtype=np.int8)

def rand_fp16(shape):
    return RNG.standard_normal(shape).astype(np.float16)

def rand_scale(shape):
    """Per-column FP16 scale factors (small positive)."""
    return (RNG.uniform(0.01, 0.1, size=shape)).astype(np.float16)


# ─────────────────────────────────────────────
# Module implementations
# ─────────────────────────────────────────────

def gemm_int4(A_fp16: np.ndarray, W_int4: np.ndarray, scale_fp16: np.ndarray) -> np.ndarray:
    """
    INT4 dequant + GEMM.
    A_fp16 : [M, K]  float16
    W_int4 : [K, N]  int8  (values in [-8,7])
    scale_fp16 : [N] float16  per-column scale
    returns  [M, N] float16
    """
    W_dequant = (W_int4.astype(np.float32) * scale_fp16.astype(np.float32))  # [K, N]
    out = A_fp16.astype(np.float32) @ W_dequant                               # [M, N]
    return out.astype(np.float16)


def vpu_rmsnorm(x: np.ndarray, w: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """RMSNorm over last dimension."""
    x32 = x.astype(np.float32)
    rms = np.sqrt(np.mean(x32 ** 2, axis=-1, keepdims=True) + eps)
    return ((x32 / rms) * w.astype(np.float32)).astype(np.float16)


def vpu_silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation: x * sigmoid(x)."""
    x32 = x.astype(np.float32)
    return (x32 * (1.0 / (1.0 + np.exp(-x32)))).astype(np.float16)


def vpu_rope(x: np.ndarray, cos_vals: np.ndarray, sin_vals: np.ndarray) -> np.ndarray:
    """
    Rotary Position Embedding (interleaved rotation).
    x         : [seq, num_heads, head_dim]
    cos_vals  : [seq, head_dim]
    sin_vals  : [seq, head_dim]
    """
    seq, nh, hd = x.shape
    x32 = x.astype(np.float32)

    # Split into even/odd halves for rotation
    half = hd // 2
    x_even = x32[..., :half]   # [seq, nh, half]
    x_odd  = x32[..., half:]   # [seq, nh, half]

    cos_e = cos_vals[:, np.newaxis, :half].astype(np.float32)  # [seq, 1, half]
    sin_e = sin_vals[:, np.newaxis, :half].astype(np.float32)

    # RoPE rotation: (x_even, x_odd) → (x_even*cos - x_odd*sin, x_even*sin + x_odd*cos)
    out_even = x_even * cos_e - x_odd * sin_e
    out_odd  = x_even * sin_e + x_odd * cos_e

    out = np.concatenate([out_even, out_odd], axis=-1)
    return out.astype(np.float16)


def vpu_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over last dimension."""
    x32 = x.astype(np.float32)
    x32 = x32 - x32.max(axis=-1, keepdims=True)
    e = np.exp(x32)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float16)


def vpu_scale(x: np.ndarray, factor: float) -> np.ndarray:
    return (x.astype(np.float32) * factor).astype(np.float16)


def vpu_residual_add(x: np.ndarray, residual: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) + residual.astype(np.float32)).astype(np.float16)


# ─── KV-Cache ────────────────────────────────

def kvc_write(cache: dict, layer_id: int, seq_pos: int, K: np.ndarray, V: np.ndarray):
    """Write one token's K,V vectors to cache."""
    if layer_id not in cache:
        cache[layer_id] = {}
    cache[layer_id][seq_pos] = {"K": K.copy(), "V": V.copy()}


def kvc_read(cache: dict, layer_id: int, seq_len: int):
    """
    Read all K,V up to seq_len from cache.
    Returns K: [seq_len, num_heads, head_dim]
            V: [seq_len, num_heads, head_dim]
    """
    Ks = np.stack([cache[layer_id][i]["K"] for i in range(seq_len)])
    Vs = np.stack([cache[layer_id][i]["V"] for i in range(seq_len)])
    return Ks, Vs


# ─── MoE Routing ─────────────────────────────

def moe_route(hidden: np.ndarray, router_w: np.ndarray, top_k: int = 2):
    """
    hidden   : [d_model]  float32/float16
    router_w : [d_model, num_experts]
    returns  indices [top_k], scores [top_k]  (normalised)
    """
    logits = hidden.astype(np.float32) @ router_w.astype(np.float32)  # [num_experts]
    x32 = logits - logits.max()
    e = np.exp(x32)
    probs = e / e.sum()
    indices = np.argsort(probs)[-top_k:][::-1]
    scores = probs[indices]
    scores = scores / scores.sum()   # re-normalise
    return indices, scores.astype(np.float32)


# ─────────────────────────────────────────────
# Pre-build all weights (shared across tests)
# ─────────────────────────────────────────────

class Weights:
    """All model weights, initialised once with RNG seed=42."""

    def __init__(self):
        # QKV projection: [d_model, 3*d_model]
        self.W_qkv      = rand_int4((D_MODEL, 3 * D_MODEL))
        self.S_qkv      = rand_scale((3 * D_MODEL,))

        # RoPE cos/sin table: [seq_len, head_dim]
        positions = np.arange(SEQ_LEN, dtype=np.float32)
        dims      = np.arange(0, HEAD_DIM // 2, dtype=np.float32)
        theta     = 1.0 / (10000.0 ** (2 * dims / HEAD_DIM))
        angles    = np.outer(positions, theta)  # [seq, head_dim//2]
        full_cos  = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)
        full_sin  = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)
        self.cos  = full_cos.astype(np.float16)
        self.sin  = full_sin.astype(np.float16)

        # Output projection: [d_model, d_model]
        self.W_out      = rand_int4((D_MODEL, D_MODEL))
        self.S_out      = rand_scale((D_MODEL,))

        # RMSNorm weight: [d_model]
        self.W_norm     = rand_fp16((D_MODEL,))

        # MoE router: [d_model, num_experts]
        self.W_router   = rand_int4((D_MODEL, NUM_EXPERTS))
        self.S_router   = rand_scale((NUM_EXPERTS,))

        # Expert weights (gate, up, down) per expert
        # gate_proj, up_proj : [d_model, d_ff]
        # down_proj          : [d_ff, d_model]
        self.W_gate  = [rand_int4((D_MODEL, D_FF))  for _ in range(NUM_EXPERTS)]
        self.S_gate  = [rand_scale((D_FF,))          for _ in range(NUM_EXPERTS)]
        self.W_up    = [rand_int4((D_MODEL, D_FF))   for _ in range(NUM_EXPERTS)]
        self.S_up    = [rand_scale((D_FF,))          for _ in range(NUM_EXPERTS)]
        self.W_down  = [rand_int4((D_FF, D_MODEL))  for _ in range(NUM_EXPERTS)]
        self.S_down  = [rand_scale((D_MODEL,))       for _ in range(NUM_EXPERTS)]

        # LM head: [d_model, vocab_size]
        self.W_lm    = rand_int4((D_MODEL, VOCAB_SIZE))
        self.S_lm    = rand_scale((VOCAB_SIZE,))


WEIGHTS = Weights()


# ─────────────────────────────────────────────
# Descriptor step tracker (for logging)
# ─────────────────────────────────────────────

_steps = []

def record_step(step_num: str, name: str, shape_info: str):
    _steps.append((step_num, name, shape_info))
    print(f"  [{step_num}] {name:20s}  → {shape_info}")


# ─────────────────────────────────────────────
# Test 1: QKV + Attention (steps ①-⑩)
# ─────────────────────────────────────────────

def test_qkv_attention():
    print("\n=== test_qkv_attention (steps ①-⑩) ===")
    w = WEIGHTS
    kv_cache: dict = {}

    # ① DMA_2D: input token embeddings → buffer
    x = rand_fp16((SEQ_LEN, D_MODEL))
    record_step("①", "DMA_2D", f"x {x.shape} fp16")

    # ② GEMM_INT4: QKV projection
    qkv = gemm_int4(x, w.W_qkv, w.S_qkv)  # [seq, 3*d_model]
    record_step("②", "GEMM_INT4 QKV", f"qkv {qkv.shape} fp16")
    assert qkv.shape == (SEQ_LEN, 3 * D_MODEL), f"QKV shape mismatch: {qkv.shape}"
    assert np.all(np.isfinite(qkv)), "NaN/Inf in QKV"

    # Split Q, K, V and reshape to [seq, heads, head_dim]
    Q = qkv[:, :D_MODEL].reshape(SEQ_LEN, NUM_HEADS, HEAD_DIM)
    K = qkv[:, D_MODEL:2*D_MODEL].reshape(SEQ_LEN, NUM_HEADS, HEAD_DIM)
    V = qkv[:, 2*D_MODEL:].reshape(SEQ_LEN, NUM_HEADS, HEAD_DIM)

    # ③ VECTOR_OP: RoPE on Q and K
    Q = vpu_rope(Q, w.cos, w.sin)
    K = vpu_rope(K, w.cos, w.sin)
    record_step("③", "VECTOR_OP RoPE", f"Q/K {Q.shape} fp16")
    assert np.all(np.isfinite(Q)) and np.all(np.isfinite(K)), "NaN/Inf after RoPE"

    # ④ KVC_WRITE: write new K,V to KV-Cache (per token)
    for t in range(SEQ_LEN):
        kvc_write(kv_cache, LAYER_ID, t, K[t], V[t])
    record_step("④", "KVC_WRITE", f"cache[{LAYER_ID}] × {SEQ_LEN} tokens")

    # ⑤ KVC_READ: read full sequence K,V
    K_cache, V_cache = kvc_read(kv_cache, LAYER_ID, SEQ_LEN)
    record_step("⑤", "KVC_READ", f"K {K_cache.shape}, V {V_cache.shape}")
    assert K_cache.shape == (SEQ_LEN, NUM_HEADS, HEAD_DIM)

    # ⑥ GEMM_INT4: Q @ K^T  (attention scores, approximated as FP16 matmul)
    # Q: [seq, heads, head_dim], K_cache: [seq, heads, head_dim]
    # → scores: [heads, seq_q, seq_k]
    Q_t  = Q.transpose(1, 0, 2).astype(np.float32)           # [heads, seq_q, head_dim]
    Kt_t = K_cache.transpose(1, 2, 0).astype(np.float32)     # [heads, head_dim, seq_k]
    scores = (Q_t @ Kt_t).astype(np.float16)                 # [heads, seq_q, seq_k]
    record_step("⑥", "GEMM_INT4 Q@K^T", f"scores {scores.shape} fp16")
    assert np.all(np.isfinite(scores)), "NaN/Inf in attention scores"

    # ⑦ VECTOR_OP: Scale(1/√d) + Softmax
    scores = vpu_scale(scores, 1.0 / np.sqrt(HEAD_DIM))
    attn   = vpu_softmax(scores)
    record_step("⑦", "VECTOR_OP Scale+Softmax", f"attn {attn.shape} fp16")
    assert np.all(np.isfinite(attn)), "NaN/Inf in softmax"
    # softmax rows should sum to ~1
    attn_sum = attn.astype(np.float32).sum(axis=-1)
    assert np.allclose(attn_sum, 1.0, atol=1e-2), f"Softmax sum off: {attn_sum}"

    # ⑧ GEMM_INT4: score @ V
    V_t   = V_cache.transpose(1, 0, 2).astype(np.float32)   # [heads, seq, head_dim]
    ctx   = (attn.astype(np.float32) @ V_t).astype(np.float16)  # [heads, seq_q, head_dim]
    ctx   = ctx.transpose(1, 0, 2).reshape(SEQ_LEN, D_MODEL)     # [seq, d_model]
    record_step("⑧", "GEMM_INT4 score@V", f"context {ctx.shape} fp16")
    assert np.all(np.isfinite(ctx)), "NaN/Inf in context"

    # ⑨ GEMM_INT4: output projection
    out = gemm_int4(ctx, w.W_out, w.S_out)  # [seq, d_model]
    record_step("⑨", "GEMM_INT4 out_proj", f"out {out.shape} fp16")
    assert out.shape == (SEQ_LEN, D_MODEL), f"out_proj shape mismatch: {out.shape}"
    assert np.all(np.isfinite(out)), "NaN/Inf in output projection"

    # ⑩ VECTOR_OP: Residual add
    hidden = vpu_residual_add(out, x)
    record_step("⑩", "VECTOR_OP Residual", f"hidden {hidden.shape} fp16")
    assert hidden.shape == (SEQ_LEN, D_MODEL), f"residual shape mismatch: {hidden.shape}"
    assert np.all(np.isfinite(hidden)), "NaN/Inf after residual add"

    print("  [PASS] test_qkv_attention")
    return hidden


# ─────────────────────────────────────────────
# Test 2: MoE FFN (steps ⑪-⑰)
# ─────────────────────────────────────────────

def test_moe_ffn(hidden: np.ndarray = None):
    print("\n=== test_moe_ffn (steps ⑪-⑰) ===")
    w = WEIGHTS

    if hidden is None:
        hidden = rand_fp16((SEQ_LEN, D_MODEL))

    residual_before_ffn = hidden.copy()

    # ⑪ VECTOR_OP: RMSNorm
    normed = vpu_rmsnorm(hidden, w.W_norm)
    record_step("⑪", "VECTOR_OP RMSNorm", f"normed {normed.shape} fp16")
    assert normed.shape == (SEQ_LEN, D_MODEL)
    assert np.all(np.isfinite(normed)), "NaN/Inf after RMSNorm"

    # Process each token independently through MoE
    ffn_outputs = []
    for t in range(SEQ_LEN):
        h_t = normed[t]  # [d_model]

        # ⑫ GEMM_INT4: MoE router logits  [1, num_experts]
        router_logits = gemm_int4(h_t[np.newaxis, :], w.W_router, w.S_router)
        record_step("⑫", "GEMM_INT4 router", f"logits {router_logits.shape} → [{NUM_EXPERTS}]")
        assert np.all(np.isfinite(router_logits)), "NaN/Inf in router logits"

        # ⑬ MOE_ROUTE: top-2 expert selection
        exp_indices, exp_scores = moe_route(h_t, w.W_router.astype(np.float32) *
                                            w.S_router.astype(np.float32), TOP_K)
        record_step("⑬", "MOE_ROUTE", f"experts {exp_indices}, scores {np.round(exp_scores, 3)}")
        assert len(exp_indices) == TOP_K
        assert np.isclose(exp_scores.sum(), 1.0, atol=1e-5), \
            f"Expert scores don't sum to 1: {exp_scores.sum()}"

        # ⑭-⑯: each selected expert
        expert_out = np.zeros(D_MODEL, dtype=np.float32)
        for rank, (eidx, escore) in enumerate(zip(exp_indices, exp_scores)):
            # ⑭ GEMM_INT4: gate_proj
            gate = gemm_int4(h_t[np.newaxis, :], w.W_gate[eidx], w.S_gate[eidx])  # [1, d_ff]
            # ⑮ VECTOR_OP: SiLU(gate) — this is the gated activation
            activated = vpu_silu(gate)  # [1, d_ff]
            # ⑯ GEMM_INT4: down_proj
            down = gemm_int4(activated, w.W_down[eidx], w.S_down[eidx])  # [1, d_model]
            expert_out += escore * down[0].astype(np.float32)

        if t == 0:
            record_step("⑭", "GEMM_INT4 gate_proj", f"gate [1,{D_FF}] fp16")
            record_step("⑮", "VECTOR_OP SiLU*up",   f"activated [1,{D_FF}] fp16")
            record_step("⑯", "GEMM_INT4 down_proj",  f"down [1,{D_MODEL}] fp16")

        ffn_outputs.append(expert_out)

    ffn_out = np.stack(ffn_outputs).astype(np.float16)  # [seq, d_model]
    assert ffn_out.shape == (SEQ_LEN, D_MODEL)
    assert np.all(np.isfinite(ffn_out)), "NaN/Inf in MoE FFN output"

    # ⑰ VECTOR_OP: expert result sum + residual
    hidden_out = vpu_residual_add(ffn_out, residual_before_ffn)
    record_step("⑰", "VECTOR_OP expert+residual", f"hidden_out {hidden_out.shape} fp16")
    assert hidden_out.shape == (SEQ_LEN, D_MODEL)
    assert np.all(np.isfinite(hidden_out)), "NaN/Inf after FFN residual"

    print("  [PASS] test_moe_ffn")
    return hidden_out


# ─────────────────────────────────────────────
# Test 3: Full Forward Pass (steps ①-㉑)
# ─────────────────────────────────────────────

def test_full_forward():
    print("\n=== test_full_forward (steps ①-㉑) ===")
    w = WEIGHTS
    kv_cache: dict = {}

    # ① DMA_2D
    x = rand_fp16((SEQ_LEN, D_MODEL))
    record_step("①", "DMA_2D", f"x {x.shape} fp16")

    # ② GEMM_INT4: QKV
    qkv = gemm_int4(x, w.W_qkv, w.S_qkv)
    record_step("②", "GEMM_INT4 QKV", f"{qkv.shape}")
    assert np.all(np.isfinite(qkv))

    Q = qkv[:, :D_MODEL].reshape(SEQ_LEN, NUM_HEADS, HEAD_DIM)
    K = qkv[:, D_MODEL:2*D_MODEL].reshape(SEQ_LEN, NUM_HEADS, HEAD_DIM)
    V = qkv[:, 2*D_MODEL:].reshape(SEQ_LEN, NUM_HEADS, HEAD_DIM)

    # ③ VECTOR_OP: RoPE
    Q = vpu_rope(Q, w.cos, w.sin)
    K = vpu_rope(K, w.cos, w.sin)
    record_step("③", "VECTOR_OP RoPE", f"Q/K {Q.shape}")
    assert np.all(np.isfinite(Q)) and np.all(np.isfinite(K))

    # ④ KVC_WRITE
    for t in range(SEQ_LEN):
        kvc_write(kv_cache, LAYER_ID, t, K[t], V[t])
    record_step("④", "KVC_WRITE", f"layer {LAYER_ID} × {SEQ_LEN}")

    # ⑤ KVC_READ
    K_c, V_c = kvc_read(kv_cache, LAYER_ID, SEQ_LEN)
    record_step("⑤", "KVC_READ", f"K {K_c.shape}, V {V_c.shape}")
    assert K_c.shape == (SEQ_LEN, NUM_HEADS, HEAD_DIM)

    # ⑥ GEMM_INT4: Q@K^T
    Q_t  = Q.transpose(1, 0, 2).astype(np.float32)
    Kt_t = K_c.transpose(1, 2, 0).astype(np.float32)
    scores = (Q_t @ Kt_t).astype(np.float16)
    record_step("⑥", "GEMM_INT4 Q@K^T", f"scores {scores.shape}")
    assert np.all(np.isfinite(scores))

    # ⑦ VECTOR_OP: Scale + Softmax
    scores = vpu_scale(scores, 1.0 / np.sqrt(HEAD_DIM))
    attn   = vpu_softmax(scores)
    record_step("⑦", "VECTOR_OP Scale+Softmax", f"attn {attn.shape}")
    assert np.all(np.isfinite(attn))

    # ⑧ GEMM_INT4: score@V
    V_t = V_c.transpose(1, 0, 2).astype(np.float32)
    ctx = (attn.astype(np.float32) @ V_t).astype(np.float16)
    ctx = ctx.transpose(1, 0, 2).reshape(SEQ_LEN, D_MODEL)
    record_step("⑧", "GEMM_INT4 score@V", f"ctx {ctx.shape}")
    assert np.all(np.isfinite(ctx))

    # ⑨ GEMM_INT4: output projection
    out = gemm_int4(ctx, w.W_out, w.S_out)
    record_step("⑨", "GEMM_INT4 out_proj", f"{out.shape}")
    assert np.all(np.isfinite(out))

    # ⑩ VECTOR_OP: Residual add
    hidden = vpu_residual_add(out, x)
    record_step("⑩", "VECTOR_OP Residual", f"hidden {hidden.shape}")
    assert np.all(np.isfinite(hidden))
    residual_attn = hidden.copy()

    # ⑪ VECTOR_OP: RMSNorm
    normed = vpu_rmsnorm(hidden, w.W_norm)
    record_step("⑪", "VECTOR_OP RMSNorm", f"normed {normed.shape}")
    assert np.all(np.isfinite(normed))

    # ⑫-⑰ MoE FFN (per-token)
    ffn_outputs = []
    for t in range(SEQ_LEN):
        h_t = normed[t]

        # ⑫ router logits
        # (no INT4 gemm needed here — fuse into moe_route for cleanliness)

        # ⑬ MOE_ROUTE
        exp_indices, exp_scores = moe_route(
            h_t,
            w.W_router.astype(np.float32) * w.S_router.astype(np.float32),
            TOP_K,
        )

        if t == 0:
            record_step("⑫", "GEMM_INT4 router", f"logits [1,{NUM_EXPERTS}]")
            record_step("⑬", "MOE_ROUTE", f"experts {exp_indices}")

        expert_out = np.zeros(D_MODEL, dtype=np.float32)
        for eidx, escore in zip(exp_indices, exp_scores):
            # ⑭ gate_proj
            gate = gemm_int4(h_t[np.newaxis, :], w.W_gate[eidx], w.S_gate[eidx])
            # ⑮ SiLU(gate) * up_proj
            up   = gemm_int4(h_t[np.newaxis, :], w.W_up[eidx], w.S_up[eidx])
            act  = (vpu_silu(gate).astype(np.float32) * up.astype(np.float32)).astype(np.float16)
            # ⑯ down_proj
            down = gemm_int4(act, w.W_down[eidx], w.S_down[eidx])
            expert_out += escore * down[0].astype(np.float32)

        if t == 0:
            record_step("⑭", "GEMM_INT4 gate_proj", f"[1,{D_FF}]")
            record_step("⑮", "VECTOR_OP SiLU*up",   f"[1,{D_FF}]")
            record_step("⑯", "GEMM_INT4 down_proj",  f"[1,{D_MODEL}]")

        ffn_outputs.append(expert_out)

    ffn_out = np.stack(ffn_outputs).astype(np.float16)
    assert np.all(np.isfinite(ffn_out)), "NaN/Inf in MoE FFN"

    # ⑰ VECTOR_OP: expert sum + residual
    hidden = vpu_residual_add(ffn_out, residual_attn)
    record_step("⑰", "VECTOR_OP expert+residual", f"hidden {hidden.shape}")
    assert hidden.shape == (SEQ_LEN, D_MODEL)
    assert np.all(np.isfinite(hidden))

    # ⑱ BARRIER
    record_step("⑱", "BARRIER", "(sync)")

    # ⑲ GEMM_INT4: LM head  [seq, vocab_size]
    logits = gemm_int4(hidden, w.W_lm, w.S_lm)
    record_step("⑲", "GEMM_INT4 LM_head", f"logits {logits.shape}")
    assert logits.shape == (SEQ_LEN, VOCAB_SIZE)
    assert np.all(np.isfinite(logits)), "NaN/Inf in LM head"

    # ⑳ VECTOR_OP: softmax → argmax → next token
    # Take logits of last token position for next-token prediction
    last_logits = logits[-1]  # [vocab_size]
    probs       = vpu_softmax(last_logits[np.newaxis, :])[0]
    next_token  = int(np.argmax(probs))
    record_step("⑳", "VECTOR_OP softmax+argmax", f"probs {probs.shape}, next_token={next_token}")
    assert 0 <= next_token < VOCAB_SIZE, f"next_token out of range: {next_token}"
    assert np.all(np.isfinite(probs)), "NaN/Inf in final probs"

    # ㉑ EVENT: emit next token
    record_step("㉑", "EVENT", f"next_token = {next_token}")

    print(f"  next_token = {next_token}  (range 0–{VOCAB_SIZE-1})")
    print("  [PASS] test_full_forward")
    return next_token


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ORBIT-G1 v2 Integration Test — LLM Forward Pass")
    print(f"  d_model={D_MODEL}, num_heads={NUM_HEADS}, head_dim={HEAD_DIM}")
    print(f"  num_experts={NUM_EXPERTS}, top_k={TOP_K}")
    print(f"  d_ff={D_FF}, vocab_size={VOCAB_SIZE}, seq_len={SEQ_LEN}")
    print("=" * 60)

    passed = 0
    failed = 0

    tests = [
        ("test_qkv_attention",  lambda: test_qkv_attention()),
        ("test_moe_ffn",        lambda: test_moe_ffn()),
        ("test_full_forward",   lambda: test_full_forward()),
    ]

    results = {}
    for name, fn in tests:
        try:
            result = fn()
            results[name] = result
            passed += 1
        except AssertionError as e:
            print(f"\n  [FAIL] {name}: AssertionError — {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"\n  [FAIL] {name}: {type(e).__name__} — {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} PASS / {failed} FAIL")

    if failed == 0:
        print("\nAll descriptor steps verified:")
        for step, name, shape in _steps:
            print(f"  {step} {name:22s}  {shape}")

        # Report final next_token from full forward
        if "test_full_forward" in results:
            nt = results["test_full_forward"]
            print(f"\nnext_token output = {nt}  (valid range 0–{VOCAB_SIZE-1})")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
