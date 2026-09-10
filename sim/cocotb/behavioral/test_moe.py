"""
test_moe.py — cocotb tests for moe_router
Flat packed interface, no async memory bus.
Parameters match RTL: NUM_EXPERTS=8, TOP_K=2, D_MODEL=16, MAX_TOKENS=4
"""
import cocotb
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
import numpy as np

# ── DUT parameters (must match RTL) ─────────────────────────────────────────
NUM_EXPERTS = 8
TOP_K       = 2
D_MODEL     = 16
MAX_TOKENS  = 4

# ── Flat packing helpers (per spec) ─────────────────────────────────────────

def load_flat(dut_signal, arr_fp16, shape):
    """Pack numpy float32 array as FP16 into flat int for DUT signal."""
    packed = np.array(arr_fp16, dtype=np.float32).reshape(shape)
    packed = packed.astype(np.float16).view(np.uint16).flatten()
    val = 0
    for i, v in enumerate(packed):
        val |= (int(v) << (i * 16))
    dut_signal.value = val

def read_flat_fp16(dut_signal, shape):
    """Read flat int from DUT signal as FP16 numpy array."""
    n = 1
    for s in shape:
        n *= s
    flat_val = int(dut_signal.value)
    result = np.zeros(n, dtype=np.uint16)
    for i in range(n):
        result[i] = (flat_val >> (i * 16)) & 0xFFFF
    return result.view(np.float16).astype(np.float32).reshape(shape)

def read_flat_u8(dut_signal, shape):
    """Read flat int from DUT signal as uint8 numpy array."""
    n = 1
    for s in shape:
        n *= s
    flat_val = int(dut_signal.value)
    result = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        result[i] = (flat_val >> (i * 8)) & 0xFF
    return result.reshape(shape)

# ── Reference implementation ─────────────────────────────────────────────────

def moe_ref(hidden, weights, top_k):
    """
    hidden:  [T, D]  float32
    weights: [D, E]  float32
    returns: indices [T, top_k] int32, scores [T, top_k] float32
    """
    logits   = hidden @ weights                          # [T, E]
    logits_f = logits.astype(np.float32)
    # softmax per token (numerically stable)
    logits_shifted = logits_f - logits_f.max(axis=1, keepdims=True)
    exp_l = np.exp(logits_shifted)
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)
    T = hidden.shape[0]
    indices = np.zeros((T, top_k), dtype=np.int32)
    scores  = np.zeros((T, top_k), dtype=np.float32)
    for t in range(T):
        idx = np.argsort(probs[t])[::-1][:top_k]
        indices[t] = idx
        scores[t]  = probs[t][idx]
    return indices, scores

# ── Reset helper ─────────────────────────────────────────────────────────────

async def reset_dut(dut):
    dut.rst_n.value      = 0
    dut.start.value      = 0
    dut.num_tokens.value = 0
    dut.hidden_flat.value  = 0
    dut.weight_flat.value  = 0
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)

async def run_moe(dut, max_cycles=50000):
    """Pulse start and wait for done."""
    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            return
    raise AssertionError("MoE Router timeout")

# ── Tests ────────────────────────────────────────────────────────────────────

@cocotb.test()
async def test_basic_routing(dut):
    """2 tokens, 8 experts, top-2. Verify indices match numpy top-k, scores ~softmax."""
    await reset_dut(dut)

    rng = np.random.default_rng(42)
    T = 2
    hidden  = rng.uniform(-1.0, 1.0, (MAX_TOKENS, D_MODEL)).astype(np.float32)
    weights = rng.uniform(-1.0, 1.0, (D_MODEL, NUM_EXPERTS)).astype(np.float32)

    # Quantize to FP16 (same as HW)
    hidden16  = hidden.astype(np.float16).astype(np.float32)
    weights16 = weights.astype(np.float16).astype(np.float32)

    # Load DUT
    load_flat(dut.hidden_flat,  hidden16,  (MAX_TOKENS, D_MODEL))
    load_flat(dut.weight_flat,  weights16, (D_MODEL, NUM_EXPERTS))
    dut.num_tokens.value = T

    await run_moe(dut)

    # Read outputs
    hw_indices = read_flat_u8(dut.indices_flat, (MAX_TOKENS, TOP_K))
    hw_scores  = read_flat_fp16(dut.scores_flat, (MAX_TOKENS, TOP_K))

    # Reference
    ref_indices, ref_scores = moe_ref(hidden16[:T], weights16, TOP_K)

    dut._log.info(f"ref_indices={ref_indices}")
    dut._log.info(f"hw_indices={hw_indices[:T]}")
    dut._log.info(f"ref_scores={ref_scores}")
    dut._log.info(f"hw_scores={hw_scores[:T]}")

    for t in range(T):
        assert set(hw_indices[t].tolist()) == set(ref_indices[t].tolist()), \
            f"Token {t}: HW indices {hw_indices[t]} != ref {ref_indices[t]}"
        for k in range(TOP_K):
            # Match score to the correct reference score by expert index
            hw_exp = int(hw_indices[t][k])
            # Find that expert's prob in ref
            ref_exp_idx = list(ref_indices[t]).index(hw_exp)
            ref_score = float(ref_scores[t][ref_exp_idx])
            hw_score  = float(hw_scores[t][k])
            err = abs(hw_score - ref_score)
            assert err < 0.02, \
                f"Token {t} expert {hw_exp}: score error {err:.4f} (hw={hw_score:.4f} ref={ref_score:.4f})"

    dut._log.info("PASS test_basic_routing")


@cocotb.test()
async def test_single_token(dut):
    """1 token — verify top-2 experts are the ones with highest probs."""
    await reset_dut(dut)

    rng = np.random.default_rng(7)
    T = 1
    hidden  = rng.uniform(-1.0, 1.0, (MAX_TOKENS, D_MODEL)).astype(np.float32)
    weights = rng.uniform(-1.0, 1.0, (D_MODEL, NUM_EXPERTS)).astype(np.float32)

    hidden16  = hidden.astype(np.float16).astype(np.float32)
    weights16 = weights.astype(np.float16).astype(np.float32)

    load_flat(dut.hidden_flat,  hidden16,  (MAX_TOKENS, D_MODEL))
    load_flat(dut.weight_flat,  weights16, (D_MODEL, NUM_EXPERTS))
    dut.num_tokens.value = T

    await run_moe(dut)

    hw_indices = read_flat_u8(dut.indices_flat, (MAX_TOKENS, TOP_K))
    hw_scores  = read_flat_fp16(dut.scores_flat, (MAX_TOKENS, TOP_K))

    ref_indices, ref_scores = moe_ref(hidden16[:T], weights16, TOP_K)

    dut._log.info(f"ref_indices={ref_indices[0]}, hw_indices={hw_indices[0]}")
    dut._log.info(f"ref_scores={ref_scores[0]}, hw_scores={hw_scores[0]}")

    assert set(hw_indices[0].tolist()) == set(ref_indices[0].tolist()), \
        f"Single token: HW {hw_indices[0]} != ref {ref_indices[0]}"

    for k in range(TOP_K):
        hw_exp = int(hw_indices[0][k])
        ref_exp_idx = list(ref_indices[0]).index(hw_exp)
        err = abs(float(hw_scores[0][k]) - float(ref_scores[0][ref_exp_idx]))
        assert err < 0.02, f"Score error for expert {hw_exp}: {err:.4f}"

    dut._log.info("PASS test_single_token")


@cocotb.test()
async def test_expert_isolation(dut):
    """Set weights so expert 0 always wins, expert 1 second — verify deterministic routing."""
    await reset_dut(dut)

    T = 3
    # Construct weights: expert 0 has large positive weight alignment with hidden,
    # expert 1 has moderate alignment, rest are zero/negative
    # Use identity-like weights: weight[:,0] = large, weight[:,1] = medium
    hidden  = np.ones((MAX_TOKENS, D_MODEL), dtype=np.float32) * 0.5
    weights = np.zeros((D_MODEL, NUM_EXPERTS), dtype=np.float32)
    # Expert 0 gets dot product = D_MODEL * 0.5 * 1.0 = 8.0
    weights[:, 0] = 1.0
    # Expert 1 gets dot product = D_MODEL * 0.5 * 0.5 = 4.0
    weights[:, 1] = 0.5
    # Others stay 0 → dot product = 0

    hidden16  = hidden.astype(np.float16).astype(np.float32)
    weights16 = weights.astype(np.float16).astype(np.float32)

    load_flat(dut.hidden_flat,  hidden16,  (MAX_TOKENS, D_MODEL))
    load_flat(dut.weight_flat,  weights16, (D_MODEL, NUM_EXPERTS))
    dut.num_tokens.value = T

    await run_moe(dut)

    hw_indices = read_flat_u8(dut.indices_flat, (MAX_TOKENS, TOP_K))
    hw_scores  = read_flat_fp16(dut.scores_flat, (MAX_TOKENS, TOP_K))

    ref_indices, ref_scores = moe_ref(hidden16[:T], weights16, TOP_K)

    dut._log.info(f"ref_indices={ref_indices}")
    dut._log.info(f"hw_indices={hw_indices[:T]}")

    for t in range(T):
        assert set(hw_indices[t].tolist()) == set(ref_indices[t].tolist()), \
            f"Token {t}: HW {hw_indices[t]} != ref {ref_indices[t]}"
        # Expert 0 must always be top-1
        assert 0 in hw_indices[t].tolist(), f"Token {t}: expert 0 not in top-k"
        assert 1 in hw_indices[t].tolist(), f"Token {t}: expert 1 not in top-k"

    # Verify determinism: all tokens should have same indices
    for t in range(1, T):
        assert set(hw_indices[t].tolist()) == set(hw_indices[0].tolist()), \
            f"Routing not deterministic: token 0={hw_indices[0]} token {t}={hw_indices[t]}"

    dut._log.info("PASS test_expert_isolation")
