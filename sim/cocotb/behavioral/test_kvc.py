"""
test_kvc.py — cocotb tests for kvc_core

KVC parameters (matching RTL defaults):
  NUM_LAYERS = 4
  NUM_HEADS  = 4
  HEAD_DIM   = 16
  MAX_SEQ    = 64
  MAX_SEQS   = 4

Flat packing convention:
  k_in_flat / v_in_flat — one token, all heads:
    element [head_id][elem] at bit offset: (head_id * HEAD_DIM + elem) * 16

  k_out_flat / v_out_flat — all tokens, all heads:
    element at token t, head h, elem e at bit offset:
    ((t * NUM_HEADS + h) * HEAD_DIM + e) * 16
"""
import cocotb
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
import numpy as np

# KVC parameters (must match RTL)
NUM_LAYERS = 4
NUM_HEADS  = 4
HEAD_DIM   = 16
MAX_SEQ    = 64
MAX_SEQS   = 4

# Operation types
OP_WRITE = 0
OP_READ  = 1


# ── FP16 helpers ──────────────────────────────────────────────────────────────

def f16_val(v: float) -> int:
    """Convert float to FP16 uint16 bit pattern."""
    return int(np.float16(v).view(np.uint16))


def f16_to_float(u: int) -> float:
    """Convert FP16 uint16 bit pattern to Python float."""
    return float(np.array(u, dtype=np.uint16).view(np.float16))


# ── Pack/unpack helpers ───────────────────────────────────────────────────────

def pack_kv_in(data: np.ndarray) -> int:
    """
    Pack K or V data for one token into k_in_flat / v_in_flat integer.

    data shape: [NUM_HEADS, HEAD_DIM], dtype float32 or float16
    Bit offset for [head_id, elem]: (head_id * HEAD_DIM + elem) * 16
    """
    result = 0
    data16 = data.astype(np.float16).view(np.uint16)
    for h in range(NUM_HEADS):
        for e in range(HEAD_DIM):
            bit_offset = (h * HEAD_DIM + e) * 16
            result |= (int(data16[h, e]) << bit_offset)
    return result


def unpack_kv_out(flat_val: int, num_tokens: int) -> np.ndarray:
    """
    Unpack k_out_flat / v_out_flat integer into array.

    Returns shape: [MAX_SEQ, NUM_HEADS, HEAD_DIM], dtype float32
    Bit offset for [tok, head, elem]: ((tok * NUM_HEADS + head) * HEAD_DIM + elem) * 16

    Note: full flat is MAX_SEQ tokens wide; only first num_tokens are meaningful.
    """
    result = np.zeros((MAX_SEQ, NUM_HEADS, HEAD_DIM), dtype=np.float32)
    for t in range(num_tokens):
        for h in range(NUM_HEADS):
            for e in range(HEAD_DIM):
                bit_offset = ((t * NUM_HEADS + h) * HEAD_DIM + e) * 16
                u16 = (flat_val >> bit_offset) & 0xFFFF
                result[t, h, e] = f16_to_float(u16)
    return result


# ── DUT control helpers ───────────────────────────────────────────────────────

async def reset_dut(dut):
    """Reset DUT and start clock."""
    dut.rst_n.value    = 0
    dut.start.value    = 0
    dut.op_type.value  = 0
    dut.seq_id.value   = 0
    dut.layer_id.value = 0
    dut.seq_pos.value  = 0
    dut.seq_len.value  = 0
    dut.k_in_flat.value = 0
    dut.v_in_flat.value = 0

    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)


async def kvc_write(dut, seq_id: int, layer_id: int, seq_pos: int,
                    k_data: np.ndarray, v_data: np.ndarray,
                    max_cycles: int = 500):
    """
    Issue a KVC_WRITE operation.

    k_data, v_data shape: [NUM_HEADS, HEAD_DIM], dtype float32
    """
    dut.op_type.value   = OP_WRITE
    dut.seq_id.value    = seq_id
    dut.layer_id.value  = layer_id
    dut.seq_pos.value   = seq_pos
    dut.seq_len.value   = 0
    dut.k_in_flat.value = pack_kv_in(k_data)
    dut.v_in_flat.value = pack_kv_in(v_data)

    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            return
    raise AssertionError(
        f"KVC_WRITE timeout: seq={seq_id} layer={layer_id} pos={seq_pos}"
    )


async def kvc_read(dut, seq_id: int, layer_id: int, seq_len: int,
                   max_cycles: int = 5000) -> np.ndarray:
    """
    Issue a KVC_READ operation and return (k_out, v_out).

    Returns:
        k_out: [MAX_SEQ, NUM_HEADS, HEAD_DIM] float32
        v_out: [MAX_SEQ, NUM_HEADS, HEAD_DIM] float32
    Only first seq_len tokens are meaningful.
    """
    dut.op_type.value   = OP_READ
    dut.seq_id.value    = seq_id
    dut.layer_id.value  = layer_id
    dut.seq_pos.value   = 0
    dut.seq_len.value   = seq_len
    dut.k_in_flat.value = 0
    dut.v_in_flat.value = 0

    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            k_flat = int(dut.k_out_flat.value)
            v_flat = int(dut.v_out_flat.value)
            k_out = unpack_kv_out(k_flat, seq_len)
            v_out = unpack_kv_out(v_flat, seq_len)
            return k_out, v_out
    raise AssertionError(
        f"KVC_READ timeout: seq={seq_id} layer={layer_id} len={seq_len}"
    )


def make_kv_token(seed: int) -> tuple:
    """Generate deterministic K and V data for a token."""
    rng = np.random.default_rng(seed)
    k = rng.uniform(-2.0, 2.0, (NUM_HEADS, HEAD_DIM)).astype(np.float32)
    v = rng.uniform(-2.0, 2.0, (NUM_HEADS, HEAD_DIM)).astype(np.float32)
    # Quantize to FP16 to match hardware
    k = k.astype(np.float16).astype(np.float32)
    v = v.astype(np.float16).astype(np.float32)
    return k, v


def check_kv_match(hw: np.ndarray, ref: np.ndarray, name: str, tol: float = 1e-3):
    """Assert that hw and ref match within tolerance."""
    err = np.max(np.abs(hw - ref))
    if err > tol:
        bad = np.argwhere(np.abs(hw - ref) > tol)
        for idx in bad[:5]:
            print(f"  {name} mismatch at {tuple(idx)}: hw={hw[tuple(idx)]:.5f} ref={ref[tuple(idx)]:.5f}")
    assert err <= tol, f"{name}: max abs error = {err:.6f} > tol={tol}"


# ── Tests ─────────────────────────────────────────────────────────────────────

@cocotb.test()
async def test_write_read_single(dut):
    """Write K/V for one token, read back, verify match."""
    await reset_dut(dut)

    seq_id   = 0
    layer_id = 0
    seq_pos  = 0

    k_ref, v_ref = make_kv_token(seed=42)

    dut._log.info(f"Writing token: seq={seq_id} layer={layer_id} pos={seq_pos}")
    await kvc_write(dut, seq_id, layer_id, seq_pos, k_ref, v_ref)

    dut._log.info(f"Reading back: seq={seq_id} layer={layer_id} len=1")
    k_out, v_out = await kvc_read(dut, seq_id, layer_id, seq_len=1)

    check_kv_match(k_out[0], k_ref, "K[0]")
    check_kv_match(v_out[0], v_ref, "V[0]")

    dut._log.info("PASS test_write_read_single")


@cocotb.test()
async def test_write_read_multi(dut):
    """Write 8 tokens sequentially, read back all, verify."""
    await reset_dut(dut)

    seq_id   = 1
    layer_id = 1
    N_TOKENS = 8

    k_refs = []
    v_refs = []

    dut._log.info(f"Writing {N_TOKENS} tokens to seq={seq_id} layer={layer_id}")
    for pos in range(N_TOKENS):
        k, v = make_kv_token(seed=100 + pos)
        k_refs.append(k)
        v_refs.append(v)
        await kvc_write(dut, seq_id, layer_id, pos, k, v)

    dut._log.info(f"Reading back {N_TOKENS} tokens")
    k_out, v_out = await kvc_read(dut, seq_id, layer_id, seq_len=N_TOKENS)

    for pos in range(N_TOKENS):
        check_kv_match(k_out[pos], k_refs[pos], f"K[{pos}]")
        check_kv_match(v_out[pos], v_refs[pos], f"V[{pos}]")
        dut._log.info(f"  token {pos} K/V verified OK")

    dut._log.info("PASS test_write_read_multi")


@cocotb.test()
async def test_multi_seq(dut):
    """Write to 2 different seq_ids, read each back, verify no cross-contamination."""
    await reset_dut(dut)

    layer_id = 2
    N_TOKENS = 4

    # Write to seq 0
    k_seq0 = []
    v_seq0 = []
    dut._log.info(f"Writing {N_TOKENS} tokens to seq=0 layer={layer_id}")
    for pos in range(N_TOKENS):
        k, v = make_kv_token(seed=200 + pos)
        k_seq0.append(k)
        v_seq0.append(v)
        await kvc_write(dut, seq_id=0, layer_id=layer_id, seq_pos=pos, k_data=k, v_data=v)

    # Write to seq 2 (different values)
    k_seq2 = []
    v_seq2 = []
    dut._log.info(f"Writing {N_TOKENS} tokens to seq=2 layer={layer_id}")
    for pos in range(N_TOKENS):
        k, v = make_kv_token(seed=300 + pos)
        k_seq2.append(k)
        v_seq2.append(v)
        await kvc_write(dut, seq_id=2, layer_id=layer_id, seq_pos=pos, k_data=k, v_data=v)

    # Read seq 0 and verify
    dut._log.info("Reading seq=0")
    k_out0, v_out0 = await kvc_read(dut, seq_id=0, layer_id=layer_id, seq_len=N_TOKENS)
    for pos in range(N_TOKENS):
        check_kv_match(k_out0[pos], k_seq0[pos], f"seq0 K[{pos}]")
        check_kv_match(v_out0[pos], v_seq0[pos], f"seq0 V[{pos}]")

    # Read seq 2 and verify
    dut._log.info("Reading seq=2")
    k_out2, v_out2 = await kvc_read(dut, seq_id=2, layer_id=layer_id, seq_len=N_TOKENS)
    for pos in range(N_TOKENS):
        check_kv_match(k_out2[pos], k_seq2[pos], f"seq2 K[{pos}]")
        check_kv_match(v_out2[pos], v_seq2[pos], f"seq2 V[{pos}]")

    # Verify no cross-contamination: seq0 data != seq2 data
    for pos in range(N_TOKENS):
        assert not np.allclose(k_out0[pos], k_out2[pos], atol=1e-4), \
            f"seq0 and seq2 K[{pos}] are identical — likely aliased!"

    dut._log.info("PASS test_multi_seq")


@cocotb.test()
async def test_multi_layer(dut):
    """Write to layer 0 and layer 2, read each back, verify isolation."""
    await reset_dut(dut)

    seq_id   = 3
    N_TOKENS = 4

    # Write to layer 0
    k_l0 = []
    v_l0 = []
    dut._log.info(f"Writing {N_TOKENS} tokens to seq={seq_id} layer=0")
    for pos in range(N_TOKENS):
        k, v = make_kv_token(seed=400 + pos)
        k_l0.append(k)
        v_l0.append(v)
        await kvc_write(dut, seq_id=seq_id, layer_id=0, seq_pos=pos, k_data=k, v_data=v)

    # Write to layer 2 (different values)
    k_l2 = []
    v_l2 = []
    dut._log.info(f"Writing {N_TOKENS} tokens to seq={seq_id} layer=2")
    for pos in range(N_TOKENS):
        k, v = make_kv_token(seed=500 + pos)
        k_l2.append(k)
        v_l2.append(v)
        await kvc_write(dut, seq_id=seq_id, layer_id=2, seq_pos=pos, k_data=k, v_data=v)

    # Read layer 0 and verify
    dut._log.info(f"Reading seq={seq_id} layer=0")
    k_out0, v_out0 = await kvc_read(dut, seq_id=seq_id, layer_id=0, seq_len=N_TOKENS)
    for pos in range(N_TOKENS):
        check_kv_match(k_out0[pos], k_l0[pos], f"layer0 K[{pos}]")
        check_kv_match(v_out0[pos], v_l0[pos], f"layer0 V[{pos}]")

    # Read layer 2 and verify
    dut._log.info(f"Reading seq={seq_id} layer=2")
    k_out2, v_out2 = await kvc_read(dut, seq_id=seq_id, layer_id=2, seq_len=N_TOKENS)
    for pos in range(N_TOKENS):
        check_kv_match(k_out2[pos], k_l2[pos], f"layer2 K[{pos}]")
        check_kv_match(v_out2[pos], v_l2[pos], f"layer2 V[{pos}]")

    # Verify layer isolation: layer 0 data != layer 2 data
    for pos in range(N_TOKENS):
        assert not np.allclose(k_out0[pos], k_out2[pos], atol=1e-4), \
            f"layer0 and layer2 K[{pos}] are identical — likely aliased!"

    dut._log.info("PASS test_multi_layer")
