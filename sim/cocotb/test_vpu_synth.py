"""
test_vpu_synth.py — cocotb tests for vpu_core_synth (Q8.8 encoding)
The synth module takes Q8.8 data in the FP16 slots.
All input values are pre-multiplied by 256 (Q8.8 encoding).
All output values are post-divided by 256 (Q8.8 decoding).

Error metric: signal-normalized absolute error (SNAE)
  snae = max_abs_err / max(max(abs(ref)), floor)
This avoids the near-zero blow-up of plain relative error.
Q8.8 has 1 LSB = 1/256 = 0.00391, so 2-3 LSB errors are expected for LUT ops.
"""
import cocotb
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
import numpy as np

DEPTH = 256


def to_q88(v):
    """Float -> Q8.8 signed 16-bit integer, returned as unsigned 16-bit for packing."""
    r = int(round(v * 256))
    if r > 32767:
        r = 32767
    if r < -32768:
        r = -32768
    return r & 0xFFFF


def from_q88(u16):
    """Unsigned 16-bit (Q8.8) -> float."""
    val = int(u16)
    if val >= 32768:
        val -= 65536  # sign extend
    return val / 256.0


def load_src_q88(dut, arr):
    val = 0
    for i, v in enumerate(arr):
        val |= (to_q88(v) << (i * 16))
    dut.src_flat.value = val


def load_aux_q88(dut, arr=None):
    if arr is None:
        dut.aux_flat.value = 0
        return
    val = 0
    for i, v in enumerate(arr):
        val |= (to_q88(v) << (i * 16))
    dut.aux_flat.value = val


def read_dst_q88(dut, length):
    flat_val = int(dut.dst_flat.value)
    result = []
    for i in range(length):
        u16 = (flat_val >> (i * 16)) & 0xFFFF
        result.append(from_q88(u16))
    return np.array(result, dtype=np.float32)


def load_imm_q88(dut, v0=0.0, v1=0.0):
    dut.imm_fp16_0.value = to_q88(v0)
    dut.imm_fp16_1.value = to_q88(v1)


def snae(hw, ref, floor=0.5):
    """Signal-Normalized Absolute Error: max(|hw-ref|) / max(max(|ref|), floor).

    Uses a floor on the denominator so near-zero reference values don't cause
    division-by-tiny-number blowup. Appropriate for Q8.8 where 1 LSB = 1/256.
    """
    abs_err = float(np.max(np.abs(hw - ref)))
    signal  = float(max(float(np.max(np.abs(ref))), floor))
    return abs_err / signal


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.op_type.value = 0
    dut.vec_len.value = 0
    dut.imm_fp16_0.value = 0
    dut.imm_fp16_1.value = 0
    dut.src_flat.value = 0
    dut.aux_flat.value = 0
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)


async def run_vpu(dut, op_type, vec_len, max_cycles=200000):
    dut.op_type.value = op_type
    dut.vec_len.value = vec_len
    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            return
    raise AssertionError(f"VPU timeout op={op_type:#x} N={vec_len}")


@cocotb.test()
async def test_elem_add(dut):
    await reset_dut(dut)
    rng = np.random.default_rng(42)
    N = 128
    # Use values that avoid near-zero catastrophic cancellation for the test
    a = rng.uniform(-4.0, 4.0, N).astype(np.float32)
    b = rng.uniform(-4.0, 4.0, N).astype(np.float32)
    # Reference: Q8.8 saturating add
    ref = np.clip((a + b), -127.996, 127.996).astype(np.float32)

    load_src_q88(dut, a)
    load_aux_q88(dut, b)
    load_imm_q88(dut)
    await run_vpu(dut, op_type=0x0, vec_len=N)

    hw  = read_dst_q88(dut, N)
    err = snae(hw, ref, floor=0.5)
    dut._log.info(f"ELEM_ADD snae={err:.4f}  ref[0:4]={ref[:4]}  hw[0:4]={hw[:4]}")
    # 1 LSB / 0.5 = 0.008, allow 2x
    assert err < 0.02, f"ELEM_ADD FAIL: snae={err:.4f}"
    dut._log.info("PASS test_elem_add")


@cocotb.test()
async def test_elem_mul(dut):
    await reset_dut(dut)
    rng = np.random.default_rng(7)
    N = 128
    a = rng.uniform(-2.0, 2.0, N).astype(np.float32)
    b = rng.uniform(-2.0, 2.0, N).astype(np.float32)
    ref = (a * b).astype(np.float32)

    load_src_q88(dut, a)
    load_aux_q88(dut, b)
    load_imm_q88(dut)
    await run_vpu(dut, op_type=0x1, vec_len=N)

    hw  = read_dst_q88(dut, N)
    err = snae(hw, ref, floor=0.5)
    dut._log.info(f"ELEM_MUL snae={err:.4f}  ref[0:4]={ref[:4]}  hw[0:4]={hw[:4]}")
    # Q8.8 mul: ~2 LSB error in worst case, signal up to 4.0 -> 2/256/4 = 0.002
    assert err < 0.02, f"ELEM_MUL FAIL: snae={err:.4f}"
    dut._log.info("PASS test_elem_mul")


@cocotb.test()
async def test_scale(dut):
    await reset_dut(dut)
    rng = np.random.default_rng(99)
    N = 64
    a     = rng.uniform(-3.0, 3.0, N).astype(np.float32)
    scale = 0.5
    ref   = (a * scale).astype(np.float32)

    load_src_q88(dut, a)
    load_aux_q88(dut)
    load_imm_q88(dut, v0=scale)
    await run_vpu(dut, op_type=0x2, vec_len=N)

    hw  = read_dst_q88(dut, N)
    err = snae(hw, ref, floor=0.5)
    dut._log.info(f"SCALE snae={err:.4f}  ref[0:4]={ref[:4]}  hw[0:4]={hw[:4]}")
    assert err < 0.02, f"SCALE FAIL: snae={err:.4f}"
    dut._log.info("PASS test_scale")


@cocotb.test()
async def test_rmsnorm(dut):
    await reset_dut(dut)
    rng = np.random.default_rng(123)
    N = 128
    a = rng.uniform(-2.0, 2.0, N).astype(np.float32)
    w = rng.uniform(0.5, 1.5, N).astype(np.float32)
    eps = 1e-5
    rms = float(np.sqrt(np.mean(a.astype(np.float64) ** 2) + eps))
    ref = (a / rms * w).astype(np.float32)

    load_src_q88(dut, a)
    load_aux_q88(dut, w)
    load_imm_q88(dut, v0=eps)
    await run_vpu(dut, op_type=0x4, vec_len=N)

    hw  = read_dst_q88(dut, N)
    err = snae(hw, ref, floor=0.5)
    dut._log.info(f"RMSNORM snae={err:.4f}  ref[0:4]={ref[:4]}  hw[0:4]={hw[:4]}")
    # isqrt LUT has ~3-4 steps error -> ~0.015 SNAE expected
    assert err < 0.08, f"RMSNORM FAIL: snae={err:.4f}"
    dut._log.info("PASS test_rmsnorm")


@cocotb.test()
async def test_silu(dut):
    await reset_dut(dut)
    rng = np.random.default_rng(55)
    N = 64
    a   = rng.uniform(-3.0, 3.0, N).astype(np.float32)
    sig = 1.0 / (1.0 + np.exp(-a.astype(np.float64)))
    ref = (a * sig).astype(np.float32)

    load_src_q88(dut, a)
    load_aux_q88(dut)
    load_imm_q88(dut)
    await run_vpu(dut, op_type=0x5, vec_len=N)

    hw  = read_dst_q88(dut, N)
    err = snae(hw, ref, floor=0.5)
    dut._log.info(f"SILU snae={err:.4f}  ref[0:4]={ref[:4]}  hw[0:4]={hw[:4]}")
    # sigmoid LUT + mul: ~2 LUT steps + 1 mul LSB
    assert err < 0.06, f"SILU FAIL: snae={err:.4f}"
    dut._log.info("PASS test_silu")


@cocotb.test()
async def test_rope(dut):
    await reset_dut(dut)
    rng = np.random.default_rng(77)
    N   = 64
    x   = rng.uniform(-2.0, 2.0, N).astype(np.float32)
    thetas = np.array([0.01 * i for i in range(N // 2)], dtype=np.float32)
    cs = np.zeros(N, dtype=np.float32)
    for i in range(N // 2):
        cs[2 * i]     = float(np.cos(thetas[i]))
        cs[2 * i + 1] = float(np.sin(thetas[i]))
    ref = np.zeros(N, dtype=np.float32)
    for i in range(N // 2):
        xe = x[2 * i];    xo = x[2 * i + 1]
        cv = cs[2 * i];   sv = cs[2 * i + 1]
        ref[2 * i]     = xe * cv - xo * sv
        ref[2 * i + 1] = xo * cv + xe * sv

    load_src_q88(dut, x)
    load_aux_q88(dut, cs)
    load_imm_q88(dut)
    await run_vpu(dut, op_type=0x6, vec_len=N)

    hw  = read_dst_q88(dut, N)
    err = snae(hw, ref, floor=0.5)
    dut._log.info(f"ROPE snae={err:.4f}  ref[0:4]={ref[:4]}  hw[0:4]={hw[:4]}")
    # two Q8.8 muls + subtract: ~3 LSB max error
    assert err < 0.05, f"ROPE FAIL: snae={err:.4f}"
    dut._log.info("PASS test_rope")


@cocotb.test()
async def test_softmax(dut):
    await reset_dut(dut)
    rng = np.random.default_rng(31)
    N = 64
    a       = rng.uniform(-3.0, 3.0, N).astype(np.float32)
    shifted = a - float(np.max(a))
    exp_v   = np.exp(shifted.astype(np.float64))
    ref     = (exp_v / np.sum(exp_v)).astype(np.float32)

    load_src_q88(dut, a)
    load_aux_q88(dut)
    load_imm_q88(dut)
    await run_vpu(dut, op_type=0x7, vec_len=N)

    hw     = read_dst_q88(dut, N)
    err    = snae(hw, ref, floor=0.1)  # softmax outputs are ~1/N ~ 0.016
    sum_hw = float(np.sum(hw))
    dut._log.info(f"SOFTMAX snae={err:.5f}  sum_hw={sum_hw:.4f}  ref[0:4]={ref[:4]}  hw[0:4]={hw[:4]}")
    assert err < 0.30, f"SOFTMAX FAIL: snae={err:.5f}"
    assert abs(sum_hw - 1.0) < 0.10, f"SOFTMAX sum={sum_hw:.4f} not ~1.0"
    dut._log.info("PASS test_softmax")


@cocotb.test()
async def test_clamp(dut):
    await reset_dut(dut)
    rng    = np.random.default_rng(11)
    N      = 64
    a      = rng.uniform(-3.0, 3.0, N).astype(np.float32)
    lo, hi = -1.0, 1.0
    ref    = np.clip(a, lo, hi).astype(np.float32)

    load_src_q88(dut, a)
    load_aux_q88(dut)
    load_imm_q88(dut, v0=lo, v1=hi)
    await run_vpu(dut, op_type=0x8, vec_len=N)

    hw  = read_dst_q88(dut, N)
    err = snae(hw, ref, floor=0.5)
    dut._log.info(f"CLAMP snae={err:.4f}  ref[0:4]={ref[:4]}  hw[0:4]={hw[:4]}")
    assert err < 0.02, f"CLAMP FAIL: snae={err:.4f}"
    dut._log.info("PASS test_clamp")


@cocotb.test()
async def test_gelu(dut):
    await reset_dut(dut)
    rng = np.random.default_rng(66)
    N   = 64
    a   = rng.uniform(-3.0, 3.0, N).astype(np.float32)
    # GELU approx: x * sigmoid(1.702 * x)
    sig = 1.0 / (1.0 + np.exp(-(1.702 * a.astype(np.float64))))
    ref = (a * sig).astype(np.float32)

    load_src_q88(dut, a)
    load_aux_q88(dut)
    load_imm_q88(dut)
    await run_vpu(dut, op_type=0x9, vec_len=N)

    hw  = read_dst_q88(dut, N)
    err = snae(hw, ref, floor=0.5)
    dut._log.info(f"GELU snae={err:.4f}  ref[0:4]={ref[:4]}  hw[0:4]={hw[:4]}")
    # sigmoid LUT (16 steps/unit -> ~1 step error) + two q88_mul
    assert err < 0.06, f"GELU FAIL: snae={err:.4f}"
    dut._log.info("PASS test_gelu")
