"""
test_vpu.py — cocotb tests for vpu_core (flat array interface)
No async memory bus — cocotb writes src_flat/aux_flat before start,
reads dst_flat after done. Zero timing race conditions.
"""
import cocotb
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
import numpy as np

DEPTH = 4096

# ── FP16 helpers ────────────────────────────────────────────────────────────

def pack_f16(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float16).view(np.uint16)

def unpack_f16(arr_u16: np.ndarray) -> np.ndarray:
    return arr_u16.view(np.float16).astype(np.float32)

def f16_scalar(v: float) -> int:
    return int(np.float16(v).view(np.uint16))

# ── DUT helpers ─────────────────────────────────────────────────────────────

def load_src(dut, arr: np.ndarray):
    """Load array into src_flat (element i → bits [i*16+15 : i*16])."""
    packed = pack_f16(arr)
    # Build one big integer from all 16-bit values (little-endian: elem0 at LSB)
    val = 0
    for i in range(len(packed)):
        val |= (int(packed[i]) << (i * 16))
    dut.src_flat.value = val

def load_aux(dut, arr: np.ndarray):
    packed = pack_f16(arr)
    val = 0
    for i in range(len(packed)):
        val |= (int(packed[i]) << (i * 16))
    dut.aux_flat.value = val

def read_dst(dut, length: int) -> np.ndarray:
    """Read dst_flat into float32 array."""
    flat_val = int(dut.dst_flat.value)
    result = np.zeros(length, dtype=np.uint16)
    for i in range(length):
        result[i] = (flat_val >> (i * 16)) & 0xFFFF
    return unpack_f16(result)

# ── Reset & run helpers ─────────────────────────────────────────────────────

async def reset_dut(dut):
    dut.rst_n.value      = 0
    dut.start.value      = 0
    dut.op_type.value    = 0
    dut.vec_len.value    = 0
    dut.imm_fp16_0.value = 0
    dut.imm_fp16_1.value = 0
    dut.src_flat.value   = 0
    dut.aux_flat.value   = 0
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)

async def run_vpu(dut, op_type: int, vec_len: int,
                  imm0: float = 0.0, imm1: float = 0.0,
                  max_cycles: int = 50000):
    """Pulse start and wait for done."""
    dut.op_type.value    = op_type
    dut.vec_len.value    = vec_len
    dut.imm_fp16_0.value = f16_scalar(imm0)
    dut.imm_fp16_1.value = f16_scalar(imm1)

    await RisingEdge(dut.clk)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            return
    raise AssertionError(f"VPU timeout op={op_type:#x} N={vec_len}")

# ── Tests ───────────────────────────────────────────────────────────────────

@cocotb.test()
async def test_elem_add(dut):
    await reset_dut(dut)
    rng = np.random.default_rng(42)
    N = 256
    a = rng.uniform(-4.0, 4.0, N).astype(np.float16).astype(np.float32)
    b = rng.uniform(-4.0, 4.0, N).astype(np.float16).astype(np.float32)
    ref = (a.astype(np.float16) + b.astype(np.float16)).astype(np.float32)

    load_src(dut, a)
    load_aux(dut, b)
    await run_vpu(dut, op_type=0x0, vec_len=N)

    hw = read_dst(dut, N)
    err = float(np.max(np.abs(hw - ref) / (np.abs(ref) + 1e-4)))
    dut._log.info(f"ELEM_ADD rel_err={err:.4f}  ref[0:4]={ref[:4]}  hw[0:4]={hw[:4]}")
    assert err < 0.01, f"ELEM_ADD FAIL: rel_err={err:.4f}"
    dut._log.info("PASS test_elem_add")


@cocotb.test()
async def test_elem_mul(dut):
    await reset_dut(dut)
    rng = np.random.default_rng(7)
    N = 128
    a = rng.uniform(-2.0, 2.0, N).astype(np.float16).astype(np.float32)
    b = rng.uniform(-2.0, 2.0, N).astype(np.float16).astype(np.float32)
    ref = (a.astype(np.float16) * b.astype(np.float16)).astype(np.float32)

    load_src(dut, a)
    load_aux(dut, b)
    await run_vpu(dut, op_type=0x1, vec_len=N)

    hw = read_dst(dut, N)
    rel = np.abs(hw - ref) / (np.abs(ref) + 1e-4)
    err = float(np.max(rel))
    bad = np.where(rel > 0.01)[0]
    for i in bad[:5]:
        dut._log.info(f"  bad[{i}]: a={a[i]:.6f} b={b[i]:.6f} ref={ref[i]:.6f} hw={hw[i]:.6f}")
    dut._log.info(f"ELEM_MUL rel_err={err:.4f}")
    assert err < 0.02, f"ELEM_MUL FAIL: rel_err={err:.4f}"
    dut._log.info("PASS test_elem_mul")


@cocotb.test()
async def test_scale(dut):
    await reset_dut(dut)
    rng = np.random.default_rng(99)
    N = 64
    a = rng.uniform(-3.0, 3.0, N).astype(np.float16).astype(np.float32)
    scale = 0.5
    ref = (a.astype(np.float16) * np.float16(scale)).astype(np.float32)

    load_src(dut, a)
    await run_vpu(dut, op_type=0x2, vec_len=N, imm0=scale)

    hw = read_dst(dut, N)
    err = float(np.max(np.abs(hw - ref) / (np.abs(ref) + 1e-4)))
    dut._log.info(f"SCALE rel_err={err:.4f}")
    assert err < 0.01, f"SCALE FAIL: rel_err={err:.4f}"
    dut._log.info("PASS test_scale")


@cocotb.test()
async def test_rmsnorm(dut):
    await reset_dut(dut)
    rng = np.random.default_rng(123)
    N = 128
    a = rng.uniform(-2.0, 2.0, N).astype(np.float32)
    w = rng.uniform(0.5, 1.5, N).astype(np.float32)
    eps = 1e-5

    a16 = a.astype(np.float16).astype(np.float32)
    w16 = w.astype(np.float16).astype(np.float32)
    rms = np.sqrt(np.mean(a16 ** 2) + eps)
    ref = (a16 / rms * w16).astype(np.float16).astype(np.float32)

    load_src(dut, a)
    load_aux(dut, w)
    await run_vpu(dut, op_type=0x4, vec_len=N, imm0=eps)

    hw = read_dst(dut, N)
    err = float(np.max(np.abs(hw - ref) / (np.abs(ref) + 1e-3)))
    dut._log.info(f"RMSNORM rel_err={err:.4f}  ref[0:4]={ref[:4]}  hw[0:4]={hw[:4]}")
    assert err < 0.05, f"RMSNORM FAIL: rel_err={err:.4f}"
    dut._log.info("PASS test_rmsnorm")


@cocotb.test()
async def test_silu(dut):
    await reset_dut(dut)
    rng = np.random.default_rng(55)
    N = 64
    a = rng.uniform(-3.0, 3.0, N).astype(np.float32)
    a16 = a.astype(np.float16).astype(np.float32)
    ref = (a16 * (1.0 / (1.0 + np.exp(-a16)))).astype(np.float16).astype(np.float32)

    load_src(dut, a)
    await run_vpu(dut, op_type=0x5, vec_len=N)

    hw = read_dst(dut, N)
    err = float(np.max(np.abs(hw - ref) / (np.abs(ref) + 1e-3)))
    dut._log.info(f"SILU rel_err={err:.4f}  ref[0:4]={ref[:4]}  hw[0:4]={hw[:4]}")
    assert err < 0.05, f"SILU FAIL: rel_err={err:.4f}"
    dut._log.info("PASS test_silu")


@cocotb.test()
async def test_rope(dut):
    await reset_dut(dut)
    rng = np.random.default_rng(77)
    N = 64  # must be even
    x = rng.uniform(-2.0, 2.0, N).astype(np.float32)
    # cos/sin interleaved: [cos0, sin0, cos1, sin1, ...]
    thetas = np.array([0.01 * i for i in range(N//2)], dtype=np.float32)
    cs = np.zeros(N, dtype=np.float32)
    for i in range(N//2):
        cs[2*i]   = np.cos(thetas[i])
        cs[2*i+1] = np.sin(thetas[i])

    x16  = x.astype(np.float16).astype(np.float32)
    cs16 = cs.astype(np.float16).astype(np.float32)
    ref = np.zeros(N, dtype=np.float32)
    for i in range(N//2):
        xe = x16[2*i];   xo = x16[2*i+1]
        cv = cs16[2*i];  sv = cs16[2*i+1]
        ref[2*i]   = xe*cv - xo*sv
        ref[2*i+1] = xo*cv + xe*sv
    ref = ref.astype(np.float16).astype(np.float32)

    load_src(dut, x)
    load_aux(dut, cs)
    await run_vpu(dut, op_type=0x6, vec_len=N)

    hw = read_dst(dut, N)
    err = float(np.max(np.abs(hw - ref) / (np.abs(ref) + 1e-3)))
    dut._log.info(f"ROPE rel_err={err:.4f}  ref[0:4]={ref[:4]}  hw[0:4]={hw[:4]}")
    assert err < 0.05, f"ROPE FAIL: rel_err={err:.4f}"
    dut._log.info("PASS test_rope")


@cocotb.test()
async def test_softmax(dut):
    await reset_dut(dut)
    rng = np.random.default_rng(31)
    N = 64
    a = rng.uniform(-3.0, 3.0, N).astype(np.float32)
    a16 = a.astype(np.float16).astype(np.float32)
    shifted = a16 - np.max(a16)
    exp_v = np.exp(shifted.astype(np.float64))
    ref = (exp_v / np.sum(exp_v)).astype(np.float32)

    load_src(dut, a)
    await run_vpu(dut, op_type=0x7, vec_len=N)

    hw = read_dst(dut, N)
    err = float(np.max(np.abs(hw - ref)))
    sum_hw = float(np.sum(hw))
    dut._log.info(f"SOFTMAX max_abs_err={err:.5f}  sum_hw={sum_hw:.4f}  ref[0:4]={ref[:4]}  hw[0:4]={hw[:4]}")
    assert err < 0.02, f"SOFTMAX FAIL: max_abs_err={err:.5f}"
    assert abs(sum_hw - 1.0) < 0.05, f"SOFTMAX sum={sum_hw:.4f} not ~1.0"
    dut._log.info("PASS test_softmax")


@cocotb.test()
async def test_clamp(dut):
    await reset_dut(dut)
    rng = np.random.default_rng(11)
    N = 64
    a = rng.uniform(-3.0, 3.0, N).astype(np.float32)
    lo, hi = -1.0, 1.0
    a16 = a.astype(np.float16).astype(np.float32)
    ref = np.clip(a16, lo, hi).astype(np.float16).astype(np.float32)

    load_src(dut, a)
    await run_vpu(dut, op_type=0x8, vec_len=N, imm0=lo, imm1=hi)

    hw = read_dst(dut, N)
    err = float(np.max(np.abs(hw - ref) / (np.abs(ref) + 1e-4)))
    dut._log.info(f"CLAMP rel_err={err:.4f}  ref[0:4]={ref[:4]}  hw[0:4]={hw[:4]}")
    assert err < 0.01, f"CLAMP FAIL: rel_err={err:.4f}"
    dut._log.info("PASS test_clamp")
