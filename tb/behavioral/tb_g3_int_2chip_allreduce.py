"""
tb_g3_int_2chip_allreduce.py — G3-INT-004: 2-Chip Distributed Training

Path: chip0 backward(dW0) + chip1 backward(dW1)
      → collective all-reduce SUM(dW0+dW1) = reduced_dW
      → optimizer0(param0, reduced_dW) + optimizer1(param1, reduced_dW)

Reduction: SUM (not average).
Active region: 16×16.
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import struct
import math

DIM = 16


def float_to_bf16(f):
    fp32_bits = struct.unpack('>I', struct.pack('>f', f))[0]
    return (fp32_bits >> 16) & 0xFFFF


def float_to_fp32(f):
    return struct.unpack('>I', struct.pack('>f', f))[0]


def fp32_to_float(bits):
    return struct.unpack('>f', struct.pack('>I', bits & 0xFFFFFFFF))[0]


def set_bf16_identity(mat, scale=1.0):
    for i in range(DIM):
        for j in range(DIM):
            mat[i][j].value = float_to_bf16(scale if i == j else 0.0)


def set_bf16_zeros(mat):
    for i in range(DIM):
        for j in range(DIM):
            mat[i][j].value = 0


def set_fp32_diagonal(mat, val):
    for i in range(DIM):
        for j in range(DIM):
            mat[i][j].value = float_to_fp32(val if i == j else 0.0)


def set_fp32_zeros(mat):
    for i in range(DIM):
        for j in range(DIM):
            mat[i][j].value = 0


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.adamw_enable.value = 0
    dut.lr_fp32.value = float_to_fp32(0.001)
    dut.beta1_fp32.value = float_to_fp32(0.9)
    dut.beta2_fp32.value = float_to_fp32(0.999)
    dut.epsilon_fp32.value = float_to_fp32(1e-8)
    dut.weight_decay_fp32.value = float_to_fp32(0.0)
    for mat in [dut.x0_bf16, dut.w0_bf16, dut.dy0_bf16,
                dut.x1_bf16, dut.w1_bf16, dut.dy1_bf16]:
        set_bf16_zeros(mat)
    for mat in [dut.param0_in, dut.m0_in, dut.v0_in,
                dut.param1_in, dut.m1_in, dut.v1_in]:
        set_fp32_zeros(mat)
    await Timer(100, unit="ns")
    dut.rst_n.value = 1
    for _ in range(5):
        await RisingEdge(dut.clk)


async def run_2chip(dut, max_wait=2000):
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    for _ in range(max_wait):
        await RisingEdge(dut.clk)
        if dut.done_pulse.value == 1:
            return True
    return False


def python_adam(p, g, m, v, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    m_new = b1 * m + (1 - b1) * g
    v_new = b2 * v + (1 - b2) * g * g
    update = lr * m_new / (math.sqrt(max(v_new, 0)) + eps)
    return p - update, m_new, v_new


@cocotb.test()
async def test_2chip_training_step_sum_reduce(dut):
    """Chip0: X=I,dY=diag(0.3). Chip1: X=I,dY=diag(0.7).
    dW0=diag(0.3), dW1=diag(0.7). reduced=diag(1.0).
    Both chips optimize with grad=diag(1.0)."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Chip0
    set_bf16_identity(dut.x0_bf16, 1.0)
    set_bf16_identity(dut.w0_bf16, 1.0)
    set_bf16_identity(dut.dy0_bf16, 0.3)
    set_fp32_diagonal(dut.param0_in, 5.0)
    set_fp32_zeros(dut.m0_in)
    set_fp32_zeros(dut.v0_in)

    # Chip1
    set_bf16_identity(dut.x1_bf16, 1.0)
    set_bf16_identity(dut.w1_bf16, 1.0)
    set_bf16_identity(dut.dy1_bf16, 0.7)
    set_fp32_diagonal(dut.param1_in, 5.0)
    set_fp32_zeros(dut.m1_in)
    set_fp32_zeros(dut.v1_in)

    ok = await run_2chip(dut)
    assert ok, "done_pulse not seen"
    assert int(dut.err_code.value) == 0, f"err={int(dut.err_code.value)}"

    # reduced_dW should be diag(0.3+0.7) = diag(1.0)
    rdw = fp32_to_float(int(dut.reduced_dw[0][0].value))
    dut._log.info(f"reduced_dw[0][0]={rdw}")
    assert abs(rdw - 1.0) < 0.05, f"Expected reduced_dw ~1.0, got {rdw}"

    # Both params should get same update: Adam(5.0, grad=1.0)
    p_ref, _, _ = python_adam(5.0, 1.0, 0.0, 0.0)
    p0 = fp32_to_float(int(dut.param0_out[0][0].value))
    p1 = fp32_to_float(int(dut.param1_out[0][0].value))
    dut._log.info(f"param0={p0}, param1={p1}, ref={p_ref}")

    assert abs(p0 - p_ref) < abs(p_ref) * 0.02 + 1e-5, f"param0: {p0} vs {p_ref}"
    assert abs(p1 - p_ref) < abs(p_ref) * 0.02 + 1e-5, f"param1: {p1} vs {p_ref}"


@cocotb.test()
async def test_equal_inputs_produce_equal_outputs(dut):
    """Same inputs on both chips → identical outputs."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    for mat0, mat1 in [(dut.x0_bf16, dut.x1_bf16),
                       (dut.w0_bf16, dut.w1_bf16),
                       (dut.dy0_bf16, dut.dy1_bf16)]:
        set_bf16_identity(mat0, 1.0)
        set_bf16_identity(mat1, 1.0)

    for mat0, mat1 in [(dut.param0_in, dut.param1_in)]:
        set_fp32_diagonal(mat0, 3.0)
        set_fp32_diagonal(mat1, 3.0)
    for mat0, mat1 in [(dut.m0_in, dut.m1_in), (dut.v0_in, dut.v1_in)]:
        set_fp32_zeros(mat0)
        set_fp32_zeros(mat1)

    ok = await run_2chip(dut)
    assert ok

    p0 = fp32_to_float(int(dut.param0_out[0][0].value))
    p1 = fp32_to_float(int(dut.param1_out[0][0].value))
    assert abs(p0 - p1) < 1e-6, f"Symmetric inputs should give equal outputs: p0={p0}, p1={p1}"


@cocotb.test()
async def test_mismatched_gradients(dut):
    """Different dY → different dW → same reduced_dW → same param update."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    set_bf16_identity(dut.x0_bf16, 1.0)
    set_bf16_identity(dut.x1_bf16, 1.0)
    set_bf16_identity(dut.w0_bf16, 1.0)
    set_bf16_identity(dut.w1_bf16, 1.0)
    set_bf16_identity(dut.dy0_bf16, 0.1)   # small grad
    set_bf16_identity(dut.dy1_bf16, 0.9)   # big grad

    set_fp32_diagonal(dut.param0_in, 2.0)
    set_fp32_diagonal(dut.param1_in, 2.0)
    set_fp32_zeros(dut.m0_in); set_fp32_zeros(dut.m1_in)
    set_fp32_zeros(dut.v0_in); set_fp32_zeros(dut.v1_in)

    ok = await run_2chip(dut)
    assert ok

    rdw = fp32_to_float(int(dut.reduced_dw[0][0].value))
    assert abs(rdw - 1.0) < 0.05, f"0.1+0.9=1.0, got {rdw}"

    p0 = fp32_to_float(int(dut.param0_out[0][0].value))
    p1 = fp32_to_float(int(dut.param1_out[0][0].value))
    assert abs(p0 - p1) < 1e-6, f"Same reduced_dW → same update: p0={p0}, p1={p1}"


@cocotb.test()
async def test_busy_done_sequence(dut):
    """busy goes high, done_pulse fires."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    set_bf16_identity(dut.x0_bf16); set_bf16_identity(dut.x1_bf16)
    set_bf16_identity(dut.w0_bf16); set_bf16_identity(dut.w1_bf16)
    set_bf16_identity(dut.dy0_bf16); set_bf16_identity(dut.dy1_bf16)
    set_fp32_diagonal(dut.param0_in, 1.0); set_fp32_diagonal(dut.param1_in, 1.0)
    set_fp32_zeros(dut.m0_in); set_fp32_zeros(dut.m1_in)
    set_fp32_zeros(dut.v0_in); set_fp32_zeros(dut.v1_in)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    saw_busy = False
    for _ in range(2000):
        await RisingEdge(dut.clk)
        if dut.busy.value == 1: saw_busy = True
        if dut.done_pulse.value == 1: break

    assert saw_busy
    assert dut.done_pulse.value == 1


@cocotb.test()
async def test_reset_mid_distributed_step(dut):
    """Reset mid-run returns to IDLE."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    set_bf16_identity(dut.x0_bf16); set_bf16_identity(dut.x1_bf16)
    set_bf16_identity(dut.dy0_bf16); set_bf16_identity(dut.dy1_bf16)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(20):
        await RisingEdge(dut.clk)

    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)

    assert dut.busy.value == 0
