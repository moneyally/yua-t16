"""
tb_optimizer_unit.py — cocotb testbench for optimizer_unit.sv

Tests:
  1. Adam zero grad: no change to params
  2. Adam known values: verify m/v/param update
  3. AdamW known values: verify weight decay effect
  4. AdamW with wd=0 matches Adam
  5. Done pulse + busy
  6. Reset mid-operation
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import struct
import math

DIM = 16


def float_to_fp32(f):
    return struct.unpack('>I', struct.pack('>f', f))[0]


def fp32_to_float(bits):
    return struct.unpack('>f', struct.pack('>I', bits & 0xFFFFFFFF))[0]


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.adamw_enable.value = 0
    dut.lr_fp32.value = float_to_fp32(0.001)
    dut.beta1_fp32.value = float_to_fp32(0.9)
    dut.beta2_fp32.value = float_to_fp32(0.999)
    dut.epsilon_fp32.value = float_to_fp32(1e-8)
    dut.weight_decay_fp32.value = float_to_fp32(0.0)
    for i in range(DIM):
        for j in range(DIM):
            dut.param_in[i][j].value = 0
            dut.grad_in[i][j].value = 0
            dut.m_in[i][j].value = 0
            dut.v_in[i][j].value = 0
    await Timer(100, unit="ns")
    dut.rst_n.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)


async def run_optimizer(dut, adamw=False, max_wait=500):
    dut.adamw_enable.value = 1 if adamw else 0
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    for _ in range(max_wait):
        await RisingEdge(dut.clk)
        if dut.done_pulse.value == 1:
            return True
    return False


def set_scalar_all(dut_matrix, val_fp32):
    """Set all DIM×DIM elements to the same FP32 value."""
    bits = float_to_fp32(val_fp32)
    for i in range(DIM):
        for j in range(DIM):
            dut_matrix[i][j].value = bits


def python_adam_step(p, g, m, v, lr, b1, b2, eps, wd=0.0, adamw=False):
    """Python reference Adam/AdamW for single element."""
    m_new = b1 * m + (1 - b1) * g
    v_new = b2 * v + (1 - b2) * g * g
    update = lr * m_new / (math.sqrt(max(v_new, 0)) + eps)
    if adamw:
        p_new = p - update - lr * wd * p
    else:
        p_new = p - update
    return p_new, m_new, v_new


@cocotb.test()
async def test_adam_zero_grad(dut):
    """Zero gradient: m/v stay 0, param unchanged."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    set_scalar_all(dut.param_in, 1.0)
    set_scalar_all(dut.grad_in, 0.0)
    set_scalar_all(dut.m_in, 0.0)
    set_scalar_all(dut.v_in, 0.0)

    ok = await run_optimizer(dut, adamw=False)
    assert ok, "done_pulse not seen"

    # param should be unchanged (update = lr * 0 / (0 + eps) = 0)
    p = fp32_to_float(int(dut.param_out[0][0].value))
    assert abs(p - 1.0) < 0.001, f"Param should be ~1.0, got {p}"

    m = fp32_to_float(int(dut.m_out[0][0].value))
    assert abs(m) < 1e-6, f"m should be ~0, got {m}"


@cocotb.test()
async def test_adam_known_values(dut):
    """Known Adam step: p=1.0, g=0.5, m=0, v=0."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    lr, b1, b2, eps = 0.001, 0.9, 0.999, 1e-8
    p0, g0, m0, v0 = 1.0, 0.5, 0.0, 0.0

    dut.lr_fp32.value = float_to_fp32(lr)
    dut.beta1_fp32.value = float_to_fp32(b1)
    dut.beta2_fp32.value = float_to_fp32(b2)
    dut.epsilon_fp32.value = float_to_fp32(eps)

    set_scalar_all(dut.param_in, p0)
    set_scalar_all(dut.grad_in, g0)
    set_scalar_all(dut.m_in, m0)
    set_scalar_all(dut.v_in, v0)

    ok = await run_optimizer(dut, adamw=False)
    assert ok

    # Python reference
    p_ref, m_ref, v_ref = python_adam_step(p0, g0, m0, v0, lr, b1, b2, eps)

    p_out = fp32_to_float(int(dut.param_out[0][0].value))
    m_out = fp32_to_float(int(dut.m_out[0][0].value))
    v_out = fp32_to_float(int(dut.v_out[0][0].value))

    # Tolerance: FP32 behavioral, allow 1% relative error
    assert abs(p_out - p_ref) < abs(p_ref) * 0.01 + 1e-6, \
        f"param: got {p_out}, expected {p_ref}"
    assert abs(m_out - m_ref) < abs(m_ref) * 0.01 + 1e-6, \
        f"m: got {m_out}, expected {m_ref}"
    assert abs(v_out - v_ref) < abs(v_ref) * 0.01 + 1e-6, \
        f"v: got {v_out}, expected {v_ref}"


@cocotb.test()
async def test_adamw_known_values(dut):
    """AdamW with weight_decay=0.01."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    lr, b1, b2, eps, wd = 0.001, 0.9, 0.999, 1e-8, 0.01
    p0, g0, m0, v0 = 1.0, 0.5, 0.0, 0.0

    dut.lr_fp32.value = float_to_fp32(lr)
    dut.beta1_fp32.value = float_to_fp32(b1)
    dut.beta2_fp32.value = float_to_fp32(b2)
    dut.epsilon_fp32.value = float_to_fp32(eps)
    dut.weight_decay_fp32.value = float_to_fp32(wd)

    set_scalar_all(dut.param_in, p0)
    set_scalar_all(dut.grad_in, g0)
    set_scalar_all(dut.m_in, m0)
    set_scalar_all(dut.v_in, v0)

    ok = await run_optimizer(dut, adamw=True)
    assert ok

    p_ref, m_ref, v_ref = python_adam_step(p0, g0, m0, v0, lr, b1, b2, eps, wd, adamw=True)

    p_out = fp32_to_float(int(dut.param_out[0][0].value))

    assert abs(p_out - p_ref) < abs(p_ref) * 0.01 + 1e-6, \
        f"AdamW param: got {p_out}, expected {p_ref}"

    # AdamW should differ from Adam
    p_adam, _, _ = python_adam_step(p0, g0, m0, v0, lr, b1, b2, eps, 0.0, adamw=False)
    assert abs(p_out - p_adam) > 1e-7, "AdamW should differ from Adam when wd>0"


@cocotb.test()
async def test_weight_decay_zero_matches_adam(dut):
    """AdamW with wd=0 should match Adam exactly."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    lr, b1, b2, eps = 0.001, 0.9, 0.999, 1e-8
    p0, g0 = 2.0, 0.3

    dut.lr_fp32.value = float_to_fp32(lr)
    dut.beta1_fp32.value = float_to_fp32(b1)
    dut.beta2_fp32.value = float_to_fp32(b2)
    dut.epsilon_fp32.value = float_to_fp32(eps)
    dut.weight_decay_fp32.value = float_to_fp32(0.0)

    set_scalar_all(dut.param_in, p0)
    set_scalar_all(dut.grad_in, g0)
    set_scalar_all(dut.m_in, 0.0)
    set_scalar_all(dut.v_in, 0.0)

    # Run Adam
    ok = await run_optimizer(dut, adamw=False)
    assert ok
    p_adam = fp32_to_float(int(dut.param_out[0][0].value))

    # Reset and run AdamW with wd=0
    await reset_dut(dut)
    dut.lr_fp32.value = float_to_fp32(lr)
    dut.beta1_fp32.value = float_to_fp32(b1)
    dut.beta2_fp32.value = float_to_fp32(b2)
    dut.epsilon_fp32.value = float_to_fp32(eps)
    dut.weight_decay_fp32.value = float_to_fp32(0.0)

    set_scalar_all(dut.param_in, p0)
    set_scalar_all(dut.grad_in, g0)
    set_scalar_all(dut.m_in, 0.0)
    set_scalar_all(dut.v_in, 0.0)

    ok = await run_optimizer(dut, adamw=True)
    assert ok
    p_adamw = fp32_to_float(int(dut.param_out[0][0].value))

    assert abs(p_adam - p_adamw) < 1e-6, \
        f"AdamW(wd=0) should match Adam: adam={p_adam}, adamw={p_adamw}"


@cocotb.test()
async def test_done_pulse_and_busy(dut):
    """busy goes high, done_pulse fires."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    set_scalar_all(dut.param_in, 1.0)
    set_scalar_all(dut.grad_in, 0.1)
    set_scalar_all(dut.m_in, 0.0)
    set_scalar_all(dut.v_in, 0.0)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    saw_busy = False
    for _ in range(500):
        await RisingEdge(dut.clk)
        if dut.busy.value == 1:
            saw_busy = True
        if dut.done_pulse.value == 1:
            break

    assert saw_busy, "busy should have been high"
    assert dut.done_pulse.value == 1, "done_pulse should fire"


@cocotb.test()
async def test_reset_mid_operation(dut):
    """Reset during optimizer returns to IDLE."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    set_scalar_all(dut.param_in, 1.0)
    set_scalar_all(dut.grad_in, 0.1)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(10):
        await RisingEdge(dut.clk)

    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    assert dut.busy.value == 0, "Should be idle after reset"
