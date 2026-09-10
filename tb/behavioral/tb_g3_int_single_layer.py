"""
tb_g3_int_single_layer.py — G3-INT-002: Single-layer training step

Path: forward(Y=X*W) → backward(dW=X^T*dY) → optimizer(W'=Adam(W,dW))
Active region: 16×16. dW path only.

Tests:
  1. Known values: X=I, W=diag(2), dY=diag(0.5) → verify param update
  2. Zero grad: dY=0 → param unchanged
  3. Busy/done sequence
  4. Reset mid-training
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


def set_bf16_identity(dut_mat, scale=1.0):
    for i in range(DIM):
        for j in range(DIM):
            dut_mat[i][j].value = float_to_bf16(scale if i == j else 0.0)


def set_bf16_zeros(dut_mat):
    for i in range(DIM):
        for j in range(DIM):
            dut_mat[i][j].value = 0


def set_fp32_scalar(dut_mat, val):
    bits = float_to_fp32(val)
    for i in range(DIM):
        for j in range(DIM):
            dut_mat[i][j].value = bits


def set_fp32_zeros(dut_mat):
    for i in range(DIM):
        for j in range(DIM):
            dut_mat[i][j].value = 0


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.adamw_enable.value = 0
    dut.lr_fp32.value = float_to_fp32(0.001)
    dut.beta1_fp32.value = float_to_fp32(0.9)
    dut.beta2_fp32.value = float_to_fp32(0.999)
    dut.epsilon_fp32.value = float_to_fp32(1e-8)
    dut.weight_decay_fp32.value = float_to_fp32(0.0)
    set_bf16_zeros(dut.x_bf16)
    set_bf16_zeros(dut.w_bf16)
    set_bf16_zeros(dut.dy_bf16)
    set_fp32_zeros(dut.param_in)
    set_fp32_zeros(dut.m_in)
    set_fp32_zeros(dut.v_in)
    await Timer(100, unit="ns")
    dut.rst_n.value = 1
    for _ in range(5):
        await RisingEdge(dut.clk)


async def run_training_step(dut, max_wait=1000):
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    for _ in range(max_wait):
        await RisingEdge(dut.clk)
        if dut.train_done_pulse.value == 1:
            return True
    return False


def python_adam(p, g, m, v, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    m_new = b1 * m + (1 - b1) * g
    v_new = b2 * v + (1 - b2) * g * g
    update = lr * m_new / (math.sqrt(max(v_new, 0)) + eps)
    return p - update, m_new, v_new


@cocotb.test()
async def test_single_layer_training_step(dut):
    """X=I, W=diag(2), dY=diag(0.5) → dW = X^T*dY = diag(0.5)
    → Adam update on param_in=diag(2) with grad=diag(0.5)."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Setup
    set_bf16_identity(dut.x_bf16, 1.0)    # X = I
    set_bf16_identity(dut.w_bf16, 2.0)    # W = 2*I
    set_bf16_identity(dut.dy_bf16, 0.5)   # dY = 0.5*I

    # Optimizer state: param=2.0 on diagonal, m=0, v=0
    for i in range(DIM):
        for j in range(DIM):
            val = 2.0 if i == j else 0.0
            dut.param_in[i][j].value = float_to_fp32(val)
    set_fp32_zeros(dut.m_in)
    set_fp32_zeros(dut.v_in)

    ok = await run_training_step(dut)
    assert ok, "train_done_pulse not seen"
    assert int(dut.err_code.value) == 0, f"err_code={int(dut.err_code.value)}"

    # dW should be diag(0.5): dW = X^T * dY = I * 0.5*I = 0.5*I
    dw00 = fp32_to_float(int(dut.dw_result[0][0].value))
    dw01 = fp32_to_float(int(dut.dw_result[0][1].value))
    dut._log.info(f"dW[0][0]={dw00}, dW[0][1]={dw01}")
    assert abs(dw00 - 0.5) < 0.05, f"dW[0][0] expected ~0.5, got {dw00}"
    assert abs(dw01) < 0.01, f"dW[0][1] expected ~0, got {dw01}"

    # param_out should be Adam(2.0, grad=0.5)
    p_ref, m_ref, v_ref = python_adam(2.0, 0.5, 0.0, 0.0)
    p_out = fp32_to_float(int(dut.param_out[0][0].value))
    m_out = fp32_to_float(int(dut.m_out[0][0].value))

    dut._log.info(f"param_out[0][0]={p_out} (ref={p_ref})")
    dut._log.info(f"m_out[0][0]={m_out} (ref={m_ref})")

    assert abs(p_out - p_ref) < abs(p_ref) * 0.02 + 1e-5, \
        f"param: got {p_out}, expected {p_ref}"
    assert abs(m_out - m_ref) < abs(m_ref) * 0.02 + 1e-5, \
        f"m: got {m_out}, expected {m_ref}"

    # Off-diagonal param should be unchanged (grad=0)
    p_off = fp32_to_float(int(dut.param_out[0][1].value))
    assert abs(p_off) < 0.001, f"Off-diag param should be ~0, got {p_off}"


@cocotb.test()
async def test_zero_grad_no_change(dut):
    """dY=0 → dW=0 → param unchanged."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    set_bf16_identity(dut.x_bf16, 1.0)
    set_bf16_identity(dut.w_bf16, 3.0)
    set_bf16_zeros(dut.dy_bf16)           # zero gradient

    set_fp32_scalar(dut.param_in, 5.0)
    set_fp32_zeros(dut.m_in)
    set_fp32_zeros(dut.v_in)

    ok = await run_training_step(dut)
    assert ok, "train_done_pulse not seen"

    # param should be unchanged (grad=0, update=0)
    p_out = fp32_to_float(int(dut.param_out[0][0].value))
    assert abs(p_out - 5.0) < 0.01, f"Param should be ~5.0, got {p_out}"


@cocotb.test()
async def test_busy_done_sequence(dut):
    """busy goes high, train_done_pulse fires at end."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    set_bf16_identity(dut.x_bf16, 1.0)
    set_bf16_identity(dut.w_bf16, 1.0)
    set_bf16_identity(dut.dy_bf16, 1.0)
    set_fp32_scalar(dut.param_in, 1.0)
    set_fp32_zeros(dut.m_in)
    set_fp32_zeros(dut.v_in)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    saw_busy = False
    for _ in range(1000):
        await RisingEdge(dut.clk)
        if dut.busy.value == 1:
            saw_busy = True
        if dut.train_done_pulse.value == 1:
            break

    assert saw_busy, "busy should have been high"
    assert dut.train_done_pulse.value == 1


@cocotb.test()
async def test_reset_mid_training(dut):
    """Reset mid-step returns to IDLE."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    set_bf16_identity(dut.x_bf16, 1.0)
    set_bf16_identity(dut.w_bf16, 1.0)
    set_bf16_identity(dut.dy_bf16, 1.0)
    set_fp32_scalar(dut.param_in, 1.0)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(15):
        await RisingEdge(dut.clk)

    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)

    assert dut.busy.value == 0, "Should be idle after reset"
