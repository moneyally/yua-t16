"""
tb_backward_engine.py — cocotb testbench for backward_engine.sv

Tests:
  1. dW zero case: all zeros → dW = 0
  2. dX zero case: all zeros → dX = 0
  3. dW known values: X=I, dY=diag(2) → dW = diag(2)
  4. dX known values: dY=I, W=diag(3) → dX = diag(3)
  5. Invalid mode error
  6. Done pulse + busy
  7. Reset mid-operation

Math:
  Forward: Y = X * W
  dW = X^T * dY   (MODE_DW = 1)
  dX = dY * W^T   (MODE_DX = 2)
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import struct

DIM = 16
MODE_DW = 1
MODE_DX = 2


def float_to_bf16(f):
    fp32_bits = struct.unpack('>I', struct.pack('>f', f))[0]
    return (fp32_bits >> 16) & 0xFFFF


def fp32_bits_to_float(bits):
    return struct.unpack('>f', struct.pack('>I', bits & 0xFFFFFFFF))[0]


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.mode.value = 0
    dut.acc_clr.value = 0
    for i in range(DIM):
        for j in range(DIM):
            dut.x_in[i][j].value = 0
            dut.w_in[i][j].value = 0
            dut.dy_in[i][j].value = 0
    await Timer(100, unit="ns")
    dut.rst_n.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)


def set_identity(dut_matrix, scale=1.0):
    """Set a DIM×DIM matrix to scale*I."""
    for i in range(DIM):
        for j in range(DIM):
            val = scale if i == j else 0.0
            dut_matrix[i][j].value = float_to_bf16(val)


def set_zeros(dut_matrix):
    for i in range(DIM):
        for j in range(DIM):
            dut_matrix[i][j].value = 0


async def run_backward(dut, mode, max_wait=200):
    """Start backward engine and wait for done."""
    dut.mode.value = mode
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(max_wait):
        await RisingEdge(dut.clk)
        if dut.done_pulse.value == 1:
            return True
    return False


@cocotb.test()
async def test_dw_zero_case(dut):
    """All zeros → dW = 0."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Clear
    dut.acc_clr.value = 1
    await RisingEdge(dut.clk)
    dut.acc_clr.value = 0
    await RisingEdge(dut.clk)

    set_zeros(dut.x_in)
    set_zeros(dut.dy_in)

    ok = await run_backward(dut, MODE_DW)
    assert ok, "done_pulse not seen"
    assert int(dut.err_code.value) == 0

    val = fp32_bits_to_float(int(dut.result[0][0].value))
    assert abs(val) < 0.01, f"dW[0][0] should be ~0, got {val}"


@cocotb.test()
async def test_dx_zero_case(dut):
    """All zeros → dX = 0."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut.acc_clr.value = 1
    await RisingEdge(dut.clk)
    dut.acc_clr.value = 0
    await RisingEdge(dut.clk)

    set_zeros(dut.dy_in)
    set_zeros(dut.w_in)

    ok = await run_backward(dut, MODE_DX)
    assert ok, "done_pulse not seen"
    assert int(dut.err_code.value) == 0

    val = fp32_bits_to_float(int(dut.result[0][0].value))
    assert abs(val) < 0.01, f"dX[0][0] should be ~0, got {val}"


@cocotb.test()
async def test_dw_known_values(dut):
    """X=I, dY=diag(2) → dW = X^T * dY = I * diag(2) = diag(2)."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut.acc_clr.value = 1
    await RisingEdge(dut.clk)
    dut.acc_clr.value = 0
    await RisingEdge(dut.clk)

    set_identity(dut.x_in, 1.0)    # X = I
    set_identity(dut.dy_in, 2.0)   # dY = 2*I

    ok = await run_backward(dut, MODE_DW)
    assert ok, "done_pulse not seen"
    assert int(dut.err_code.value) == 0

    # dW[0][0] should be 2.0
    val00 = fp32_bits_to_float(int(dut.result[0][0].value))
    assert abs(val00 - 2.0) < 0.1, f"dW[0][0] expected ~2.0, got {val00}"

    # dW[0][1] should be 0
    val01 = fp32_bits_to_float(int(dut.result[0][1].value))
    assert abs(val01) < 0.01, f"dW[0][1] expected ~0, got {val01}"

    # dW[5][5] should be 2.0
    val55 = fp32_bits_to_float(int(dut.result[5][5].value))
    assert abs(val55 - 2.0) < 0.1, f"dW[5][5] expected ~2.0, got {val55}"


@cocotb.test()
async def test_dx_known_values(dut):
    """dY=I, W=diag(3) → dX = dY * W^T = I * diag(3)^T = diag(3)."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut.acc_clr.value = 1
    await RisingEdge(dut.clk)
    dut.acc_clr.value = 0
    await RisingEdge(dut.clk)

    set_identity(dut.dy_in, 1.0)   # dY = I
    set_identity(dut.w_in, 3.0)    # W = 3*I

    ok = await run_backward(dut, MODE_DX)
    assert ok, "done_pulse not seen"
    assert int(dut.err_code.value) == 0

    val00 = fp32_bits_to_float(int(dut.result[0][0].value))
    assert abs(val00 - 3.0) < 0.1, f"dX[0][0] expected ~3.0, got {val00}"

    val01 = fp32_bits_to_float(int(dut.result[0][1].value))
    assert abs(val01) < 0.01, f"dX[0][1] expected ~0, got {val01}"


@cocotb.test()
async def test_invalid_mode_error(dut):
    """Mode=0 → err_code=1."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    ok = await run_backward(dut, 0)  # invalid
    assert ok, "done_pulse not seen on error"
    assert int(dut.err_code.value) == 1, f"Expected err_code=1, got {int(dut.err_code.value)}"


@cocotb.test()
async def test_done_pulse_and_busy(dut):
    """busy goes high, done_pulse fires at end."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut.acc_clr.value = 1
    await RisingEdge(dut.clk)
    dut.acc_clr.value = 0
    await RisingEdge(dut.clk)

    set_identity(dut.x_in)
    set_identity(dut.dy_in)

    dut.mode.value = MODE_DW
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    saw_busy = False
    for _ in range(100):
        await RisingEdge(dut.clk)
        if dut.busy.value == 1:
            saw_busy = True
        if dut.done_pulse.value == 1:
            break

    assert saw_busy, "busy should have been high"
    assert dut.done_pulse.value == 1, "done_pulse should fire"


@cocotb.test()
async def test_reset_mid_operation(dut):
    """Reset during backward should return to IDLE cleanly."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    set_identity(dut.x_in)
    set_identity(dut.dy_in)

    dut.mode.value = MODE_DW
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait a few cycles then reset
    for _ in range(5):
        await RisingEdge(dut.clk)

    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    assert dut.busy.value == 0, "Should be idle after reset"
    assert dut.done_pulse.value == 0, "No spurious done after reset"
