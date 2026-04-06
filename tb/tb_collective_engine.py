"""
tb_collective_engine.py — cocotb testbench for collective_engine.sv

Tests:
  1. All-reduce SUM: zeros
  2. All-reduce SUM: known values (1.0 + 2.0 = 3.0)
  3. All-reduce SUM: negative values
  4. Invalid op_type error
  5. Invalid peer_mask error
  6. Done pulse + busy
  7. Reset mid-operation
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import struct

DIM = 16
OP_ALL_REDUCE_SUM = 0x01
OP_ALL_GATHER = 0x02
VALID_PEER_MASK = 0x03


def float_to_fp32(f):
    return struct.unpack('>I', struct.pack('>f', f))[0]


def fp32_to_float(bits):
    return struct.unpack('>f', struct.pack('>I', bits & 0xFFFFFFFF))[0]


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
    dut.op_type.value = 0
    dut.peer_mask.value = 0
    set_fp32_zeros(dut.local_in)
    set_fp32_zeros(dut.peer_in)
    await Timer(100, unit="ns")
    dut.rst_n.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)


async def run_collective(dut, op, mask=VALID_PEER_MASK, max_wait=500):
    dut.op_type.value = op
    dut.peer_mask.value = mask
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    for _ in range(max_wait):
        await RisingEdge(dut.clk)
        if dut.done_pulse.value == 1:
            return True
    return False


@cocotb.test()
async def test_all_reduce_sum_zero(dut):
    """0 + 0 = 0."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    set_fp32_zeros(dut.local_in)
    set_fp32_zeros(dut.peer_in)

    ok = await run_collective(dut, OP_ALL_REDUCE_SUM)
    assert ok, "done_pulse not seen"
    assert int(dut.err_code.value) == 0

    val = fp32_to_float(int(dut.result_out[0][0].value))
    assert abs(val) < 0.001, f"Expected 0, got {val}"


@cocotb.test()
async def test_all_reduce_sum_known_values(dut):
    """1.0 + 2.0 = 3.0 everywhere."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    set_fp32_scalar(dut.local_in, 1.0)
    set_fp32_scalar(dut.peer_in, 2.0)

    ok = await run_collective(dut, OP_ALL_REDUCE_SUM)
    assert ok

    val = fp32_to_float(int(dut.result_out[0][0].value))
    assert abs(val - 3.0) < 0.01, f"Expected 3.0, got {val}"

    val_last = fp32_to_float(int(dut.result_out[15][15].value))
    assert abs(val_last - 3.0) < 0.01, f"Expected 3.0 at [15][15], got {val_last}"


@cocotb.test()
async def test_all_reduce_sum_negative(dut):
    """(-1.5) + 0.5 = -1.0."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    set_fp32_scalar(dut.local_in, -1.5)
    set_fp32_scalar(dut.peer_in, 0.5)

    ok = await run_collective(dut, OP_ALL_REDUCE_SUM)
    assert ok
    assert int(dut.err_code.value) == 0

    val = fp32_to_float(int(dut.result_out[7][7].value))
    assert abs(val - (-1.0)) < 0.01, f"Expected -1.0, got {val}"


@cocotb.test()
async def test_invalid_op_error(dut):
    """ALL_GATHER (unsupported) → err_code=1."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    set_fp32_scalar(dut.local_in, 1.0)
    set_fp32_scalar(dut.peer_in, 1.0)

    ok = await run_collective(dut, OP_ALL_GATHER)
    assert ok, "done_pulse should fire even on error"
    assert int(dut.err_code.value) == 1, f"Expected err_code=1, got {int(dut.err_code.value)}"


@cocotb.test()
async def test_invalid_peer_mask_error(dut):
    """Invalid peer_mask (0x01 = self-only) → err_code=2."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    set_fp32_scalar(dut.local_in, 1.0)
    set_fp32_scalar(dut.peer_in, 1.0)

    ok = await run_collective(dut, OP_ALL_REDUCE_SUM, mask=0x01)
    assert ok
    assert int(dut.err_code.value) == 2, f"Expected err_code=2, got {int(dut.err_code.value)}"


@cocotb.test()
async def test_done_pulse_and_busy(dut):
    """busy high during reduction, done_pulse at end."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    set_fp32_scalar(dut.local_in, 1.0)
    set_fp32_scalar(dut.peer_in, 1.0)

    dut.op_type.value = OP_ALL_REDUCE_SUM
    dut.peer_mask.value = VALID_PEER_MASK
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
    assert dut.done_pulse.value == 1


@cocotb.test()
async def test_reset_mid_operation(dut):
    """Reset during collective returns to IDLE."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    set_fp32_scalar(dut.local_in, 5.0)
    set_fp32_scalar(dut.peer_in, 5.0)

    dut.op_type.value = OP_ALL_REDUCE_SUM
    dut.peer_mask.value = VALID_PEER_MASK
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for _ in range(10):
        await RisingEdge(dut.clk)

    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)

    assert dut.busy.value == 0, "Should be idle after reset"
    assert dut.done_pulse.value == 0
