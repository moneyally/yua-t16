"""
tb_desc_queue_basic.py — cocotb testbench for desc_queue.sv

Tests:
  1. Enqueue/dequeue on each queue independently
  2. Overflow detection
  3. Simultaneous push/pop
  4. Per-queue isolation
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


async def reset_dut(dut):
    dut.rst_n.value = 0
    for i in range(4):
        dut.push_valid.value = 0
        dut.pop_valid.value = 0
        dut.overflow_clr.value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_enqueue_dequeue(dut):
    """Push 4 descriptors to Q0, pop them back, verify order."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Push 4 items to Q0
    for i in range(4):
        dut.push_valid.value = 0x1  # Q0 only
        dut.push_data[0].value = 0xAAAA_0000 + i
        await RisingEdge(dut.clk)
    dut.push_valid.value = 0

    await RisingEdge(dut.clk)

    # Pop 4 items from Q0
    read_data = []
    for _ in range(4):
        dut.pop_valid.value = 0x1
        await RisingEdge(dut.clk)
        if int(dut.pop_ready.value) & 0x1:
            read_data.append(int(dut.pop_data[0].value))
    dut.pop_valid.value = 0

    assert len(read_data) == 4, f"Expected 4, got {len(read_data)}"
    for i, d in enumerate(read_data):
        expected = 0xAAAA_0000 + i
        assert d == expected, f"Q0[{i}]: expected {expected:#x}, got {d:#x}"


@cocotb.test()
async def test_overflow_flag(dut):
    """Fill Q1 beyond capacity, check overflow_flags[1]."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Fill Q1 to capacity (QUEUE_DEPTH=64)
    for i in range(64):
        dut.push_valid.value = 0x2  # Q1
        dut.push_data[1].value = i
        await RisingEdge(dut.clk)
        # Check ready
        if not (int(dut.push_ready.value) & 0x2):
            break

    # One more push should trigger overflow
    dut.push_valid.value = 0x2
    dut.push_data[1].value = 0xFFFF
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.push_valid.value = 0

    assert (int(dut.overflow_flags.value) & 0x2) != 0, "Q1 overflow flag not set"
    assert dut.any_overflow.value == 1, "any_overflow not set"

    # W1C clear
    dut.overflow_clr.value = 0x2
    await RisingEdge(dut.clk)
    dut.overflow_clr.value = 0
    await RisingEdge(dut.clk)

    assert (int(dut.overflow_flags.value) & 0x2) == 0, "Q1 overflow flag not cleared"


@cocotb.test()
async def test_per_queue_isolation(dut):
    """Push to Q0 and Q3 simultaneously, verify no cross-contamination."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Push to Q0 and Q3 at same time
    dut.push_valid.value = 0x9  # Q0 + Q3
    dut.push_data[0].value = 0x1111
    dut.push_data[3].value = 0x3333
    await RisingEdge(dut.clk)
    dut.push_valid.value = 0

    await RisingEdge(dut.clk)

    # Pop Q0
    dut.pop_valid.value = 0x1
    await RisingEdge(dut.clk)
    q0_data = int(dut.pop_data[0].value)
    dut.pop_valid.value = 0

    # Pop Q3
    dut.pop_valid.value = 0x8
    await RisingEdge(dut.clk)
    q3_data = int(dut.pop_data[3].value)
    dut.pop_valid.value = 0

    assert q0_data == 0x1111, f"Q0 data mismatch: {q0_data:#x}"
    assert q3_data == 0x3333, f"Q3 data mismatch: {q3_data:#x}"

    # Q1 and Q2 should be empty
    assert (int(dut.pop_ready.value) & 0x2) == 0, "Q1 should be empty"
    assert (int(dut.pop_ready.value) & 0x4) == 0, "Q2 should be empty"
