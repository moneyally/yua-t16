"""
tb_trace_ring_freeze.py — cocotb testbench for trace_ring.sv freeze/filter

Tests:
  1. Freeze mode: events are dropped, drop_count increments
  2. Fatal-only mode: non-fatal events are ignored
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.trace_valid.value = 0
    dut.trace_type.value = 0
    dut.trace_fatal.value = 0
    dut.trace_payload.value = 0
    dut.ctrl_enable.value = 0
    dut.ctrl_freeze.value = 0
    dut.ctrl_fatal_only.value = 0
    dut.rd_addr.value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


async def write_event(dut, fatal=0, payload=0):
    dut.trace_valid.value = 1
    dut.trace_type.value = 1
    dut.trace_payload.value = payload
    dut.trace_fatal.value = fatal
    await RisingEdge(dut.clk)
    dut.trace_valid.value = 0


@cocotb.test()
async def test_freeze_drops_events(dut):
    """In freeze mode, events are dropped and drop_count increments."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut.ctrl_enable.value = 1

    # Write 5 events normally
    for i in range(5):
        await write_event(dut, payload=i)
    await RisingEdge(dut.clk)
    assert int(dut.ring_tail.value) == 5

    # Enable freeze
    dut.ctrl_freeze.value = 1

    # Try to write 3 more
    for i in range(3):
        await write_event(dut, payload=100 + i)
    await RisingEdge(dut.clk)

    # Tail should not have moved
    assert int(dut.ring_tail.value) == 5, "Tail moved despite freeze"
    # Drop count should be 3
    assert int(dut.drop_count.value) == 3, \
        f"drop_count={int(dut.drop_count.value)}, expected 3"


@cocotb.test()
async def test_fatal_only_filter(dut):
    """Fatal-only mode: only fatal events are recorded."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut.ctrl_enable.value = 1
    dut.ctrl_fatal_only.value = 1

    # Write non-fatal events
    for i in range(5):
        await write_event(dut, fatal=0, payload=i)
    await RisingEdge(dut.clk)
    assert int(dut.ring_tail.value) == 0, "Non-fatal events should be filtered"

    # Write fatal events
    for i in range(3):
        await write_event(dut, fatal=1, payload=100 + i)
    await RisingEdge(dut.clk)
    assert int(dut.ring_tail.value) == 3, "Fatal events should be recorded"
