"""
tb_trace_ring_wrap.py — cocotb testbench for trace_ring.sv wraparound

Tests:
  1. Write 1024+ events, verify wraparound behavior
  2. Head advances when tail overwrites
  3. wrap_irq_pulse fires on wraparound
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


DEPTH = 1024


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


async def write_event(dut, event_type=1, payload=0, fatal=0):
    dut.trace_valid.value = 1
    dut.trace_type.value = event_type
    dut.trace_payload.value = payload
    dut.trace_fatal.value = fatal
    await RisingEdge(dut.clk)
    dut.trace_valid.value = 0


@cocotb.test()
async def test_basic_write_and_tail_advance(dut):
    """Write events, tail should advance."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut.ctrl_enable.value = 1

    for i in range(10):
        await write_event(dut, payload=i)

    await RisingEdge(dut.clk)
    assert int(dut.ring_tail.value) == 10, f"tail={int(dut.ring_tail.value)}, expected 10"
    assert int(dut.ring_head.value) == 0, "head should be 0"


@cocotb.test()
async def test_wraparound(dut):
    """Write DEPTH+10 events, head should advance past 0."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut.ctrl_enable.value = 1

    for i in range(DEPTH + 10):
        await write_event(dut, payload=i)

    await RisingEdge(dut.clk)

    tail = int(dut.ring_tail.value)
    head = int(dut.ring_head.value)

    assert tail == 10, f"tail={tail}, expected 10 (after wrap)"
    assert head > 0, f"head should have advanced, got {head}"


@cocotb.test()
async def test_wrap_irq_pulse(dut):
    """wrap_irq_pulse should fire when tail wraps around and meets head."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut.ctrl_enable.value = 1

    irq_seen = False
    for i in range(DEPTH + 5):
        await write_event(dut, payload=i)
        await RisingEdge(dut.clk)
        if dut.wrap_irq_pulse.value == 1:
            irq_seen = True

    assert irq_seen, "wrap_irq_pulse never fired"
