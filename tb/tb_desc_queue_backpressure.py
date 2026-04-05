"""
tb_desc_queue_backpressure.py — cocotb testbench for desc_queue.sv

Tests:
  1. Backpressure: push stalls when queue full
  2. Simultaneous push and pop at full capacity
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.push_valid.value = 0
    dut.pop_valid.value = 0
    dut.overflow_clr.value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_backpressure_stall(dut):
    """Fill Q2, verify push_ready deasserts, then drain and resume."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    pushed = 0
    # Fill Q2
    for i in range(70):  # more than DEPTH=64
        dut.push_valid.value = 0x4  # Q2
        dut.push_data[2].value = i
        await RisingEdge(dut.clk)
        if int(dut.push_ready.value) & 0x4:
            pushed += 1
        else:
            break
    dut.push_valid.value = 0

    assert pushed == 64, f"Expected 64 pushes before full, got {pushed}"

    # Push ready should be 0 for Q2
    assert (int(dut.push_ready.value) & 0x4) == 0, "Q2 push_ready should be 0 when full"

    # Pop one item
    dut.pop_valid.value = 0x4
    await RisingEdge(dut.clk)
    dut.pop_valid.value = 0
    await RisingEdge(dut.clk)

    # Now push_ready should be back
    assert (int(dut.push_ready.value) & 0x4) != 0, "Q2 push_ready should recover after pop"


@cocotb.test()
async def test_simultaneous_push_pop(dut):
    """Push and pop on Q0 simultaneously — steady state throughput."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Pre-fill Q0 with some data
    for i in range(8):
        dut.push_valid.value = 0x1
        dut.push_data[0].value = i
        await RisingEdge(dut.clk)
    dut.push_valid.value = 0
    await RisingEdge(dut.clk)

    # Simultaneous push and pop for 16 cycles
    popped = []
    for i in range(16):
        dut.push_valid.value = 0x1
        dut.push_data[0].value = 100 + i
        dut.pop_valid.value = 0x1
        await RisingEdge(dut.clk)
        if int(dut.pop_ready.value) & 0x1:
            popped.append(int(dut.pop_data[0].value))

    dut.push_valid.value = 0
    dut.pop_valid.value = 0

    # Should have gotten data out every cycle (queue was non-empty)
    assert len(popped) >= 8, f"Expected at least 8 pops, got {len(popped)}"
