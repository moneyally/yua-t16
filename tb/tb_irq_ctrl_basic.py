"""
tb_irq_ctrl_basic.py — cocotb testbench for irq_ctrl.sv

Tests:
  1. Source fires -> pending set -> irq_out asserted
  2. Mask blocks IRQ output
  3. Force inject sets pending
  4. Fatal cause latch
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.irq_sources.value = 0
    dut.pending_w1c_en.value = 0
    dut.pending_w1c_data.value = 0
    dut.mask_wr_en.value = 0
    dut.mask_wr_data.value = 0
    dut.force_wr_en.value = 0
    dut.force_wr_data.value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_source_sets_pending(dut):
    """IRQ source pulse sets pending bit."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Unmask bit 0 (IRQ_DESC_DONE)
    dut.mask_wr_en.value = 1
    dut.mask_wr_data.value = 0xFFFF_FFFE  # unmask bit 0
    await RisingEdge(dut.clk)
    dut.mask_wr_en.value = 0

    # Fire source 0
    dut.irq_sources.value = 0x0000_0001
    await RisingEdge(dut.clk)
    dut.irq_sources.value = 0

    await RisingEdge(dut.clk)
    assert (int(dut.irq_pending.value) & 0x1) == 1, "Pending bit 0 not set"
    assert dut.irq_out.value == 1, "irq_out not asserted"


@cocotb.test()
async def test_mask_blocks_output(dut):
    """Masked IRQ: pending is set but irq_out is not asserted."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Leave all masked (default after reset)
    # Fire source 1
    dut.irq_sources.value = 0x0000_0002
    await RisingEdge(dut.clk)
    dut.irq_sources.value = 0

    await RisingEdge(dut.clk)
    assert (int(dut.irq_pending.value) & 0x2) == 2, "Pending bit 1 not set"
    assert dut.irq_out.value == 0, "irq_out should be 0 when masked"


@cocotb.test()
async def test_force_inject(dut):
    """IRQ_FORCE sets pending without hardware source."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Unmask bit 5
    dut.mask_wr_en.value = 1
    dut.mask_wr_data.value = 0xFFFF_FFDF  # unmask bit 5
    await RisingEdge(dut.clk)
    dut.mask_wr_en.value = 0

    # Force inject bit 5
    dut.force_wr_en.value = 1
    dut.force_wr_data.value = 0x0000_0020
    await RisingEdge(dut.clk)
    dut.force_wr_en.value = 0

    await RisingEdge(dut.clk)
    assert (int(dut.irq_pending.value) & 0x20) == 0x20, "Force inject did not set pending"
    assert dut.irq_out.value == 1, "irq_out not asserted after force"


@cocotb.test()
async def test_fatal_cause_latch(dut):
    """Fatal IRQ source latches in irq_cause_last."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Fire fatal source: bit 2 (IRQ_DMA_ERROR)
    dut.irq_sources.value = 0x0000_0004
    await RisingEdge(dut.clk)
    dut.irq_sources.value = 0

    await RisingEdge(dut.clk)
    assert (int(dut.irq_cause_last.value) & 0x4) == 0x4, \
        "Fatal cause not latched for DMA_ERROR"

    # Fire another fatal: bit 10 (IRQ_WATCHDOG)
    dut.irq_sources.value = 0x0000_0400
    await RisingEdge(dut.clk)
    dut.irq_sources.value = 0

    await RisingEdge(dut.clk)
    # cause_last should now be WATCHDOG (overwrites previous)
    assert (int(dut.irq_cause_last.value) & 0x400) == 0x400, \
        "Fatal cause not latched for WATCHDOG"
