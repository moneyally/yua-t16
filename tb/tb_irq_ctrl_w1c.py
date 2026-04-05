"""
tb_irq_ctrl_w1c.py — cocotb testbench for irq_ctrl.sv W1C behavior

Tests:
  1. W1C clears specific pending bits
  2. W1C only clears written bits (others preserved)
  3. Source re-fires after clear
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
async def test_w1c_clear(dut):
    """Write 1 to pending bit clears it."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Set bits 0 and 1
    dut.irq_sources.value = 0x0000_0003
    await RisingEdge(dut.clk)
    dut.irq_sources.value = 0
    await RisingEdge(dut.clk)

    assert (int(dut.irq_pending.value) & 0x3) == 0x3

    # W1C clear bit 0 only
    dut.pending_w1c_en.value = 1
    dut.pending_w1c_data.value = 0x0000_0001
    await RisingEdge(dut.clk)
    dut.pending_w1c_en.value = 0

    await RisingEdge(dut.clk)
    pending = int(dut.irq_pending.value)
    assert (pending & 0x1) == 0, "Bit 0 should be cleared"
    assert (pending & 0x2) == 0x2, "Bit 1 should still be set"


@cocotb.test()
async def test_source_refire_after_clear(dut):
    """After W1C clear, same source can re-set the pending bit."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Set bit 3
    dut.irq_sources.value = 0x0000_0008
    await RisingEdge(dut.clk)
    dut.irq_sources.value = 0
    await RisingEdge(dut.clk)
    assert (int(dut.irq_pending.value) & 0x8) == 0x8

    # Clear bit 3
    dut.pending_w1c_en.value = 1
    dut.pending_w1c_data.value = 0x0000_0008
    await RisingEdge(dut.clk)
    dut.pending_w1c_en.value = 0
    await RisingEdge(dut.clk)
    assert (int(dut.irq_pending.value) & 0x8) == 0

    # Re-fire bit 3
    dut.irq_sources.value = 0x0000_0008
    await RisingEdge(dut.clk)
    dut.irq_sources.value = 0
    await RisingEdge(dut.clk)
    assert (int(dut.irq_pending.value) & 0x8) == 0x8, "Re-fire failed after W1C"


@cocotb.test()
async def test_set_wins_over_clear(dut):
    """SET WINS policy: HW source + W1C on same bit same cycle -> bit stays SET."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # First set bit 0 so there's something to clear
    dut.irq_sources.value = 0x0000_0001
    await RisingEdge(dut.clk)
    dut.irq_sources.value = 0
    await RisingEdge(dut.clk)
    assert (int(dut.irq_pending.value) & 0x1) == 1

    # Same cycle: HW fires bit 0 AND SW clears bit 0
    dut.irq_sources.value = 0x0000_0001    # HW set
    dut.pending_w1c_en.value = 1
    dut.pending_w1c_data.value = 0x0000_0001  # SW clear
    await RisingEdge(dut.clk)
    dut.irq_sources.value = 0
    dut.pending_w1c_en.value = 0

    await RisingEdge(dut.clk)
    # SET WINS: bit 0 should still be pending
    assert (int(dut.irq_pending.value) & 0x1) == 1, \
        "SET WINS violated: bit cleared despite concurrent HW fire"


@cocotb.test()
async def test_msix_pulse_on_new_irq(dut):
    """msix_req pulses when a new unmasked IRQ appears."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Unmask all
    dut.mask_wr_en.value = 1
    dut.mask_wr_data.value = 0
    await RisingEdge(dut.clk)
    dut.mask_wr_en.value = 0

    # Fire source 0
    dut.irq_sources.value = 0x0000_0001
    await RisingEdge(dut.clk)
    dut.irq_sources.value = 0

    # Check for msix_req pulse (1 cycle delay for edge detect)
    await RisingEdge(dut.clk)
    assert dut.msix_req.value == 1, "msix_req not pulsed"

    # Next cycle should not pulse (no new IRQ)
    await RisingEdge(dut.clk)
    assert dut.msix_req.value == 0, "msix_req should not persist"
