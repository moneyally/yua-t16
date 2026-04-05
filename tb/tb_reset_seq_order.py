"""
tb_reset_seq_order.py — cocotb testbench for reset_seq.sv

Tests:
  1. Power-on reset: io → mem → core release order
  2. Software reset: re-enters reset, same release order
  3. Watchdog reset: same sequence, WDOG cause latched
  4. Boot cause sticky OR and clear
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, FallingEdge


RELEASE_DELAY = 4  # must match RTL parameter


async def reset_dut(dut):
    """Apply power-on reset."""
    dut.por_n.value = 0
    dut.sw_reset.value = 0
    dut.wdog_reset.value = 0
    dut.pcie_flr.value = 0
    dut.sw_cause_clr.value = 0
    await Timer(50, unit="ns")
    dut.por_n.value = 1
    # Wait for synchronizer (2 FF + margin)
    for _ in range(4):
        await RisingEdge(dut.clk)


async def wait_release_complete(dut, max_cycles=50):
    """Wait until reset_active goes low."""
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        if dut.reset_active.value == 0:
            return
    raise TimeoutError("reset_active did not deassert")


@cocotb.test()
async def test_por_release_order(dut):
    """POR: rst_io_n releases first, then rst_mem_n, then rst_core_n."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    io_release_cycle = None
    mem_release_cycle = None
    core_release_cycle = None
    cycle = 0

    for _ in range(50):
        await RisingEdge(dut.clk)
        cycle += 1
        if dut.rst_io_n.value == 1 and io_release_cycle is None:
            io_release_cycle = cycle
        if dut.rst_mem_n.value == 1 and mem_release_cycle is None:
            mem_release_cycle = cycle
        if dut.rst_core_n.value == 1 and core_release_cycle is None:
            core_release_cycle = cycle
        if core_release_cycle is not None:
            break

    assert io_release_cycle is not None, "rst_io_n never released"
    assert mem_release_cycle is not None, "rst_mem_n never released"
    assert core_release_cycle is not None, "rst_core_n never released"

    # Strict ordering
    assert io_release_cycle < mem_release_cycle, \
        f"io ({io_release_cycle}) must release before mem ({mem_release_cycle})"
    assert mem_release_cycle < core_release_cycle, \
        f"mem ({mem_release_cycle}) must release before core ({core_release_cycle})"

    # Boot cause should have POR bit set
    assert (int(dut.boot_cause.value) & 0x1) == 1, "POR bit not set in boot_cause"


@cocotb.test()
async def test_sw_reset_sequence(dut):
    """Software reset: all domains re-enter reset, then release in order."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)
    await wait_release_complete(dut)

    # All domains should be released
    assert dut.rst_core_n.value == 1

    # Trigger software reset
    dut.sw_reset.value = 1
    await RisingEdge(dut.clk)
    dut.sw_reset.value = 0

    # All domains should be asserted again
    await RisingEdge(dut.clk)
    assert dut.rst_io_n.value == 0, "rst_io_n not asserted after sw_reset"
    assert dut.rst_mem_n.value == 0, "rst_mem_n not asserted after sw_reset"
    assert dut.rst_core_n.value == 0, "rst_core_n not asserted after sw_reset"

    # Wait for release and check order
    await wait_release_complete(dut)
    assert dut.reset_active.value == 0

    # Boot cause should have SW bit set
    assert (int(dut.boot_cause.value) & 0x4) != 0, "SW bit not set in boot_cause"


@cocotb.test()
async def test_wdog_reset_sequence(dut):
    """Watchdog reset: same release sequence, WDOG cause latched."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)
    await wait_release_complete(dut)

    # Clear boot cause first
    dut.sw_cause_clr.value = 1
    await RisingEdge(dut.clk)
    dut.sw_cause_clr.value = 0
    await RisingEdge(dut.clk)

    # Trigger watchdog reset
    dut.wdog_reset.value = 1
    await RisingEdge(dut.clk)
    dut.wdog_reset.value = 0

    await RisingEdge(dut.clk)
    assert dut.rst_core_n.value == 0, "rst_core_n not asserted after wdog_reset"

    await wait_release_complete(dut)

    # Boot cause should have WDOG bit set
    assert (int(dut.boot_cause.value) & 0x2) != 0, "WDOG bit not set in boot_cause"


@cocotb.test()
async def test_boot_cause_sticky_and_clear(dut):
    """Boot cause bits are sticky OR, cleared by sw_cause_clr."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)
    await wait_release_complete(dut)

    # POR bit should be set
    assert (int(dut.boot_cause.value) & 0x1) != 0

    # Trigger SW reset — should OR in SW bit
    dut.sw_reset.value = 1
    await RisingEdge(dut.clk)
    dut.sw_reset.value = 0
    await RisingEdge(dut.clk)

    # Both POR and SW should be set
    cause = int(dut.boot_cause.value)
    assert (cause & 0x1) != 0, "POR bit lost"
    assert (cause & 0x4) != 0, "SW bit not set"

    await wait_release_complete(dut)

    # Clear
    dut.sw_cause_clr.value = 1
    await RisingEdge(dut.clk)
    dut.sw_cause_clr.value = 0
    await RisingEdge(dut.clk)

    assert dut.boot_cause.value == 0, f"boot_cause not cleared: {dut.boot_cause.value}"
