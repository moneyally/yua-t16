"""
tb_oom_guard_race.py — cocotb testbench for oom_guard.sv race conditions

Tests:
  1. In-flight DMA reserve/de-reserve
  2. Simultaneous alloc + dma_reserve
  3. Underflow detection
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.alloc_inc.value = 0
    dut.alloc_dec.value = 0
    dut.alloc_bytes.value = 0
    dut.dma_reserve_inc.value = 0
    dut.dma_reserve_dec.value = 0
    dut.dma_rsv_bytes.value = 0
    dut.prefetch_reserve_inc.value = 0
    dut.prefetch_reserve_dec.value = 0
    dut.prefetch_rsv_bytes.value = 0
    dut.thresh_pressure.value = 1000
    dut.thresh_critical.value = 2000
    dut.thresh_emergency.value = 3000
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_dma_reserve_release(dut):
    """Reserve DMA bytes, effective usage increases. Release, it decreases."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Reserve 1500 bytes for DMA
    dut.dma_reserve_inc.value = 1
    dut.dma_rsv_bytes.value = 1500
    await RisingEdge(dut.clk)
    dut.dma_reserve_inc.value = 0

    await RisingEdge(dut.clk)
    # effective_usage should be 1500 -> PRESSURE
    assert int(dut.effective_usage.value) == 1500
    assert int(dut.pressure_state.value) == 1  # PRESSURE

    # Complete DMA — release reservation
    dut.dma_reserve_dec.value = 1
    dut.dma_rsv_bytes.value = 1500
    await RisingEdge(dut.clk)
    dut.dma_reserve_dec.value = 0

    await RisingEdge(dut.clk)
    assert int(dut.effective_usage.value) == 0
    assert int(dut.pressure_state.value) == 0  # NORMAL


@cocotb.test()
async def test_simultaneous_alloc_and_reserve(dut):
    """Alloc + DMA reserve on same cycle — both should count."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Alloc 800 + DMA reserve 800 on same cycle -> effective = 1600
    dut.alloc_inc.value = 1
    dut.alloc_bytes.value = 800
    dut.dma_reserve_inc.value = 1
    dut.dma_rsv_bytes.value = 800
    await RisingEdge(dut.clk)
    dut.alloc_inc.value = 0
    dut.dma_reserve_inc.value = 0

    await RisingEdge(dut.clk)
    assert int(dut.effective_usage.value) == 1600
    assert int(dut.pressure_state.value) == 1  # PRESSURE


@cocotb.test()
async def test_underflow_detection(dut):
    """Dealloc more than allocated should set underflow_error."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Dealloc without any prior allocation
    dut.alloc_dec.value = 1
    dut.alloc_bytes.value = 100
    await RisingEdge(dut.clk)
    dut.alloc_dec.value = 0

    await RisingEdge(dut.clk)
    assert dut.underflow_error.value == 1, "Underflow error not detected"
