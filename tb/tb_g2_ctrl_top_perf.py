"""Perf counter and freeze tests."""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

A_MXU_BCYC_LO = 0x6_0000
A_MXU_TILE     = 0x6_0010
A_VPU_BCYC_LO = 0x6_0008
A_PERF_FREEZE  = 0x6_0018

async def reset_dut(dut):
    dut.por_n.value=0; dut.reg_addr.value=0; dut.reg_wr_en.value=0; dut.reg_wr_data.value=0
    dut.rd_req_ready.value=1; dut.rd_done.value=0
    dut.rd_data_valid.value=0; dut.rd_data.value=0; dut.rd_data_last.value=0
    dut.wr_req_ready.value=1; dut.wr_done.value=0; dut.wr_data_ready.value=1
    await Timer(100, unit="ns"); dut.por_n.value=1
    for _ in range(30):
        await RisingEdge(dut.clk)
        if dut.reset_active.value==0: break
    await RisingEdge(dut.clk)

async def reg_write(dut, a, d):
    dut.reg_addr.value=a; dut.reg_wr_en.value=1; dut.reg_wr_data.value=d
    await RisingEdge(dut.clk); dut.reg_wr_en.value=0

async def reg_read(dut, a):
    dut.reg_addr.value=a; dut.reg_wr_en.value=0
    await RisingEdge(dut.clk); return int(dut.reg_rd_data.value)

@cocotb.test()
async def test_mxu_busy_zero_at_idle(dut):
    """MXU busy cycles should be 0 at idle (no GEMM dispatched)."""
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    v = await reg_read(dut, A_MXU_BCYC_LO)
    assert v == 0, f"MXU busy should be 0 at idle, got {v}"

@cocotb.test()
async def test_vpu_read_as_zero(dut):
    """VPU counters should read-as-zero in Proto-A."""
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    v = await reg_read(dut, A_VPU_BCYC_LO)
    assert v == 0, f"VPU busy should be 0, got {v}"

@cocotb.test()
async def test_perf_freeze_holds_counters(dut):
    """Setting PERF_FREEZE should stop counters from advancing."""
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    # Freeze
    await reg_write(dut, A_PERF_FREEZE, 1)
    c0 = await reg_read(dut, A_MXU_BCYC_LO)
    for _ in range(20): await RisingEdge(dut.clk)
    c1 = await reg_read(dut, A_MXU_BCYC_LO)
    assert c1 == c0, f"Frozen counter should not advance: {c0} -> {c1}"

@cocotb.test()
async def test_tile_count_zero_at_idle(dut):
    """MXU tile count should be 0 before any GEMM."""
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    v = await reg_read(dut, A_MXU_TILE)
    assert v == 0, f"Tile count should be 0, got {v}"
