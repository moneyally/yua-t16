"""Watchdog stub + BOOT_CAUSE tests."""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

A_BOOT_CAUSE = 0x0_1000
A_WDOG_CTRL  = 0x0_1014

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
async def test_por_boot_cause(dut):
    """After POR, BOOT_CAUSE[0] (POR) should be set."""
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    bc = await reg_read(dut, A_BOOT_CAUSE)
    assert (bc&1)==1, f"POR bit should be set, got {bc:#x}"

@cocotb.test()
async def test_wdog_test_inject(dut):
    """Writing WDOG_CTRL with bit[31]=1 triggers watchdog reset."""
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    # Clear boot_cause first
    await reg_write(dut, A_BOOT_CAUSE, 0)
    await RisingEdge(dut.clk)
    # Inject watchdog reset via bit[31]
    await reg_write(dut, A_WDOG_CTRL, 0x8000_0000)
    # Wait for reset sequence to complete
    for _ in range(30):
        await RisingEdge(dut.clk)
        if dut.reset_active.value==0: break
    await RisingEdge(dut.clk)
    bc = await reg_read(dut, A_BOOT_CAUSE)
    assert (bc&0x2)!=0, f"WDOG bit should be set after test inject, got {bc:#x}"

@cocotb.test()
async def test_wdog_ctrl_readback(dut):
    """WDOG_CTRL register read/write."""
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    await reg_write(dut, A_WDOG_CTRL, 0x0001_0001)  # enable + window value
    await RisingEdge(dut.clk)
    v = await reg_read(dut, A_WDOG_CTRL)
    assert v==0x0001_0001, f"WDOG_CTRL readback mismatch: {v:#010x}"
