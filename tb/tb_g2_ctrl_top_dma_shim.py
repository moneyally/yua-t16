"""DMA status shim register tests."""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

A_DMA_STATUS = 0x1_0010
A_DMA_ERR    = 0x1_0014

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

async def reg_read(dut, a):
    dut.reg_addr.value=a; dut.reg_wr_en.value=0
    await RisingEdge(dut.clk); return int(dut.reg_rd_data.value)

@cocotb.test()
async def test_dma_status_idle(dut):
    """DMA_STATUS should be 0 at idle (no BUSY, no DONE, no ERR)."""
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    st = await reg_read(dut, A_DMA_STATUS)
    assert (st&0xF)==0, f"DMA_STATUS lower nibble should be 0, got {st:#x}"

@cocotb.test()
async def test_dma_err_code_zero(dut):
    """DMA_ERR_CODE should be 0 at idle."""
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    ec = await reg_read(dut, A_DMA_ERR)
    assert ec==0, f"DMA_ERR_CODE should be 0, got {ec}"
