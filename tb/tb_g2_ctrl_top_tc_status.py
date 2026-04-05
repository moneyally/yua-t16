"""TC0 RUNSTATE / CTRL / FAULT_STATUS register tests."""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

DESC_SIZE = 64
A_TC0_RUNSTATE = 0x4_0000
A_TC0_CTRL     = 0x4_0004
A_TC0_PCYC_LO  = 0x4_0010
A_TC0_FAULT    = 0x4_0018
A_DESC_STAGE   = 0x0_2100
A_Q0_DOORBELL  = 0x0_2000
A_TRACE_CTRL   = 0xA_0010

def crc8(d):
    c = 0
    for b in d:
        c ^= b
        for _ in range(8):
            c = ((c<<1)^0x07)&0xFF if c&0x80 else (c<<1)&0xFF
    return c

def make_desc(op, kt=4):
    d = [0]*DESC_SIZE; d[0]=op&0xFF
    for i in range(4): d[40+i]=(kt>>(8*i))&0xFF
    d[DESC_SIZE-1] = crc8(d[:DESC_SIZE-1])
    return d

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

async def reg_write(dut, addr, data):
    dut.reg_addr.value=addr; dut.reg_wr_en.value=1; dut.reg_wr_data.value=data
    await RisingEdge(dut.clk); dut.reg_wr_en.value=0

async def reg_read(dut, addr):
    dut.reg_addr.value=addr; dut.reg_wr_en.value=0
    await RisingEdge(dut.clk); return int(dut.reg_rd_data.value)

async def stage_and_bell(dut, desc, q=0):
    for i in range(16):
        w=0
        for b in range(4):
            idx=i*4+b
            if idx<DESC_SIZE: w|=desc[idx]<<(8*b)
        await reg_write(dut, A_DESC_STAGE+i*4, w)
    await reg_write(dut, A_Q0_DOORBELL+q*4, 1)

@cocotb.test()
async def test_idle_runstate(dut):
    """TC0 RUNSTATE should be IDLE(0) after reset."""
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    rs = await reg_read(dut, A_TC0_RUNSTATE)
    assert (rs&0x7)==0, f"Expected IDLE(0), got {rs&0x7}"

@cocotb.test()
async def test_ctrl_enable_default(dut):
    """TC0 CTRL.ENABLE should be 1 by default."""
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    ctrl = await reg_read(dut, A_TC0_CTRL)
    assert (ctrl&1)==1, f"ENABLE should be 1, got {ctrl&1}"

@cocotb.test()
async def test_halt_stops_dispatch(dut):
    """Setting HALT should prevent descriptor dispatch."""
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    await reg_write(dut, A_TRACE_CTRL, 1)
    # Set HALT
    await reg_write(dut, A_TC0_CTRL, 0x03)  # ENABLE=1, HALT=1
    # Submit NOP
    await stage_and_bell(dut, make_desc(0x01))
    for _ in range(30): await RisingEdge(dut.clk)
    # RUNSTATE should be IDLE (not processing because halted)
    rs = await reg_read(dut, A_TC0_RUNSTATE)
    assert (rs&0x7)==0, f"Halted TC0 should be IDLE, got {rs&0x7}"

@cocotb.test()
async def test_fault_status_w1c(dut):
    """Bad descriptor -> FAULT_STATUS latched, W1C clears it."""
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    await reg_write(dut, A_TRACE_CTRL, 1)
    # Submit illegal opcode
    await stage_and_bell(dut, make_desc(0xFF))
    for _ in range(30): await RisingEdge(dut.clk)
    fault = await reg_read(dut, A_TC0_FAULT)
    assert fault != 0, f"FAULT_STATUS should be non-zero, got {fault}"
    # W1C clear
    await reg_write(dut, A_TC0_FAULT, fault)
    await RisingEdge(dut.clk)
    fault2 = await reg_read(dut, A_TC0_FAULT)
    assert fault2 == 0, f"FAULT_STATUS should be cleared, got {fault2}"

@cocotb.test()
async def test_perf_cycles_count(dut):
    """PERF_CYCLES should advance during NOP processing."""
    clock=Clock(dut.clk,10,unit="ns"); cocotb.start_soon(clock.start())
    await reset_dut(dut)
    await reg_write(dut, A_TRACE_CTRL, 1)
    c0 = await reg_read(dut, A_TC0_PCYC_LO)
    await stage_and_bell(dut, make_desc(0x01))
    for _ in range(30): await RisingEdge(dut.clk)
    c1 = await reg_read(dut, A_TC0_PCYC_LO)
    assert c1 > c0, f"PERF_CYCLES should advance: {c0} -> {c1}"
