"""
tb_g2_ctrl_top_oom.py — OOM guard state verification

Tests:
  1. Enqueue multiple descriptors, OOM usage increases
  2. Complete descriptors, OOM usage decreases back
  3. OOM state reads correctly via register
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

DESC_SIZE = 64
A_DESC_STAGE  = 0x0_2100
A_Q0_DOORBELL = 0x0_2000
A_OOM_USAGE   = 0x2_0000
A_OOM_EFF     = 0x2_0010
A_OOM_STATE   = 0x2_001C
A_TRACE_CTRL  = 0xA_0010
A_IRQ_MASK    = 0x9_0004

DESC_COST = 4096  # must match g2_ctrl_top parameter


def crc8(data_bytes):
    crc = 0x00
    for byte in data_bytes:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80: crc = ((crc << 1) ^ 0x07) & 0xFF
            else: crc = (crc << 1) & 0xFF
    return crc


def make_nop():
    desc = [0] * DESC_SIZE
    desc[0] = 0x01  # NOP
    desc[DESC_SIZE-1] = crc8(desc[:DESC_SIZE-1])
    return desc


async def reset_dut(dut):
    dut.por_n.value = 0
    dut.reg_addr.value = 0; dut.reg_wr_en.value = 0; dut.reg_wr_data.value = 0
    dut.rd_req_ready.value = 1; dut.rd_done.value = 0
    dut.rd_data_valid.value = 0; dut.rd_data.value = 0; dut.rd_data_last.value = 0
    dut.wr_req_ready.value = 1; dut.wr_done.value = 0; dut.wr_data_ready.value = 1
    await Timer(100, unit="ns")
    dut.por_n.value = 1
    for _ in range(30):
        await RisingEdge(dut.clk)
        if dut.reset_active.value == 0: break
    await RisingEdge(dut.clk)


async def reg_write(dut, addr, data):
    dut.reg_addr.value = addr
    dut.reg_wr_en.value = 1
    dut.reg_wr_data.value = data
    await RisingEdge(dut.clk)
    dut.reg_wr_en.value = 0


async def reg_read(dut, addr):
    dut.reg_addr.value = addr
    dut.reg_wr_en.value = 0
    await RisingEdge(dut.clk)
    return int(dut.reg_rd_data.value)


async def stage_and_doorbell(dut, desc, queue=0):
    for i in range(16):
        word = 0
        for b in range(4):
            idx = i*4+b
            if idx < DESC_SIZE: word |= desc[idx] << (8*b)
        await reg_write(dut, A_DESC_STAGE + i*4, word)
    await reg_write(dut, A_Q0_DOORBELL + queue*4, 1)


@cocotb.test()
async def test_oom_usage_increases_on_enqueue(dut):
    """Push NOP descriptors, OOM allocated_bytes should increase."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    await reg_write(dut, A_TRACE_CTRL, 0x01)

    # Check baseline
    usage0 = await reg_read(dut, A_OOM_USAGE)
    assert usage0 == 0, f"Initial OOM usage should be 0, got {usage0}"

    # Stage once, then 3 rapid doorbells (NOP completes fast so we need
    # to push quickly before FSM drains them)
    desc = make_nop()
    for i in range(16):
        word = 0
        for b in range(4):
            idx = i*4+b
            if idx < DESC_SIZE: word |= desc[idx] << (8*b)
        await reg_write(dut, A_DESC_STAGE + i*4, word)

    # 3 rapid doorbells
    await reg_write(dut, A_Q0_DOORBELL, 1)
    await reg_write(dut, A_Q0_DOORBELL, 1)
    await reg_write(dut, A_Q0_DOORBELL, 1)

    # Read effective usage immediately (alloc should be 3*DESC_COST,
    # but FSM may have already started processing first NOP)
    # Check effective >= DESC_COST (at least 1 in-flight)
    eff = await reg_read(dut, A_OOM_EFF)
    assert eff >= DESC_COST, f"Effective usage should be >= {DESC_COST} during processing, got {eff}"


@cocotb.test()
async def test_oom_usage_decreases_on_completion(dut):
    """Push NOPs, let FSM complete them, usage should return to 0."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    await reg_write(dut, A_TRACE_CTRL, 0x01)

    desc = make_nop()
    await stage_and_doorbell(dut, desc)

    # Wait for NOP to complete (FSM processes quickly)
    for _ in range(50):
        await RisingEdge(dut.clk)

    usage = await reg_read(dut, A_OOM_USAGE)
    assert usage == 0, f"After NOP completion, usage should be 0, got {usage}"


@cocotb.test()
async def test_oom_state_readable(dut):
    """OOM_STATE register should reflect NORMAL state."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    state_reg = await reg_read(dut, A_OOM_STATE)
    oom_st = state_reg & 0x3
    admit_stop = (state_reg >> 8) & 0x1
    pf_clamp = (state_reg >> 9) & 0x1

    assert oom_st == 0, f"OOM state should be NORMAL(0), got {oom_st}"
    assert admit_stop == 0, "admission_stop should be 0"
    assert pf_clamp == 0, "prefetch_clamp should be 0"
