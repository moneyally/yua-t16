"""
tb_g2_ctrl_top_trace_dump.py — Trace read window verification

Tests:
  1. Execute a descriptor, then read trace entries via MMIO read window
  2. Verify trace entry content: type, queue_class, opcode
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

DESC_SIZE = 64
A_DESC_STAGE  = 0x0_2100
A_Q0_DOORBELL = 0x0_2000
A_IRQ_MASK    = 0x9_0004
A_TRACE_CTRL  = 0xA_0010
A_TRACE_TAIL  = 0xA_0004
A_TRACE_WIN   = 0xA_0100  # lo32 at +0, hi32 at +4
A_TRACE_META  = 0xA_3000  # type+fatal at +0


def crc8(data_bytes):
    crc = 0x00
    for byte in data_bytes:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80: crc = ((crc << 1) ^ 0x07) & 0xFF
            else: crc = (crc << 1) & 0xFF
    return crc


def make_desc(opcode, kt=4):
    desc = [0] * DESC_SIZE
    desc[0] = opcode & 0xFF
    for i in range(4): desc[40+i] = (kt >> (8*i)) & 0xFF
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
    doorbell_addr = A_Q0_DOORBELL + queue * 4
    await reg_write(dut, doorbell_addr, 1)


async def run_nop(dut, queue=0):
    """Submit NOP descriptor and wait for completion."""
    desc = make_desc(opcode=0x01)
    await stage_and_doorbell(dut, desc, queue)
    for _ in range(50):
        await RisingEdge(dut.clk)


@cocotb.test()
async def test_trace_dump_after_nop(dut):
    """Submit NOP, read trace entries via read window, verify content."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Enable trace
    await reg_write(dut, A_TRACE_CTRL, 0x01)

    # Submit NOP on Q0
    await run_nop(dut, queue=0)

    # Read trace tail
    tail = await reg_read(dut, A_TRACE_TAIL)
    assert tail > 0, f"Trace tail should advance after NOP, got {tail}"

    # Read first trace entry via read window
    lo = await reg_read(dut, A_TRACE_WIN + 0)  # entry 0, lo32
    hi = await reg_read(dut, A_TRACE_WIN + 4)  # entry 0, hi32
    meta = await reg_read(dut, A_TRACE_META + 0)  # entry 0 meta

    dut._log.info(f"Trace[0]: lo={lo:#010x} hi={hi:#010x} meta={meta:#010x}")

    # meta format: {24'b0, type[3:0], 3'b0, fatal}
    # bits [7:4] = type, bit [0] = fatal
    entry_type = (meta >> 4) & 0xF
    entry_fatal = meta & 0x1

    # Should be DISPATCH (type=1) or DONE (type=2) event
    assert entry_type in [1, 2], f"Unexpected trace type: {entry_type}"
    assert entry_fatal == 0, "NOP should not be fatal"

    # payload: {46'b0, qclass[1:0], opcode[7:0], 8'b0}
    payload = lo | (hi << 32)
    opcode_field = (payload >> 8) & 0xFF
    qclass_field = (payload >> 16) & 0x3

    dut._log.info(f"  opcode={opcode_field:#04x} qclass={qclass_field}")
    assert opcode_field == 0x01, f"Expected NOP opcode 0x01, got {opcode_field:#04x}"
    assert qclass_field == 0, f"Expected qclass 0 (Q0), got {qclass_field}"
