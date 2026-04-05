"""
tb_g2_ctrl_top_smoke.py — E2E smoke test for g2_ctrl_top.sv

E2E flow: reset → register read → enqueue descriptor → doorbell →
          dispatch → DMA emulation → core_done → IRQ pending → trace entry

This test validates the full control-plane path.
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# Register addresses (offset from 0x8030_0000)
A_G2_ID       = 0x0_0000
A_G2_VERSION  = 0x0_0004
A_BOOT_CAUSE  = 0x0_1000
A_DESC_STAGE  = 0x0_2100
A_Q0_DOORBELL = 0x0_2000
A_Q0_STATUS   = 0x0_3000
A_IRQ_PENDING = 0x9_0000
A_IRQ_MASK    = 0x9_0004
A_TRACE_CTRL  = 0xA_0010
A_TRACE_TAIL  = 0xA_0004

DESC_SIZE = 64


def crc8(data_bytes):
    crc = 0x00
    for byte in data_bytes:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def make_gemm_descriptor(act=0x1000, wgt=0x2000, out=0x3000, kt=4):
    """Build a 64-byte GEMM descriptor with valid CRC."""
    desc = [0] * DESC_SIZE
    desc[0] = 0x02  # GEMM opcode
    for i in range(8):
        desc[16 + i] = (act >> (8 * i)) & 0xFF
        desc[24 + i] = (wgt >> (8 * i)) & 0xFF
        desc[32 + i] = (out >> (8 * i)) & 0xFF
    for i in range(4):
        desc[40 + i] = (kt >> (8 * i)) & 0xFF
    desc[DESC_SIZE - 1] = crc8(desc[:DESC_SIZE - 1])
    return desc


async def reset_dut(dut):
    """Power-on reset via por_n."""
    dut.por_n.value = 0
    dut.reg_addr.value = 0
    dut.reg_wr_en.value = 0
    dut.reg_wr_data.value = 0
    dut.rd_req_ready.value = 0
    dut.rd_done.value = 0
    dut.rd_data_valid.value = 0
    dut.rd_data.value = 0
    dut.rd_data_last.value = 0
    dut.wr_req_ready.value = 0
    dut.wr_done.value = 0
    dut.wr_data_ready.value = 0
    await Timer(100, unit="ns")
    dut.por_n.value = 1
    # Wait for reset_seq to release all domains
    for _ in range(30):
        await RisingEdge(dut.clk)
        if dut.reset_active.value == 0:
            break
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


async def dma_responder(dut, num_beats=4):
    """Simple DMA read responder: provides num_beats of zero data."""
    # Wait for rd_req_valid
    for _ in range(500):
        await RisingEdge(dut.clk)
        if dut.rd_req_valid.value == 1:
            dut.rd_req_ready.value = 1
            await RisingEdge(dut.clk)
            dut.rd_req_ready.value = 0
            # Send data beats
            for beat in range(num_beats):
                dut.rd_data_valid.value = 1
                dut.rd_data.value = beat
                dut.rd_data_last.value = 1 if beat == num_beats - 1 else 0
                await RisingEdge(dut.clk)
            dut.rd_data_valid.value = 0
            dut.rd_data_last.value = 0
            return
    raise TimeoutError("DMA read request never arrived")


async def dma_write_responder(dut, num_beats=64):
    """Simple DMA write responder: accepts write data."""
    for _ in range(1000):
        await RisingEdge(dut.clk)
        if dut.wr_req_valid.value == 1:
            dut.wr_req_ready.value = 1
            await RisingEdge(dut.clk)
            dut.wr_req_ready.value = 0
            # Accept data beats
            dut.wr_data_ready.value = 1
            for _ in range(num_beats + 10):
                await RisingEdge(dut.clk)
                if dut.wr_data_valid.value == 1 and dut.wr_data_last.value == 1:
                    break
            dut.wr_data_ready.value = 0
            return
    raise TimeoutError("DMA write request never arrived")


@cocotb.test()
async def test_e2e_reset_read_enqueue_dispatch_irq_trace(dut):
    """Full E2E: reset -> reg read -> enqueue -> dispatch -> done -> irq -> trace."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # ── Step 1: Reset ──
    await reset_dut(dut)
    assert dut.reset_active.value == 0, "Reset should be complete"

    # ── Step 2: Register reads ──
    val = await reg_read(dut, A_G2_ID)
    assert val == 0x4732_0001, f"G2_ID mismatch: {val:#010x}"

    cause = await reg_read(dut, A_BOOT_CAUSE)
    assert (cause & 0x1) != 0, "POR bit should be set after power-on"

    # ── Step 3: Enable trace ──
    await reg_write(dut, A_TRACE_CTRL, 0x0000_0001)  # enable

    # ── Step 4: Unmask IRQ bit 0 (DESC_DONE) ──
    await reg_write(dut, A_IRQ_MASK, 0xFFFF_FFFE)  # unmask bit 0

    # ── Step 5: Stage descriptor ──
    desc = make_gemm_descriptor(act=0x1000, wgt=0x2000, out=0x3000, kt=4)
    for i in range(16):
        word = 0
        for b in range(4):
            byte_idx = i * 4 + b
            if byte_idx < DESC_SIZE:
                word |= desc[byte_idx] << (8 * b)
        await reg_write(dut, A_DESC_STAGE + i * 4, word)

    # ── Step 6: Doorbell -> enqueue ──
    await reg_write(dut, A_Q0_DOORBELL, 0x0000_0001)

    # ── Step 7: Wait for dispatch + DMA emulation ──
    # gemm_top will: decode desc -> read act -> read wgt -> compute -> write out
    # We need to respond to 2 DMA reads (act + wgt) and 1 DMA write (out)
    kt = 4

    # DMA read: activation
    await dma_responder(dut, num_beats=kt)
    # DMA read: weights
    await dma_responder(dut, num_beats=kt)
    # DMA write: output
    await dma_write_responder(dut, num_beats=64)

    # Wait for done
    for _ in range(100):
        await RisingEdge(dut.clk)

    # ── Step 8: Check IRQ pending ──
    irq_pend = await reg_read(dut, A_IRQ_PENDING)
    # Bit 0 (DESC_DONE) should be set
    assert (irq_pend & 0x1) != 0, f"IRQ_DESC_DONE not pending: {irq_pend:#010x}"

    # IRQ output should be asserted (bit 0 unmasked)
    assert dut.irq_out.value == 1, "irq_out not asserted"

    # ── Step 9: Check trace ring ──
    trace_t = await reg_read(dut, A_TRACE_TAIL)
    assert trace_t > 0, f"Trace tail should have advanced, got {trace_t}"

    # ── Step 10: Clear IRQ via W1C ──
    await reg_write(dut, A_IRQ_PENDING, 0x0000_0001)
    await RisingEdge(dut.clk)

    irq_pend2 = await reg_read(dut, A_IRQ_PENDING)
    assert (irq_pend2 & 0x1) == 0, f"IRQ bit 0 should be cleared after W1C: {irq_pend2:#010x}"


@cocotb.test()
async def test_register_defaults_after_reset(dut):
    """Verify key register defaults immediately after reset."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Queue status should be empty (head == tail == 0)
    q0 = await reg_read(dut, A_Q0_STATUS)
    assert q0 == 0, f"Q0_STATUS should be 0 after reset, got {q0:#010x}"

    # IRQ pending should be 0
    irq = await reg_read(dut, A_IRQ_PENDING)
    assert irq == 0, f"IRQ_PENDING should be 0 after reset, got {irq:#010x}"

    # Trace tail should be 0
    tt = await reg_read(dut, A_TRACE_TAIL)
    assert tt == 0, f"TRACE_TAIL should be 0 after reset, got {tt}"
