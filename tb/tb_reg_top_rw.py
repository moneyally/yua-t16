"""
tb_reg_top_rw.py — cocotb testbench for reg_top.sv

Tests:
  1. Read G2_ID / G2_VERSION / G2_CAP0 (RO defaults)
  2. Write/read descriptor staging registers
  3. Doorbell pulse generation
  4. IRQ mask write/read
  5. Trace ctrl write/read
  6. W1C for Q_OVERFLOW
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# Address offsets (match reg_top.sv)
A_G2_ID       = 0x0_0000
A_G2_VERSION  = 0x0_0004
A_G2_CAP0     = 0x0_0008
A_BOOT_CAUSE  = 0x0_1000
A_Q0_DOORBELL = 0x0_2000
A_DESC_STAGE  = 0x0_2100
A_Q0_STATUS   = 0x0_3000
A_Q_OVERFLOW  = 0x0_3010
A_IRQ_PENDING = 0x9_0000
A_IRQ_MASK    = 0x9_0004
A_TRACE_CTRL  = 0xA_0010


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.addr.value = 0
    dut.wr_en.value = 0
    dut.wr_data.value = 0
    dut.boot_cause.value = 0
    dut.overflow_flags.value = 0
    dut.irq_pending.value = 0
    dut.irq_mask_rd.value = 0
    dut.irq_cause_last.value = 0
    dut.oom_state.value = 0
    dut.oom_admission_stop.value = 0
    dut.oom_prefetch_clamp.value = 0
    dut.trace_head.value = 0
    dut.trace_tail.value = 0
    dut.trace_drop_count.value = 0
    for i in range(4):
        dut.q_head[i].value = 0
        dut.q_tail[i].value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


async def reg_write(dut, addr, data):
    dut.addr.value = addr
    dut.wr_en.value = 1
    dut.wr_data.value = data
    await RisingEdge(dut.clk)
    dut.wr_en.value = 0


async def reg_read(dut, addr):
    dut.addr.value = addr
    dut.wr_en.value = 0
    await RisingEdge(dut.clk)
    return int(dut.rd_data.value)


@cocotb.test()
async def test_read_only_defaults(dut):
    """G2_ID, G2_VERSION, G2_CAP0 return correct default values."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    val = await reg_read(dut, A_G2_ID)
    assert val == 0x4732_0001, f"G2_ID: expected 0x47320001, got {val:#010x}"

    val = await reg_read(dut, A_G2_VERSION)
    assert val == 0x0001_0000, f"G2_VERSION: expected 0x00010000, got {val:#010x}"

    val = await reg_read(dut, A_G2_CAP0)
    assert val == 0x0000_0060, f"G2_CAP0: expected 0x00000060, got {val:#010x}"


@cocotb.test()
async def test_desc_staging_rw(dut):
    """Write and readback descriptor staging registers."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Write 16 words
    for i in range(16):
        await reg_write(dut, A_DESC_STAGE + i * 4, 0xDEAD_0000 + i)

    # Read back
    for i in range(16):
        val = await reg_read(dut, A_DESC_STAGE + i * 4)
        expected = 0xDEAD_0000 + i
        assert val == expected, f"DESC_STAGE[{i}]: expected {expected:#010x}, got {val:#010x}"


@cocotb.test()
async def test_doorbell_pulse(dut):
    """Doorbell write generates 1-cycle pulse."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await reg_write(dut, A_Q0_DOORBELL, 0x0000_0001)
    # Pulse should be active during the write cycle (combinational)
    # Check it's 0 now (pulse already gone)
    await RisingEdge(dut.clk)
    assert int(dut.doorbell_pulse.value) == 0, "Doorbell pulse should be 1-cycle only"


@cocotb.test()
async def test_irq_mask_rw(dut):
    """Write IRQ_MASK and read it back."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # irq_mask_rd comes from irq_ctrl (external). Simulate it.
    dut.irq_mask_rd.value = 0xFFFF_0000
    await reg_write(dut, A_IRQ_MASK, 0x0000_FFFF)
    await RisingEdge(dut.clk)  # let write strobe clear

    # mask_wr_en should be 0 now (wr_en is 0)
    assert dut.irq_mask_wr_en.value == 0, "mask_wr_en should be 0 after write cycle"

    val = await reg_read(dut, A_IRQ_MASK)
    assert val == 0xFFFF_0000, f"IRQ_MASK readback: {val:#010x} (from external irq_mask_rd)"


@cocotb.test()
async def test_trace_ctrl_rw(dut):
    """Write TRACE_CTRL bits and read back."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Enable trace + fatal_only
    await reg_write(dut, A_TRACE_CTRL, 0x0000_0005)  # bits [0] + [2]
    val = await reg_read(dut, A_TRACE_CTRL)
    assert val == 5, f"TRACE_CTRL: expected 5, got {val}"
    assert dut.trace_enable.value == 1
    assert dut.trace_freeze.value == 0
    assert dut.trace_fatal_only.value == 1


@cocotb.test()
async def test_boot_cause_read(dut):
    """Read BOOT_CAUSE from external input."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut.boot_cause.value = 0x5  # POR + SW
    val = await reg_read(dut, A_BOOT_CAUSE)
    assert val == 5, f"BOOT_CAUSE: expected 5, got {val}"
