"""
tb_cdc_fifo_async.py — cocotb testbench for cdc_fifo.sv

Tests:
  1. Basic async write/read with different clock frequencies
  2. Full/empty edge behavior
  3. Continuous streaming (producer faster than consumer)
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


async def reset_fifo(dut):
    """Reset both domains."""
    dut.wr_rst_n.value = 0
    dut.rd_rst_n.value = 0
    dut.wr_valid.value = 0
    dut.rd_valid.value = 0
    dut.wr_data.value = 0
    await Timer(100, unit="ns")
    dut.wr_rst_n.value = 1
    dut.rd_rst_n.value = 1
    # Let synchronizers settle
    await Timer(100, unit="ns")


@cocotb.test()
async def test_basic_async_rw(dut):
    """Write 8 items on wr_clk, read them on rd_clk (different freq)."""
    # wr_clk = 100 MHz, rd_clk = 73 MHz (intentionally async)
    cocotb.start_soon(Clock(dut.wr_clk, 10, unit="ns").start())
    cocotb.start_soon(Clock(dut.rd_clk, 13.7, unit="ns").start())

    await reset_fifo(dut)

    # Write 8 items
    written = []
    for i in range(8):
        dut.wr_valid.value = 1
        dut.wr_data.value = 0xDEAD_0000 + i
        await RisingEdge(dut.wr_clk)
        while dut.wr_ready.value == 0:
            await RisingEdge(dut.wr_clk)
        written.append(0xDEAD_0000 + i)
    dut.wr_valid.value = 0

    # Wait for CDC sync
    await Timer(200, unit="ns")

    # Read items
    read_data = []
    for _ in range(8):
        dut.rd_valid.value = 1
        await RisingEdge(dut.rd_clk)
        for attempt in range(50):
            if dut.rd_ready.value == 1:
                read_data.append(int(dut.rd_data.value))
                break
            await RisingEdge(dut.rd_clk)
    dut.rd_valid.value = 0

    assert len(read_data) == 8, f"Expected 8 reads, got {len(read_data)}"
    # FIFO is ordered — data should match write order
    for i, (w, r) in enumerate(zip(written, read_data)):
        assert w == r, f"Mismatch at {i}: wrote {w:#x}, read {r:#x}"


@cocotb.test()
async def test_full_empty_flags(dut):
    """Fill FIFO to full, verify full flag. Drain, verify empty flag."""
    cocotb.start_soon(Clock(dut.wr_clk, 10, unit="ns").start())
    cocotb.start_soon(Clock(dut.rd_clk, 10, unit="ns").start())

    await reset_fifo(dut)

    # FIFO should start empty
    assert dut.empty.value == 1, "FIFO not empty after reset"

    # Fill FIFO (DEPTH=16)
    for i in range(16):
        dut.wr_valid.value = 1
        dut.wr_data.value = i
        await RisingEdge(dut.wr_clk)
        # Wait for ready if needed
        while dut.wr_ready.value == 0:
            await RisingEdge(dut.wr_clk)
    dut.wr_valid.value = 0

    # Wait for full flag to propagate
    await Timer(100, unit="ns")
    assert dut.full.value == 1, "FIFO not full after writing 16 items"

    # Drain FIFO
    for _ in range(16):
        dut.rd_valid.value = 1
        await RisingEdge(dut.rd_clk)
        while dut.rd_ready.value == 0:
            await RisingEdge(dut.rd_clk)
    dut.rd_valid.value = 0

    # Wait for empty flag to propagate
    await Timer(100, unit="ns")
    assert dut.empty.value == 1, "FIFO not empty after draining"


@cocotb.test()
async def test_continuous_streaming(dut):
    """Producer pushes continuously, consumer pulls at half rate."""
    cocotb.start_soon(Clock(dut.wr_clk, 10, unit="ns").start())
    cocotb.start_soon(Clock(dut.rd_clk, 10, unit="ns").start())

    await reset_fifo(dut)

    total = 32
    read_data = []

    async def producer():
        for i in range(total):
            dut.wr_valid.value = 1
            dut.wr_data.value = i
            await RisingEdge(dut.wr_clk)
            while dut.wr_ready.value == 0:
                await RisingEdge(dut.wr_clk)
        dut.wr_valid.value = 0

    async def consumer():
        toggle = 0
        while len(read_data) < total:
            toggle ^= 1
            if toggle:
                dut.rd_valid.value = 1
            else:
                dut.rd_valid.value = 0
            await RisingEdge(dut.rd_clk)
            if dut.rd_valid.value == 1 and dut.rd_ready.value == 1:
                read_data.append(int(dut.rd_data.value))
            if len(read_data) >= total:
                break
            # Safety timeout
            if len(read_data) == 0:
                await Timer(500, unit="ns")

    cocotb.start_soon(producer())
    cocotb.start_soon(consumer())

    # Timeout
    await Timer(5000, unit="ns")

    assert len(read_data) >= total // 2, \
        f"Consumer too slow: only got {len(read_data)}/{total}"
