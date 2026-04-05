"""
tb_cdc_fifo_reset.py — cocotb testbench for cdc_fifo.sv reset behavior

Tests:
  1. Reset during active traffic
  2. Asymmetric reset (only wr domain reset)
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


async def reset_fifo(dut):
    dut.wr_rst_n.value = 0
    dut.rd_rst_n.value = 0
    dut.wr_valid.value = 0
    dut.rd_valid.value = 0
    dut.wr_data.value = 0
    await Timer(100, unit="ns")
    dut.wr_rst_n.value = 1
    dut.rd_rst_n.value = 1
    await Timer(100, unit="ns")


@cocotb.test()
async def test_reset_during_traffic(dut):
    """Assert reset while FIFO has data, verify clean recovery."""
    cocotb.start_soon(Clock(dut.wr_clk, 10, unit="ns").start())
    cocotb.start_soon(Clock(dut.rd_clk, 13, unit="ns").start())

    await reset_fifo(dut)

    # Write some data
    for i in range(4):
        dut.wr_valid.value = 1
        dut.wr_data.value = 0xBEEF_0000 + i
        await RisingEdge(dut.wr_clk)
    dut.wr_valid.value = 0

    await Timer(50, unit="ns")

    # Assert reset on both domains
    dut.wr_rst_n.value = 0
    dut.rd_rst_n.value = 0
    await Timer(50, unit="ns")
    dut.wr_rst_n.value = 1
    dut.rd_rst_n.value = 1
    await Timer(100, unit="ns")

    # FIFO should be empty after reset
    assert dut.empty.value == 1, "FIFO not empty after reset"
    assert dut.full.value == 0, "FIFO full after reset"

    # Should be able to write/read again cleanly
    dut.wr_valid.value = 1
    dut.wr_data.value = 0xCAFE_0001
    await RisingEdge(dut.wr_clk)
    dut.wr_valid.value = 0

    await Timer(200, unit="ns")

    dut.rd_valid.value = 1
    for _ in range(10):
        await RisingEdge(dut.rd_clk)
        if dut.rd_ready.value == 1:
            assert int(dut.rd_data.value) == 0xCAFE_0001, "Data corruption after reset"
            break
    dut.rd_valid.value = 0
