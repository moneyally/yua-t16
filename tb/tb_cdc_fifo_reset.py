"""
tb_cdc_fifo_reset.py — cocotb testbench for cdc_fifo.sv reset behavior

Tests:
  1. Reset during active traffic
  2. Asymmetric reset (only wr domain reset)
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge, Timer


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


async def read_n(dut, n, guard=60):
    """cdc_fifo 에서 n 개를 읽는다. 타이밍 두 가지를 맞춰야 한다.

    1. `rd_data` 는 **레지스터 출력**이다 (rtl/cdc_fifo.sv "registered output").
       RisingEdge 직후에 읽으면 논블로킹 대입 전이라 **이전 값**을 본다.
    2. `rd_ready`(=~empty) 도 레지스터 출력이다. rising edge N 의 핸드셰이크는
       **N 직전** rd_ready 로 결정되고, 그 결과 데이터는 **N 직후** rd_data 에 나온다.

    그래서 falling edge 에서 샘플링한다. falling F_n (R_n 과 R_{n+1} 사이)에서:
        rd_data  = R_n 핸드셰이크의 결과
        rd_ready = R_{n+1} 에서 핸드셰이크가 일어날지
    따라서 "직전 falling 의 rd_ready" 가 이번 rd_data 의 유효성이다.
    시드는 rising edge 를 소비하면 안 된다 (핸드셰이크 1개를 잃는다).

    예전 테스트는 1번을 놓쳐 첫 읽기에서 리셋값 0 을 보고 실패했다
    (docs/BUGS.md BUG-008a/b). **RTL 은 정상이다** — 8개가 순서대로 나온다.
    """
    out = []
    dut.rd_valid.value = 1
    await FallingEdge(dut.rd_clk)          # rising 을 소비하지 않는 시드
    prev_ready = int(dut.rd_ready.value)
    for _ in range(guard):
        await FallingEdge(dut.rd_clk)
        if prev_ready:
            out.append(int(dut.rd_data.value))
            if len(out) >= n:
                break
        prev_ready = int(dut.rd_ready.value)
    dut.rd_valid.value = 0
    return out


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

    got = await read_n(dut, 1)
    assert got and got[0] == 0xCAFE_0001, f"Data corruption after reset: got {got}"
