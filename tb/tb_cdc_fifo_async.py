"""
tb_cdc_fifo_async.py — cocotb testbench for cdc_fifo.sv

Tests:
  1. Basic async write/read with different clock frequencies
  2. Full/empty edge behavior
  3. Continuous streaming (producer faster than consumer)
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, ReadOnly, RisingEdge, Timer


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

    read_data = await read_n(dut, 8)

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

    # 시뮬레이션이 끝날 때 코루틴이 살아 있으면 verilator + cocotb 2.x 가
    # 종료 중에 죽는다 (rc=-11 SIGSEGV). 핸들을 잡아두고 반드시 정리한다.
    # (docs/BUGS.md BUG-008a — 이 테스트의 세그폴트는 RTL 문제가 아니었다.)
    prod = cocotb.start_soon(producer())
    cons = cocotb.start_soon(consumer())

    # Timeout
    await Timer(5000, unit="ns")

    for t in (prod, cons):
        if not t.done():
            t.cancel()
    dut.wr_valid.value = 0
    dut.rd_valid.value = 0
    await RisingEdge(dut.wr_clk)

    assert len(read_data) >= total // 2, \
        f"Consumer too slow: only got {len(read_data)}/{total}"
