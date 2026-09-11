"""기대값 출처: spec/deltarule.md 3.5절 평탄화·바이트 순서 + tools/orbit_pack.py.
메모리 내용 자체는 테스트가 쓴 값이므로 골든 모델이 필요 없다 — 계약만 검사한다.

tb_dr1_scratch.py — rtl/dr1/dr1_scratch.sv (PLAN W7)

  C1. 호스트 포트 쓰기 → 읽기가 같은 값
  C2. 읽기 지연이 **정확히 1사이클** (state_sram 과 같은 계약)
  C3. DR1 포트와 호스트 포트가 **서로 다른 주소**에서 동시에 동작한다
  C4. DR1 포트가 쓴 것을 호스트 포트가 읽는다 (o 벡터 회수 경로)
  C5. 벡터 한 개를 orbit_pack 바이트 순서대로 넣고 그대로 되읽는다

레지스터 출력이므로 falling edge 에서 샘플링한다 (tb/tb_cdc_fifo_async.py 교훈).
"""

import os
import sys

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.orbit_pack import from_bits, to_bits  # noqa: E402

W = 16
DEPTH = 256


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.h_en.value = 0
    dut.h_we.value = 0
    dut.h_addr.value = 0
    dut.h_wdata.value = 0
    dut.d_rd_en.value = 0
    dut.d_rd_addr.value = 0
    dut.d_wr_en.value = 0
    dut.d_wr_addr.value = 0
    dut.d_wr_data.value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def host_write(dut, addr, value):
    dut.h_en.value = 1
    dut.h_we.value = 1
    dut.h_addr.value = addr
    dut.h_wdata.value = to_bits(int(value), W)
    await RisingEdge(dut.clk)
    dut.h_en.value = 0
    dut.h_we.value = 0


async def host_read(dut, addr):
    """읽기 지연 1사이클: 요청 edge 다음 falling edge 에 데이터가 있다."""
    dut.h_en.value = 1
    dut.h_we.value = 0
    dut.h_addr.value = addr
    await RisingEdge(dut.clk)
    dut.h_en.value = 0
    await FallingEdge(dut.clk)
    return from_bits(int(dut.h_rdata.value), W)


async def dr1_write(dut, addr, value):
    dut.d_wr_en.value = 1
    dut.d_wr_addr.value = addr
    dut.d_wr_data.value = to_bits(int(value), W)
    await RisingEdge(dut.clk)
    dut.d_wr_en.value = 0


async def dr1_read(dut, addr):
    dut.d_rd_en.value = 1
    dut.d_rd_addr.value = addr
    await RisingEdge(dut.clk)
    dut.d_rd_en.value = 0
    await FallingEdge(dut.clk)
    return from_bits(int(dut.d_rd_data.value), W)


@cocotb.test()
async def test_c1_host_write_read(dut):
    """C1: 호스트가 쓴 값을 호스트가 읽는다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    vals = {0: 1234, 1: -5678, 17: 32767, 255: -32768}
    for a, v in vals.items():
        await host_write(dut, a, v)
    for a, v in vals.items():
        got = await host_read(dut, a)
        assert got == v, f"C1: addr={a} 기대 {v} 실제 {got}"


@cocotb.test()
async def test_c2_read_latency_exactly_one(dut):
    """C2: 읽기 지연이 정확히 1사이클. 0도 2도 아니다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await host_write(dut, 5, 0x0ABC)
    await host_write(dut, 6, 0x0DEF)

    # 요청을 걸고 **같은** edge 직후에는 아직 옛 값이어야 한다
    dut.h_en.value = 1
    dut.h_we.value = 0
    dut.h_addr.value = 5
    await FallingEdge(dut.clk)      # 요청 전 — 출력은 직전 읽기 결과
    before = int(dut.h_rdata.value)
    await RisingEdge(dut.clk)       # 이 edge 에서 읽기가 일어난다
    dut.h_en.value = 0
    await FallingEdge(dut.clk)      # 여기서 데이터가 보여야 한다
    after = int(dut.h_rdata.value)

    assert after == 0x0ABC, f"C2: 1사이클 뒤에 값이 없다 (before={before:#x} after={after:#x})"


@cocotb.test()
async def test_c3_two_ports_different_addresses(dut):
    """C3: 두 포트가 서로 다른 주소에서 같은 사이클에 동작한다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await host_write(dut, 10, 111)
    await host_write(dut, 20, 222)

    # 같은 사이클에 host 는 10번을 읽고 dr1 은 20번을 읽는다
    dut.h_en.value = 1
    dut.h_we.value = 0
    dut.h_addr.value = 10
    dut.d_rd_en.value = 1
    dut.d_rd_addr.value = 20
    await RisingEdge(dut.clk)
    dut.h_en.value = 0
    dut.d_rd_en.value = 0
    await FallingEdge(dut.clk)

    assert from_bits(int(dut.h_rdata.value), W) == 111, "C3: host 포트가 틀렸다"
    assert from_bits(int(dut.d_rd_data.value), W) == 222, "C3: dr1 포트가 틀렸다"


@cocotb.test()
async def test_c4_dr1_write_host_read(dut):
    """C4: DR1 이 쓴 결과(o 벡터)를 호스트가 회수한다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for i in range(8):
        await dr1_write(dut, 100 + i, -1000 * (i + 1))
    for i in range(8):
        got = await host_read(dut, 100 + i)
        assert got == -1000 * (i + 1), f"C4: addr={100+i} 기대 {-1000*(i+1)} 실제 {got}"


@cocotb.test()
async def test_c5_vector_round_trip(dut):
    """C5: 벡터 하나를 원소 순서대로 넣고 DR1 포트로 그대로 읽는다.

    spec/deltarule.md 3.5절: 원소 i 가 낮은 주소. 이 순서가 어긋나면
    q 와 k 가 뒤집혀 들어가도 테스트가 통과해 버린다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    rng = np.random.default_rng(77)
    vec = rng.integers(-32768, 32768, size=16, dtype=np.int64)

    base = 32
    for i, x in enumerate(vec):
        await host_write(dut, base + i, int(x))

    got = []
    for i in range(16):
        got.append(await dr1_read(dut, base + i))

    assert got == [int(x) for x in vec], (
        f"C5: 원소 순서가 어긋났다\n  쓴 것  ={vec.tolist()}\n  읽은 것={got}"
    )
