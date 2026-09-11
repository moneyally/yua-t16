"""기대값 출처: tools/orbit_pack.py 평탄화 규칙 + rtl/dr1/state_sram.sv 타이밍 계약 (프로토콜 테스트).

tb_state_sram.py — rtl/dr1/state_sram.sv (PLAN W5)

이 모듈은 골든 함수와 1:1 대응하는 연산이 없다 (저장소다). 검사 대상은 **계약**이다:

  S1. 전 행 쓰기 → 읽기 비트 일치
  S2. 읽기 지연이 **정확히 1사이클**
  S3. clr_start 후 전부 0, clr_busy 가 **정확히 D사이클**
  S4. 다른 행 동시 읽기/쓰기 — 서로 간섭 없음
  S5. **같은 행** 동시 읽기/쓰기 — 읽기는 **이전 값** (read-before-write)

`rd_data` 는 레지스터 출력이므로 **falling edge 에서 샘플링**한다.
RisingEdge 직후에 읽으면 논블로킹 대입 전이라 이전 사이클 값을 본다 —
`cdc_fifo` 가 정확히 이걸로 틀렸다 (docs/BUGS.md BUG-008a/b).
"""

import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.orbit_pack import pack_vec, unpack_vec  # noqa: E402

D = 16
W = 16


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.rd_en.value = 0
    dut.rd_row.value = 0
    dut.wr_en.value = 0
    dut.wr_row.value = 0
    dut.wr_data.value = 0
    dut.clr_start.value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def write_row(dut, row: int, flat: int):
    dut.wr_en.value = 1
    dut.wr_row.value = row
    dut.wr_data.value = flat
    await RisingEdge(dut.clk)
    dut.wr_en.value = 0


async def read_row(dut, row: int) -> int:
    """읽기 지연 1사이클을 지켜서 한 행을 읽는다.

    rd_en 을 올린 채 rising edge 를 넘기면 그 엣지에서 rd_data 가 갱신된다.
    다음 falling edge 에서 정착된 값을 읽는다.
    """
    dut.rd_en.value = 1
    dut.rd_row.value = row
    await RisingEdge(dut.clk)      # 이 엣지에서 mem[row] → rd_data
    dut.rd_en.value = 0
    await FallingEdge(dut.clk)     # 정착 후 샘플링
    return int(dut.rd_data.value)


def row_pattern(r: int):
    """행마다 다른, 부호까지 섞인 패턴."""
    return [((r * 37 + i * 11) % 65536) - 32768 for i in range(D)]


@cocotb.test()
async def test_s1_write_then_read_all_rows(dut):
    """S1: 전 행에 쓰고 읽어서 비트 일치. 평탄화는 tools/orbit_pack.pack_vec."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    want = {}
    for r in range(D):
        vec = row_pattern(r)
        want[r] = vec
        await write_row(dut, r, pack_vec(vec, W))

    for r in range(D):
        got_flat = await read_row(dut, r)
        got = unpack_vec(got_flat, D, W)
        assert got.tolist() == want[r], (
            f"S1: 행 {r} 불일치\n  기대={want[r]}\n  실제={got.tolist()}"
        )
    dut._log.info(f"S1: {D}행 전부 비트 일치")


@cocotb.test()
async def test_s2_read_latency_is_exactly_one_cycle(dut):
    """S2: 읽기 지연이 정확히 1사이클. 0사이클(조합)도 2사이클도 아니어야 한다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    a, b = 3, 9
    flat_a = pack_vec([1] * D, W)
    flat_b = pack_vec([-1] * D, W)
    await write_row(dut, a, flat_a)
    await write_row(dut, b, flat_b)

    # 먼저 b 를 읽어 rd_data 를 flat_b 로 만든다
    assert await read_row(dut, b) == flat_b

    # read_row 가 falling edge 에서 끝났으므로 지금은 사이클 중간이다.
    # a 를 요청하고 **rising edge 를 넘기지 않은 채로** 읽으면 아직 flat_b 여야 한다.
    # (여기서 FallingEdge 를 또 기다리면 사이클을 하나 넘겨서 이미 갱신된 값을 본다 —
    #  처음에 이 테스트가 그렇게 틀렸다.)
    dut.rd_en.value = 1
    dut.rd_row.value = a
    await Timer(1, unit="ns")          # 신호 전파만. 클럭 엣지 없음
    same_cycle = int(dut.rd_data.value)
    assert same_cycle == flat_b, (
        f"S2: 요청한 사이클에 이미 새 값이 나왔다 → 조합 읽기다 (지연 0). "
        f"0x{same_cycle:x}"
    )

    await RisingEdge(dut.clk)       # 이 엣지에서 갱신
    dut.rd_en.value = 0
    await FallingEdge(dut.clk)
    next_cycle = int(dut.rd_data.value)
    assert next_cycle == flat_a, (
        f"S2: 1사이클 뒤에 새 값이 안 나왔다 → 지연이 2 이상이다. 0x{next_cycle:x}"
    )
    dut._log.info("S2: 읽기 지연 정확히 1사이클")


@cocotb.test()
async def test_s3_clear_zeroes_all_and_busy_is_exactly_d_cycles(dut):
    """S3: clr_start 후 전부 0, clr_busy 가 정확히 D사이클."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for r in range(D):
        await write_row(dut, r, pack_vec(row_pattern(r), W))

    assert int(dut.clr_busy.value) == 0, "S3: 시작 전에 clr_busy 가 1이다"

    dut.clr_start.value = 1
    await RisingEdge(dut.clk)
    dut.clr_start.value = 0

    # clr_busy 를 falling edge 에서 사이클마다 세어, 연속 구간 길이를 재다.
    busy_cycles = 0
    for _ in range(4 * D):
        await FallingEdge(dut.clk)
        if int(dut.clr_busy.value):
            busy_cycles += 1
        elif busy_cycles:
            break

    assert busy_cycles == D, f"S3: clr_busy 가 {busy_cycles}사이클이다 (D={D} 이어야 한다)"

    zero = pack_vec([0] * D, W)
    for r in range(D):
        got = await read_row(dut, r)
        assert got == zero, f"S3: clr 후 행 {r} 이 0 이 아니다: 0x{got:x}"
    dut._log.info(f"S3: clr_busy {busy_cycles}사이클, 전 행 0 확인")


@cocotb.test()
async def test_s4_different_row_simultaneous_rw(dut):
    """S4: 다른 행 동시 읽기/쓰기 — 서로 간섭하지 않는다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    r_rd, r_wr = 2, 7
    old_rd = pack_vec([100] * D, W)
    old_wr = pack_vec([200] * D, W)
    new_wr = pack_vec([-300] * D, W)
    await write_row(dut, r_rd, old_rd)
    await write_row(dut, r_wr, old_wr)

    # 같은 엣지에서 r_rd 읽기 + r_wr 쓰기
    dut.rd_en.value = 1
    dut.rd_row.value = r_rd
    dut.wr_en.value = 1
    dut.wr_row.value = r_wr
    dut.wr_data.value = new_wr
    await RisingEdge(dut.clk)
    dut.rd_en.value = 0
    dut.wr_en.value = 0
    await FallingEdge(dut.clk)
    assert int(dut.rd_data.value) == old_rd, "S4: 다른 행 쓰기가 읽기를 오염시켰다"

    assert await read_row(dut, r_wr) == new_wr, "S4: 쓰기가 반영되지 않았다"
    assert await read_row(dut, r_rd) == old_rd, "S4: 읽던 행이 바뀌었다"
    dut._log.info("S4: 다른 행 동시 R/W 간섭 없음")


@cocotb.test()
async def test_s5_same_row_simultaneous_rw_reads_old_value(dut):
    """S5: 같은 행 동시 읽기/쓰기 → 읽기는 **이전 값** (read-before-write).

    matvec_unit/update_unit 이 같은 행을 읽고 쓰는 파이프라인에서 이 정의에 의존한다.
    헤더에 명시된 계약이므로 여기서 못박는다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    r = 5
    old = pack_vec([1234] * D, W)
    new = pack_vec([-5678] * D, W)
    await write_row(dut, r, old)

    dut.rd_en.value = 1
    dut.rd_row.value = r
    dut.wr_en.value = 1
    dut.wr_row.value = r
    dut.wr_data.value = new
    await RisingEdge(dut.clk)
    dut.rd_en.value = 0
    dut.wr_en.value = 0
    await FallingEdge(dut.clk)
    got = int(dut.rd_data.value)
    assert got == old, (
        f"S5: 같은 행 동시 R/W 에서 읽기가 **이전 값**이어야 한다 "
        f"(read-before-write). 0x{got:x} != 0x{old:x}"
    )
    assert await read_row(dut, r) == new, "S5: 쓰기가 반영되지 않았다"
    dut._log.info("S5: 같은 행 동시 R/W = 이전 값 읽기 확인")
