"""기대값 출처: sim/golden/deltarule.py step() — d=64 에서도 골든과 비트 일치해야 한다.

tb_dr1_d64.py — **d=64 확장 검증** (PLAN W8 "d=64 확장은 W8 완료 후에만")

같은 RTL 을 파라미터만 바꿔 돌린다. **파일을 복제하지 않는다** — 복제하면
한쪽만 고치는 날이 온다.

    PARAM_D=64 PARAM_SCRATCH=8192 python3 tb/run_tb.py dr1_tb_wrap tb_dr1_d64 <소스들>
    (scripts/run_dr1_tb.sh 가 이 형태로 부른다)

  X1. d=64 에서 INIT → DUMP 가 골든 초기 상태(0)와 일치
  X2. d=64 **1토큰** 비트 일치
  X3. d=64 **20토큰** 비트 일치 + 포화 수 일치
  X4. d=64 사이클 실측 (DESIGN 6.1 표에 기입)

왜 20토큰인가: d=64 는 토큰당 골든 연산이 d=16 의 16배다 (matvec 2회 + 행 갱신 d회
= O(d²)). 1,000토큰은 파이썬 골든이 너무 느려서 d=16 쪽에서 본다. 여기서 보는 것은
**차원이 바뀌어도 같은 RTL 이 맞는가**이고, 그건 20토큰이면 드러난다
(BUG-009 는 2토큰째에 드러났다).
"""

import os
import sys

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from sim.golden import deltarule as G  # noqa: E402
from tb_dr1_top import CocotbDut, make_token_sequence, run_harness_async  # noqa: E402
from tools.orbit_mmio_map import dr1_scratch_layout  # noqa: E402

D = 64
SCRATCH = 8192


class Dut64(CocotbDut):
    """d=64 용 DUT. 스크래치가 커서 배치도 큰 쪽으로 계산한다.

    `CocotbDut` 은 기본 스크래치(1024)를 가정하는 `dr1_scratch_layout(d)` 를 쓴다.
    d=64 는 덤프(64²=4096)가 거기 안 들어가므로 크기를 명시해서 다시 계산한다.
    **배치 규칙 자체는 같은 함수에서 온다** — 여기서 따로 만들지 않는다.
    """

    def __init__(self, dut, d: int, scratch_words: int):
        super().__init__(dut, 16)          # 부모의 검사만 통과시키고
        lay = dr1_scratch_layout(d, scratch_words)
        self.d = d
        self.q_elem = lay["q"]
        self.k_elem = lay["k"]
        self.v_elem = lay["v"]
        self.o_elem = lay["o"]
        self.dump_elem = lay["dump"]
        self.timeout_cycles = 65536        # d=64 는 덤프가 d²+ 사이클이다


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.cmd_valid.value = 0
    dut.cmd_opcode.value = 0
    dut.cmd_slot.value = 0
    dut.cmd_q_addr.value = 0
    dut.cmd_k_addr.value = 0
    dut.cmd_v_addr.value = 0
    dut.cmd_dst_addr.value = 0
    dut.cmd_alpha.value = 0x8000
    dut.cmd_beta.value = 0
    dut.h_en.value = 0
    dut.h_we.value = 0
    dut.h_addr.value = 0
    dut.h_wdata.value = 0
    dut.sat_count_clr.value = 0
    dut.clamp_count_clr.value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


def check_param(dut):
    """DUT 가 정말 d=64 로 빌드됐는지 확인한다.

    파라미터를 안 넘기면 d=16 짜리 DUT 를 d=64 로 착각하고 테스트가 조용히
    엉뚱한 것을 본다. 덤프 폭으로 확인한다 (dump_data 는 D*W 비트다).
    """
    width = len(dut.dump_data.value)
    assert width == D * 16, (
        f"DUT 가 d={width // 16} 로 빌드됐다 (d={D} 이어야 한다). "
        f"PARAM_D={D} PARAM_SCRATCH={SCRATCH} 로 돌려야 한다."
    )


@cocotb.test()
async def test_x1_d64_init_dump(dut):
    """X1: d=64 에서 INIT → DUMP 가 골든 초기 상태(전부 0)와 일치."""
    check_param(dut)
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    res = await run_harness_async(Dut64(dut, D, SCRATCH), tokens=[], seed=0)
    dut._log.info(res.report())
    assert res.ok, f"X1: d=64 초기 상태가 골든과 다르다\n{res.report(max_lines=5)}"


@cocotb.test()
async def test_x2_d64_one_token(dut):
    """X2: d=64 토큰 1개가 골든 step() 과 비트 일치."""
    check_param(dut)
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    tokens = make_token_sequence(D, 1, seed=64)
    res = await run_harness_async(Dut64(dut, D, SCRATCH), tokens, seed=64)
    dut._log.info(res.report(max_lines=5))
    assert res.ok, f"X2: d=64 1토큰이 골든과 다르다\n{res.report(max_lines=5)}"
    dut._log.info(f"X2: d=64 1토큰 비트 일치. 사이클={res.cycles}")


@cocotb.test()
async def test_x3_d64_twenty_tokens(dut):
    """X3: d=64 토큰 20개 연속 + 포화 수 일치.

    상태가 누적되는 경로를 본다 — BUG-009(이전 행 사용)는 2토큰째에 드러났다.
    """
    check_param(dut)
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    tokens = make_token_sequence(D, 20, seed=65)
    res = await run_harness_async(Dut64(dut, D, SCRATCH), tokens, seed=65)
    dut._log.info(res.report(max_lines=5))
    assert res.ok, f"X3: d=64 20토큰이 골든과 다르다\n{res.report(max_lines=5)}"
    assert res.dut_sat_total == res.golden_sat_total, (
        f"X3: 포화 수 RTL={res.dut_sat_total} 골든={res.golden_sat_total}"
    )


@cocotb.test()
async def test_x4_d64_cycles(dut):
    """X4: d=64 토큰당 사이클 실측. **고치지 말고 기록한다** (DESIGN 6.1)."""
    check_param(dut)
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    tokens = make_token_sequence(D, 4, seed=66)
    res = await run_harness_async(Dut64(dut, D, SCRATCH), tokens, seed=66)
    assert res.ok

    cycles = sorted(set(res.cycles))
    dut._log.info(f"X4: d=64 DELTA_STEP 실측 사이클 = {cycles}")
    assert len(cycles) == 1, f"X4: 토큰마다 사이클이 다르다 {cycles}"

    # d=16 이 126 이었다. 구조상 대략 4배여야 한다 (전부 d 에 비례하는 단계다).
    # 정확히 맞을 필요는 없고, **자릿수가 맞는지**만 본다 — 아니면 어딘가 상수 경로가 있다.
    assert 400 <= cycles[0] <= 600, (
        f"X4: d=64 사이클이 {cycles[0]} 이다. d=16 의 126 에서 대략 4배(~500)를 "
        f"기대했다 — 벗어나면 d 에 비례하지 않는 단계가 생긴 것이다"
    )
