"""기대값 출처: sim/golden/deltarule.py — 하네스(tb/tb_dr1_top.py)가 골든과 비트 비교한다.
불변조건 I1(초기 상태 S=0)은 docs/DESIGN.md 7절.

tb_dr1_harness_rtl.py — 하네스를 **실제 RTL(rtl/dr1/dr1_tb_wrap.sv)** 에 붙인다 (PLAN W7)

  R1. I1 — INIT 직후 DUMP 가 골든 초기 상태(전부 0)와 비트 일치
  R2. 오류 주입 — 덤프 1워드를 뒤집으면 하네스가 **위치를 찍어** 잡는가
  R3. slot 초과 → Dr1Fault(0x05) (done_pulse 가 아니라 done_err 를 본다)
  R4. **토큰 1개**가 골든 step() 과 o·S 비트 일치
  R5. **토큰 10개** 연속
  R6. **토큰 100개** 연속 (PLAN W7 완료 기준)
  R7. 포화가 나는 입력에서도 sat 개수까지 골든과 일치
  R8. 사이클 수 실측 (DESIGN 6.1 표에 기입)

토큰 시퀀스는 하네스의 make_token_sequence 가 만든다 — 골든과 RTL 이 같은 입력을 본다.
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
from tb_dr1_top import (  # noqa: E402
    CocotbDut,
    Dr1Fault,
    make_token_sequence,
    run_harness_async,
)
from tools.orbit_mmio_map import FaultCode  # noqa: E402

D = 16


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


@cocotb.test()
async def test_r1_invariant_i1_on_real_rtl(dut):
    """R1: 실 RTL 의 INIT→DUMP 가 골든 초기 상태와 비트 일치 (불변조건 I1)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    res = await run_harness_async(CocotbDut(dut, D), tokens=[], seed=0)
    dut._log.info(res.report())
    assert res.ok, f"R1: 실 RTL 이 골든과 다르다\n{res.report()}"


@cocotb.test()
async def test_r2_fault_injection_on_rtl_path(dut):
    """R2: 덤프 1워드를 뒤집으면 하네스가 위치를 찍어 잡는가.

    하네스가 실제로 비교를 하는지 확인하는 유일한 방법이다. R1 만 있으면
    "아무것도 안 하고 통과" 와 구분되지 않는다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    class FlipDut(CocotbDut):
        """RTL 에서 받은 덤프의 한 칸만 1 LSB 틀리게 만든다. RTL 은 그대로다."""

        def __init__(self, dut, d, cell):
            super().__init__(dut, d)
            self.cell = cell

        async def dump(self, slot: int = 0):
            S = np.asarray(await super().dump(slot), dtype=np.int64).copy()
            r, c = self.cell
            S[r][c] += 1
            return S

    res = await run_harness_async(FlipDut(dut, D, (3, 11)), tokens=[], seed=0)
    dut._log.info(res.report(max_lines=3))
    assert not res.ok, "R2: 오류를 주입했는데 하네스가 통과시켰다 — 하네스가 죽어 있다"
    m = res.first
    assert (m.row, m.col) == (3, 11), f"R2: 위치를 잘못 찍었다 {m}"
    assert m.expected == 0 and m.actual == 1, f"R2: 값을 잘못 찍었다 {m}"


@cocotb.test()
async def test_r3_bad_slot_raises_fault(dut):
    """R3: slot 초과 → Dr1Fault(0x05). done_pulse 를 완료로 읽지 않는다는 뜻이다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    cd = CocotbDut(dut, D)
    try:
        await cd.init(slot=1)
    except Dr1Fault as e:
        assert int(e.fault_code) == int(FaultCode.DR1_BAD_SLOT), f"R3: fault={e}"
    else:
        raise AssertionError("R3: slot=1 인데 성공했다 — done_ok 가 떴다는 뜻이다")


@cocotb.test()
async def test_r4_one_token_bit_exact(dut):
    """R4: **토큰 1개**가 골든 step() 과 비트 일치 (o 와 최종 상태 S 둘 다).

    PLAN W7: "골든 대비 1토큰 비트 일치 → 10토큰 → 100토큰".
    여기가 그 첫 칸이다. 여기서 깨지면 10·100 을 볼 이유가 없다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    tokens = make_token_sequence(D, 1, seed=1)
    res = await run_harness_async(CocotbDut(dut, D), tokens, seed=1)
    dut._log.info(res.report(max_lines=5))
    assert res.ok, f"R4: 1토큰이 골든과 다르다\n{res.report(max_lines=5)}"
    assert res.dut_sat_total == res.golden_sat_total, (
        f"R4: 포화 수가 다르다 RTL={res.dut_sat_total} 골든={res.golden_sat_total}"
    )
    dut._log.info(f"R4: 1토큰 비트 일치. 사이클={res.cycles}")


@cocotb.test()
async def test_r5_ten_tokens(dut):
    """R5: 토큰 10개 연속. 상태가 누적되는 경로를 본다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    tokens = make_token_sequence(D, 10, seed=2)
    res = await run_harness_async(CocotbDut(dut, D), tokens, seed=2)
    dut._log.info(res.report(max_lines=5))
    assert res.ok, f"R5: 10토큰이 골든과 다르다\n{res.report(max_lines=5)}"
    assert res.dut_sat_total == res.golden_sat_total


@cocotb.test()
async def test_r6_hundred_tokens(dut):
    """R6: **토큰 100개** — PLAN W7 완료 기준."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    tokens = make_token_sequence(D, 100, seed=3)
    res = await run_harness_async(CocotbDut(dut, D), tokens, seed=3)
    dut._log.info(res.report(max_lines=5))
    assert res.ok, f"R6: 100토큰이 골든과 다르다\n{res.report(max_lines=5)}"
    assert res.dut_sat_total == res.golden_sat_total, (
        f"R6: 포화 수 RTL={res.dut_sat_total} 골든={res.golden_sat_total}"
    )
    cycles = sorted(set(res.cycles))
    dut._log.info(f"R6: 100토큰 비트 일치. 토큰당 사이클 = {cycles}")


@cocotb.test()
async def test_r7_saturating_tokens(dut):
    """R7: 포화가 실제로 나는 입력에서도 sat 개수까지 일치.

    기본 시퀀스는 scale 을 작게 잡아 포화가 안 난다 (make_token_sequence).
    포화 경로를 안 보면 "포화 수가 일치한다"는 말이 공허하다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # β=1.0, v 를 최대 근처로 → 갱신 항이 커져서 재양자화가 포화한다
    rng = np.random.default_rng(99)
    tokens = []
    for _ in range(12):
        tokens.append((
            G.random_vec(rng, D, 0.9),
            G.random_vec(rng, D, 0.9),
            np.full(D, 32767, dtype=np.int64),
            G.UQ15_ONE,
            G.UQ15_ONE,
        ))

    res = await run_harness_async(CocotbDut(dut, D), tokens, seed=99)
    dut._log.info(res.report(max_lines=5))
    assert res.ok, f"R7: 포화 입력에서 골든과 다르다\n{res.report(max_lines=5)}"
    assert res.golden_sat_total > 0, "R7: 포화를 유발하려 했는데 골든도 0 이다 — 입력이 약하다"
    assert res.dut_sat_total == res.golden_sat_total, (
        f"R7: 포화 수 RTL={res.dut_sat_total} 골든={res.golden_sat_total}"
    )
    dut._log.info(f"R7: 포화 {res.golden_sat_total}회까지 일치")


@cocotb.test()
async def test_r9_thousand_tokens_three_seeds(dut):
    """R9: **1,000토큰 × 시드 3개** — PLAN W8 완료 기준 (d=16 완료 선언 조건).

    100토큰까지는 초기 상태 근처만 본다. 1,000토큰은 상태가 충분히 돌아간 뒤에도
    비트 일치가 유지되는지를 본다 — Q1.15 누적 오차가 갈라지면 여기서 드러난다.

    시드 3개인 이유: 한 시드로는 "그 시드에서만 맞는" 경로를 구분할 수 없다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for seed in (1, 2, 3):
        tokens = make_token_sequence(D, 1000, seed=seed)
        res = await run_harness_async(CocotbDut(dut, D), tokens, seed=seed)
        dut._log.info(res.report(max_lines=3))
        assert res.ok, f"R9: seed={seed} 1000토큰이 골든과 다르다\n{res.report(max_lines=5)}"
        assert res.dut_sat_total == res.golden_sat_total, (
            f"R9: seed={seed} 포화 수 RTL={res.dut_sat_total} 골든={res.golden_sat_total}"
        )
    dut._log.info("R9: 1,000토큰 × 시드 3개 전부 비트 일치")


@cocotb.test()
async def test_r10_sat_and_clamp_events(dut):
    """R10: 트레이스 이벤트 신호가 실제로 뜬다 (spec/deltarule.md 5.2절).

    SAT_EVENT / CLAMP_EVENT 는 dr1_top 이 내는 1사이클 펄스다. g2_ctrl_top 이
    그것을 트레이스 링에 넣는다. 여기서는 **펄스가 뜨는지와 카운터가 맞는지**를
    본다 (링에 들어가는 것은 tb_g2_ctrl_top_dr1_fault / host_e2e 가 본다).
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    cd = CocotbDut(dut, D)
    await cd.init(0)

    # 클램프: α, β 둘 다 1.0 초과로 보낸다 (골든은 거부하므로 직접 디스크립터를 쓴다)
    clamp_before = int(dut.dr1_clamp_count.value)
    await cd._issue(
        0x51, 0,
        q_addr=cd.q_elem * 2, k_addr=cd.k_elem * 2, v_addr=cd.v_elem * 2,
        dst_addr=cd.o_elem * 2, alpha=0xFFFF, beta=0x9000,
    )
    clamp_after = int(dut.dr1_clamp_count.value)
    assert clamp_after == clamp_before + 2, (
        f"R10: 클램프 2회가 세어져야 한다 ({clamp_before} → {clamp_after})"
    )

    # 포화: v 를 최대로, β=1.0 → 갱신 항이 커져 재양자화가 포화한다
    sat_before = int(dut.dr1_sat_count.value)
    for i in range(D):
        await cd._scr_write(cd.v_elem + i, 32767)
        await cd._scr_write(cd.k_elem + i, 32767)
    for _ in range(3):
        await cd._issue(
            0x51, 0,
            q_addr=cd.q_elem * 2, k_addr=cd.k_elem * 2, v_addr=cd.v_elem * 2,
            dst_addr=cd.o_elem * 2, alpha=0x8000, beta=0x8000,
        )
    sat_after = int(dut.dr1_sat_count.value)
    assert sat_after > sat_before, (
        f"R10: 포화를 유발했는데 SAT 카운트가 안 늘었다 ({sat_before} → {sat_after})"
    )
    dut._log.info(f"R10: 클램프 +2, 포화 +{sat_after - sat_before}")


@cocotb.test()
async def test_r8_cycles_measured(dut):
    """R8: 토큰당 사이클을 실측한다. **고치지 말고 기록한다** (DESIGN 6.1)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    tokens = make_token_sequence(D, 8, seed=5)
    res = await run_harness_async(CocotbDut(dut, D), tokens, seed=5)
    assert res.ok

    cycles = sorted(set(res.cycles))
    dut._log.info(f"R8: DELTA_STEP 실측 사이클 = {cycles} (d={D})")
    assert len(cycles) == 1, f"R8: 토큰마다 사이클이 다르다 {cycles} — 입력 의존 경로가 있다"
