"""기대값 출처: sim/golden/deltarule.py requantize_q15() — 골든 함수와 비트 단위 비교.

tb_requant_q15.py — rtl/dr1/requant_q15.sv (PLAN W5)

순수 조합 논리라서 클럭이 필요 없다. 입력을 넣고 정착시킨 뒤 출력을 읽는다.
경계값 + 무작위 1만 개를 `golden.requantize_q15` 와 대조한다.

round-half-to-even 은 두 곳(골든·RTL)에 같은 규칙이 있어야 한다. 이 테스트가
그 두 곳이 갈라지지 않았음을 보장한다.
"""

import os
import sys

import cocotb
from cocotb.triggers import Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.golden.deltarule import FRAC_BITS, Q15_MAX, Q15_MIN, requantize_q15  # noqa: E402

ACC_W = 40
ACC_MAX = (1 << (ACC_W - 1)) - 1
ACC_MIN = -(1 << (ACC_W - 1))
ONE = 1 << FRAC_BITS


def _to_signed(raw: int, width: int) -> int:
    raw &= (1 << width) - 1
    return raw - (1 << width) if raw >> (width - 1) else raw


async def apply_and_read(dut, acc: int):
    """acc 를 넣고 조합 논리를 정착시킨 뒤 (q15, sat) 를 읽는다."""
    dut.acc.value = acc & ((1 << ACC_W) - 1)
    await Timer(1, unit="ns")
    return _to_signed(int(dut.q15.value), 16), int(dut.sat.value)


async def check(dut, acc: int, label: str = ""):
    got_q, got_s = await apply_and_read(dut, acc)
    want_q, want_s = requantize_q15(acc)
    assert (got_q, got_s) == (want_q, int(want_s)), (
        f"[{label}] acc={acc} ({acc / ONE:+.6f})  "
        f"RTL=({got_q}, {got_s})  골든=({want_q}, {int(want_s)})"
    )
    return got_q, got_s


@cocotb.test()
async def test_boundaries(dut):
    """경계값: 정확히 ±0.5 LSB, 포화 양·음 끝, 0."""
    cases = [
        (0, "0"),
        (ONE, "+1.0 LSB"),
        (-ONE, "-1.0 LSB"),
        (ONE // 2, "정확히 +0.5 LSB (짝수로 → 0)"),
        (-(ONE // 2), "정확히 -0.5 LSB (짝수로 → 0)"),
        (ONE + ONE // 2, "정확히 +1.5 LSB (홀수 → 2)"),
        (-(ONE + ONE // 2), "정확히 -1.5 LSB (→ -2)"),
        (2 * ONE + ONE // 2, "정확히 +2.5 LSB (짝수 → 2)"),
        (ONE // 2 - 1, "+0.5 LSB 바로 아래 (→ 0)"),
        (ONE // 2 + 1, "+0.5 LSB 바로 위 (→ 1)"),
        (Q15_MAX * ONE, "Q15_MAX 정확"),
        (Q15_MAX * ONE + ONE // 2, "Q15_MAX + 0.5 (짝수? → 포화 여부 확인)"),
        ((Q15_MAX + 1) * ONE, "Q15_MAX 초과 → 포화"),
        (Q15_MIN * ONE, "Q15_MIN 정확"),
        ((Q15_MIN - 1) * ONE, "Q15_MIN 미달 → 포화"),
        (ACC_MAX, "누산기 최대"),
        (ACC_MIN, "누산기 최소"),
    ]
    for acc, label in cases:
        q, s = await check(dut, acc, label)
        dut._log.info(f"  acc={acc:>14d}  → q15={q:>7d} sat={s}   {label}")


@cocotb.test()
async def test_every_half_lsb_near_zero(dut):
    """0 주변에서 0.5 LSB 격자를 전부 — round-half-to-even 이 갈라지기 쉬운 구간."""
    for n in range(-16, 17):
        await check(dut, n * (ONE // 2), f"{n}/2 LSB")


@cocotb.test()
async def test_random_10k(dut):
    """무작위 1만 개를 골든과 비트 비교. 포화 구간도 섞는다."""
    import random

    rnd = random.Random(20260911)
    n_sat = 0
    for i in range(10000):
        if i % 5 == 0:
            # 포화가 나도록 크게
            acc = rnd.randint(Q15_MAX * ONE, ACC_MAX) * rnd.choice([1, -1])
        else:
            acc = rnd.randint(Q15_MIN * ONE, Q15_MAX * ONE)
        acc = max(ACC_MIN, min(ACC_MAX, acc))
        _, s = await check(dut, acc, f"random#{i}")
        n_sat += s
    dut._log.info(f"무작위 10000개 통과. 그중 포화 {n_sat}개")
    assert n_sat > 100, f"포화 케이스가 너무 적다 ({n_sat}) — 테스트가 그 경로를 안 본다"
