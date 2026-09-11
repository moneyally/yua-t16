"""기대값 출처: sim/golden/deltarule.py matvec(S, x) — 골든 함수와 비트 단위 비교.

tb_matvec_unit.py — rtl/dr1/matvec_unit.sv + state_sram 실제 인스턴스 (PLAN W5)

DUT 는 `rtl/dr1/matvec_tb_wrap.sv` (matvec_unit + state_sram 결선).

  M1. 무작위 S·x 500쌍을 golden.matvec 과 비트 일치. 포화 케이스 **최소 50개 강제**
  M2. sat_count 가 골든의 sat_count 와 일치
  M3. done 폭이 정확히 1사이클
  M4. 사이클 수 실측 (DESIGN.md 6절 표에 기입)

평탄화는 `tools/orbit_pack.py` 만 쓴다. 레지스터 출력은 falling edge 샘플링.
"""

import os
import sys

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.golden.deltarule import Q15_MAX, Q15_MIN, matvec  # noqa: E402
from tools.orbit_pack import pack_vec, unpack_vec  # noqa: E402

D = 16
W = 16
CYCLE_BUDGET = D + 4        # 계약 상한 (DESIGN.md 6절)


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.wr_en.value = 0
    dut.wr_row.value = 0
    dut.wr_data.value = 0
    dut.clr_start.value = 0
    dut.start.value = 0
    dut.x_flat.value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def load_state(dut, S):
    for r in range(D):
        dut.wr_en.value = 1
        dut.wr_row.value = r
        dut.wr_data.value = pack_vec(S[r], W)
        await RisingEdge(dut.clk)
    dut.wr_en.value = 0
    await RisingEdge(dut.clk)


async def run_matvec(dut, x):
    """start 를 1펄스 주고 done 까지 기다린다. done 폭도 센다."""
    dut.x_flat.value = pack_vec(x, W)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    done_high = 0
    n = 0
    for _ in range(8 * D):
        await FallingEdge(dut.clk)
        n += 1
        if int(dut.done.value):
            done_high += 1
            y = int(dut.y_flat.value)
            sat = int(dut.sat_count.value)
            cyc = int(dut.cycles.value)
            # done 이 다음 사이클에 내려가는지 확인
            await FallingEdge(dut.clk)
            assert int(dut.done.value) == 0, (
                "M3: done 이 2사이클 이상 유지된다 — 폭은 정확히 1이어야 한다 "
                "(DESIGN.md 5.1)"
            )
            return y, sat, cyc, done_high, n
        await RisingEdge(dut.clk)
    raise AssertionError(f"done 이 {8 * D} 사이클 안에 오지 않았다")


def rand_state(rng, scale):
    return np.array(
        [[int(rng.integers(-scale, scale + 1)) for _ in range(D)] for _ in range(D)],
        dtype=np.int64,
    )


def rand_vec(rng, scale):
    return np.array([int(rng.integers(-scale, scale + 1)) for _ in range(D)], dtype=np.int64)


@cocotb.test()
async def test_m1_random_500_pairs_bit_exact(dut):
    """M1/M2: 무작위 500쌍이 골든 matvec 과 비트 일치. 포화 최소 50개 포함."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    rng = np.random.default_rng(20260911)
    n_sat_cases = 0
    cycles_seen = set()

    for trial in range(500):
        # 1/4 은 포화가 나도록 크게 (|S|,|x| 가 크면 40비트 누산이 Q1.15 를 넘는다)
        if trial % 4 == 0:
            S = rand_state(rng, 32767)
            x = rand_vec(rng, 32767)
        else:
            S = rand_state(rng, 4000)
            x = rand_vec(rng, 4000)

        await load_state(dut, S)
        y_flat, sat_rtl, cyc, _, _ = await run_matvec(dut, x)
        y_rtl = unpack_vec(y_flat, D, W)

        y_gold, sat_gold = matvec(S, x)
        if sat_gold:
            n_sat_cases += 1
        cycles_seen.add(cyc)

        if not np.array_equal(y_rtl, y_gold):
            bad = np.argwhere(y_rtl != y_gold).reshape(-1)
            i = int(bad[0])
            raise AssertionError(
                f"M1: trial {trial} 비트 불일치\n"
                f"  첫 불일치 y[{i}]: RTL={int(y_rtl[i])} 골든={int(y_gold[i])} "
                f"diff={int(y_rtl[i]) - int(y_gold[i])}\n"
                f"  RTL  ={y_rtl.tolist()}\n"
                f"  골든 ={y_gold.tolist()}"
            )
        assert sat_rtl == sat_gold, (
            f"M2: trial {trial} 포화 수 불일치 RTL={sat_rtl} 골든={sat_gold}"
        )

    dut._log.info(f"M1/M2: 500쌍 비트 일치. 포화 발생 trial {n_sat_cases}개")
    dut._log.info(f"M4: 관측된 사이클 수 = {sorted(cycles_seen)}")
    assert n_sat_cases >= 50, (
        f"포화 케이스가 {n_sat_cases}개뿐이다 (50개 이상 필요) — 포화 경로를 안 본다"
    )
    assert max(cycles_seen) <= CYCLE_BUDGET, (
        f"M4: 사이클 {max(cycles_seen)} 이 예산 {CYCLE_BUDGET} 초과 "
        "(고치지 말고 숫자만 기록 — PLAN W5)"
    )


@cocotb.test()
async def test_m3_done_is_single_cycle_pulse(dut):
    """M3: done 폭이 정확히 1사이클 (DESIGN.md 5.1 — 레벨이 아니다)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    S = np.array([[1000 if i == j else 0 for j in range(D)] for i in range(D)], dtype=np.int64)
    await load_state(dut, S)
    x = np.array([500] * D, dtype=np.int64)

    _, _, cyc, done_high, _ = await run_matvec(dut, x)
    assert done_high == 1, f"M3: done 이 {done_high}회 관측됐다 (1회여야 한다)"
    dut._log.info(f"M3: done 폭 1사이클 확인. cycles={cyc}")


@cocotb.test()
async def test_m4_cycle_count_measured(dut):
    """M4: 사이클 수 실측. DESIGN.md 6절 표에 기입한다. 예산 초과면 기록만."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    rng = np.random.default_rng(7)
    seen = []
    for _ in range(10):
        S = rand_state(rng, 3000)
        await load_state(dut, S)
        _, _, cyc, _, _ = await run_matvec(dut, rand_vec(rng, 3000))
        seen.append(cyc)
    dut._log.info(f"M4: D={D} matvec 사이클 = {seen[0]} (10회 전부 {set(seen)})")
    assert len(set(seen)) == 1, f"M4: 사이클 수가 입력에 따라 다르다: {set(seen)} (상수여야 한다)"


@cocotb.test()
async def test_m5_identity_matrix_is_passthrough(dut):
    """기대값 출처: 골든 matvec — S=I·(1.0) 이면 y ≈ x. 손으로도 확인되는 케이스."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    ONE = 1 << 15
    S = np.zeros((D, D), dtype=np.int64)
    for i in range(D):
        S[i, i] = Q15_MAX          # ≈1.0 (부호 있는 Q1.15 최대)
    await load_state(dut, S)

    x = np.array([(i - 8) * 1000 for i in range(D)], dtype=np.int64)
    y_flat, sat, cyc, _, _ = await run_matvec(dut, x)
    y = unpack_vec(y_flat, D, W)
    y_gold, sat_gold = matvec(S, x)

    assert np.array_equal(y, y_gold), f"RTL={y.tolist()} 골든={y_gold.tolist()}"
    assert sat == sat_gold == 0
    # ≈1.0 이라 1 LSB 이내여야 한다
    assert np.max(np.abs(y - x)) <= 1, (
        f"S=I·Q15_MAX 인데 y 가 x 와 {int(np.max(np.abs(y - x)))} LSB 다르다: "
        f"x={x.tolist()} y={y.tolist()}"
    )
    dut._log.info(f"M5: S=I 통과 확인 (최대 오차 {int(np.max(np.abs(y - x)))} LSB)")
