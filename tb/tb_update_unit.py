"""기대값 출처: sim/golden/deltarule.py update_row(S_row, α, β, err_i, k) — 골든과 비트 비교.

tb_update_unit.py — rtl/dr1/update_unit.sv (PLAN W6)

  U1. 무작위 300세트가 골든 update_row 와 비트 일치 (포화 케이스 강제 포함)
  U2. sat_count 가 골든의 sat_count 와 일치
  U3. 특수 케이스: α=β=0x8000(=1.0), β=0(갱신 없음), α=0(상태 지움), 포화 최대치
  U4. done 폭이 정확히 1사이클 (DESIGN.md 5.1)
  U5. 사이클 실측 ≤ D+6 (DESIGN.md 6.1 표에 기입)

평탄화는 `tools/orbit_pack.py` 만 쓴다. 레지스터 출력은 falling edge 샘플링.
"""

import os
import sys

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.golden.deltarule import UQ15_ONE, update_row  # noqa: E402
from tools.orbit_pack import pack_vec, to_bits, unpack_vec  # noqa: E402

D = 16
W = 16
CYCLE_BUDGET = D + 6        # 계약 상한 (DESIGN.md 6.1). 실측은 테스트가 찍는다


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.alpha_uq15.value = 0
    dut.beta_uq15.value = 0
    dut.err_i.value = 0
    dut.s_row_flat.value = 0
    dut.k_flat.value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def run_update(dut, s_row, k, alpha, beta, err):
    """start 1펄스 → done 까지. (row, sat_count, cycles, done 폭) 반환."""
    dut.s_row_flat.value = pack_vec(s_row, W)
    dut.k_flat.value = pack_vec(k, W)
    dut.alpha_uq15.value = int(alpha)
    dut.beta_uq15.value = int(beta)
    dut.err_i.value = to_bits(int(err), W)      # 2의 보수로 넣는다
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    done_high = 0
    row = sat = cyc = None
    for _ in range(8 * D):
        await FallingEdge(dut.clk)
        if int(dut.done.value):
            done_high += 1
            row = int(dut.row_flat.value)
            sat = int(dut.sat_count.value)
            cyc = int(dut.cycles.value)
        elif done_high:
            break
    assert row is not None, "done 이 오지 않았다"
    return row, sat, cyc, done_high


def rand_vec(rng, lim):
    return rng.integers(-lim, lim + 1, size=D, dtype=np.int64)


async def check_one(dut, s_row, k, alpha, beta, err, tag):
    row_flat, sat_rtl, cyc, _ = await run_update(dut, s_row, k, alpha, beta, err)
    row_rtl = unpack_vec(row_flat, D, W)
    row_gold, sat_gold = update_row(s_row, int(alpha), int(beta), int(err), k)

    if not np.array_equal(row_rtl, row_gold):
        bad = np.argwhere(row_rtl != row_gold).reshape(-1)
        j = int(bad[0])
        raise AssertionError(
            f"{tag} 비트 불일치 (α=0x{int(alpha):04X} β=0x{int(beta):04X} err={int(err)})\n"
            f"  첫 불일치 row[{j}]: RTL={int(row_rtl[j])} 골든={int(row_gold[j])} "
            f"diff={int(row_rtl[j]) - int(row_gold[j])}\n"
            f"  S_row={np.asarray(s_row).tolist()}\n"
            f"  k    ={np.asarray(k).tolist()}\n"
            f"  RTL  ={row_rtl.tolist()}\n"
            f"  골든 ={row_gold.tolist()}"
        )
    assert sat_rtl == sat_gold, (
        f"{tag} 포화 수 불일치 RTL={sat_rtl} 골든={sat_gold} "
        f"(α=0x{int(alpha):04X} β=0x{int(beta):04X} err={int(err)})"
    )
    return cyc, sat_gold


@cocotb.test()
async def test_u1_random_300_sets_bit_exact(dut):
    """U1/U2: 무작위 300세트가 골든 update_row 와 비트 일치. 포화 케이스 포함."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    rng = np.random.default_rng(20260912)
    n_sat = 0
    cycles_seen = set()

    for trial in range(300):
        if trial % 4 == 0:
            # 포화 유발: 큰 상태 + α=1.0 + 큰 err·k
            s_row = rand_vec(rng, 32767)
            k = rand_vec(rng, 32767)
            alpha = UQ15_ONE
            beta = UQ15_ONE
            err = int(rng.integers(-32768, 32768))
        else:
            s_row = rand_vec(rng, 20000)
            k = rand_vec(rng, 20000)
            alpha = int(rng.integers(0, UQ15_ONE + 1))
            beta = int(rng.integers(0, UQ15_ONE + 1))
            err = int(rng.integers(-32768, 32768))

        cyc, sat = await check_one(dut, s_row, k, alpha, beta, err, f"U1 trial {trial}")
        cycles_seen.add(cyc)
        if sat:
            n_sat += 1

    dut._log.info(f"U1/U2: 300세트 비트 일치. 포화 발생 세트 {n_sat}개")
    dut._log.info(f"U5: 관측된 사이클 수 = {sorted(cycles_seen)}")
    assert n_sat >= 30, f"포화 케이스가 {n_sat}개뿐이다 — 포화 경로를 안 본다"
    assert max(cycles_seen) <= CYCLE_BUDGET, (
        f"U5: 사이클 {max(cycles_seen)} 이 예산 {CYCLE_BUDGET} 초과"
    )


@cocotb.test()
async def test_u3_special_cases(dut):
    """U3: α=β=1.0 / β=0 / α=0 / 최대 포화 — 골든과 비트 일치.

    기대값은 전부 골든 update_row 에서 온다. 손으로 쓴 상수를 비교하지 않는다.
    다만 의미는 주석으로 남긴다:
      β=0  → row = requant(α·S)  (갱신 없음)
      α=0  → row = requant(berr·k) (이전 상태를 완전히 버림)
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    rng = np.random.default_rng(7)
    s_row = rand_vec(rng, 30000)
    k = rand_vec(rng, 30000)

    # α = β = 1.0 (0x8000). Q1.15 에 없는 1.0 을 UQ1.15 가 정확히 표현한다.
    await check_one(dut, s_row, k, UQ15_ONE, UQ15_ONE, 12345, "U3 α=β=1.0")

    # β = 0 → 갱신 항이 죽는다. row = requant(α·S)
    row_flat, sat, _, _ = await run_update(dut, s_row, k, UQ15_ONE, 0, 32767)
    row_rtl = unpack_vec(row_flat, D, W)
    row_gold, sat_gold = update_row(s_row, UQ15_ONE, 0, 32767, k)
    assert np.array_equal(row_rtl, row_gold), f"U3 β=0: {row_rtl.tolist()} != {row_gold.tolist()}"
    assert sat == sat_gold
    # α=1.0, β=0 이면 상태가 그대로 남아야 한다 (requant(1.0·S) = S)
    assert np.array_equal(row_rtl, np.asarray(s_row)), (
        f"U3 β=0 인데 상태가 변했다: {row_rtl.tolist()} != {np.asarray(s_row).tolist()}"
    )

    # α = 0 → 이전 상태를 완전히 버린다
    await check_one(dut, s_row, k, 0, UQ15_ONE, -32768, "U3 α=0")

    # 포화 최대치: α=1.0, S=최대, β=1.0, err=최대, k=최대 → 전 열 포화
    s_max = np.full(D, 32767, dtype=np.int64)
    k_max = np.full(D, 32767, dtype=np.int64)
    _, sat_gold = await check_one(dut, s_max, k_max, UQ15_ONE, UQ15_ONE, 32767, "U3 포화 최대")
    assert sat_gold >= D, f"U3: 전 열이 포화해야 하는데 골든 sat={sat_gold}"


@cocotb.test()
async def test_u4_done_is_single_cycle_pulse(dut):
    """U4: done 폭이 정확히 1사이클 (DESIGN.md 5.1 — 레벨이 아니다)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    rng = np.random.default_rng(99)
    s_row = rand_vec(rng, 1000)
    k = rand_vec(rng, 1000)
    _, _, _, done_high = await run_update(dut, s_row, k, UQ15_ONE, UQ15_ONE, 500)
    assert done_high == 1, f"U4: done 이 {done_high} 사이클 동안 높다 (1이어야 한다)"


@cocotb.test()
async def test_u5_cycles_constant_and_back_to_back(dut):
    """U5: 사이클 수가 입력과 무관한 상수이고, 연속 호출이 같은 값을 낸다.

    dr1_top 이 D행을 이 유닛으로 스트리밍하므로(W7) 행마다 같은 비용이어야 한다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    rng = np.random.default_rng(4242)
    seen = []
    for _ in range(D):      # 행 D개를 연속으로 흘린다
        s_row = rand_vec(rng, 25000)
        k = rand_vec(rng, 25000)
        cyc, _ = await check_one(
            dut, s_row, k, UQ15_ONE, int(rng.integers(0, UQ15_ONE + 1)),
            int(rng.integers(-32768, 32768)), "U5",
        )
        seen.append(cyc)

    assert len(set(seen)) == 1, f"U5: 사이클이 흔들린다 {seen}"
    dut._log.info(f"U5: 행당 {seen[0]} 사이클 고정 (D={D}행이면 {seen[0] * D} 사이클)")
    assert seen[0] <= CYCLE_BUDGET
