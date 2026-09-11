"""기대값 출처: sim/golden/deltarule.py compute_err(v, p, alpha_uq15) — 골든과 비트 비교.

tb_err_unit.py — rtl/dr1/err_unit.sv (PLAN W7)

  E1. 무작위 500세트가 골든 compute_err 와 비트 일치 (포화 강제 포함)
  E2. sat_count 가 골든과 일치. **포화가 두 종류**라 따로 확인한다:
      재양자화 포화(α·p 가 Q1.15 를 넘음) / 뺄셈 포화(v−ap 가 넘음)
  E3. α=0x8000(1.0) 이면 err = v − p (정확히), α=0 이면 err = v
  E4. 순수 조합 — 클럭 없이 입력만 바꿔도 출력이 따라온다

err_unit 은 조합 논리라 클럭이 없다. Timer 로만 진행한다.
"""

import os
import sys

import cocotb
import numpy as np
from cocotb.triggers import Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.golden.deltarule import UQ15_ONE, compute_err  # noqa: E402
from tools.orbit_pack import pack_vec, unpack_vec  # noqa: E402

D = 16
W = 16


async def apply(dut, v, p, alpha):
    dut.v_flat.value = pack_vec(v, W)
    dut.p_flat.value = pack_vec(p, W)
    dut.alpha_uq15.value = int(alpha)
    await Timer(1, unit="ns")
    return unpack_vec(int(dut.err_flat.value), D, W), int(dut.sat_count.value)


async def check_one(dut, v, p, alpha, tag):
    err_rtl, sat_rtl = await apply(dut, v, p, alpha)
    err_gold, sat_gold = compute_err(v, p, int(alpha))

    if not np.array_equal(err_rtl, err_gold):
        bad = np.argwhere(err_rtl != err_gold).reshape(-1)
        i = int(bad[0])
        raise AssertionError(
            f"{tag} 비트 불일치 (α=0x{int(alpha):04X})\n"
            f"  첫 불일치 err[{i}]: RTL={int(err_rtl[i])} 골든={int(err_gold[i])} "
            f"diff={int(err_rtl[i]) - int(err_gold[i])}\n"
            f"  v={np.asarray(v).tolist()}\n"
            f"  p={np.asarray(p).tolist()}"
        )
    assert sat_rtl == sat_gold, (
        f"{tag} 포화 수 불일치 RTL={sat_rtl} 골든={sat_gold} (α=0x{int(alpha):04X})"
    )
    return sat_gold


def rand_vec(rng, lim):
    return rng.integers(-lim, lim + 1, size=D, dtype=np.int64)


@cocotb.test()
async def test_e1_random_500_bit_exact(dut):
    """E1/E2: 무작위 500세트가 골든 compute_err 와 비트 일치."""
    rng = np.random.default_rng(20260913)
    n_sat = 0

    for trial in range(500):
        if trial % 3 == 0:
            # 포화 유발: v 와 α·p 가 반대 부호로 크게
            v = rng.integers(16000, 32768, size=D, dtype=np.int64)
            p = -rng.integers(16000, 32768, size=D, dtype=np.int64)
            alpha = UQ15_ONE
        else:
            v = rand_vec(rng, 20000)
            p = rand_vec(rng, 20000)
            alpha = int(rng.integers(0, UQ15_ONE + 1))

        sat = await check_one(dut, v, p, alpha, f"E1 trial {trial}")
        if sat:
            n_sat += 1

    dut._log.info(f"E1/E2: 500세트 비트 일치. 포화 발생 {n_sat}세트")
    assert n_sat >= 100, f"포화 케이스가 {n_sat}개뿐이다 — 포화 경로를 안 본다"


@cocotb.test()
async def test_e2_both_saturation_kinds(dut):
    """E2: 포화 두 종류를 **각각** 만들어서 골든과 개수가 맞는지.

    골든은 q15_mul(재양자화)과 sat_q15(뺄셈)에서 따로 센다. RTL 도 그래야 한다.
    """
    # (a) 재양자화 포화만: α=1.0, p=-32768 → α·p = -32768 (포화 없음)
    #     α·p 가 Q1.15 를 넘으려면 α>1 이어야 하는데 클램프되므로 여기서는
    #     |α·p| ≤ 32768 이다. -32768 은 Q1.15 최소값이라 포화가 아니다.
    #     따라서 재양자화 포화는 **구조적으로 나기 어렵다** — 그것도 골든과 같아야 한다.
    v = np.zeros(D, dtype=np.int64)
    p = np.full(D, -32768, dtype=np.int64)
    sat = await check_one(dut, v, p, UQ15_ONE, "E2 재양자화")
    dut._log.info(f"E2 재양자화 경계: sat={sat} (골든과 일치)")

    # (b) 뺄셈 포화만: v=32767, α·p 가 음수로 크다 → v-ap > 32767
    v = np.full(D, 32767, dtype=np.int64)
    p = np.full(D, -32768, dtype=np.int64)
    sat = await check_one(dut, v, p, UQ15_ONE, "E2 뺄셈")
    assert sat >= D, f"E2: 전 원소가 뺄셈 포화해야 하는데 sat={sat}"
    dut._log.info(f"E2 뺄셈 포화: sat={sat}")


@cocotb.test()
async def test_e3_alpha_extremes(dut):
    """E3: α=1.0 → err=v−p, α=0 → err=v. 기대값은 골든에서 오지만 의미를 확인한다."""
    rng = np.random.default_rng(31)
    v = rand_vec(rng, 10000)
    p = rand_vec(rng, 10000)

    err_rtl, _ = await apply(dut, v, p, UQ15_ONE)
    await check_one(dut, v, p, UQ15_ONE, "E3 α=1.0")
    assert np.array_equal(err_rtl, np.asarray(v) - np.asarray(p)), (
        f"E3: α=1.0 이면 err=v−p 여야 한다\n  RTL={err_rtl.tolist()}"
    )

    err_rtl, _ = await apply(dut, v, p, 0)
    await check_one(dut, v, p, 0, "E3 α=0")
    assert np.array_equal(err_rtl, np.asarray(v)), (
        f"E3: α=0 이면 err=v 여야 한다\n  RTL={err_rtl.tolist()}"
    )


@cocotb.test()
async def test_e4_purely_combinational(dut):
    """E4: 클럭 없이 입력만 바꿔도 출력이 바뀐다 (레지스터가 섞여 있지 않다)."""
    rng = np.random.default_rng(4)
    seen = set()
    for _ in range(8):
        v = rand_vec(rng, 30000)
        p = rand_vec(rng, 30000)
        err, _ = await apply(dut, v, p, UQ15_ONE)
        seen.add(tuple(int(x) for x in err))
    assert len(seen) == 8, (
        f"E4: 입력 8세트에 서로 다른 출력이 8개 나와야 하는데 {len(seen)}개다 — "
        "출력이 래치돼 있을 수 있다"
    )
