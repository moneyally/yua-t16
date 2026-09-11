"""기대값 출처: tools/orbit_pack.py pack_vec (spec/deltarule.md 3.5절 평탄화 규칙).

tb_vec_regs.py — rtl/dr1/vec_regs.sv (PLAN W5)

  V1. q, k, v 세 벡터를 적재한 뒤 flat 출력이 pack_vec 결과와 비트 일치
  V2. 리셋 후 전부 0
  V3. 한 벡터만 적재하면 나머지는 안 변한다 (ld_sel 이 실제로 동작하는지)
  V4. ld_en=0 이면 아무것도 안 바뀐다

평탄화 규칙을 테스트벤치가 직접 계산하지 않고 `tools/orbit_pack.pack_vec` 를 쓴다 —
규칙이 두 곳에 있으면 한쪽을 고칠 때 다른 쪽이 조용히 틀린다.
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
SEL_Q, SEL_K, SEL_V = 0, 1, 2


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.ld_en.value = 0
    dut.ld_sel.value = 0
    dut.ld_idx.value = 0
    dut.ld_data.value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def load_vec(dut, sel: int, vec):
    """원소를 하나씩 적재한다."""
    for i, x in enumerate(vec):
        dut.ld_en.value = 1
        dut.ld_sel.value = sel
        dut.ld_idx.value = i
        dut.ld_data.value = int(x) & 0xFFFF
        await RisingEdge(dut.clk)
    dut.ld_en.value = 0
    await FallingEdge(dut.clk)


@cocotb.test()
async def test_v2_reset_is_zero(dut):
    """V2: 리셋 후 세 출력 전부 0."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await FallingEdge(dut.clk)
    for name in ("q_flat", "k_flat", "v_flat"):
        assert int(getattr(dut, name).value) == 0, f"V2: 리셋 후 {name} 이 0 이 아니다"
    dut._log.info("V2: 리셋 후 전부 0")


@cocotb.test()
async def test_v1_load_three_vectors(dut):
    """V1: 세 벡터 적재 후 flat 이 pack_vec 결과와 비트 일치."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    qv = [((i * 4099) % 65536) - 32768 for i in range(D)]
    kv = [(-1) ** i * (i * 137 + 1) for i in range(D)]
    vv = [32767 if i % 3 == 0 else -32768 if i % 3 == 1 else 0 for i in range(D)]

    await load_vec(dut, SEL_Q, qv)
    await load_vec(dut, SEL_K, kv)
    await load_vec(dut, SEL_V, vv)

    for name, vec in (("q_flat", qv), ("k_flat", kv), ("v_flat", vv)):
        got = int(getattr(dut, name).value)
        want = pack_vec(vec, W)
        assert got == want, (
            f"V1: {name} 불일치\n  기대 0x{want:x}\n  실제 0x{got:x}\n"
            f"  풀어서: 기대={vec}\n         실제={unpack_vec(got, D, W).tolist()}"
        )
    dut._log.info("V1: q/k/v 세 벡터 pack_vec 과 비트 일치")


@cocotb.test()
async def test_v3_sel_isolates_vectors(dut):
    """V3: ld_sel 이 실제로 벡터를 구분하는지 — 하나만 바꾸면 나머지는 그대로."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    base = [i - 8 for i in range(D)]
    await load_vec(dut, SEL_Q, base)
    await load_vec(dut, SEL_K, base)
    await load_vec(dut, SEL_V, base)
    want = pack_vec(base, W)

    # k 만 바꾼다
    other = [1000 + i for i in range(D)]
    await load_vec(dut, SEL_K, other)

    assert int(dut.q_flat.value) == want, "V3: k 를 적재했는데 q 가 바뀌었다"
    assert int(dut.v_flat.value) == want, "V3: k 를 적재했는데 v 가 바뀌었다"
    assert int(dut.k_flat.value) == pack_vec(other, W), "V3: k 가 안 바뀌었다"
    dut._log.info("V3: ld_sel 구분 동작")


@cocotb.test()
async def test_v4_ld_en_low_does_nothing(dut):
    """V4: ld_en=0 이면 ld_data/ld_idx 를 흔들어도 아무것도 안 바뀐다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    base = [i * 3 for i in range(D)]
    await load_vec(dut, SEL_Q, base)
    snapshot = int(dut.q_flat.value)

    dut.ld_en.value = 0
    for i in range(D):
        dut.ld_sel.value = SEL_Q
        dut.ld_idx.value = i
        dut.ld_data.value = 0xDEAD
        await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    assert int(dut.q_flat.value) == snapshot, "V4: ld_en=0 인데 값이 바뀌었다"
    dut._log.info("V4: ld_en=0 무동작 확인")
