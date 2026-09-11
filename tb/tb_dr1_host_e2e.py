"""기대값 출처: sim/golden/deltarule.py step() — 호스트가 받은 o 와 S 를 골든과 비트 비교.

tb_dr1_host_e2e.py — **호스트 스택으로** DR1 을 돌린다 (PLAN W9·W10)

지금까지의 DR1 테스트는 dr1_top 의 포트를 직접 흔들었다. 여기서는 실제 경로다:

    OrbitDevice (tools/orbit_device.py)
      → CocotbBackend (MMIO 레지스터 버스)
        → g2_ctrl_top (desc_queue → desc_fsm_v2 → dr1_top)
          → dr1_scratch / state_sram

즉 **호스트가 쓰는 것과 같은 코드**로 검증한다. 디스크립터 패킹, CRC, 도어벨,
IRQ 폴링, 스크래치 MMIO 창까지 전부 실물이다.

  H1. delta_init → delta_dump 가 골든 초기 상태(S=0)와 일치 (불변조건 I1)
  H2. delta_step 1토큰이 골든 step() 과 o·S 비트 일치
  H3. 10토큰 연속 비트 일치 + 포화 수 일치
  H4. 스크래치 MMIO 창 왕복 (호스트가 쓴 값을 호스트가 읽는다)
  H5. 호스트가 잘못된 디스크립터를 보내면 DeviceError 로 올라온다
  H6. **오류 주입** — 호스트가 받은 o 를 한 칸 비틀면 비교가 잡아낸다
"""

import os
import sys

import cocotb
import numpy as np
from cocotb.clock import Clock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.golden import deltarule as G  # noqa: E402
from tools.orbit_cocotb_backend import CocotbBackend  # noqa: E402
from tools.orbit_device import DeviceError, OrbitDevice  # noqa: E402
from tools.orbit_mmio_map import (  # noqa: E402
    G2_ID,
    IRQ_MASK,
    IrqBit,
    TRACE_CTRL,
    TRACE_ENABLE,
    dr1_scratch_layout,
)

D = 16


async def bringup(dut):
    """클럭·리셋·IRQ 언마스크. tb_g2_ctrl_top_host_e2e.py 와 같은 절차."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    backend = CocotbBackend(dut)
    await backend.reset()

    dev = OrbitDevice(backend)
    gid = backend.read(G2_ID.offset)
    assert gid == 0x4732_0001, f"G2_ID 가 다르다: {gid:#x}"

    await backend.async_write(TRACE_CTRL.offset, TRACE_ENABLE)
    await backend.async_write(
        IRQ_MASK.offset,
        0xFFFF_FFFF & ~((1 << IrqBit.DESC_DONE) | (1 << IrqBit.TC0_FAULT)),
    )
    return backend, dev


@cocotb.test()
async def test_h1_init_dump_is_zero_state(dut):
    """H1: 호스트가 delta_init → delta_dump 를 하면 골든 초기 상태와 같다 (I1)."""
    _, dev = await bringup(dut)

    await dev.delta_init()
    S = np.array(await dev.delta_dump(D), dtype=np.int64)

    assert S.shape == (D, D), f"H1: 덤프 모양이 {S.shape}"
    assert np.array_equal(S, np.zeros((D, D), dtype=np.int64)), (
        f"H1: INIT 직후 상태가 0 이 아니다\n{S}"
    )
    dut._log.info("H1: 호스트 경로로 I1 확인")


@cocotb.test()
async def test_h2_one_token_matches_golden(dut):
    """H2: delta_step 1토큰이 골든 step() 과 o·S 비트 일치."""
    _, dev = await bringup(dut)
    await dev.delta_init()

    rng = np.random.default_rng(11)
    q = G.random_vec(rng, D, 0.2)
    k = G.random_vec(rng, D, 0.2)
    v = G.random_vec(rng, D, 0.2)
    alpha = G.float_to_uq15(0.95)
    beta = G.float_to_uq15(0.25)

    o_host, sat, cycles = await dev.delta_step(q, k, v, alpha, beta)
    S_next_gold, o_gold, sat_gold = G.step(
        np.zeros((D, D), dtype=np.int64), q, k, v, alpha, beta
    )

    o_host = np.array(o_host, dtype=np.int64)
    if not np.array_equal(o_host, o_gold):
        bad = np.argwhere(o_host != o_gold).reshape(-1)
        i = int(bad[0])
        raise AssertionError(
            f"H2: o 비트 불일치\n"
            f"  첫 불일치 o[{i}]: 호스트={int(o_host[i])} 골든={int(o_gold[i])} "
            f"diff={int(o_host[i]) - int(o_gold[i])}\n"
            f"  호스트={o_host.tolist()}\n  골든  ={o_gold.tolist()}"
        )
    assert sat == sat_gold, f"H2: 포화 수 호스트={sat} 골든={sat_gold}"

    S_host = np.array(await dev.delta_dump(D), dtype=np.int64)
    assert np.array_equal(S_host, S_next_gold), (
        f"H2: 상태 불일치\n호스트=\n{S_host}\n골든=\n{S_next_gold}"
    )
    dut._log.info(f"H2: 1토큰 비트 일치 (사이클 {cycles}, 포화 {sat})")


@cocotb.test()
async def test_h3_ten_tokens(dut):
    """H3: 10토큰 연속. 상태 누적 경로를 호스트 스택으로 본다."""
    _, dev = await bringup(dut)
    await dev.delta_init()

    rng = np.random.default_rng(12)
    S_ref = np.zeros((D, D), dtype=np.int64)
    sat_total_host = 0
    sat_total_gold = 0

    for t in range(10):
        q = G.random_vec(rng, D, 0.2)
        k = G.random_vec(rng, D, 0.2)
        v = G.random_vec(rng, D, 0.2)
        alpha = G.float_to_uq15(0.9)
        beta = G.float_to_uq15(0.2)

        o_host, sat, _ = await dev.delta_step(q, k, v, alpha, beta)
        S_ref, o_gold, sat_gold = G.step(S_ref, q, k, v, alpha, beta)

        sat_total_host += sat
        sat_total_gold += sat_gold
        o_host = np.array(o_host, dtype=np.int64)
        if not np.array_equal(o_host, o_gold):
            bad = np.argwhere(o_host != o_gold).reshape(-1)
            i = int(bad[0])
            raise AssertionError(
                f"H3: 토큰 {t} o[{i}] 호스트={int(o_host[i])} 골든={int(o_gold[i])}"
            )

    S_host = np.array(await dev.delta_dump(D), dtype=np.int64)
    assert np.array_equal(S_host, S_ref), "H3: 10토큰 뒤 상태가 다르다"
    assert sat_total_host == sat_total_gold, (
        f"H3: 포화 누적 호스트={sat_total_host} 골든={sat_total_gold}"
    )
    dut._log.info("H3: 10토큰 비트 일치")


@cocotb.test()
async def test_h4_scratch_mmio_round_trip(dut):
    """H4: 스크래치 MMIO 창 왕복. 읽기를 두 번 하는 규약까지 확인한다."""
    _, dev = await bringup(dut)

    lay = dr1_scratch_layout(D)
    vec = [1, -1, 32767, -32768, 1234, -4321, 0, 9999,
           -9999, 100, -100, 7, -7, 300, -300, 512]
    await dev.dr1_write_vec(lay["v"], vec)
    got = await dev.dr1_read_vec(lay["v"], D)
    assert got == vec, f"H4: 스크래치 왕복 실패\n  쓴 것  ={vec}\n  읽은 것={got}"
    dut._log.info("H4: 스크래치 MMIO 왕복 OK (부호·경계값 포함)")


@cocotb.test()
async def test_h5_bad_descriptor_raises(dut):
    """H5: 잘못된 디스크립터는 호스트 쪽에서 DeviceError 로 올라온다.

    슬롯 초과는 호스트 패커가 먼저 잡고, 우회하면 하드웨어가 fault 를 낸다.
    둘 다 **조용히 성공하지 않는다**는 것이 요점이다.
    """
    from tools.orbit_desc import Dr1FieldError, crc8, pack_delta_init
    from tools.orbit_mmio_map import DESC_SIZE, DR1_SLOT_OFF

    _, dev = await bringup(dut)

    try:
        pack_delta_init(slot=1)
    except Dr1FieldError:
        pass
    else:
        raise AssertionError("H5: 호스트 패커가 잘못된 슬롯을 안 잡았다")

    # 호스트 검사를 우회해서 하드웨어까지 보낸다
    raw = list(pack_delta_init(slot=0))
    raw[DR1_SLOT_OFF] = 1
    raw[DESC_SIZE - 1] = crc8(raw[: DESC_SIZE - 1])

    try:
        await dev._dr1_run(bytes(raw))
    except DeviceError as e:
        assert "0x05" in str(e), f"H5: fault_code 가 보고되지 않았다: {e}"
        dut._log.info(f"H5: 하드웨어 fault 가 호스트까지 올라왔다 — {e}")
    else:
        raise AssertionError("H5: 잘못된 슬롯인데 성공했다")


@cocotb.test()
async def test_h6_fault_injection_host_path(dut):
    """H6: 호스트가 받은 o 를 한 칸 비틀면 비교가 잡는가.

    H2/H3 가 통과한다는 말이 의미를 가지려면, 비교가 실제로 동작한다는 증거가
    있어야 한다. 골든-대-골든은 항상 통과하기 때문이다.
    """
    _, dev = await bringup(dut)
    await dev.delta_init()

    rng = np.random.default_rng(13)
    q = G.random_vec(rng, D, 0.2)
    k = G.random_vec(rng, D, 0.2)
    v = G.random_vec(rng, D, 0.2)
    alpha = G.float_to_uq15(0.9)
    beta = G.float_to_uq15(0.3)

    o_host, _, _ = await dev.delta_step(q, k, v, alpha, beta)
    _, o_gold, _ = G.step(np.zeros((D, D), dtype=np.int64), q, k, v, alpha, beta)

    o_bad = np.array(o_host, dtype=np.int64).copy()
    o_bad[5] += 1                                  # 고의로 1 LSB 틀리게
    assert not np.array_equal(o_bad, o_gold), (
        "H6: 1 LSB 를 틀렸는데 비교가 같다고 한다 — 비교가 죽어 있다"
    )
    diff = np.argwhere(o_bad != o_gold).reshape(-1)
    assert list(diff) == [5], f"H6: 비교가 위치를 잘못 찍었다 {diff}"
    dut._log.info("H6: 오류 주입이 o[5] 에서 잡혔다")
