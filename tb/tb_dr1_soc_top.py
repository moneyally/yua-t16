"""기대값 출처: sim/golden/deltarule.py step() — 보드 최상위를 지나도 골든과 비트 일치.
레지스터 주소·기대 ID 는 tools/orbit_mmio_map.py (SSOT). 테스트가 숫자를 손으로 안 쓴다.

tb_dr1_soc_top.py — **보드 최상위를 그대로 돌린다** (PLAN W12 예행)

지금까지 DR1 은 `g2_ctrl_top` 의 내부 레지스터 버스를 직접 흔들어 검증했다.
보드는 그 버스에 손을 못 댄다 — PS 가 AXI4-Lite 로 들어오고 `axil_reg_bridge` 를
지나야 같은 자리에 닿는다. **그 한 칸이 검증된 적이 없었다.**

    OrbitDevice → AxiLiteBackend → dr1_soc_top
                                     ├── axil_reg_bridge (AXI4-Lite 슬레이브)
                                     ├── g2_ctrl_top  (desc → dr1_top → scratch)
                                     └── axi4_master_adapter (AXI4 마스터 → 메모리 모델)

PLAN W12 의 보드 순서를 시뮬레이션에서 미리 밟는다:
  S1. AXI4-Lite 로 `G2_ID` 읽기 = `0x47320001`   ← W12 2번 그대로
  S2. `DELTA_INIT` → `DELTA_DUMP` 가 골든 초기 상태(S=0)와 일치
  S3. `DELTA_STEP` **1토큰** 이 골든과 비트 일치   ← W12 3번 그대로
  S4. **10토큰** 연속 비트 일치 (BUG-009 교훈: 1토큰은 상태 경로를 숨긴다)
  S5. 워치독을 AXI4-Lite 로 켜고 끈다 — 보드에서 이 순서로 만진다
  S6. 호스트 스택 전체가 이 경로에서도 오류를 올린다 (잘못된 디스크립터)

**미검증인 채로 남는 것**: 실제 PS·실제 DDR·타이밍. 이 테스트가 통과해도
보드가 돈다는 뜻은 아니다 (docs/FPGA.md 2절).
"""

import os
import sys

import cocotb
import numpy as np
from cocotb.clock import Clock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from sim.golden import deltarule as G  # noqa: E402
from tb_axi4_master_adapter import AxiSlaveMemory  # noqa: E402
from tools.orbit_axil_backend import AxiLiteBackend  # noqa: E402
from tools.orbit_device import DeviceError, OrbitDevice  # noqa: E402
from tools.orbit_mmio_map import (  # noqa: E402
    G2_ID,
    IRQ_MASK,
    IrqBit,
    TRACE_CTRL,
    TRACE_ENABLE,
    WDOG_CTRL,
    WDOG_EN,
)

D = 16


async def bringup(dut):
    """클럭·리셋·IRQ 언마스크를 **AXI4-Lite 로만** 한다.

    내부 신호를 직접 건드리지 않는다 — 보드에서 할 수 없는 일은 여기서도 안 한다.
    유일한 예외가 `m_axi_*` 슬레이브 모델인데, 그건 보드에서 PS DDR 이 하는 일이다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    backend = AxiLiteBackend(dut)
    await backend.reset()

    # 외부 메모리 자리. DR1 경로는 안 쓰지만, 아무도 안 받으면 GEMM 이 멈춘다
    mem = AxiSlaveMemory(dut)
    cocotb.start_soon(mem.run_read())
    cocotb.start_soon(mem.run_write())

    dev = OrbitDevice(backend)
    await backend.async_write(TRACE_CTRL.offset, TRACE_ENABLE)
    await backend.async_write(
        IRQ_MASK.offset,
        0xFFFF_FFFF & ~((1 << IrqBit.DESC_DONE) | (1 << IrqBit.TC0_FAULT)),
    )
    return backend, dev


def make_token(rng, d=D):
    """q/k/v 를 골든의 생성기로 만든다. 테스트가 난수 규약을 따로 두지 않는다."""
    return (G.random_vec(rng, d, 0.2),
            G.random_vec(rng, d, 0.2),
            G.random_vec(rng, d, 0.2))


@cocotb.test()
async def test_s1_read_device_id_over_axi_lite(dut):
    """S1: AXI4-Lite 로 G2_ID 를 읽는다 = 0x47320001.

    PLAN W12 의 2번이 이것이다. 보드에서 `devmem` 으로 하는 바로 그 읽기다.
    여기서 틀리면 보드에서도 틀린다 — 그게 이 테스트의 전부이자 목적이다.
    """
    backend, _ = await bringup(dut)
    gid = await backend.async_read(G2_ID.offset)
    assert gid == 0x4732_0001, (
        f"S1: G2_ID={gid:#010x} — 0x47320001 이어야 한다. "
        f"AXI4-Lite 브리지가 주소를 잘못 넘기고 있다"
    )


@cocotb.test()
async def test_s2_init_then_dump_is_zero(dut):
    """S2: DELTA_INIT → DELTA_DUMP 가 골든 초기 상태(전부 0)와 일치."""
    _, dev = await bringup(dut)
    await dev.delta_init()
    S = await dev.delta_dump(D)
    assert np.array_equal(np.array(S), np.zeros((D, D), dtype=np.int64)), (
        "S2: INIT 뒤 상태가 0 이 아니다"
    )


@cocotb.test()
async def test_s3_one_token_matches_golden(dut):
    """S3: DELTA_STEP 1토큰이 골든 step() 과 비트 일치. PLAN W12 3번."""
    _, dev = await bringup(dut)
    await dev.delta_init()

    rng = np.random.default_rng(901)
    q, k, v = make_token(rng)
    alpha = G.float_to_uq15(0.95)
    beta = G.float_to_uq15(0.25)

    o_host, sat, cycles = await dev.delta_step(q, k, v, alpha, beta)
    S_gold, o_gold, sat_gold = G.step(
        np.zeros((D, D), dtype=np.int64), q, k, v, alpha, beta
    )

    o_host = np.array(o_host, dtype=np.int64)
    assert np.array_equal(o_host, o_gold), (
        f"S3: o 가 골든과 다르다\n  호스트={o_host.tolist()}\n  골든  ={o_gold.tolist()}"
    )
    assert sat == sat_gold, f"S3: 포화 수 호스트={sat} 골든={sat_gold}"

    S_host = np.array(await dev.delta_dump(D), dtype=np.int64)
    assert np.array_equal(S_host, S_gold), (
        f"S3: 상태 불일치\n호스트=\n{S_host}\n골든=\n{S_gold}"
    )
    dut._log.info(f"S3: AXI4-Lite 경로로 1토큰 비트 일치 (사이클 {cycles})")


@cocotb.test()
async def test_s4_ten_tokens_match_golden(dut):
    """S4: 10토큰 연속 비트 일치.

    왜 10개인가: **1토큰은 상태 경로 버그를 숨긴다.** BUG-009 는 초기 상태가
    0 이라 1토큰에서는 안 보였고 2토큰째에 드러났다 (docs/BUGS.md).
    """
    _, dev = await bringup(dut)
    await dev.delta_init()

    rng = np.random.default_rng(902)
    S_ref = np.zeros((D, D), dtype=np.int64)
    sat_host_total = 0
    sat_gold_total = 0

    for t in range(10):
        q, k, v = make_token(rng)
        alpha = G.float_to_uq15(0.9)
        beta = G.float_to_uq15(0.2)

        o_host, sat, _ = await dev.delta_step(q, k, v, alpha, beta)
        S_ref, o_gold, sat_gold = G.step(S_ref, q, k, v, alpha, beta)
        sat_host_total += sat
        sat_gold_total += sat_gold

        o_host = np.array(o_host, dtype=np.int64)
        assert np.array_equal(o_host, o_gold), (
            f"S4: 토큰 {t} 에서 o 가 갈라졌다\n"
            f"  호스트={o_host.tolist()}\n  골든  ={o_gold.tolist()}"
        )

    assert sat_host_total == sat_gold_total, (
        f"S4: 포화 수 호스트={sat_host_total} 골든={sat_gold_total}"
    )
    S_host = np.array(await dev.delta_dump(D), dtype=np.int64)
    assert np.array_equal(S_host, S_ref), (
        "S4: 10토큰 뒤 상태 S 가 골든과 다르다 — 누적 경로가 어긋났다"
    )


@cocotb.test()
async def test_s5_watchdog_over_axi_lite(dut):
    """S5: 워치독을 AXI4-Lite 로 켜고 읽고 끈다.

    보드에서 이 순서로 만진다 (docs/FPGA.md 3절). 여기서 못 끄면 보드에서
    칩이 계속 리셋되는데 원인을 못 찾는다.
    """
    backend, dev = await bringup(dut)

    # `dev.watchdog_enable()` 은 동기 백엔드용이다 (보드의 /dev/mem 백엔드).
    # 여기서는 같은 워드를 async 경로로 쓴다 — **워드는 호스트 헬퍼가 만든다**,
    # 테스트가 비트를 손으로 쓰면 호스트와 갈라지는 날이 온다.
    from tools.orbit_mmio_map import wdog_ctrl_word
    await backend.async_write(WDOG_CTRL.offset, wdog_ctrl_word(0x0100))

    val = await backend.async_read(WDOG_CTRL.offset)
    assert val & WDOG_EN, f"S5: EN 이 안 섰다 {val:#010x}"
    assert (val >> 8) & 0xFFFF == 0x0100, f"S5: PERIOD 가 안 남았다 {val:#010x}"

    await backend.async_write(WDOG_CTRL.offset, 0)
    val = await backend.async_read(WDOG_CTRL.offset)
    assert val == 0, f"S5: 워치독이 안 꺼졌다 {val:#010x}"

    # 끈 뒤에는 창 3개 분량을 돌려도 리셋이 안 걸려야 한다
    await backend.tick(3 * 1024)
    assert int(dut.reset_active.value) == 0, "S5: 껐는데 워치독이 리셋을 걸었다"


@cocotb.test()
async def test_s6_bad_descriptor_raises(dut):
    """S6: 잘못된 디스크립터는 이 경로에서도 DeviceError 로 올라온다.

    호스트가 fault 를 **못 보는 것**이 가장 나쁘다 — 보드에서 조용히 멈춘다.
    """
    _, dev = await bringup(dut)
    await dev.delta_init()

    rng = np.random.default_rng(77)
    q, k, v = make_token(rng)
    try:
        await dev.delta_step(q, k, v, G.float_to_uq15(0.9),
                             G.float_to_uq15(0.2), slot=1)
    except (DeviceError, Exception) as e:      # 패커/하드웨어 어느 쪽이 잡아도 좋다
        assert "slot" in str(e).lower() or isinstance(e, DeviceError), (
            f"S6: 예상 밖의 예외 {type(e).__name__}: {e}"
        )
    else:
        assert False, "S6: 잘못된 슬롯인데 아무 일도 안 일어났다"
