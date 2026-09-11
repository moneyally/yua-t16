"""기대값 출처: spec/watchdog.md 1·2절 + tools/orbit_mmio_map.py (레지스터맵 SSOT).
호스트가 쓰는 워드는 `wdog_ctrl_word()` 가 만든다 — 테스트가 비트를 손으로 안 쓴다.

tb_g2_ctrl_top_wdog.py — 워치독이 **레지스터에서 리셋까지** 이어지는지

`tb/tb_wdog_timer.py` 는 타이머 단독을 본다. 여기서 보는 것은 그 위의 결선이다:
WDOG_CTRL 쓰기 → 필드 디코드 → wdog_timer → reset_seq → BOOT_CAUSE[1].
어느 한 칸이 빠져도 타이머는 맞는데 보드는 안 산다.

  W1. WDOG_CTRL 읽기값: EN·PERIOD 는 남고 **KICK/TEST_FIRE 는 0 으로 읽힌다**
      (쓰기 펄스이지 상태가 아니다 — spec 1절)
  W2. EN=0 이면 창 3개 분량을 돌려도 리셋이 안 걸린다
  W3. EN=1, PERIOD=0 → 리셋이 걸리고 **BOOT_CAUSE[1]=WDOG** 가 선다
  W4. 킥을 계속하면 리셋이 안 걸린다 (같은 시간, 킥만 다르다)
  W5. TEST_FIRE(bit31) 는 **EN 과 무관하게** 즉시 건다 (기존 동작 회귀)
  W6. 워치독 리셋 뒤 BOOT_CAUSE 를 클리어하고 다시 쓸 수 있다
  W7. **DELTA_STEP 한가운데** 리셋 → 풀린 뒤 골든과 비트 일치 (진짜 상황)
  W8. 그 리셋이 **유령 DESC_DONE 을 남기지 않는다** (BUG-001 과 같은 성격)
"""

import os
import sys

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sim.golden import deltarule as G  # noqa: E402
from tools.orbit_cocotb_backend import CocotbBackend  # noqa: E402
from tools.orbit_desc import desc_to_words, pack_delta_step  # noqa: E402
from tools.orbit_device import OrbitDevice  # noqa: E402
from tools.orbit_mmio_map import (  # noqa: E402
    BASE,
    BOOT_CAUSE,
    BOOT_CAUSE_WDOG,
    DESC_STAGE_BASE,
    DR1_STATUS,
    DR1_STATUS_BUSY,
    IRQ_MASK,
    IRQ_PENDING,
    IrqBit,
    QUEUE_DOORBELLS,
    TRACE_CTRL,
    TRACE_ENABLE,
    WDOG_CTRL,
    WDOG_EN,
    WDOG_KICK,
    WDOG_PRESCALE,
    WDOG_TEST_FIRE,
    dr1_scratch_layout,
    wdog_ctrl_word,
)

D = 16
A_WDOG = WDOG_CTRL.addr - BASE
A_BOOT_CAUSE = BOOT_CAUSE.addr - BASE
A_DESC_STAGE = DESC_STAGE_BASE - BASE
A_Q0_DOORBELL = QUEUE_DOORBELLS[0].offset
A_IRQ_PENDING = IRQ_PENDING.offset
A_IRQ_MASK = IRQ_MASK.offset
A_DR1_STATUS = DR1_STATUS.addr - BASE
IRQ_UNMASK = 0xFFFF_FFFF & ~((1 << IrqBit.DESC_DONE) | (1 << IrqBit.TC0_FAULT))


async def reset_dut(dut):
    dut.por_n.value = 0
    dut.reg_addr.value = 0
    dut.reg_wr_en.value = 0
    dut.reg_wr_data.value = 0
    dut.rd_req_ready.value = 1
    dut.rd_done.value = 0
    dut.rd_data_valid.value = 0
    dut.rd_data.value = 0
    dut.rd_data_last.value = 0
    dut.wr_req_ready.value = 1
    dut.wr_done.value = 0
    dut.wr_data_ready.value = 1
    await Timer(100, unit="ns")
    dut.por_n.value = 1
    for _ in range(30):
        await RisingEdge(dut.clk)
        if dut.reset_active.value == 0:
            break
    await RisingEdge(dut.clk)


async def reg_write(dut, addr, data):
    dut.reg_addr.value = addr
    dut.reg_wr_en.value = 1
    dut.reg_wr_data.value = data
    await RisingEdge(dut.clk)
    dut.reg_wr_en.value = 0


async def reg_read(dut, addr):
    dut.reg_addr.value = addr
    dut.reg_wr_en.value = 0
    await RisingEdge(dut.clk)
    return int(dut.reg_rd_data.value)


async def start(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)


async def watch_reset(dut, cycles, *, kick_every=0):
    """`cycles` 사이클 돌면서 reset_active 가 뜨는지 본다.

    `kick_every` > 0 이면 그 간격으로 WDOG_CTRL 에 KICK 을 쓴다.
    **리셋이 걸리면 그 사이클 번호를 돌려준다** (안 걸리면 None).
    """
    for cyc in range(cycles):
        if kick_every and cyc and cyc % kick_every == 0:
            # EN/PERIOD 를 유지한 채 창만 다시 연다 — 호스트의 watchdog_kick() 과 같다
            cur = await reg_read(dut, A_WDOG)
            await reg_write(dut, A_WDOG, (cur & 0x00FF_FF00) | (cur & WDOG_EN) | WDOG_KICK)
            continue
        await RisingEdge(dut.clk)
        if int(dut.reset_active.value):
            return cyc
    return None


@cocotb.test()
async def test_w1_readback_drops_pulse_bits(dut):
    """W1: KICK/TEST_FIRE 는 0 으로 읽힌다. EN/PERIOD 는 남는다."""
    await start(dut)
    await reg_write(dut, A_WDOG, wdog_ctrl_word(0x1234, enable=True, kick=True))
    val = await reg_read(dut, A_WDOG)

    assert (val >> 8) & 0xFFFF == 0x1234, f"W1: PERIOD 가 안 남았다 {val:#010x}"
    assert val & WDOG_EN, f"W1: EN 이 안 남았다 {val:#010x}"
    assert not (val & WDOG_KICK), (
        f"W1: KICK 이 0 으로 안 읽힌다 {val:#010x} — 쓰기 펄스지 상태가 아니다"
    )
    assert not (val & WDOG_TEST_FIRE), f"W1: TEST_FIRE 가 남았다 {val:#010x}"


@cocotb.test()
async def test_w2_disabled_never_resets(dut):
    """W2: EN=0 이면 창 3개 분량을 돌려도 리셋이 안 걸린다."""
    await start(dut)
    await reg_write(dut, A_WDOG, wdog_ctrl_word(0, enable=False, kick=False))
    at = await watch_reset(dut, 3 * WDOG_PRESCALE + 200)
    assert at is None, f"W2: EN=0 인데 {at} 사이클에서 리셋이 걸렸다"


@cocotb.test()
async def test_w3_timeout_resets_and_sets_boot_cause(dut):
    """W3: EN=1, PERIOD=0 → 리셋 + BOOT_CAUSE[1]=WDOG.

    이게 이 모듈의 존재 이유다. **DELTA_STEP 이 멈춰도 칩이 스스로 빠져나온다.**
    """
    await start(dut)
    await reg_write(dut, A_WDOG, wdog_ctrl_word(0))

    at = await watch_reset(dut, 3 * WDOG_PRESCALE)
    assert at is not None, "W3: EN=1 인데 리셋이 안 걸렸다 — 타이머가 안 물렸다"
    # 창은 1024 사이클이다. 레지스터 쓰기·읽기 오버헤드가 있으니 자릿수만 본다.
    assert WDOG_PRESCALE // 2 <= at <= 2 * WDOG_PRESCALE, (
        f"W3: {at} 사이클에서 걸렸다 — {WDOG_PRESCALE} 근처여야 한다"
    )

    # 리셋이 풀릴 때까지 기다린 뒤 원인을 읽는다
    for _ in range(200):
        await RisingEdge(dut.clk)
        if not int(dut.reset_active.value):
            break
    cause = await reg_read(dut, A_BOOT_CAUSE)
    assert cause & BOOT_CAUSE_WDOG, (
        f"W3: BOOT_CAUSE={cause:#x} — WDOG 비트(0x2)가 안 섰다. "
        f"리셋은 걸렸는데 원인이 기록 안 되면 보드에서 원인을 못 찾는다"
    )


@cocotb.test()
async def test_w4_kick_prevents_reset(dut):
    """W4: 같은 시간 동안 킥만 계속하면 리셋이 안 걸린다.

    W3 와 **차이는 킥 하나뿐**이다 — 그래서 이 둘이 짝이다.
    """
    await start(dut)
    await reg_write(dut, A_WDOG, wdog_ctrl_word(0))
    at = await watch_reset(dut, 3 * WDOG_PRESCALE, kick_every=600)
    assert at is None, f"W4: 킥하는데 {at} 사이클에서 리셋이 걸렸다"


@cocotb.test()
async def test_w5_test_fire_still_works(dut):
    """W5: bit[31] 즉시 발사는 EN 과 무관하게 그대로 동작한다 (회귀 방지).

    이 경로는 워치독을 만들기 전부터 있었다. 기존 테스트가 여기에 기댄다.
    """
    await start(dut)
    await reg_write(dut, A_WDOG, WDOG_TEST_FIRE)       # EN=0 인 채로 쏜다
    at = await watch_reset(dut, 50)
    assert at is not None and at < 10, (
        f"W5: TEST_FIRE 가 즉시 안 걸렸다 (at={at})"
    )


@cocotb.test()
async def test_w6_recovers_after_wdog_reset(dut):
    """W6: 워치독 리셋 뒤에도 레지스터가 다시 살아난다.

    리셋은 걸렸는데 그 뒤로 아무것도 안 되면 워치독이 칩을 죽인 것이다.
    """
    await start(dut)
    await reg_write(dut, A_WDOG, wdog_ctrl_word(0))
    at = await watch_reset(dut, 3 * WDOG_PRESCALE)
    assert at is not None, "W6: 리셋이 안 걸렸다"

    for _ in range(200):
        await RisingEdge(dut.clk)
        if not int(dut.reset_active.value):
            break
    else:
        assert False, "W6: 리셋이 안 풀렸다"

    # 다시 쓰고 읽을 수 있어야 한다 — 이번엔 꺼 둔다
    await reg_write(dut, A_WDOG, wdog_ctrl_word(0x00AB, enable=False, kick=False))
    val = await reg_read(dut, A_WDOG)
    assert (val >> 8) & 0xFFFF == 0x00AB and not (val & WDOG_EN), (
        f"W6: 리셋 뒤 WDOG_CTRL 이 안 산다 {val:#010x}"
    )
    at2 = await watch_reset(dut, 2 * WDOG_PRESCALE)
    assert at2 is None, f"W6: 껐는데 다시 걸렸다 ({at2})"


# ─────────────────────────────────────────────────────────────────────
# 워치독이 **동작 중에** 터지는 경우 — 이게 실제로 일어나는 유일한 경우다
# ─────────────────────────────────────────────────────────────────────
# W1~W6 은 칩이 놀고 있을 때 워치독을 본다. 그런데 워치독이 실제로 터지는 때는
# **칩이 멈췄을 때**, 즉 DELTA_STEP 한가운데다. 그때 칩이 되살아나지 않으면
# 워치독은 구조 장치가 아니라 벽돌 만드는 장치다.
#
#   W7. STEP 한가운데 리셋 → 풀린 뒤 INIT+STEP 이 **골든과 비트 일치**
#   W8. 그 리셋이 **유령 DESC_DONE 을 남기지 않는다** (BUG-001 과 같은 성격)

async def _wait_dr1_busy(dut, limit=400):
    """`DR1_STATUS[0]` 이 설 때까지 기다린다. **몇 사이클째인지 세지 않는다.**

    "40사이클쯤이면 한가운데겠지" 로 하면, 디스크립터 디코드/CRC 가 길어지는 날
    리셋이 **STEP 이 시작되기도 전에** 떨어진다. 그러면 테스트는 여전히 통과하는데
    보려던 것(datapath 한가운데 리셋)은 안 본다. busy 를 직접 본다.
    """
    for _ in range(limit):
        await RisingEdge(dut.clk)
        if await reg_read(dut, A_DR1_STATUS) & DR1_STATUS_BUSY:
            return True
    return False


async def _start_a_step(dut):
    """DELTA_STEP 디스크립터를 쏘고 **완료를 기다리지 않고** 돌아온다."""
    lay = dr1_scratch_layout(D)
    desc = pack_delta_step(
        lay["q"] * 2, lay["k"] * 2, lay["v"] * 2, lay["o"] * 2,
        G.float_to_uq15(0.9), G.float_to_uq15(0.2), slot=0,
    )
    for i, w in enumerate(desc_to_words(desc)):
        await reg_write(dut, A_DESC_STAGE + i * 4, w)
    await reg_write(dut, A_Q0_DOORBELL, 0x0001)


@cocotb.test()
async def test_w7_reset_midway_then_recovers(dut):
    """W7: DELTA_STEP 한가운데 워치독 리셋 → 그 뒤 칩이 **정상으로 돌아온다**.

    워치독이 터지는 때는 칩이 멈췄을 때다. 즉 **STEP 한가운데**가 정상 상황이다.
    되살아나지 않으면 워치독은 구조 장치가 아니라 벽돌 만드는 장치다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await _start_a_step(dut)
    assert await _wait_dr1_busy(dut), "W7: DR1 이 busy 가 안 됐다 — STEP 이 시작도 안 했다"
    # busy 를 본 뒤 몇 사이클 더 흘려 확실히 한가운데로 간다 (STEP 은 127사이클)
    for _ in range(30):
        await RisingEdge(dut.clk)
    assert await reg_read(dut, A_DR1_STATUS) & DR1_STATUS_BUSY, (
        "W7: 30사이클 만에 STEP 이 끝났다 — 한가운데가 아니다"
    )
    assert int(dut.reset_active.value) == 0, "W7: 아직 리셋이 걸리면 안 된다"

    await reg_write(dut, A_WDOG, WDOG_TEST_FIRE)     # 한가운데서 발사
    at = await watch_reset(dut, 50)
    assert at is not None, "W7: 동작 중에는 리셋이 안 걸린다?"

    for _ in range(400):
        await RisingEdge(dut.clk)
        if not int(dut.reset_active.value):
            break
    else:
        assert False, "W7: 리셋이 안 풀렸다 — 칩이 벽돌이 됐다"

    # 이제 호스트 스택으로 **처음부터** 돌려서 골든과 맞는지 본다
    backend = CocotbBackend(dut)
    dev = OrbitDevice(backend)
    await backend.async_write(TRACE_CTRL.offset, TRACE_ENABLE)
    await backend.async_write(IRQ_MASK.offset, IRQ_UNMASK)

    await dev.delta_init()
    rng = np.random.default_rng(7007)
    q = G.random_vec(rng, D, 0.2)
    k = G.random_vec(rng, D, 0.2)
    v = G.random_vec(rng, D, 0.2)
    alpha, beta = G.float_to_uq15(0.9), G.float_to_uq15(0.2)

    o_host, sat, _ = await dev.delta_step(q, k, v, alpha, beta)
    S_gold, o_gold, sat_gold = G.step(
        np.zeros((D, D), dtype=np.int64), q, k, v, alpha, beta
    )
    assert np.array_equal(np.array(o_host, dtype=np.int64), o_gold), (
        f"W7: 워치독 리셋 뒤 결과가 골든과 다르다\n"
        f"  호스트={list(o_host)}\n  골든  ={o_gold.tolist()}"
    )
    assert sat == sat_gold, f"W7: 포화 수 {sat} != {sat_gold}"

    S_host = np.array(await dev.delta_dump(D), dtype=np.int64)
    assert np.array_equal(S_host, S_gold), (
        "W7: 리셋 뒤 상태 S 가 골든과 다르다 — state_sram 이 깨끗하게 안 돌아왔다"
    )


@cocotb.test()
async def test_w8_midway_reset_leaves_no_phantom_done(dut):
    """W8: 한가운데 리셋이 **유령 `DESC_DONE` 을 남기지 않는다**.

    BUG-001 과 같은 성격이다: 끝나지 않은 디스크립터가 완료 IRQ 를 올리면
    호스트는 없는 결과를 읽는다. 리셋으로 잘린 디스크립터도 마찬가지다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await reg_write(dut, A_IRQ_MASK, IRQ_UNMASK)
    # 남아 있을지 모르는 것을 먼저 지운다 (W1C)
    pend = await reg_read(dut, A_IRQ_PENDING)
    if pend:
        await reg_write(dut, A_IRQ_PENDING, pend)

    await _start_a_step(dut)
    assert await _wait_dr1_busy(dut), "W8: DR1 이 busy 가 안 됐다"
    for _ in range(30):
        await RisingEdge(dut.clk)
    await reg_write(dut, A_WDOG, WDOG_TEST_FIRE)

    for _ in range(500):
        await RisingEdge(dut.clk)
        if not int(dut.reset_active.value):
            break

    pend = await reg_read(dut, A_IRQ_PENDING)
    assert not (pend & (1 << IrqBit.DESC_DONE)), (
        f"W8: IRQ_PENDING={pend:#x} — 잘린 디스크립터가 DESC_DONE 을 올렸다. "
        f"호스트가 없는 결과를 읽으러 간다 (BUG-001 과 같은 성격)"
    )
