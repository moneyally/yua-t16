"""기대값 출처: spec/watchdog.md 2절(동작)을 파이썬으로 다시 적은 `WdogModel`.
계산이 없는 블록이라 sim/golden/ 대응이 없다 — **스펙 표가 정답지**다.

tb_wdog_timer.py — rtl/wdog_timer.sv

`reg_top.sv` 의 "Proto-A stub: register only, no actual timer" 를 닫는 타이머다.
`reset_seq` 는 이미 `wdog_reset` 입력과 `BOOT_CAUSE[1]` 래치를 갖고 있었다 —
**없던 것은 타이머 하나뿐**이었다.

  T1. EN=0 이면 영원히 안 터진다 (창 3개 분량을 돌려 본다)
  T2. EN=1, PERIOD=0 → **정확히 1024 사이클** 뒤 1펄스
  T3. PERIOD=1 → 2048 사이클. 그리고 펄스는 **1사이클**이고 스스로 리로드해서
      같은 주기로 다시 터진다
  T4. 킥하면 안 터진다 — 창 안에서 계속 킥하다가 멈추면 그때부터 새로 센다
  T5. PERIOD 를 창 도중에 바꿔도 **진행 중인 창은 안 변한다** (다음 리로드부터)
  T6. EN 을 0 으로 내렸다 다시 1 로 올리면 **처음부터** 센다 (잔값으로 즉시 안 터진다)
  T7. KICK 없이 켜면 첫 창은 **옛 PERIOD** 로 돈다 — `wdog_ctrl_word()` 가 KICK 을
      항상 싣는 이유를 못 박는다

왜 PRESCALE 을 테스트에서 줄이지 않는가: 줄이면 보드에서 도는 것과 **다른 회로**를
검증하게 된다 (spec/watchdog.md 1절). 대신 PERIOD 를 0~1 로 작게 쓴다.
"""

import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PRESCALE = 1024          # spec/watchdog.md 1절. rtl 의 PRESCALE_LOG2=10 과 같아야 한다


class WdogModel:
    """spec/watchdog.md 2절을 그대로 옮긴 참조 모델. RTL 을 보지 않는다."""

    def __init__(self, prescale=PRESCALE):
        self.prescale = prescale
        self.pre = 0
        self.win = 0
        self.timeout = 0

    def step(self, en, kick, period):
        """한 클럭. 반환값은 **이 사이클 이후** 관측되는 timeout."""
        self.timeout = 0
        if (not en) or kick:
            self.pre = 0
            self.win = period
        else:
            if self.pre == self.prescale - 1:
                self.pre = 0
                if self.win == 0:
                    self.timeout = 1
                    self.win = period
                else:
                    self.win -= 1
            else:
                self.pre += 1
        return self.timeout


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.en.value = 0
    dut.kick.value = 0
    dut.period.value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def start(dut):
    """클럭·리셋을 세우고, DUT 와 같은 자리에서 출발하는 모델을 돌려준다.

    **여기서 딱 한 번 falling edge 에 정렬한다.** 이후 `run_cycles` 는 매 반복이
    falling edge 에서 끝나므로 다시 정렬하지 않는다 — 호출 사이에 정렬을 또 하면
    모델은 안 세는데 rising edge 가 하나 지나가서 DUT 만 1사이클 앞선다
    (T4·T6 이 정확히 그렇게 틀렸다. docs/BUGS.md 테스트벤치 실수 표 참조).
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await FallingEdge(dut.clk)
    return WdogModel()


async def run_cycles(dut, model, n, *, en=1, period=0, kick_at=(), period_at=None):
    """n 사이클 돌리며 **모델과 매 사이클 비교**한다. 터진 사이클 번호를 돌려준다.

    `model` 은 **호출 사이에 이어진다.** 새로 만들면 DUT 의 카운터는 이어지는데
    모델만 0 에서 다시 출발해 엉뚱한 곳에서 불일치가 난다 (T4 가 정확히 그랬다).

    kick_at    — 이 사이클 번호들에서 kick 을 1 로 올린다
    period_at  — {사이클: 새 PERIOD} — 창 도중 변경을 본다
    """
    fires = []
    cur_period = period
    kick_at = set(kick_at)

    # 정렬하지 않는다 — 이미 falling edge 다 (start() 또는 앞선 호출의 마지막 반복).
    for cyc in range(n):
        if period_at and cyc in period_at:
            cur_period = period_at[cyc]
        do_kick = 1 if cyc in kick_at else 0

        dut.en.value = en
        dut.kick.value = do_kick
        dut.period.value = cur_period

        exp = model.step(en, do_kick, cur_period)

        await RisingEdge(dut.clk)
        await FallingEdge(dut.clk)          # 등록된 출력은 falling edge 에서 본다
        got = int(dut.timeout.value)
        assert got == exp, (
            f"사이클 {cyc}: timeout RTL={got} 모델={exp} "
            f"(en={en} kick={do_kick} period={cur_period})"
        )
        if got:
            fires.append(cyc)

    dut.kick.value = 0
    return fires


@cocotb.test()
async def test_t1_disabled_never_fires(dut):
    """T1: EN=0 이면 안 터진다. 창 3개 분량(3072 사이클)을 돌려 본다."""
    m = await start(dut)
    fires = await run_cycles(dut, m, 3 * PRESCALE + 10, en=0, period=0)
    assert fires == [], f"T1: EN=0 인데 {len(fires)}번 터졌다 {fires[:5]}"


@cocotb.test()
async def test_t2_period0_is_1024_cycles(dut):
    """T2: PERIOD=0 → 정확히 1024 사이클 뒤 1펄스.

    **정확히**가 중요하다. 1 사이클 어긋나면 보드에서 창이 미묘하게 틀어지고,
    그건 아무도 안 본다.
    """
    m = await start(dut)
    fires = await run_cycles(dut, m, PRESCALE + 50, en=1, period=0)
    assert len(fires) == 1, f"T2: {len(fires)}번 터졌다 {fires}"
    assert fires[0] == PRESCALE - 1, (
        f"T2: {fires[0]}번째 사이클에서 터졌다 — {PRESCALE - 1} 이어야 한다 "
        f"(리로드 후 {PRESCALE} 사이클째)"
    )


@cocotb.test()
async def test_t3_period1_and_repeats(dut):
    """T3: PERIOD=1 → 2048 사이클. 펄스는 1사이클, 그리고 같은 주기로 반복된다.

    **KICK 과 함께 켠다.** 호스트도 그렇게 한다 (`wdog_ctrl_word` 가 KICK 을 항상
    싣는다 — spec/watchdog.md 3절). 이유는 T7 이 보여 준다.
    """
    m = await start(dut)
    fires = await run_cycles(
        dut, m, 2 * (2 * PRESCALE) + 50, en=1, period=1, kick_at=(0,)
    )
    assert len(fires) == 2, f"T3: {len(fires)}번 터졌다 {fires}"
    assert fires[0] == 2 * PRESCALE, f"T3: 첫 발화 {fires[0]}"
    assert fires[1] - fires[0] == 2 * PRESCALE, (
        f"T3: 두 번째 발화 간격이 {fires[1] - fires[0]} — {2 * PRESCALE} 이어야 한다"
    )


@cocotb.test()
async def test_t4_kick_prevents_fire(dut):
    """T4: 창 안에서 킥하면 안 터진다. 킥을 멈추면 그때부터 새로 센다."""
    m = await start(dut)
    # 700 사이클마다 킥 → 1024 창은 절대 안 닫힌다
    kicks = tuple(range(700, 3000, 700))
    fires = await run_cycles(dut, m, 3000, en=1, period=0, kick_at=kicks)
    assert fires == [], f"T4: 킥하는데 터졌다 {fires}"

    # 마지막 킥은 2800 사이클째다. 거기서 창이 다시 열렸으므로 2800+1024 에서
    # 터진다 — 위 호출이 3000 에서 끝났으니 이어지는 호출 기준 824 사이클째다.
    # 모델이 호출 사이에 이어지므로 **매 사이클 비교가 이미 이것을 잡는다**;
    # 아래 assert 는 "결국 터진다"가 아니라 **정확히 언제**를 못박는 것이다.
    at = PRESCALE - (3000 - 2800)
    fires2 = await run_cycles(dut, m, at + 50, en=1, period=0)
    assert fires2 == [at], f"T4: 킥을 멈춘 뒤 {fires2} — [{at}] 이어야 한다"


@cocotb.test()
async def test_t5_period_change_midwindow(dut):
    """T5: 창 도중 PERIOD 를 바꿔도 진행 중인 창은 안 변한다.

    왜 중요한가 (spec/watchdog.md 2절): 즉시 반영이면 호스트가 PERIOD 를 계속
    다시 쓰는 것만으로 워치독이 영원히 안 터진다. 그건 워치독이 아니다.
    """
    m = await start(dut)
    # PERIOD=0 으로 시작해서 500 사이클째에 5 로 올린다.
    # 진행 중인 창은 그대로이므로 **1024 에서 터져야 한다**.
    fires = await run_cycles(
        dut, m, PRESCALE + 50, en=1, period=0, period_at={500: 5}
    )
    assert len(fires) == 1 and fires[0] == PRESCALE - 1, (
        f"T5: PERIOD 변경이 진행 중인 창에 영향을 줬다 {fires}"
    )


@cocotb.test()
async def test_t6_enable_restarts_window(dut):
    """T6: EN 을 내렸다 올리면 처음부터 센다 (잔값으로 즉시 안 터진다)."""
    m = await start(dut)
    # 900 사이클 돌린 뒤 (창 1024 중 900 소모)
    fires = await run_cycles(dut, m, 900, en=1, period=0)
    assert fires == [], f"T6: 창이 안 닫혔는데 터졌다 {fires}"

    # EN=0 으로 200 사이클 (원래대로면 여기서 창이 닫혔을 구간이다)
    fires = await run_cycles(dut, m, 200, en=0, period=0)
    assert fires == [], f"T6: EN=0 구간에서 터졌다 {fires}"

    # 다시 EN=1 → **1024 사이클을 처음부터** 센다
    fires = await run_cycles(dut, m, PRESCALE + 50, en=1, period=0)
    assert len(fires) == 1 and fires[0] == PRESCALE - 1, (
        f"T6: EN 재개 후 {fires} — 잔값으로 일찍 터졌다면 {fires[0]} < {PRESCALE - 1}"
    )


@cocotb.test()
async def test_t7_enable_without_kick_uses_old_window(dut):
    """T7: KICK 없이 EN 과 PERIOD 를 같이 올리면 **첫 창은 옛 PERIOD 로 돈다**.

    왜 이걸 테스트로 못 박는가: 이게 `tools/orbit_mmio_map.wdog_ctrl_word()` 가
    KICK 을 항상 싣는 이유다. 근거가 테스트로 남아 있지 않으면 누가 "KICK 은
    불필요해 보인다"며 빼고, 보드에서 첫 창만 짧아지는 걸 아무도 못 본다.

    안전한 방향이긴 하다 — 창이 **짧아지지** 길어지지 않는다. 그래도 놀랄 일은
    없는 편이 낫다.
    """
    m = await start(dut)
    # 리셋 직후 win 은 0 이다 (EN=0 이 매 사이클 PERIOD=0 을 리로드해 왔다).
    # 여기서 KICK 없이 PERIOD=3, EN=1 을 동시에 올린다 → 첫 창은 1024 다.
    fires = await run_cycles(dut, m, PRESCALE + 50, en=1, period=3)
    assert fires == [PRESCALE - 1], (
        f"T7: {fires} — KICK 없이 켜면 첫 창은 옛 PERIOD(0)로 {PRESCALE} 사이클이다"
    )
    # 그 다음 창부터는 새 PERIOD 가 먹는다 (발화 때 win<=period 로 리로드했다)
    fires2 = await run_cycles(dut, m, 4 * PRESCALE + 50, en=1, period=3)
    assert len(fires2) == 1, f"T7: 두 번째 창에서 {fires2}"
