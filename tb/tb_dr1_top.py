"""기대값 출처: sim/golden/deltarule.py (골든 모델). RTL 은 아직 없다 — PLAN 1단계는 RTL 금지.

tb_dr1_top.py — ORBIT-DR1 검증 하네스 골격 (PLAN W4-2)

## 이 파일의 목적

`docs/DESIGN.md` 7절의 검증 계약을 코드로 만든다:

    1. 골든 모델로 무작위 시드 N 토큰 시퀀스의 S_t, o_t 생성
    2. 같은 입력을 RTL 에 넣고 DELTA_DUMP 로 상태를 꺼내 **비트 단위 비교**
    3. 불일치 시 첫 불일치 토큰 인덱스, 행, 열, 기대/실제 값을 출력

RTL 이 없는 지금은 **골든을 DUT 자리에 놓고 하네스 자체를 검증한다.**
`docs/PLAN.md` W4-2 의 완료 기준: **"RTL 이 오면 CocotbDUT 한 클래스만 채운다"** 가 참인 것.

## 왜 고의로 틀린 DUT 를 넣는 테스트가 있는가

골든-대-골든은 **항상 통과한다.** 그래서 그것만으로는 하네스가 실제로 비교를 하는지,
아니면 아무것도 안 하고 통과하는지 구분할 수 없다. `OffByOneDut` 은 1 LSB 를 고의로
더해서 **하네스가 불일치를 잡고 위치를 정확히 찍는지** 확인한다.
이것이 없으면 하네스는 "항상 통과하는 장식"일 수 있다.

## 실행

    python3 -m pytest tests/test_dr1_harness.py -q      # 골든-대-골든 + 오류 주입
    python3 tb/tb_dr1_top.py                            # 직접 실행 (요약 출력)
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.golden import deltarule as G  # noqa: E402
from tools.orbit_mmio_map import (  # noqa: E402
    DR1_ADDR_ALIGN,
    DR1_NUM_SLOTS,
    FaultCode,
    UQ15_ONE,
)

# ═══════════════════════════════════════════════════════════════════════════
# 실패 보고
# ═══════════════════════════════════════════════════════════════════════════


class Dr1Fault(Exception):
    """DUT 가 `done_err` 를 냈다 (docs/DESIGN.md 5.1, spec/deltarule.md 4절).

    성공 완료(`done_ok`)와 실패 종료(`done_err`)는 한 트랜잭션에 정확히 하나만
    나온다. 하네스에서는 "값을 돌려주면 done_ok, 이 예외를 던지면 done_err" 로
    표현한다.
    """

    def __init__(self, fault_code: int, detail: str = ""):
        self.fault_code = int(fault_code)
        name = FaultCode(self.fault_code).name if self.fault_code in [int(f) for f in FaultCode] else "UNKNOWN"
        super().__init__(f"done_err fault_code=0x{self.fault_code:02X} ({name}) {detail}".strip())


@dataclass(frozen=True)
class Mismatch:
    """비트 불일치 하나. 출력 형식을 여기서 못박는다 (DESIGN.md 7절 3항)."""

    token: int          # 토큰 인덱스. 최종 상태 비교는 -1
    kind: str           # "o" (출력 벡터) 또는 "S" (상태 행렬)
    row: int
    col: int | None     # "o" 는 None
    expected: int
    actual: int

    @property
    def diff(self) -> int:
        return self.actual - self.expected

    def __str__(self) -> str:
        where = f"token={self.token:>5d}" if self.token >= 0 else "token=  end"
        idx = f"{self.kind}[{self.row}]" if self.col is None else f"{self.kind}[{self.row}][{self.col}]"
        return (
            f"[MISMATCH] {where}  {idx:<14s} "
            f"expected=0x{self.expected & 0xFFFF:04X} ({self.expected:+7d})  "
            f"actual=0x{self.actual & 0xFFFF:04X} ({self.actual:+7d})  "
            f"diff={self.diff:+d}"
        )


@dataclass
class HarnessResult:
    d: int
    n_tokens: int
    seed: int
    mismatches: list[Mismatch] = field(default_factory=list)
    dut_sat_total: int = 0
    golden_sat_total: int = 0
    cycles: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches

    @property
    def first(self) -> Mismatch | None:
        return self.mismatches[0] if self.mismatches else None

    def report(self, max_lines: int = 10) -> str:
        head = (
            f"DR1 harness  d={self.d}  N={self.n_tokens}  seed={self.seed}  "
            f"→ {'PASS' if self.ok else f'FAIL ({len(self.mismatches)} mismatches)'}"
        )
        lines = [head]
        if self.cycles:
            lines.append(
                f"  cycles/token: min={min(self.cycles)} max={max(self.cycles)} "
                f"mean={sum(self.cycles) / len(self.cycles):.1f}"
            )
        lines.append(f"  sat: dut={self.dut_sat_total} golden={self.golden_sat_total}")
        for m in self.mismatches[:max_lines]:
            lines.append("  " + str(m))
        if len(self.mismatches) > max_lines:
            lines.append(f"  ... 그 외 {len(self.mismatches) - max_lines}건")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# DUT 인터페이스
# ═══════════════════════════════════════════════════════════════════════════


class Dr1Dut(ABC):
    """DR1 하드웨어의 3개 opcode 를 파이썬 인터페이스로.

    `spec/deltarule.md` 2절의 opcode 와 1:1 이다:
        init  ↔ DELTA_INIT (0x50)
        step  ↔ DELTA_STEP (0x51)
        dump  ↔ DELTA_DUMP (0x52)

    실패(`done_err`)는 `Dr1Fault` 로 던진다. 성공은 값을 돌려준다.
    """

    d: int

    @abstractmethod
    def init(self, slot: int = 0) -> None:
        """DELTA_INIT — 상태 슬롯을 0 으로."""

    @abstractmethod
    def step(self, slot: int, q, k, v, alpha: int, beta: int):
        """DELTA_STEP — 토큰 1개.

        alpha, beta 는 **UQ1.15** (0x8000 = 1.0). Q1.15 가 아니다.
        반환: (o, sat, cycles)
            o      : (d,) Q1.15 int
            sat    : 이 토큰에서 발생한 포화 횟수 (DR1_SAT_COUNT 증가분)
            cycles : 이 토큰이 걸린 사이클 수. **모델은 None** (사이클 개념이 없다)
        """

    @abstractmethod
    def dump(self, slot: int = 0):
        """DELTA_DUMP — 상태 S 전체 (d, d) Q1.15."""

    def check_step_args(self, slot: int, q, k, v, alpha: int, beta: int) -> None:
        """spec/deltarule.md 4절 실패 조건을 DUT 공통으로 검사한다.

        주소 정렬(0x06 DR1_UNALIGNED)은 하네스에 주소 개념이 없어서 검사하지 않는다.
        그건 `tools/orbit_desc.py` 의 패커와 `tests/test_dr1_spec_consistency.py` 가 본다.
        """
        if not (0 <= slot < DR1_NUM_SLOTS):
            raise Dr1Fault(FaultCode.DR1_BAD_SLOT, f"slot={slot}, 유효 범위 0..{DR1_NUM_SLOTS - 1}")
        for name, vec in (("q", q), ("k", k), ("v", v)):
            if np.asarray(vec).shape != (self.d,):
                raise ValueError(f"{name} 의 모양이 ({self.d},) 가 아니다: {np.asarray(vec).shape}")
        # alpha/beta 범위 위반은 fault 가 아니다 — RTL 은 클램프한다 (spec 1절).
        # 골든은 ValueError 를 내므로, 클램프는 DUT 구현 쪽 책임이다.


class GoldenDut(Dr1Dut):
    """골든 모델을 DUT 자리에 놓는다. 하네스 자체를 검증하는 용도.

    RTL 이 오면 이 클래스는 **참조 쪽**으로만 쓰이고, DUT 자리는 CocotbDut 가 받는다.
    """

    def __init__(self, d: int):
        self.d = d
        self._S = {s: np.zeros((d, d), dtype=np.int64) for s in range(DR1_NUM_SLOTS)}
        self.sat_total = 0
        self.clamp_total = 0

    def init(self, slot: int = 0) -> None:
        if not (0 <= slot < DR1_NUM_SLOTS):
            raise Dr1Fault(FaultCode.DR1_BAD_SLOT, f"slot={slot}")
        self._S[slot] = np.zeros((self.d, self.d), dtype=np.int64)

    def step(self, slot: int, q, k, v, alpha: int, beta: int):
        self.check_step_args(slot, q, k, v, alpha, beta)
        # RTL 은 1.0 초과를 0x8000 으로 클램프하고 CLAMP_EVENT 를 남긴다 (spec 1절).
        # DUT 역할일 때는 하드웨어와 같은 쪽에 선다.
        a, b = int(alpha), int(beta)
        if a > UQ15_ONE:
            a = UQ15_ONE
            self.clamp_total += 1
        if b > UQ15_ONE:
            b = UQ15_ONE
            self.clamp_total += 1
        S_next, o, sat = G.step(self._S[slot], q, k, v, a, b)
        self._S[slot] = S_next
        self.sat_total += sat
        return o, sat, None       # 모델에는 사이클 개념이 없다

    def dump(self, slot: int = 0):
        if not (0 <= slot < DR1_NUM_SLOTS):
            raise Dr1Fault(FaultCode.DR1_BAD_SLOT, f"slot={slot}")
        return self._S[slot].copy()


class CocotbDut(Dr1Dut):
    """RTL DUT — **`rtl/dr1/dr1_top.sv` 를 직접 몬다** (PLAN W6).

    ───────────────────────────────────────────────────────────────────────
    W6 에서 채운 것 / 아직 빈 것
    ───────────────────────────────────────────────────────────────────────
      init  ✔  DELTA_INIT (0x50)  — 구현됨
      dump  ✔  DELTA_DUMP (0x52)  — 구현됨, 덤프 스트림을 받아 (d,d) 로 되돌린다
      step  ✘  DELTA_STEP (0x51)  — **W7**. 지금 부르면 NotImplementedError.
                                     RTL 은 fault_code 0x07(DR1_UNIMPL)로 실패한다.

    ───────────────────────────────────────────────────────────────────────
    왜 dr1_top 이고 g2_ctrl_top 이 아닌가
    ───────────────────────────────────────────────────────────────────────
    W6 의 덤프는 **스트림 포트**로만 나온다. 스크래치 메모리 쓰기 경로가 아직
    없어서 `g2_ctrl_top` 바깥에서는 상태를 볼 수 없다 (spec/deltarule.md 3.4절
    "W6 현재 상태"). 그래서 하네스는 `dr1_top` 을 직접 몬다.
    **디스크립터·IRQ 경로는 따로 본다** — `tb/tb_g2_ctrl_top_dr1_fault.py`.
    스크래치 쓰기 경로가 생기면(W7~W8) 이 클래스를 g2_ctrl_top 으로 올린다.

    ───────────────────────────────────────────────────────────────────────
    완료 판정 (docs/DESIGN.md 5.1, docs/BUGS.md BUG-001)
    ───────────────────────────────────────────────────────────────────────
    **`done_pulse` 를 보지 않는다.** 그것은 리타이어(성공+실패)라 fault 를
    성공으로 읽는다. `done_ok` 면 값을 돌려주고, `done_err` 면 `fault_code` 를
    담아 `Dr1Fault` 를 던진다.

    ───────────────────────────────────────────────────────────────────────
    샘플링 규칙 (docs/BUGS.md BUG-008a/b — cdc_fifo 에서 배운 것)
    ───────────────────────────────────────────────────────────────────────
    **레지스터 출력은 falling edge 에서 읽는다.** `RisingEdge` 직후에 읽으면
    논블로킹 대입 전이라 **이전 사이클 값**을 본다. cdc_fifo 테스트가 정확히
    이것으로 틀렸다 — RTL 은 정상인데 테스트가 리셋값을 보고 실패했다.
    여기서 `dump_valid`/`dump_data`/`done_*` 를 전부 falling edge 에서 읽는 이유다.
    참고 구현: `tb/tb_cdc_fifo_async.py` 의 `read_n()`.
    """

    # spec/deltarule.md 4절: 주소는 16바이트 정렬이어야 한다 (DR1_ADDR_ALIGN)
    DUMP_ADDR = 0x1000

    def __init__(self, dut, d: int, *, timeout_cycles: int = 512):
        from tools.orbit_mmio_map import DR1_ADDR_ALIGN as _align
        assert self.DUMP_ADDR % _align == 0, "덤프 주소가 정렬되지 않았다"
        self.dut = dut
        self.d = d
        self.timeout_cycles = timeout_cycles

    # ── 저수준: 디스크립터 1개를 몰고 완료를 본다 ────────────────────────
    async def _issue(self, opcode: int, slot: int, dst_addr: int = 0):
        from cocotb.triggers import FallingEdge, RisingEdge

        dut = self.dut
        dut.cmd_opcode.value = int(opcode)
        dut.cmd_slot.value = int(slot)
        dut.cmd_dst_addr.value = int(dst_addr)

        # valid/ready 계약: **ready 를 볼 때까지 valid 를 유지한다.**
        # 처음에 이걸 안 지키고 valid 를 1사이클만 냈다가 두 번째 명령을 통째로
        # 잃었다 (DUMP 타임아웃). 앞 명령이 ST_DONE_OK 에 있는 동안 cmd_ready 가
        # 0 이기 때문이다 — RTL 은 정상이고 테스트가 틀렸다. cdc_fifo 와 같은 종류.
        dut.cmd_valid.value = 1
        for _ in range(self.timeout_cycles):
            await FallingEdge(dut.clk)
            if int(dut.cmd_ready.value):
                break
        else:
            raise TimeoutError(f"opcode=0x{int(opcode):02X}: cmd_ready 가 오지 않았다")
        await RisingEdge(dut.clk)      # 이 edge 에서 수락된다
        dut.cmd_valid.value = 0

        rows = {}
        fault = None
        cycles = None
        for _ in range(self.timeout_cycles):
            await FallingEdge(dut.clk)
            if int(dut.dump_valid.value):
                rows[int(dut.dump_row.value)] = int(dut.dump_data.value)
            if int(dut.done_err.value):
                fault = int(dut.fault_code.value)
                cycles = int(dut.dr1_cycles.value)
            elif int(dut.done_ok.value):
                cycles = int(dut.dr1_cycles.value)
            if cycles is not None and not int(dut.busy.value):
                # FSM 이 IDLE 로 돌아온 뒤에 돌려준다. 완료 상태에서 바로 나가면
                # 다음 명령의 handshake 가 한 사이클 어긋난다.
                if fault is not None:
                    raise Dr1Fault(fault, f"opcode=0x{int(opcode):02X} slot={slot}")
                return rows, cycles
        raise TimeoutError(
            f"opcode=0x{int(opcode):02X} 가 {self.timeout_cycles} 사이클 안에 끝나지 않았다"
        )

    async def init(self, slot: int = 0) -> None:
        """DELTA_INIT — 상태 슬롯을 0 으로."""
        await self._issue(0x50, slot)

    async def step(self, slot: int, q, k, v, alpha: int, beta: int):
        raise NotImplementedError(
            "DELTA_STEP 은 W7 이다 (PLAN '한 번에 하나'). "
            "RTL 에 보내면 fault_code 0x07 DR1_UNIMPL 로 실패한다 — "
            "tb/tb_dr1_top_fsm.py test_f3 가 그것을 본다."
        )

    async def dump(self, slot: int = 0):
        """DELTA_DUMP — 상태 S 전체 (d, d) Q1.15.

        행 우선 스트림을 받아 `tools/orbit_pack.unpack_state` 로 되돌린다.
        평탄화 규칙은 그 모듈 한 곳에만 있다 (spec/deltarule.md 3.5절).
        """
        from tools.orbit_pack import unpack_state

        rows, _ = await self._issue(0x52, slot, self.DUMP_ADDR)
        missing = [r for r in range(self.d) if r not in rows]
        if missing:
            raise AssertionError(f"덤프에 빠진 행이 있다: {missing}")
        return unpack_state([rows[r] for r in range(self.d)], self.d)




class OffByOneDut(GoldenDut):
    """**고의로 틀리는 DUT.** 하네스가 실제로 비교를 하는지 확인하는 용도.

    지정한 토큰의 `o[idx]` 에, 또는 최종 `S[r][c]` 에 1 LSB 를 더한다.
    골든-대-골든은 항상 통과하므로 이것 없이는 하네스가 살아 있는지 알 수 없다.
    """

    def __init__(self, d: int, o_token: int | None = None, o_index: int = 0,
                 s_cell: tuple[int, int] | None = None, delta: int = 1):
        super().__init__(d)
        self.o_token = o_token
        self.o_index = o_index
        self.s_cell = s_cell
        self.delta = delta
        self._t = 0

    def step(self, slot: int, q, k, v, alpha: int, beta: int):
        o, sat, cyc = super().step(slot, q, k, v, alpha, beta)
        if self.o_token is not None and self._t == self.o_token:
            o = o.copy()
            o[self.o_index] += self.delta
        self._t += 1
        return o, sat, cyc

    def dump(self, slot: int = 0):
        S = super().dump(slot)
        if self.s_cell is not None:
            r, c = self.s_cell
            S[r, c] += self.delta
        return S


# ═══════════════════════════════════════════════════════════════════════════
# 토큰 시퀀스 생성 + 하네스
# ═══════════════════════════════════════════════════════════════════════════


def make_token_sequence(d: int, n: int, seed: int, scale: float = 0.2,
                        alpha_range=(0.80, 0.999), beta_range=(0.05, 0.30)):
    """무작위 시드 N 토큰. 같은 seed → 같은 시퀀스 (DESIGN.md 7절 I5).

    scale 을 작게 두어 기본 시퀀스에서는 포화가 나지 않게 한다. 포화 경로는
    별도 테스트에서 강제로 만든다.
    alpha/beta 는 **UQ1.15** 로 만든다.
    """
    r = np.random.default_rng(seed)
    tokens = []
    for _ in range(n):
        tokens.append((
            G.random_vec(r, d, scale),
            G.random_vec(r, d, scale),
            G.random_vec(r, d, scale),
            G.float_to_uq15(r.uniform(*alpha_range)),
            G.float_to_uq15(r.uniform(*beta_range)),
        ))
    return tokens


def _harness_core(dut: Dr1Dut, tokens: Sequence, res: "HarnessResult", slot: int,
                  compare_every_token: bool):
    """하네스의 **유일한** 비교 로직. DUT 호출을 `yield` 하고 결과를 `send` 로 받는다.

    이렇게 나눈 이유: cocotb DUT 의 init/step/dump 는 **코루틴**이라 `await` 이 필요하고,
    골든 DUT 는 평범한 함수다. 비교 로직을 두 벌 쓰면 한쪽만 고치는 날이 온다
    (BUG-006 과 같은 종류). 그래서 로직은 여기 한 곳에 두고, `run_harness`(동기)와
    `run_harness_async`(코루틴)는 호출 방식만 다른 얇은 껍데기다.

    yield 형식: ("init"|"step"|"dump", args_tuple)
    """
    d = dut.d
    yield ("init", (slot,))
    S_ref = np.zeros((d, d), dtype=np.int64)

    for t, (q, k, v, alpha, beta) in enumerate(tokens):
        o_dut, sat_dut, cycles = yield ("step", (slot, q, k, v, alpha, beta))
        S_ref, o_ref, sat_ref = G.step(S_ref, q, k, v, alpha, beta)

        res.dut_sat_total += int(sat_dut)
        res.golden_sat_total += int(sat_ref)
        if cycles is not None:
            res.cycles.append(int(cycles))

        if compare_every_token:
            o_dut = np.asarray(o_dut, dtype=np.int64)
            for i in range(d):
                if int(o_dut[i]) != int(o_ref[i]):
                    res.mismatches.append(
                        Mismatch(token=t, kind="o", row=i, col=None,
                                 expected=int(o_ref[i]), actual=int(o_dut[i]))
                    )

    S_dut = np.asarray((yield ("dump", (slot,))), dtype=np.int64)
    if S_dut.shape != S_ref.shape:
        raise AssertionError(f"dump 모양이 다르다: {S_dut.shape} vs {S_ref.shape}")
    bad = np.argwhere(S_dut != S_ref)
    for (r_, c_) in bad:
        res.mismatches.append(
            Mismatch(token=-1, kind="S", row=int(r_), col=int(c_),
                     expected=int(S_ref[r_, c_]), actual=int(S_dut[r_, c_]))
        )


def run_harness(dut: Dr1Dut, tokens: Sequence, *, seed: int = 0, slot: int = 0,
                compare_every_token: bool = True) -> HarnessResult:
    """DUT 를 골든과 비트 단위로 비교한다 (DESIGN.md 7절). **동기 DUT 용.**

    순서:
      1. DUT.init(slot) + 골든 상태를 0 으로 — 같은 출발점
      2. 토큰마다 DUT.step, 골든 step → o 를 비트 비교
      3. 마지막에 DUT.dump(slot) 를 골든 상태와 비트 비교

    불일치는 **전부 모으고** 첫 번째를 `result.first` 로 둔다.
    """
    res = HarnessResult(d=dut.d, n_tokens=len(tokens), seed=seed)
    gen = _harness_core(dut, tokens, res, slot, compare_every_token)
    try:
        op, args = next(gen)
        while True:
            op, args = gen.send(getattr(dut, op)(*args))
    except StopIteration:
        return res


async def run_harness_async(dut: Dr1Dut, tokens: Sequence, *, seed: int = 0, slot: int = 0,
                            compare_every_token: bool = True) -> HarnessResult:
    """`run_harness` 와 **같은 비교 로직**을, init/step/dump 가 코루틴인 DUT 에.

    cocotb 테스트에서 쓴다 (`CocotbDut`). 비교 규칙이 여기 따로 있지 않다 —
    `_harness_core` 하나뿐이다.
    """
    res = HarnessResult(d=dut.d, n_tokens=len(tokens), seed=seed)
    gen = _harness_core(dut, tokens, res, slot, compare_every_token)
    try:
        op, args = next(gen)
        while True:
            op, args = gen.send(await getattr(dut, op)(*args))
    except StopIteration:
        return res


# ═══════════════════════════════════════════════════════════════════════════
# 직접 실행 — 요약
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=== DR1 하네스 골든-대-골든 (RTL 없음) ===")
    worst = 0
    for d in (16, 64):
        for seed in (1, 2, 3):
            n = 1000
            tokens = make_token_sequence(d, n, seed)
            res = run_harness(GoldenDut(d), tokens, seed=seed)
            print(res.report())
            worst = max(worst, len(res.mismatches))
    print()
    print("=== 오류 주입 (하네스가 살아 있는지) ===")
    tokens = make_token_sequence(16, 50, 7)
    res = run_harness(OffByOneDut(16, o_token=13, o_index=5, s_cell=(2, 9)), tokens, seed=7)
    print(res.report(max_lines=3))
    assert not res.ok, "오류를 주입했는데 하네스가 통과시켰다 — 하네스가 죽어 있다"
    print()
    print("=== 결과 ===" )
    print("골든-대-골든 불일치 0" if worst == 0 else f"골든-대-골든 불일치 {worst}건 (버그)")
    return 0 if worst == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
