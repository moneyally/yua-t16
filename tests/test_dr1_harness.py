"""기대값 출처: sim/golden/deltarule.py + docs/DESIGN.md 7절 I1~I6. RTL 없음 (PLAN 1단계 RTL 금지).

tests/test_dr1_harness.py — PLAN W4-2 하네스 자체 검증

`tb/tb_dr1_top.py` 의 하네스가 **실제로 동작하는지** 검사한다. 두 축이다:

  (1) 골든-대-골든  : GoldenDut 을 DUT 자리에 놓고 N=1000, d=16·64, 시드 3개 통과.
                      하네스가 정상 경로에서 오탐하지 않는지.
  (2) 오류 주입      : 1 LSB 를 고의로 더한 DUT 로 하네스가 불일치를 잡고
                      **위치를 정확히 찍는지**. 이게 없으면 (1) 이 통과해도
                      하네스가 "항상 통과하는 장식"인지 알 수 없다.

그리고 불변조건 I1~I6 을 **하네스 수준에서** 다시 본다 (골든 모델 수준 검사는
tests/test_golden_deltarule.py).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tb"))

from sim.golden import deltarule as G  # noqa: E402
from tb.tb_dr1_top import (  # noqa: E402
    CocotbDut,
    Dr1Fault,
    GoldenDut,
    Mismatch,
    OffByOneDut,
    make_token_sequence,
    run_harness,
)
from tools.orbit_mmio_map import DR1_NUM_SLOTS, FaultCode, UQ15_ONE  # noqa: E402

SEEDS = [1, 2, 3]


# ═══════════════════════════════════════════════════════════════════════════
# (1) 골든-대-골든
# ═══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("seed", SEEDS)
def test_golden_vs_golden_d16_1000_tokens(seed):
    """기대값 출처: 골든 모델. d=16, N=1000 골든-대-골든은 불일치 0이어야 한다."""
    tokens = make_token_sequence(16, 1000, seed)
    res = run_harness(GoldenDut(16), tokens, seed=seed)
    assert res.ok, res.report()
    assert res.dut_sat_total == res.golden_sat_total == 0, res.report()


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_golden_vs_golden_d64_1000_tokens(seed):
    """기대값 출처: 골든 모델. d=64, N=1000 골든-대-golden 불일치 0. (시드당 ~13초)"""
    tokens = make_token_sequence(64, 1000, seed)
    res = run_harness(GoldenDut(64), tokens, seed=seed)
    assert res.ok, res.report()


@pytest.mark.parametrize("d", [16, 64])
def test_same_seed_same_sequence(d):
    """기대값 출처: DESIGN.md 7절 I5 — 같은 시드는 같은 토큰 시퀀스를 만들어야 한다."""
    a = make_token_sequence(d, 20, 42)
    b = make_token_sequence(d, 20, 42)
    for i, (ta, tb) in enumerate(zip(a, b)):
        for j in range(3):
            assert np.array_equal(ta[j], tb[j]), f"토큰 {i} 벡터 {j} 가 다르다"
        assert ta[3] == tb[3] and ta[4] == tb[4], f"토큰 {i} 의 alpha/beta 가 다르다"
    c = make_token_sequence(d, 20, 43)
    assert not np.array_equal(a[0][0], c[0][0]), "다른 시드인데 같은 시퀀스가 나왔다"


# ═══════════════════════════════════════════════════════════════════════════
# (2) 오류 주입 — 하네스가 살아 있는가
# ═══════════════════════════════════════════════════════════════════════════
def test_injected_o_error_is_caught_at_exact_location():
    """기대값 출처: 하네스 계약 — o 에 1 LSB 를 더하면 그 토큰·인덱스를 정확히 찍어야 한다."""
    d, n, seed = 16, 60, 7
    tok, idx = 13, 5
    tokens = make_token_sequence(d, n, seed)
    res = run_harness(OffByOneDut(d, o_token=tok, o_index=idx), tokens, seed=seed)

    assert not res.ok, "1 LSB 를 주입했는데 하네스가 통과시켰다 — 하네스가 비교를 안 하고 있다"
    o_bad = [m for m in res.mismatches if m.kind == "o"]
    assert len(o_bad) == 1, f"o 불일치가 정확히 1건이어야 한다: {[str(m) for m in o_bad]}"
    m = o_bad[0]
    assert m.token == tok, f"토큰 위치가 틀렸다: {m.token} != {tok}"
    assert m.row == idx, f"인덱스가 틀렸다: {m.row} != {idx}"
    assert m.col is None, "o 는 열 인덱스가 없어야 한다"
    assert m.diff == 1, f"차이가 +1 이어야 한다: {m.diff}"
    assert res.first is m, "첫 불일치가 o 쪽이어야 한다 (상태 비교보다 먼저 일어난다)"


def test_injected_state_error_is_caught_at_exact_cell():
    """기대값 출처: 하네스 계약 — 최종 S 의 한 칸에 1 LSB 를 더하면 행·열을 정확히 찍어야 한다."""
    d, n, seed = 16, 40, 11
    cell = (2, 9)
    tokens = make_token_sequence(d, n, seed)
    res = run_harness(OffByOneDut(d, s_cell=cell), tokens, seed=seed)

    assert not res.ok, "상태에 1 LSB 를 주입했는데 하네스가 통과시켰다"
    s_bad = [m for m in res.mismatches if m.kind == "S"]
    assert len(s_bad) == 1, f"S 불일치가 정확히 1건이어야 한다: {[str(m) for m in s_bad]}"
    m = s_bad[0]
    assert (m.row, m.col) == cell, f"셀 위치가 틀렸다: {(m.row, m.col)} != {cell}"
    assert m.token == -1, "최종 상태 비교는 token=-1 로 표시해야 한다"
    assert m.diff == 1


def test_injected_error_negative_delta_also_caught():
    """기대값 출처: 하네스 계약 — 부호가 반대인 오차(−1 LSB)도 잡아야 한다."""
    tokens = make_token_sequence(16, 30, 5)
    res = run_harness(OffByOneDut(16, o_token=0, o_index=15, delta=-1), tokens, seed=5)
    assert not res.ok
    assert res.first.diff == -1, f"diff 가 −1 이어야 한다: {res.first}"


def test_mismatch_report_format_is_stable():
    """기대값 출처: DESIGN.md 7절 3항 — 첫 불일치의 토큰/행/열/기대/실제가 출력에 있어야 한다."""
    m = Mismatch(token=42, kind="o", row=7, col=None, expected=0x1234, actual=0x1235)
    s = str(m)
    for piece in ("token=   42", "o[7]", "expected=0x1234", "actual=0x1235", "diff=+1"):
        assert piece in s, f"출력 형식에 {piece!r} 이 없다: {s}"

    m2 = Mismatch(token=-1, kind="S", row=3, col=11, expected=-1, actual=0)
    s2 = str(m2)
    assert "token=  end" in s2 and "S[3][11]" in s2, s2


# ═══════════════════════════════════════════════════════════════════════════
# 불변조건 I1~I6 — 하네스 수준
# ═══════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("d", [16, 64])
def test_i1_init_zeroes_state_via_dut(d):
    """기대값 출처: DESIGN.md 7절 I1 — DELTA_INIT 후 dump 는 전부 0."""
    dut = GoldenDut(d)
    tokens = make_token_sequence(d, 5, 1)
    for q, k, v, a, b in tokens:
        dut.step(0, q, k, v, a, b)
    assert np.any(dut.dump(0) != 0), "토큰을 돌렸는데 상태가 0 그대로다 (테스트 전제 실패)"
    dut.init(0)
    assert np.all(dut.dump(0) == 0), "I1: DELTA_INIT 후 dump 가 0 이 아니다"


@pytest.mark.parametrize("d", [16, 64])
def test_i2_alpha_one_beta_zero_keeps_state_via_dut(d):
    """기대값 출처: DESIGN.md 7절 I2 — α=1(UQ1.15 정확), β=0 이면 dump 가 그대로."""
    dut = GoldenDut(d)
    for q, k, v, a, b in make_token_sequence(d, 3, 2):
        dut.step(0, q, k, v, a, b)
    before = dut.dump(0)
    r = np.random.default_rng(9)
    dut.step(0, G.random_vec(r, d), G.random_vec(r, d), G.random_vec(r, d), UQ15_ONE, 0)
    assert np.array_equal(dut.dump(0), before), "I2: α=1, β=0 인데 상태가 변했다"


@pytest.mark.parametrize("d", [16, 64])
def test_i3_beta_zero_output_is_prev_state_times_q_via_dut(d):
    """기대값 출처: DESIGN.md 7절 I3 — β=0, α=1 이면 o == S_{t-1}·q."""
    dut = GoldenDut(d)
    for q, k, v, a, b in make_token_sequence(d, 3, 3):
        dut.step(0, q, k, v, a, b)
    S_prev = dut.dump(0)
    r = np.random.default_rng(4)
    qv = G.random_vec(r, d)
    o, _, _ = dut.step(0, qv, G.random_vec(r, d), G.random_vec(r, d), UQ15_ONE, 0)
    want = np.array(
        [G.q15_from_acc(sum(int(S_prev[i, j]) * int(qv[j]) for j in range(d))) for i in range(d)],
        dtype=np.int64,
    )
    assert np.array_equal(np.asarray(o), want), "I3: β=0 인데 o 가 S_{t-1}·q 와 다르다"


@pytest.mark.slow
@pytest.mark.parametrize("d", [16, 64])
def test_i4_dut_sat_count_matches_golden_via_harness(d):
    """기대값 출처: DESIGN.md 7절 I4 — DUT 의 포화 수가 골든과 일치해야 한다.

    포화를 강제로 만든다: 상태를 키우는 시퀀스(scale 큼, β 큼)를 돌려
    포화가 실제로 발생하게 한 뒤, 하네스가 센 dut/golden 값을 비교한다.
    """
    tokens = make_token_sequence(d, 40, 21, scale=0.95,
                                 alpha_range=(0.99, 0.999), beta_range=(0.9, 1.0))
    res = run_harness(GoldenDut(d), tokens, seed=21)
    assert res.ok, res.report()
    assert res.golden_sat_total > 0, (
        f"포화를 유발하려 했는데 0회다 (scale/β 를 키워야 한다): {res.report()}"
    )
    assert res.dut_sat_total == res.golden_sat_total, (
        f"I4: DUT 포화 {res.dut_sat_total} != 골든 {res.golden_sat_total}"
    )


@pytest.mark.parametrize("d", [16, 64])
def test_i5_harness_is_deterministic(d):
    """기대값 출처: DESIGN.md 7절 I5 — 같은 시드로 두 번 돌리면 결과가 같다."""
    tokens = make_token_sequence(d, 50, 77)
    a = run_harness(GoldenDut(d), tokens, seed=77)
    b = run_harness(GoldenDut(d), tokens, seed=77)
    assert a.ok and b.ok
    assert a.dut_sat_total == b.dut_sat_total
    assert [str(m) for m in a.mismatches] == [str(m) for m in b.mismatches]


def test_i6_completion_contract_bad_slot_raises_done_err():
    """기대값 출처: DESIGN.md 5.1 + spec/deltarule.md 4절 — 실패는 done_err(Dr1Fault).

    하네스에서 done_ok 는 "값을 돌려준다", done_err 는 "Dr1Fault 를 던진다" 로 표현한다.
    한 트랜잭션에 정확히 하나만 일어나야 한다 — 예외를 던지면서 상태를 바꾸면 안 된다.
    """
    d = 16
    dut = GoldenDut(d)
    for q, k, v, a, b in make_token_sequence(d, 3, 6):
        dut.step(0, q, k, v, a, b)
    before = dut.dump(0)

    r = np.random.default_rng(1)
    args = (G.random_vec(r, d), G.random_vec(r, d), G.random_vec(r, d), UQ15_ONE, 0x4000)
    with pytest.raises(Dr1Fault) as ei:
        dut.step(DR1_NUM_SLOTS, *args)          # 범위 밖 슬롯
    assert ei.value.fault_code == int(FaultCode.DR1_BAD_SLOT), (
        f"fault_code 가 0x05 DR1_BAD_SLOT 이어야 한다: 0x{ei.value.fault_code:02X}"
    )
    assert np.array_equal(dut.dump(0), before), (
        "I6: done_err 가 났는데 상태가 변했다 — ok/err 중 하나만 일어나야 한다"
    )

    with pytest.raises(Dr1Fault):
        dut.init(DR1_NUM_SLOTS)
    with pytest.raises(Dr1Fault):
        dut.dump(DR1_NUM_SLOTS)

    # 정상 경로는 값을 돌려준다 (done_ok)
    o, sat, cycles = dut.step(0, *args)
    assert np.asarray(o).shape == (d,)
    assert isinstance(sat, int) and sat >= 0
    assert cycles is None, "골든 모델에는 사이클 개념이 없으므로 None 이어야 한다"


def test_alpha_beta_above_one_is_clamped_not_faulted():
    """기대값 출처: spec/deltarule.md 1절 — α/β 가 0x8000 초과면 fault 가 아니라 클램프.

    DUT 역할일 때는 하드웨어와 같은 쪽에 선다 (클램프 + CLAMP_EVENT 카운트).
    골든 모델을 직접 호출하면 ValueError 가 난다 — 검증 기준이라 미정의 입력을 받지 않는다.
    """
    d = 8
    dut = GoldenDut(d)
    r = np.random.default_rng(2)
    q, k, v = G.random_vec(r, d), G.random_vec(r, d), G.random_vec(r, d)

    o1, _, _ = dut.step(0, q, k, v, UQ15_ONE + 1234, UQ15_ONE)     # 클램프됨
    assert dut.clamp_total == 1, f"CLAMP 카운트가 1 이어야 한다: {dut.clamp_total}"

    dut2 = GoldenDut(d)
    o2, _, _ = dut2.step(0, q, k, v, UQ15_ONE, UQ15_ONE)           # 이미 1.0
    assert np.array_equal(np.asarray(o1), np.asarray(o2)), "클램프 결과가 1.0 과 같아야 한다"

    with pytest.raises(ValueError):
        G.step(np.zeros((d, d), dtype=np.int64), q, k, v, UQ15_ONE + 1, 0)


# ═══════════════════════════════════════════════════════════════════════════
# CocotbDut 자리
# ═══════════════════════════════════════════════════════════════════════════
def test_cocotb_dut_is_a_stub_with_guidance():
    """기대값 출처: PLAN W4-2 — CocotbDut 은 아직 자리만 있고, 채울 방법이 적혀 있어야 한다."""
    with pytest.raises(NotImplementedError, match="RTL"):
        CocotbDut()

    doc = CocotbDut.__doc__
    for piece in ("falling edge", "done_ok", "16바이트", "pack_delta_step", "BUG-008", "BUG-001"):
        assert piece in doc, f"CocotbDut docstring 에 {piece!r} 안내가 없다"


def test_harness_only_needs_the_dut_interface():
    """기대값 출처: PLAN W4-2 완료 기준 — "RTL 이 오면 CocotbDut 한 클래스만 채운다".

    DUT 를 **엄격한 프록시**로 감싼다. `init` / `step` / `dump` / `d` 외의 속성에
    접근하면 AttributeError 를 낸다. 하네스가 이 프록시로 끝까지 돌아가면,
    하네스는 DUT 인터페이스 3개(+ d)만 쓴다는 것이 증명된다.

    (`__getattribute__` 로 접근을 세는 방식은 약하다 — DUT 자신의 메서드가 내부적으로
    `self.foo` 를 건드리는 것까지 같이 세어서, 하네스가 쓴 것과 구분되지 않는다.
    프록시는 그 구분을 강제한다.)
    """
    ALLOWED = {"init", "step", "dump", "d"}

    class StrictProxy:
        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)
            object.__setattr__(self, "touched", set())

        def __getattr__(self, name):
            if name not in ALLOWED:
                raise AttributeError(
                    f"하네스가 DUT 인터페이스 밖의 속성 {name!r} 을 건드렸다. "
                    f"허용: {sorted(ALLOWED)}. 이러면 CocotbDut 만 채워서는 안 돌아간다."
                )
            object.__getattribute__(self, "touched").add(name)
            return getattr(object.__getattribute__(self, "_inner"), name)

    proxy = StrictProxy(GoldenDut(16))
    tokens = make_token_sequence(16, 8, 1)
    res = run_harness(proxy, tokens, seed=1)      # AttributeError 가 나면 여기서 실패한다
    assert res.ok, res.report()

    touched = proxy.touched
    assert {"init", "step", "dump"} <= touched, (
        f"하네스가 3개를 다 쓰지 않는다: {sorted(touched)}"
    )
    assert touched <= ALLOWED, f"허용 밖 접근: {sorted(touched - ALLOWED)}"


def test_harness_rejects_wrong_shaped_dump():
    """기대값 출처: 하네스 계약 — dump 모양이 다르면 조용히 통과하지 말고 실패해야 한다."""
    class WrongShape(GoldenDut):
        def dump(self, slot: int = 0):
            return super().dump(slot)[:-1]        # 한 행 모자라게

    tokens = make_token_sequence(8, 3, 1)
    with pytest.raises(AssertionError, match="dump 모양"):
        run_harness(WrongShape(8), tokens, seed=1)
