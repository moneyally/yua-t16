"""기대값 출처: 실수(float64) 델타룰 `step_float()` — Q1.15 경로가 그것을 따라가야 한다.
계약 수치는 `docs/DESIGN.md` 6.5절 표(실측)에서 왔다.

test_golden_drift.py — **긴 시퀀스에서 상태가 발산하지 않는가**

## 왜 이 파일이 있는가

다른 모든 검증은 "RTL == 골든" 을 본다. 그건 **골든이 진짜 델타룰인가**에는
아무 답도 주지 않는다. Q1.15 상태가 1,000토큰 뒤에도 실수 델타룰을 따라가고
있는지는 별개의 질문이고, 이 레포에서 **가장 중요한 질문**이다:

    북극성은 "상태가 연산기 옆 SRAM 에 상주하고 토큰이 지나가며 제자리에서
    갱신되는" 유닛이다. 상태가 **오래 머무는 것**이 전제다. 상태가 조금씩
    표류해서 200토큰 뒤에 다른 것이 되어 있다면, 비트 정확도는 의미가 없다.

이 검사는 2026-09-11 까지 `sim/golden/deltarule.py` 의 `__main__` 안에만 있었다
(`python3 sim/golden/deltarule.py` 로 손으로 돌려야 보였다). 즉 **아무도 안 지키고
있었다.** 여기로 옮겨 pytest 가 지킨다.

## 무엇을 단언하는가

1. **발산하지 않는다** — 400토큰 드리프트가 100토큰 드리프트보다 크게 나쁘지 않다.
   이게 핵심이다. 절대값보다 **기울기**가 중요하다.
2. **절대 상한** — α 대역별 관측값에 여유를 둔 상한.

α < 1 이라 옛 오차는 α^n 으로 사그라든다. 그래서 평탄해지는 것이 이론적으로
맞고, 실측도 그렇다 (DESIGN 6.5). α 가 1 에 가까울수록 느리게 잊으므로 더 크다 —
그래서 대역을 나눈다. **α 를 표현 가능한 최대값(0x7FFF, 사실상 망각 없음)으로
고정해도** 400→3200토큰에서 ~11 LSB 로 평탄하다 (DESIGN 6.5 마지막 줄).

## 이 테스트가 무는가 (2026-09-11 확인)

| 넣어 본 실수 | 결과 |
|---|---|
| 정수 경로가 α 를 무시 (안 잊음) | **잡는다** — `scripts/mutation_test.py` 의 `state_never_decays` 로 상시 확인 |
| 반올림을 half-to-even → half-down | **못 잡는다.** 그래도 발산하지 않기 때문이다 |

두 번째 줄은 구멍이 아니라 **측정 결과**다. 반올림 방식은
`tests/test_golden_deltarule.py` 가 지킨다 (거기가 맞는 자리다). 여기서 지키는
것은 "Q1.15 가 긴 문맥에서 충분한가" 하나뿐이고, 그건 다른 어디서도 안 본다.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.golden.deltarule import drift_series  # noqa: E402

D = 16
EARLY, LATE = 100, 400
SEEDS = (11, 22, 33)

# (α 대역, 400토큰 절대 상한 LSB). 실측은 DESIGN 6.5 표 — 여유를 2배 남짓 둔다.
#   0.80-0.99    관측 ~2.8  → 6
#   0.99-0.999   관측 ~8.0  → 18
#   0.999-0.9999 관측 ~9.1  → 24   (가장 느리게 잊는 경우)
#   α 고정 최대  관측 ~10.9 → 24  ← **최악의 경우. 사실상 망각 없음**
BANDS = [
    ((0.80, 0.99), 6.0),
    ((0.99, 0.999), 18.0),
    ((0.999, 0.9999), 24.0),
    ((0.999969, 0.999969), 24.0),   # 0x7FFF — Q1.15/UQ1.15 로 낼 수 있는 최대 α
]


@pytest.mark.parametrize("alpha_range,cap", BANDS, ids=[f"a{a[0]}-{a[1]}" for a, _ in BANDS])
def test_drift_does_not_diverge(alpha_range, cap):
    """드리프트가 **토큰 수에 따라 자라지 않는다**.

    자란다면 상태 고정 설계 자체가 긴 문맥에서 못 쓴다는 뜻이다 — Q1.15 를
    버리거나(Q2.30 상태), 주기적으로 상태를 다시 씻어야 한다. 그건 설계 변경이지
    버그 수정이 아니므로, **지금 알고 있어야** 한다.
    """
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        series, sat = drift_series(D, LATE, rng, [EARLY, LATE],
                                   alpha_range=alpha_range)
        early, late = series[EARLY], series[LATE]

        assert late <= cap, (
            f"α∈{alpha_range} seed={seed}: {LATE}토큰 드리프트 {late:.2f} LSB > 상한 {cap}. "
            f"({EARLY}토큰에서는 {early:.2f}). 상태가 실수 델타룰에서 멀어지고 있다"
        )
        # **기울기**가 본체다. 무작위라 오르내리므로 3배까지 봐준다.
        assert late <= max(3.0 * early, 2.0), (
            f"α∈{alpha_range} seed={seed}: 드리프트가 자란다 — "
            f"{EARLY}토큰 {early:.2f} → {LATE}토큰 {late:.2f} LSB. "
            f"평탄해져야 한다 (α<1 이라 옛 오차는 α^n 으로 사그라든다)"
        )
        assert sat == 0, (
            f"α∈{alpha_range} seed={seed}: 포화가 {sat}회 났다. 이 시험은 포화 없는 "
            f"영역에서 드리프트만 보려던 것이다 — 입력 스케일이 바뀌었는지 확인할 것"
        )


def test_drift_is_actually_measured_not_zero():
    """드리프트가 **0 이 아니어야** 한다 — 0 이면 이 테스트가 아무것도 안 보고 있다.

    `step_float` 이 실수로 정수 경로를 호출하게 되면 드리프트가 정확히 0 이 되고,
    위 테스트는 전부 통과한다. 그 구멍을 막는다.
    """
    rng = np.random.default_rng(5)
    series, _ = drift_series(D, 50, rng, [50])
    assert series[50] > 0.1, (
        f"드리프트가 {series[50]:.4f} LSB 다. 정수 경로와 float 경로가 같은 것을 "
        f"계산하고 있지 않은지 확인할 것 — 그러면 이 파일 전체가 무의미하다"
    )


def test_drift_grows_with_alpha_near_one():
    """α 가 1 에 가까울수록 드리프트가 **커야** 한다 (느리게 잊으니까).

    방향이 반대면 이해가 틀렸거나 구현이 틀린 것이다. 숫자 하나를 외우는 대신
    **관계**를 못박는다 — 관계는 구현이 바뀌어도 살아남는다.
    """
    rng_lo = np.random.default_rng(101)
    rng_hi = np.random.default_rng(101)
    lo, _ = drift_series(D, 400, rng_lo, [400], alpha_range=(0.80, 0.90))
    hi, _ = drift_series(D, 400, rng_hi, [400], alpha_range=(0.999, 0.9999))
    assert hi[400] > lo[400], (
        f"α≈0.9999 드리프트 {hi[400]:.2f} ≤ α≈0.85 드리프트 {lo[400]:.2f} LSB. "
        f"느리게 잊을수록 오차가 더 쌓여야 한다 — 방향이 반대다"
    )


def test_drift_bounded_at_d64():
    """d=64 에서도 유계인가 — **실제로 쓰려는 차원**이 여기다.

    d=16 은 먼저 끝내려고 고른 크기다 (`docs/DESIGN.md` 4절). 델타룰 헤드의
    현실적인 차원은 64 이상이고, 상태 원소가 16배라 오차가 더 쌓일 여지가 있다.
    d=16 만 보고 "유계다" 라고 말하면 그건 확인 안 한 것을 말하는 것이다.

    토큰당 골든 연산이 d² 라 느리다 — 200토큰까지만 본다. 자라는지 아닌지는
    그 구간에서 이미 드러난다 (d=16 표에서 100→200 이 가장 많이 오르는 구간이다).
    """
    rng = np.random.default_rng(64064)
    series, sat = drift_series(64, 200, rng, [50, 200], alpha_range=(0.99, 0.999))
    early, late = series[50], series[200]
    assert late <= 18.0, (
        f"d=64 200토큰 드리프트 {late:.2f} LSB > 18. (50토큰에서는 {early:.2f}) "
        f"— d=16 에서 유계라고 d=64 도 그렇다고 가정하면 안 된다"
    )
    assert late <= max(3.0 * early, 2.0), (
        f"d=64 에서 드리프트가 자란다 — 50토큰 {early:.2f} → 200토큰 {late:.2f} LSB"
    )
    assert sat == 0, f"d=64: 포화가 {sat}회 났다"
