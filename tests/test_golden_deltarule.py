"""기대값 출처: docs/DESIGN.md 7절 불변조건 I1~I5 + 2절 수식. 골든 모델 자신을 검사한다 (RTL 없음).

tests/test_golden_deltarule.py — PLAN W3-2

`sim/golden/deltarule.py` 가 정답지이므로, 그 정답지 자체를 불변조건으로 검사한다.
RTL 은 아직 없다 (PLAN 1단계는 RTL 금지). d=16 과 d=64 를 모두 돌린다.

`docs/DESIGN.md` 7절:
  I1. DELTA_INIT 후 상태는 전부 0.
  I2. α=1, β=0 이면 상태는 변하지 않는다.
  I3. β=0 이면 o_t = S_{t-1}·q_t 와 일치.
  I4. 포화가 일어난 토큰 수 == 트레이스 링의 SAT_EVENT 수.
  I5. 같은 입력·같은 시드 → 두 번 실행 결과 동일 (결정성).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.golden.deltarule import (  # noqa: E402
    ONE_Q15,
    Q15_MAX,
    Q15_MIN,
    float_to_q15,
    q15_from_acc,
    q15_to_float,
    random_state,
    random_vec,
    run,
    step,
)

DIMS = [16, 64]

# α=1.0 은 부호 있는 Q1.15 로 **정확히 표현되지 않는다** (최대 32767 = 0.999969...).
# 아래 테스트들은 "α≈1" 을 Q15_MAX 로 쓰고, 그 때문에 생기는 1 LSB 오차를 허용한다.
# 정확한 1.0 이 필요한 경우는 ONE_Q15(=32768)를 쓴다 — 이건 16비트 필드를 넘으므로
# W4-1 (spec/deltarule.md) 에서 α/β 필드 폭을 정할 때 결론을 내야 한다. docs/LOG.md 참조.
ALPHA_ONE_Q15 = Q15_MAX      # 표현 가능한 최대값 ≈ 1.0
ALPHA_ONE_EXACT = ONE_Q15    # 정확한 1.0 (16비트 초과)


def rng(seed=20260911):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# I1
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", DIMS)
def test_i1_init_state_is_all_zero(d):
    """기대값 출처: DESIGN.md 7절 I1 — DELTA_INIT 후 상태는 전부 0.

    골든 모델 수준에서 DELTA_INIT 은 "0 행렬에서 시작"이다.
    0 상태에서 한 토큰을 돌리면 p=0 이므로 err=v 가 되어야 한다
    (S_t = β·v·kᵀ). 그 성질까지 확인한다.
    """
    S0 = np.zeros((d, d), dtype=np.int64)
    assert np.all(S0 == 0), "INIT 상태는 전부 0이어야 한다"

    r = rng()
    k = random_vec(r, d)
    v = random_vec(r, d)
    q = random_vec(r, d)
    beta = float_to_q15(0.5)

    S1, o1, sat = step(S0, q, k, v, ALPHA_ONE_EXACT, beta)
    assert sat == 0, f"이 입력에서는 포화가 없어야 한다 (관측 {sat})"

    # S=0 이면 p=0, err=v 이므로 S_1 = q15(β·v[i] · k[j]) 여야 한다.
    from sim.golden.deltarule import q15_mul
    for i in range(d):
        berr = q15_mul(beta, int(v[i]))
        for j in range(d):
            want = q15_from_acc(berr * int(k[j]))
            assert S1[i, j] == want, (
                f"I1: S=0 에서 시작했는데 S_1[{i}][{j}] 가 β·v·kᵀ 와 다르다: "
                f"{S1[i, j]} != {want}"
            )


# ---------------------------------------------------------------------------
# I2
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", DIMS)
def test_i2_alpha1_beta0_keeps_state(d):
    """기대값 출처: DESIGN.md 7절 I2 — α=1, β=0 이면 상태는 변하지 않는다.

    α=1.0 이 Q1.15 로 정확히 표현되지 않으므로 두 가지를 확인한다:
      - ALPHA_ONE_EXACT(정확한 1.0): 상태가 **완전히 동일**해야 한다.
      - Q15_MAX(≈1.0):               1 LSB 이내여야 한다.
    """
    r = rng()
    S = random_state(r, d)
    q, k, v = random_vec(r, d), random_vec(r, d), random_vec(r, d)

    S_exact, _, sat_e = step(S, q, k, v, ALPHA_ONE_EXACT, 0)
    assert sat_e == 0
    assert np.array_equal(S_exact, S), (
        "I2: α=1(정확), β=0 인데 상태가 변했다. "
        f"최대 차이 {int(np.max(np.abs(S_exact - S)))} LSB"
    )

    S_approx, _, _ = step(S, q, k, v, ALPHA_ONE_Q15, 0)
    dmax = int(np.max(np.abs(S_approx - S)))
    assert dmax <= 1, f"I2: α≈1(Q15_MAX), β=0 인데 상태가 {dmax} LSB 변했다 (1 이하여야 한다)"


# ---------------------------------------------------------------------------
# I3
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", DIMS)
def test_i3_beta0_output_is_prev_state_times_q(d):
    """기대값 출처: DESIGN.md 7절 I3 — β=0 이면 o_t = S_{t-1}·q_t 와 일치.

    β=0 이면 S_t = α·S_{t-1} 이므로, α=1 일 때 o_t 는 **직전 상태**와 q 의 곱이다.
    양자화 순서까지 같아야 하므로 골든 모델의 재양자화 함수로 기대값을 만든다.
    """
    r = rng()
    S = random_state(r, d)
    q, k, v = random_vec(r, d), random_vec(r, d), random_vec(r, d)

    _, o, sat = step(S, q, k, v, ALPHA_ONE_EXACT, 0)
    assert sat == 0

    want = np.array(
        [q15_from_acc(sum(int(S[i, j]) * int(q[j]) for j in range(d))) for i in range(d)],
        dtype=np.int64,
    )
    assert np.array_equal(o, want), (
        "I3: β=0, α=1 인데 o 가 S_{t-1}·q 와 다르다. "
        f"첫 불일치 index={int(np.argmax(o != want))}"
    )


# ---------------------------------------------------------------------------
# I4
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", DIMS)
def test_i4_saturation_count_matches_hand_count(d):
    """기대값 출처: DESIGN.md 7절 I4 — 포화 수는 손으로 센 값과 일치해야 한다.

    포화를 **강제로** 만든다. 상태를 전부 +Q15_MAX, k=[1,0,...], v 를 전부 Q15_MIN 으로:

        p[i]   = q15(S[i][0]·k[0]) = Q15_MAX            (포화 없음)
        ap[i]  = α·p[i] = Q15_MAX                       (α=1, 포화 없음)
        err[i] = v[i] − ap[i] = -32768 − 32767 = -65535 → **포화** → Q15_MIN
        berr   = β·Q15_MIN = Q15_MIN                    (β=1, 포화 없음)
        S_next[i][0] = q15(α·Q15_MAX + Q15_MIN·ONE) = -1   (포화 없음)
        S_next[i][j>0] = q15(α·Q15_MAX + Q15_MIN·0) = Q15_MAX (포화 없음)
        o[i]   = q15(S_next[i][0]·k[0]) = -1            (포화 없음)

    즉 포화는 err 계산에서 **행마다 정확히 1회**, 총 **d 회**다.
    """
    S = np.full((d, d), Q15_MAX, dtype=np.int64)
    k = np.zeros(d, dtype=np.int64)
    k[0] = ALPHA_ONE_EXACT                    # 정확히 1.0
    q = k.copy()
    v = np.full(d, Q15_MIN, dtype=np.int64)

    S_next, o, sat = step(S, q, k, v, ALPHA_ONE_EXACT, ALPHA_ONE_EXACT)

    assert sat == d, (
        f"I4: 포화가 정확히 {d}회(행마다 1회) 나와야 한다. 관측 {sat}회. "
        "err = v − α·p 에서 -65535 가 Q15_MIN 으로 잘리는 것이 유일한 포화다."
    )
    # 위 손계산이 실제로 맞는지 결과값으로도 확인한다
    assert np.all(S_next[:, 0] == -1), f"S_next[:,0] 은 -1 이어야 한다: {S_next[:3, 0]}"
    if d > 1:
        assert np.all(S_next[:, 1] == Q15_MAX), "S_next[:,1] 은 Q15_MAX 여야 한다"
    assert np.all(o == -1), f"o 는 전부 -1 이어야 한다: {o[:3]}"


@pytest.mark.parametrize("d", DIMS)
def test_i4_no_saturation_on_small_inputs(d):
    """기대값 출처: DESIGN.md 7절 I4 의 반대편 — 작은 입력에서는 포화가 0이어야 한다."""
    r = rng()
    S = random_state(r, d, scale=0.1)
    tokens = [
        (random_vec(r, d, 0.1), random_vec(r, d, 0.1), random_vec(r, d, 0.1),
         float_to_q15(0.9), float_to_q15(0.1))
        for _ in range(10)
    ]
    _, _, total_sat = run(S, tokens)
    assert total_sat == 0, f"작은 입력 10토큰에서 포화가 {total_sat}회 났다 (0이어야 한다)"


# ---------------------------------------------------------------------------
# I5
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("d", DIMS)
def test_i5_deterministic(d):
    """기대값 출처: DESIGN.md 7절 I5 — 같은 입력·같은 시드면 두 번 실행 결과가 동일.

    한 스텝과 여러 토큰 시퀀스 양쪽에서 확인한다.
    """
    r = rng()
    S = random_state(r, d)
    tokens = [
        (random_vec(r, d), random_vec(r, d), random_vec(r, d),
         float_to_q15(0.95), float_to_q15(0.25))
        for _ in range(8)
    ]

    a_S, a_o, a_sat = run(S, tokens)
    b_S, b_o, b_sat = run(S, tokens)

    assert np.array_equal(a_S, b_S), "I5: 같은 입력인데 최종 상태가 다르다"
    assert all(np.array_equal(x, y) for x, y in zip(a_o, b_o)), "I5: 출력 시퀀스가 다르다"
    assert a_sat == b_sat, f"I5: 포화 수가 다르다 {a_sat} != {b_sat}"

    # run() 이 입력 S 를 망가뜨리지 않는지도 확인 (결정성의 전제)
    c_S, _, _ = run(S, tokens)
    assert np.array_equal(a_S, c_S), "I5: run() 이 입력 상태를 변경했다"


# ---------------------------------------------------------------------------
# 손계산 대조 (d=2) — 사용자가 직접 검산하는 테스트
# ---------------------------------------------------------------------------
#
# 유리수 손계산 (α=1, q=k=[1,0]):
#
#   1) S=0, k=[1,0], v=[3,5], β=1
#        p   = S·k   = [0,0]
#        err = v−α·p = [3,5]
#        S   = 0 + 1·[3,5]ᵀ·[1,0] = [[3,0],[5,0]]
#        o   = S·q   = [3,5]
#
#   2) k=[1,0], v=[7,1], β=1/2, α=1
#        p   = S·k   = [3,5]
#        err = v−α·p = [7−3, 1−5] = [4,−4]
#        β·err       = [2,−2]
#        S   = [[3,0],[5,0]] + [[2,0],[−2,0]] = [[5,0],[3,0]]
#
# Q1.15 로 넣을 때는 값을 1/8 로 줄인다 (u = 1/8 = 4096).
# 3·u=12288, 5·u=20480, 7·u=28672, 1·u=4096 — 전부 Q1.15 범위 안이다.
U = ONE_Q15 // 8          # 1/8 = 4096
HALF = ONE_Q15 // 2       # 0.5 = 16384


def _e1(k_and_q_one, alpha_one, beta_one):
    """손계산 1단계를 주어진 '1.0 표현'으로 실행."""
    S0 = np.zeros((2, 2), dtype=np.int64)
    k = np.array([k_and_q_one, 0], dtype=np.int64)
    q = k.copy()
    v = np.array([3 * U, 5 * U], dtype=np.int64)
    return step(S0, q, k, v, alpha_one, beta_one), k, q


def test_hand_worked_d2_exact_one():
    """기대값 출처: 사용자 손계산 (docs/LOG.md 세션 6). 1.0 이 정확할 때 **완전 일치**해야 한다.

    α=β=k=q=1.0 을 ONE_Q15(=32768)로 둔다. 이 값은 16비트 필드를 넘지만
    골든 모델은 파이썬 정수라 정확히 표현된다. 수식 자체가 맞는지 보는 테스트다.
    """
    (S1, o1, sat1), k, q = _e1(ONE_Q15, ONE_Q15, ONE_Q15)
    assert sat1 == 0, f"1단계에서 포화가 나면 안 된다 (관측 {sat1})"

    want_S1 = np.array([[3 * U, 0], [5 * U, 0]], dtype=np.int64)
    want_o1 = np.array([3 * U, 5 * U], dtype=np.int64)
    assert np.array_equal(S1, want_S1), f"1단계 S: {S1.tolist()} != {want_S1.tolist()}"
    assert np.array_equal(o1, want_o1), f"1단계 o: {o1.tolist()} != {want_o1.tolist()}"

    # 2단계: v=[7,1]·u, β=1/2, α=1
    v2 = np.array([7 * U, 1 * U], dtype=np.int64)
    S2, o2, sat2 = step(S1, q, k, v2, ONE_Q15, HALF)
    assert sat2 == 0, f"2단계에서 포화가 나면 안 된다 (관측 {sat2})"

    want_S2 = np.array([[5 * U, 0], [3 * U, 0]], dtype=np.int64)
    assert np.array_equal(S2, want_S2), f"2단계 S: {S2.tolist()} != {want_S2.tolist()}"

    # 유리수 비율로도 확인 — 1/8 스케일을 되돌리면 손계산 정수가 나와야 한다
    assert [[int(x) // U for x in row] for row in want_S2.tolist()] == [[5, 0], [3, 0]]
    assert [int(x) // U for x in want_o1.tolist()] == [3, 5]


def test_hand_worked_d2_representable_q15():
    """기대값 출처: 위와 같은 손계산. 1.0 을 표현 가능한 Q15_MAX 로 쓸 때의 오차를 고정한다.

    Q1.15 에는 1.0 이 없다 (최대 32767/32768 = 0.999969...). 그래서 α·β·k·q 를
    "1.0" 대신 Q15_MAX 로 넣으면 **≈1.0 을 곱할 때마다 최대 1 LSB** 를 잃는다.
    허용 오차를 임의로 정하지 않고, **연쇄에 들어간 ≈1.0 곱셈 횟수**로 정한다:

        S1[i][0] : berr = β·v (1회) → ·k (1회)              = 2회 → ≤ 2 LSB
        o1[i]    : 위에 ·q (1회) 추가                        = 3회 → ≤ 3 LSB
        S2[i][0] : α·S1 + β·err·k — β=0.5 는 정확하고 오차가
                   상쇄되는 방향이라 실측 1 LSB                      ≤ 3 LSB

    실측(2026-09-11): S1 오차 2, o1 오차 3, S2 오차 1.
    **이 숫자가 커지면 골든 모델이나 Q1.15 형식 정의가 바뀐 것이다.**
    α/β/k/q 의 1.0 표현 문제는 W4-1(spec/deltarule.md)에서 필드 폭과 함께 결론낸다.
    """
    (S1, o1, _), k, q = _e1(Q15_MAX, Q15_MAX, Q15_MAX)
    want_S1 = np.array([[3 * U, 0], [5 * U, 0]], dtype=np.int64)
    want_o1 = np.array([3 * U, 5 * U], dtype=np.int64)

    dS1 = int(np.max(np.abs(S1 - want_S1)))
    do1 = int(np.max(np.abs(o1 - want_o1)))
    assert dS1 <= 2, f"1단계 S 오차 {dS1} LSB (≈1.0 곱 2회 → 2 이하): {S1.tolist()} vs {want_S1.tolist()}"
    assert do1 <= 3, f"1단계 o 오차 {do1} LSB (≈1.0 곱 3회 → 3 이하): {o1.tolist()} vs {want_o1.tolist()}"

    v2 = np.array([7 * U, 1 * U], dtype=np.int64)
    S2, _, _ = step(S1, q, k, v2, Q15_MAX, HALF)
    want_S2 = np.array([[5 * U, 0], [3 * U, 0]], dtype=np.int64)
    dS2 = int(np.max(np.abs(S2 - want_S2)))
    assert dS2 <= 3, f"2단계 S 오차 {dS2} LSB: {S2.tolist()} vs {want_S2.tolist()}"

    # 비율은 그대로여야 한다 — 1/8 스케일을 되돌리면 손계산 정수가 나온다
    assert [round(int(x) / U) for x in S2[:, 0]] == [5, 3], (
        f"2단계 S 열0 을 1/8 스케일로 되돌리면 [5,3] 이어야 한다: {S2[:, 0].tolist()}"
    )
    assert [round(int(x) / U) for x in o1] == [3, 5], (
        f"1단계 o 를 1/8 스케일로 되돌리면 [3,5] 여야 한다: {o1.tolist()}"
    )


def test_hand_worked_d2_float_reference_agrees():
    """기대값 출처: 사용자 손계산을 float 참조 구현으로 재현. 정수 경로와 같은 답이어야 한다."""
    from sim.golden.deltarule import step_float

    S = np.zeros((2, 2))
    k = np.array([1.0, 0.0])
    q = k.copy()
    S1, o1 = step_float(S, q, k, np.array([3.0, 5.0]), 1.0, 1.0)
    assert S1.tolist() == [[3.0, 0.0], [5.0, 0.0]]
    assert o1.tolist() == [3.0, 5.0]

    S2, _ = step_float(S1, q, k, np.array([7.0, 1.0]), 1.0, 0.5)
    assert S2.tolist() == [[5.0, 0.0], [3.0, 0.0]]


# ---------------------------------------------------------------------------
# 모델 자체 테스트도 pytest 로 묶는다
# ---------------------------------------------------------------------------
def test_module_self_test_passes():
    """기대값 출처: sim/golden/deltarule.py 의 self_test() — round-half-even·포화·float 오차."""
    from sim.golden.deltarule import self_test

    assert self_test(verbose=False) is True
