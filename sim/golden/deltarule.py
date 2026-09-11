"""deltarule.py — ORBIT-DR1 델타룰 헤드 골든 모델 (PLAN W3-1)

**이 파일이 정답지다.** RTL(W5~W7)은 이것과 비트 단위로 일치해야 한다.
`docs/DESIGN.md` 2절(수학)·3절(숫자 형식)이 SSOT이고, 이 파일은 그 구현이다.
문서와 이 파일이 다르면 문서를 먼저 고친다.

수식 (DESIGN.md 2절)
--------------------
    S_t = α·S_{t-1}·(I − β·k·kᵀ) + β·v·kᵀ
    o_t = S_t · q

DESIGN.md 2절 "주의"가 못박은 **정확한 전개**:

    S_t = α·S − α·β·p·kᵀ + β·v·kᵀ        (p = S_{t-1}·k)

이 식은 하드웨어 친화형 한 줄로 정확히 접힌다:

    S_t = α·S + β·(v − α·p)·kᵀ

    왜냐하면  α·S + β·(v − α·p)·kᵀ = α·S + β·v·kᵀ − α·β·p·kᵀ  ✔

따라서 **err = v − α·p** 를 쓴다. DESIGN.md 가 경고한 `err = v − p` 는
α=1 일 때만 같다. 이 모델은 모든 α 에 대해 정확한 쪽을 쓴다.
(DESIGN.md 2절 본문의 전개는 `err = v − p` 로 적혀 있다 — 문구 보완 제안은 docs/LOG.md)

숫자 형식 (DESIGN.md 3절)
-------------------------
    Q1.15  : int16, 값 = raw / 2^15, 범위 [-1.0, +0.999969...]
    곱     : Q1.15 × Q1.15 = Q2.30 (int32)
    누산   : 40비트
    재양자화: Q2.30 → Q1.15, **round-half-to-even**
    오버플로: **포화**(saturate), 발생 횟수를 센다 (RTL 트레이스 링의 SAT_EVENT 와 대응)

float 로 계산한 뒤 변환하지 않는다. 전부 정수 연산이다.

연산 순서 (RTL 이 맞춰야 하는 계약)
-----------------------------------
토큰 하나당, 이 순서 그대로:

    1. p_acc[i]    = Σ_j S[i][j] · k[j]          40비트 누산 (Q2.30 곱)
    2. p[i]        = q15(p_acc[i])               Q2.30 → Q1.15
    3. ap[i]       = q15(α · p[i])               Q1.15
    4. err[i]      = sat(v[i] − ap[i])           Q1.15
    5. berr[i]     = q15(β · err[i])             Q1.15
    6. acc[i][j]   = α · S[i][j]                 Q2.30   (scale_unit → 누산기 로드)
    7. acc[i][j]  += berr[i] · k[j]              Q2.30   (mac_array 외적 누산)
    8. S_next[i][j]= q15(acc[i][j])              Q1.15   (한 번만 재양자화)
    9. o_acc[i]    = Σ_j S_next[i][j] · q[j]     40비트 누산
   10. o[i]        = q15(o_acc[i])               Q1.15

자체 테스트: `python3 sim/golden/deltarule.py`
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Q1.15 상수
# ---------------------------------------------------------------------------
FRAC_BITS = 15
ONE_Q15 = 1 << FRAC_BITS          # 1.0 은 Q1.15 로 표현할 수 없다 (32768 > 32767)
Q15_MAX = (1 << 15) - 1           # +32767  = +0.999969482421875
Q15_MIN = -(1 << 15)              # -32768  = -1.0
ACC_BITS = 40                     # DESIGN.md 3절: 누산 40비트
ACC_MAX = (1 << (ACC_BITS - 1)) - 1
ACC_MIN = -(1 << (ACC_BITS - 1))


class SatCounter:
    """포화 발생 횟수. RTL 의 SAT_EVENT 와 1:1 대응한다."""

    __slots__ = ("n",)

    def __init__(self) -> None:
        self.n = 0

    def bump(self, k: int = 1) -> None:
        self.n += k


# ---------------------------------------------------------------------------
# 원시 연산
# ---------------------------------------------------------------------------
def _rshift_round_half_even(x: int, shift: int) -> int:
    """x >> shift, 단 버려지는 비트에 대해 round-half-to-even.

    음수도 정확해야 하므로 파이썬의 floor 시프트(`>>`)와 **음이 아닌 나머지**를 쓴다.
    - 나머지 > 절반      → 올림
    - 나머지 == 절반     → 결과가 홀수일 때만 올림 (짝수로 맞춘다)
    - 나머지 < 절반      → 버림

    예: 1.5 → 2, 2.5 → 2, -1.5 → -2, -2.5 → -2
    """
    if shift <= 0:
        return x
    half = 1 << (shift - 1)
    q = x >> shift                       # floor
    rem = x - (q << shift)               # 0 <= rem < 2^shift
    if rem > half:
        q += 1
    elif rem == half and (q & 1):
        q += 1
    return q


def _sat_q15(x: int, sat: SatCounter | None = None) -> int:
    """Q1.15 범위로 포화."""
    if x > Q15_MAX:
        if sat is not None:
            sat.bump()
        return Q15_MAX
    if x < Q15_MIN:
        if sat is not None:
            sat.bump()
        return Q15_MIN
    return x


def _sat_acc(x: int, sat: SatCounter | None = None) -> int:
    """40비트 누산기 범위로 포화."""
    if x > ACC_MAX:
        if sat is not None:
            sat.bump()
        return ACC_MAX
    if x < ACC_MIN:
        if sat is not None:
            sat.bump()
        return ACC_MIN
    return x


def q15_mul(a: int, b: int, sat: SatCounter | None = None) -> int:
    """Q1.15 × Q1.15 → Q1.15. 곱은 Q2.30(int32), 재양자화는 round-half-to-even."""
    prod = int(a) * int(b)                       # Q2.30
    return _sat_q15(_rshift_round_half_even(prod, FRAC_BITS), sat)


def q15_from_acc(acc: int, sat: SatCounter | None = None) -> int:
    """40비트 Q2.30 누산기 → Q1.15."""
    return _sat_q15(_rshift_round_half_even(int(acc), FRAC_BITS), sat)


def float_to_q15(x: float) -> int:
    """float → Q1.15 raw. 경계에서 포화한다. 입력 생성용이며 연산 경로에는 쓰지 않는다."""
    v = _rshift_round_half_even(int(round(x * (1 << (FRAC_BITS + 8)))), 8)
    return _sat_q15(v)


def q15_to_float(x: int) -> float:
    """Q1.15 raw → float. 비교·보고용."""
    return float(x) / ONE_Q15


# ---------------------------------------------------------------------------
# 정수 델타룰 스텝 — 이것이 RTL 의 계약이다
# ---------------------------------------------------------------------------
def step(S, q, k, v, alpha: int, beta: int):
    """델타룰 한 토큰. 전부 Q1.15 정수 연산.

    인자
    ----
    S     : (d, d) int, Q1.15 상태 행렬
    q,k,v : (d,)   int, Q1.15 벡터
    alpha : int, Q1.15 게이트  α ∈ (0, 1]
    beta  : int, Q1.15 학습률  β ∈ (0, 1]

    반환
    ----
    (S_next, o, sat_count)
      S_next : (d, d) int64 ndarray, Q1.15
      o      : (d,)   int64 ndarray, Q1.15
      sat_count : 이 스텝에서 발생한 포화 횟수

    연산 순서는 모듈 docstring 의 1~10 단계 그대로다. RTL 은 이 순서를 지켜야
    비트 일치가 나온다.
    """
    S = np.asarray(S, dtype=np.int64)
    q = np.asarray(q, dtype=np.int64)
    k = np.asarray(k, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    d = S.shape[0]
    assert S.shape == (d, d), f"S 는 정사각이어야 한다: {S.shape}"
    assert q.shape == k.shape == v.shape == (d,), "q/k/v 는 (d,) 여야 한다"

    sat = SatCounter()

    # 1~2. p = S·k   (40비트 누산 후 재양자화)
    p = np.empty(d, dtype=np.int64)
    for i in range(d):
        acc = 0
        for j in range(d):
            acc = _sat_acc(acc + int(S[i, j]) * int(k[j]), sat)
        p[i] = q15_from_acc(acc, sat)

    # 3~5. err = v − α·p  (DESIGN.md 2절 "주의"의 정확한 식), β·err
    berr = np.empty(d, dtype=np.int64)
    for i in range(d):
        ap = q15_mul(alpha, int(p[i]), sat)
        err = _sat_q15(int(v[i]) - ap, sat)
        berr[i] = q15_mul(beta, err, sat)

    # 6~8. S_next = α·S + berr·kᵀ
    #
    # **Q2.30 누산기 안에서 더한 뒤 한 번만 재양자화한다.**
    # 항마다 Q1.15 로 내렸다가 더하면 재양자화가 2번 일어나 오차가 1 LSB 를 넘는다
    # (실측 1.06~1.65 LSB). 이 순서는 DESIGN.md 8절의 재사용 계획
    # "acc 초기값을 α·S 행으로 로드하는 경로 추가" 와도 일치한다 —
    # mac_array 의 누산기에 α·S 를 싣고 외적을 누산한 뒤 마지막에 한 번 내린다.
    S_next = np.empty((d, d), dtype=np.int64)
    for i in range(d):
        for j in range(d):
            acc = _sat_acc(int(alpha) * int(S[i, j]), sat)          # Q2.30 로드
            acc = _sat_acc(acc + int(berr[i]) * int(k[j]), sat)     # Q2.30 누산
            S_next[i, j] = q15_from_acc(acc, sat)                   # 한 번만 재양자화

    # 9~10. o = S_next·q
    o = np.empty(d, dtype=np.int64)
    for i in range(d):
        acc = 0
        for j in range(d):
            acc = _sat_acc(acc + int(S_next[i, j]) * int(q[j]), sat)
        o[i] = q15_from_acc(acc, sat)

    return S_next, o, sat.n


def run(S, tokens, dtype=np.int64):
    """토큰 시퀀스를 순서대로 돌린다.

    tokens : [(q, k, v, alpha, beta), ...]
    반환   : (S_final, [o_0, o_1, ...], 총 포화 횟수)
    """
    S = np.asarray(S, dtype=dtype).copy()
    outs, total_sat = [], 0
    for (q, k, v, alpha, beta) in tokens:
        S, o, n = step(S, q, k, v, alpha, beta)
        outs.append(o)
        total_sat += n
    return S, outs, total_sat


# ---------------------------------------------------------------------------
# float 참조 구현 — 비교용. **연산 경로가 아니다.**
# ---------------------------------------------------------------------------
def step_float(S, q, k, v, alpha: float, beta: float):
    """DESIGN.md 2절 정확식을 float64 로. 양자화 없음.

        p   = S·k
        S_t = α·S − α·β·p·kᵀ + β·v·kᵀ
        o   = S_t·q
    """
    S = np.asarray(S, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    p = S @ k
    S_next = alpha * S - (alpha * beta) * np.outer(p, k) + beta * np.outer(v, k)
    o = S_next @ q
    return S_next, o


def step_float_from_q15(S_q15, q_q15, k_q15, v_q15, alpha_q15: int, beta_q15: int):
    """Q1.15 입력을 float 로 바꿔 참조 계산. 비교 편의용."""
    f = q15_to_float
    return step_float(
        np.vectorize(f)(np.asarray(S_q15, dtype=np.int64)),
        np.vectorize(f)(np.asarray(q_q15, dtype=np.int64)),
        np.vectorize(f)(np.asarray(k_q15, dtype=np.int64)),
        np.vectorize(f)(np.asarray(v_q15, dtype=np.int64)),
        f(alpha_q15),
        f(beta_q15),
    )


# ---------------------------------------------------------------------------
# 입력 생성 (테스트용)
# ---------------------------------------------------------------------------
def random_vec(rng, d, scale=0.25):
    """Q1.15 범위 안쪽에서 랜덤 벡터. scale 로 포화를 피한다."""
    return np.array([float_to_q15(x) for x in rng.uniform(-scale, scale, d)], dtype=np.int64)


def random_state(rng, d, scale=0.25):
    return np.array(
        [[float_to_q15(x) for x in row] for row in rng.uniform(-scale, scale, (d, d))],
        dtype=np.int64,
    )


# ---------------------------------------------------------------------------
# 자체 테스트 (PLAN W3-1: "정수 구현과 float 구현의 오차가 1 LSB 이내인지")
# ---------------------------------------------------------------------------
def _check_round_half_even():
    cases = [
        (3, 1, 2), (1, 1, 0), (5, 1, 2), (7, 1, 4),       # 1.5→2, 0.5→0, 2.5→2, 3.5→4
        (-1, 1, 0), (-3, 1, -2), (-5, 1, -2), (-7, 1, -4),  # -0.5→0, -1.5→-2, -2.5→-2
        (2, 1, 1), (-2, 1, -1),
    ]
    for x, sh, want in cases:
        got = _rshift_round_half_even(x, sh)
        assert got == want, f"round-half-even {x}>>{sh}: {got} != {want}"
    print("  round-half-to-even        OK  (10 케이스)")


def _check_saturation():
    sat = SatCounter()
    assert _sat_q15(Q15_MAX + 1, sat) == Q15_MAX
    assert _sat_q15(Q15_MIN - 1, sat) == Q15_MIN
    assert sat.n == 2
    # -1.0 × -1.0 = +1.0 은 Q1.15 로 표현 불가 → 포화해야 한다
    s2 = SatCounter()
    assert q15_mul(Q15_MIN, Q15_MIN, s2) == Q15_MAX and s2.n == 1
    print("  포화(saturate)             OK  (경계 3케이스, SAT 카운트 일치)")


def _check_invariants(d, rng):
    """DESIGN.md 7절 불변조건 중 골든 모델 수준에서 검사 가능한 것."""
    S = random_state(rng, d)

    # I2. α=1, β=0 이면 상태가 변하지 않는다
    S1, _, _ = step(S, random_vec(rng, d), random_vec(rng, d), random_vec(rng, d),
                    Q15_MAX, 0)
    # α=Q15_MAX 는 정확히 1.0 이 아니다 (32767/32768). 그래서 1 LSB 오차를 허용한다.
    dmax = int(np.max(np.abs(S1 - S)))
    assert dmax <= 1, f"I2: α≈1,β=0 인데 상태가 {dmax} LSB 변했다"

    # I3. β=0 이면 o = (α·S)·q
    qv = random_vec(rng, d)
    S2, o2, _ = step(S, qv, random_vec(rng, d), random_vec(rng, d), Q15_MAX, 0)
    o_ref = np.array([q15_from_acc(sum(int(S2[i, j]) * int(qv[j]) for j in range(d)))
                      for i in range(d)], dtype=np.int64)
    assert np.array_equal(o2, o_ref), "I3: β=0 일 때 o 가 S_next·q 와 다르다"

    # I5. 결정성 — 같은 입력 두 번 → 같은 결과
    args = (S, qv, random_vec(rng, d), random_vec(rng, d), 30000, 12000)
    a = step(*args)
    b = step(*args)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2], \
        "I5: 같은 입력인데 결과가 다르다"
    print(f"  불변조건 I2/I3/I5 (d={d})   OK")


def _check_vs_float(d, rng, n_trials=20):
    """정수 구현 vs float 참조. 오차를 **세 가지로 분해해서** 실측한다.

    1. S 오차        : 정수 S_next  vs  float S_next (같은 양자화 입력에서)
                       → 상태 갱신 경로 자체의 양자화 오차. **계약: ≤ 1 LSB**
    2. o 오차(고립)  : 정수 o  vs  float(정수 S_next) @ float(q)
                       → o = S_next·q 단계만의 양자화 오차. **계약: ≤ 1 LSB**
    3. o 오차(전체)  : 정수 o  vs  float S_next @ float q
                       → 위 1번 오차가 d 번 누산을 거쳐 전파된 것까지 포함.
                         **이건 1 LSB 로 묶을 수 없다.** d 가 커질수록 커진다.
                         정보로 보고만 하고 게이트하지 않는다.

    왜 3번을 게이트하지 않는가: RTL 검증(DESIGN.md 7절)은 **골든 모델과 비트 일치**로
    한다. 둘 다 정수라 float 드리프트는 판정에 쓰이지 않는다. float 비교는 정수 구현의
    부호·시프트 같은 **굵직한 실수**를 잡으려고 두는 것이고, 1.2 LSB 수준의 전파 오차는
    그런 실수의 징후가 아니다.
    """
    f = q15_to_float
    worst = {"S": 0.0, "o_iso": 0.0, "o_e2e": 0.0}
    for _ in range(n_trials):
        S = random_state(rng, d)
        qv, kv, vv = random_vec(rng, d), random_vec(rng, d), random_vec(rng, d)
        alpha = float_to_q15(rng.uniform(0.5, 0.999))
        beta = float_to_q15(rng.uniform(0.05, 0.5))

        S_i, o_i, sat_n = step(S, qv, kv, vv, alpha, beta)
        assert sat_n == 0, "이 입력 범위에서는 포화가 없어야 한다"
        S_f, o_f = step_float_from_q15(S, qv, kv, vv, alpha, beta)

        S_i_f = S_i.astype(np.float64) / ONE_Q15
        o_i_f = o_i.astype(np.float64) / ONE_Q15
        q_f = np.vectorize(f)(qv)

        worst["S"] = max(worst["S"], np.max(np.abs(S_i_f - S_f)) * ONE_Q15)
        worst["o_iso"] = max(worst["o_iso"],
                             np.max(np.abs(o_i_f - (S_i_f @ q_f))) * ONE_Q15)
        worst["o_e2e"] = max(worst["o_e2e"], np.max(np.abs(o_i_f - o_f)) * ONE_Q15)
    return worst


def drift_over_tokens(d, n_tokens, rng, seed_state=None):
    """N 토큰을 돌린 뒤 정수 경로와 float 경로의 상태 차이를 LSB 로 보고한다.

    게이트가 아니라 **관측**이다. DESIGN.md 9절 "Q1.15 상태는 장시간 누적 시 정밀도
    손실" 항목의 근거 숫자를 만든다. W8 에서 1,000토큰까지 볼 때 쓴다.
    """
    f = q15_to_float
    S_i = random_state(rng, d) if seed_state is None else np.asarray(seed_state).copy()
    S_f = np.vectorize(f)(S_i).astype(np.float64)
    total_sat = 0
    for _ in range(n_tokens):
        qv, kv, vv = random_vec(rng, d), random_vec(rng, d), random_vec(rng, d)
        a = float_to_q15(rng.uniform(0.8, 0.999))
        b = float_to_q15(rng.uniform(0.05, 0.3))
        S_i, _, n = step(S_i, qv, kv, vv, a, b)
        total_sat += n
        S_f, _ = step_float(S_f, np.vectorize(f)(qv), np.vectorize(f)(kv),
                            np.vectorize(f)(vv), f(a), f(b))
    drift = np.max(np.abs(S_i.astype(np.float64) / ONE_Q15 - S_f)) * ONE_Q15
    return drift, total_sat


def self_test(verbose=True):
    """PLAN W3-1 자체 테스트. 실패하면 AssertionError."""
    rng = np.random.default_rng(20260911)
    if verbose:
        print("=== sim/golden/deltarule.py 자체 테스트 ===")
    _check_round_half_even()
    _check_saturation()

    for d in (16, 64):
        _check_invariants(d, rng)

    ok = True
    for d in (16, 64):
        w = _check_vs_float(d, rng, n_trials=20 if d == 16 else 5)
        bad = (w["S"] > 1.0) or (w["o_iso"] > 1.0)
        if bad:
            ok = False
        if verbose:
            print(f"  float 대비 (d={d:2d})  {'초과' if bad else 'OK '}  "
                  f"S {w['S']:.3f} / o(고립) {w['o_iso']:.3f} LSB   [기준 1 LSB]")
            print(f"                     참고: o(전체) {w['o_e2e']:.3f} LSB "
                  f"— S 양자화가 {d}회 누산으로 전파된 것. 게이트 아님")

    if verbose:
        for d, n in ((16, 100), (64, 20)):
            drift, sat_n = drift_over_tokens(d, n, rng)
            print(f"  {n:4d}토큰 드리프트 (d={d:2d})  {drift:8.1f} LSB, 포화 {sat_n}회  — 관측용")

    if verbose:
        print("=== 전부 통과 ===" if ok else "=== 오차 기준 초과 — 위 숫자 확인 ===")
    assert ok, "정수 구현과 float 참조의 오차가 1 LSB 를 넘었다"
    return True


if __name__ == "__main__":
    self_test()
