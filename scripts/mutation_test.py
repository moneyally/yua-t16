#!/usr/bin/env python3
"""mutation_test.py — 골든 모델(`sim/golden/deltarule.py`)에 대한 뮤테이션 테스트

## 왜 필요한가

`sim/golden/deltarule.py` 는 **정답지**다. RTL 은 이것과 비트 비교해서만 통과한다
(CLAUDE.md 규칙 2). 그런데 골든이 틀리면 RTL 도 같이 틀린 채로 "통과"한다.
골든을 지키는 것은 `tests/test_golden_deltarule.py` 뿐이다.

그 테스트가 **실제로 뭔가를 지키는지** 확인하는 방법이 뮤테이션 테스트다:
골든 소스를 한 군데씩 고의로 망가뜨리고, 테스트가 **반드시 실패하는지** 본다.

- 뮤턴트가 죽는다(killed) = 테스트가 그 실수를 잡는다 ✔
- 뮤턴트가 산다(survived) = **테스트에 구멍이 있다** ✘ — 그 구멍을 메워야 한다

## 어떻게 도는가

레포를 건드리지 않는다. 임시 디렉터리에 `sim/`, `tools/`, `tests/` 를 복사하고
거기서 골든을 망가뜨린 뒤 **진짜 pytest 를 돌린다**. 통과하면 그 뮤턴트는 생존이다.

    python3 scripts/mutation_test.py           # 전체
    python3 scripts/mutation_test.py --quick   # 대표 뮤턴트만
    python3 scripts/mutation_test.py --list    # 목록만
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GOLDEN_REL = Path("sim/golden/deltarule.py")
ORACLE = "tests/test_golden_deltarule.py"


@dataclass(frozen=True)
class Mutant:
    name: str
    old: str
    new: str
    why: str          # 이 실수가 실제로 일어나면 무엇이 깨지는가
    quick: bool = False


# 각 뮤턴트는 **실제로 저지를 법한 실수** 여야 한다. 아무 문자나 바꾸면
# "테스트가 잡았다"가 의미 없어진다.
MUTANTS = [
    Mutant(
        "round_half_up",
        "elif rem == half and (q & 1):",
        "elif rem == half:",
        "round-half-to-even 이 round-half-up 이 된다. RTL requant_q15 와 갈라진다.",
        quick=True,
    ),
    Mutant(
        "no_rounding",
        "if rem > half:",
        "if False:",
        "반올림이 사라지고 버림이 된다. 1 LSB 씩 체계적으로 치우친다.",
        quick=True,
    ),
    Mutant(
        "no_saturation_hi",
        "    if x > Q15_MAX:",
        "    if False:",
        "상한 포화가 사라진다. 오버플로가 조용히 감긴다 (SAT_EVENT 도 안 뜬다).",
        quick=True,
    ),
    Mutant(
        "q15_min_off_by_one",
        "Q15_MIN = -(1 << 15)",
        "Q15_MIN = -(1 << 15) + 1",
        "Q1.15 최소값이 -1.0 이 아니게 된다. 경계에서만 틀린다.",
    ),
    Mutant(
        "err_sign_flip",
        "err[i] = sat_q15(int(v[i]) - ap, sat)",
        "err[i] = sat_q15(int(v[i]) + ap, sat)",
        "err = v − α·p 가 v + α·p 가 된다. DESIGN 2절 위반.",
        quick=True,
    ),
    Mutant(
        "err_drops_alpha",
        "ap = q15_mul(a, int(p[i]), sat)",
        "ap = int(p[i])",
        "err = v − p 가 된다. **α=1 일 때만 맞는다** — DESIGN 2절이 경고한 그 실수다.",
        quick=True,
    ),
    Mutant(
        "update_sign_flip",
        "acc = _sat_acc(acc + berr * int(k[j]), sat)",
        "acc = _sat_acc(acc - berr * int(k[j]), sat)",
        "외적 누산의 부호가 뒤집힌다.",
    ),
    Mutant(
        "update_drops_beta",
        "berr = q15_mul(b, int(err_i), sat)",
        "berr = int(err_i)",
        "학습률 β 가 무시된다. β=1 일 때만 맞는다.",
    ),
    Mutant(
        "update_double_requant",
        "acc = _sat_acc(a * int(S_row[j]), sat)",
        "acc = q15_mul(a, int(S_row[j]), sat) << 15",
        "α·S 를 **먼저 내렸다가** 다시 올린다 = 재양자화 2회. "
        "DESIGN 3절이 금지한 것이고, 실측 오차가 0.63 → 1.06 LSB 로 뛴다.",
        quick=True,
    ),
    Mutant(
        "matvec_sign_flip",
        "acc = _sat_acc(acc + int(S[i, j]) * int(x[j]), sat)",
        "acc = _sat_acc(acc - int(S[i, j]) * int(x[j]), sat)",
        "행렬-벡터 곱의 부호가 뒤집힌다.",
    ),
    Mutant(
        "o_uses_old_state",
        "o, n = matvec(S_next, q)",
        "o, n = matvec(S, q)",
        "o 를 **갱신 전** 상태로 계산한다. 골든 9~10단계 순서 위반 — "
        "RTL 이 이걸 따라 하면 토큰마다 한 칸씩 밀린다.",
        quick=True,
    ),
    Mutant(
        "uq15_gate_off",
        "    if v > UQ15_MAX_DEFINED:",
        "    if False:",
        "α, β > 1.0 을 조용히 받는다. 상태가 발산해도 아무도 모른다.",
    ),
    Mutant(
        "sat_not_counted",
        "            sat.bump()\n        return Q15_MAX",
        "            pass\n        return Q15_MAX",
        "포화를 세지 않는다. DR1_SAT_COUNT 가 항상 0 이 되고 I4 가 죽는다.",
    ),
]


def build_sandbox(tmp: Path) -> Path:
    """레포에서 필요한 것만 복사한다. 원본은 절대 건드리지 않는다."""
    for sub in ("sim", "tools", "tests"):
        shutil.copytree(REPO / sub, tmp / sub,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return tmp


def run_oracle(sandbox: Path, timeout: int = 300) -> tuple[bool, str]:
    """샌드박스에서 골든 테스트를 돌린다. (통과 여부, 마지막 줄)"""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", ORACLE, "-x", "-q", "--no-header"],
        cwd=sandbox, capture_output=True, text=True, timeout=timeout,
    )
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    return r.returncode == 0, tail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="대표 뮤턴트만")
    ap.add_argument("--list", action="store_true", help="목록만 출력")
    args = ap.parse_args()

    mutants = [m for m in MUTANTS if m.quick] if args.quick else MUTANTS

    if args.list:
        for m in mutants:
            print(f"{m.name:24s} {m.why}")
        return 0

    print("=== 골든 모델 뮤테이션 테스트 ===")
    print(f"대상  : {GOLDEN_REL}")
    print(f"오라클: {ORACLE}")
    print(f"뮤턴트: {len(mutants)}개\n")

    src = (REPO / GOLDEN_REL).read_text()

    # 0) 원본은 반드시 통과해야 한다. 아니면 뮤테이션 결과가 무의미하다.
    with tempfile.TemporaryDirectory() as td:
        sb = build_sandbox(Path(td))
        ok, tail = run_oracle(sb)
        if not ok:
            print(f"  원본이 이미 실패한다: {tail}")
            print("  뮤테이션 테스트는 원본이 통과할 때만 의미가 있다.")
            return 2
        print(f"  기준선(원본) 통과 확인: {tail}\n")

    survived = []
    for m in mutants:
        n = src.count(m.old)
        if n != 1:
            print(f"  SKIP    {m.name:24s} 패턴이 {n}번 나온다 (1번이어야 한다) — "
                  f"골든이 바뀌었으면 뮤턴트도 고쳐야 한다")
            survived.append((m, f"패턴 {n}회"))
            continue

        with tempfile.TemporaryDirectory() as td:
            sb = build_sandbox(Path(td))
            (sb / GOLDEN_REL).write_text(src.replace(m.old, m.new, 1))
            try:
                ok, tail = run_oracle(sb)
            except subprocess.TimeoutExpired:
                ok, tail = True, "시간 초과"

        if ok:
            print(f"  SURVIVED {m.name:23s} ← **테스트 구멍** ({tail})")
            survived.append((m, tail))
        else:
            print(f"  killed  {m.name:24s} {tail}")

    print()
    if survived:
        print(f"=== 생존 {len(survived)}개 — 테스트에 구멍이 있다 ===")
        for m, tail in survived:
            print(f"  {m.name}: {m.why}")
        return 1

    print(f"=== 전부 죽었다 ({len(mutants)}/{len(mutants)}) — 골든 테스트가 살아 있다 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
