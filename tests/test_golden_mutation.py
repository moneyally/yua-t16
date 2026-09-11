"""기대값 출처: scripts/mutation_test.py 의 뮤턴트 목록 — 각 뮤턴트는 **반드시 죽어야 한다**.

test_golden_mutation.py — **불변 코드**의 테스트가 살아 있는지 검사한다

## 왜 이 파일이 있는가

`tests/test_golden_deltarule.py` 가 통과한다는 사실만으로는 그 테스트가 **뭔가를
지킨다**는 증거가 되지 않는다. 아무것도 검사하지 않는 테스트도 통과한다.

뮤테이션 테스트는 소스를 한 군데씩 고의로 망가뜨리고 그 테스트가 반드시
실패하는지 본다. 실패하지 않으면 = 그 실수를 아무도 안 잡는다 = **구멍**이다.

실제로 이 방법으로 구멍을 찾았다: α/β 정의역(>0x8000) 검사가 골든 테스트에
아예 없었다 (`uq15_gate_off` 뮤턴트가 생존). 지금은 메워져 있다.

## 대상은 골든만이 아니다

각 뮤턴트가 **자기 대상 파일과 오라클 테스트를 들고 있다** (`Mutant.target`,
`Mutant.oracle`). 골든 모델 외에 `tools/orbit_mmio_map.py` 의 레지스터 비트
배치처럼 **한 번 정하면 안 바뀌고, 틀려도 조용히 도는** 코드가 대상이다.

## 빠른 부분집합만 돈다

전체는 `python3 scripts/mutation_test.py` 로 돌린다 (~10초).
여기서는 `quick=True` 로 표시된 대표 뮤턴트만 돌려서 pytest 를 느리게 만들지 않는다.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.mutation_test import (  # noqa: E402
    MUTANTS,
    build_sandbox,
    run_oracle,
)

QUICK = [m for m in MUTANTS if m.quick]


def test_mutant_list_is_not_empty():
    """뮤턴트 목록이 비면 이 파일 전체가 무의미해진다."""
    assert len(MUTANTS) >= 15, f"뮤턴트가 {len(MUTANTS)}개뿐이다"
    assert len(QUICK) >= 4, f"quick 뮤턴트가 {len(QUICK)}개뿐이다"


def test_every_mutant_pattern_is_unique():
    """각 뮤턴트 패턴이 **자기 대상 파일에** 정확히 한 번 나와야 한다.

    0번이면 대상이 바뀌어 뮤턴트가 죽은 코드가 된 것이고,
    2번 이상이면 어디를 망가뜨렸는지 알 수 없다. 둘 다 조용히 무의미해지는 길이다.
    """
    cache = {}
    bad = {}
    for m in MUTANTS:
        src = cache.setdefault(m.target, (REPO / m.target).read_text())
        n = src.count(m.old)
        if n != 1:
            bad[m.name] = f"{m.target} 에 {n}번"
    assert not bad, (
        f"대상 소스와 어긋난 뮤턴트: {bad}. "
        f"대상을 고쳤으면 scripts/mutation_test.py 의 패턴도 고쳐야 한다."
    )


def test_baseline_passes(tmp_path):
    """뮤턴트가 쓰는 **모든 오라클**이 원본에서 통과해야 한다.

    아니면 아래 결과가 전부 무의미하다 — 죽은 게 뮤턴트 때문인지 원래 깨져
    있었기 때문인지 구분이 안 된다.
    """
    sb = build_sandbox(tmp_path)
    for oracle in sorted({m.oracle for m in MUTANTS}):
        ok, tail = run_oracle(sb, oracle)
        assert ok, f"원본에서 {oracle} 이 이미 실패한다: {tail}"


@pytest.mark.parametrize("mutant", QUICK, ids=[m.name for m in QUICK])
def test_mutant_is_killed(mutant, tmp_path):
    """뮤턴트를 넣으면 골든 테스트가 **반드시 실패**해야 한다.

    통과하면 그 실수를 아무도 안 잡는다는 뜻이다 — 테스트를 보강해야 한다.
    """
    src = (REPO / mutant.target).read_text()
    sb = build_sandbox(tmp_path)
    (sb / mutant.target).write_text(src.replace(mutant.old, mutant.new, 1))

    ok, tail = run_oracle(sb, mutant.oracle)
    assert not ok, (
        f"뮤턴트 '{mutant.name}' 가 살아남았다 ({tail}).\n"
        f"  망가뜨린 것: {mutant.why}\n"
        f"  {mutant.target}: {mutant.old!r} → {mutant.new!r}\n"
        f"  {mutant.oracle} 에 이 실수를 잡는 테스트가 없다."
    )
