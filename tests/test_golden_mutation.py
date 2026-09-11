"""기대값 출처: scripts/mutation_test.py 의 뮤턴트 목록 — 각 뮤턴트는 **반드시 죽어야 한다**.

test_golden_mutation.py — 골든 모델 테스트가 살아 있는지 검사한다

## 왜 이 파일이 있는가

`tests/test_golden_deltarule.py` 가 통과한다는 사실만으로는 그 테스트가 **뭔가를
지킨다**는 증거가 되지 않는다. 아무것도 검사하지 않는 테스트도 통과한다.

뮤테이션 테스트는 골든 소스를 한 군데씩 고의로 망가뜨리고 그 테스트가 반드시
실패하는지 본다. 실패하지 않으면 = 그 실수를 아무도 안 잡는다 = **구멍**이다.

실제로 이 방법으로 구멍을 찾았다: α/β 정의역(>0x8000) 검사가 골든 테스트에
아예 없었다 (`uq15_gate_off` 뮤턴트가 생존). 지금은 메워져 있다.

## 빠른 부분집합만 돈다

전체(13개)는 `python3 scripts/mutation_test.py` 로 돌린다 (~10초).
여기서는 `quick=True` 로 표시된 대표 뮤턴트만 돌려서 pytest 를 느리게 만들지 않는다.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.mutation_test import (  # noqa: E402
    GOLDEN_REL,
    MUTANTS,
    build_sandbox,
    run_oracle,
)

QUICK = [m for m in MUTANTS if m.quick]


def test_mutant_list_is_not_empty():
    """뮤턴트 목록이 비면 이 파일 전체가 무의미해진다."""
    assert len(MUTANTS) >= 10, f"뮤턴트가 {len(MUTANTS)}개뿐이다"
    assert len(QUICK) >= 4, f"quick 뮤턴트가 {len(QUICK)}개뿐이다"


def test_every_mutant_pattern_is_unique(tmp_path):
    """각 뮤턴트 패턴이 골든 소스에 **정확히 한 번** 나와야 한다.

    0번이면 골든이 바뀌어 뮤턴트가 죽은 코드가 된 것이고,
    2번 이상이면 어디를 망가뜨렸는지 알 수 없다. 둘 다 조용히 무의미해지는 길이다.
    """
    src = (REPO / GOLDEN_REL).read_text()
    bad = {m.name: src.count(m.old) for m in MUTANTS if src.count(m.old) != 1}
    assert not bad, (
        f"골든 소스와 어긋난 뮤턴트: {bad}. "
        f"골든을 고쳤으면 scripts/mutation_test.py 의 패턴도 고쳐야 한다."
    )


def test_baseline_passes(tmp_path):
    """원본 골든은 통과해야 한다. 아니면 아래 결과가 전부 무의미하다."""
    sb = build_sandbox(tmp_path)
    ok, tail = run_oracle(sb)
    assert ok, f"원본 골든이 이미 실패한다: {tail}"


@pytest.mark.parametrize("mutant", QUICK, ids=[m.name for m in QUICK])
def test_mutant_is_killed(mutant, tmp_path):
    """뮤턴트를 넣으면 골든 테스트가 **반드시 실패**해야 한다.

    통과하면 그 실수를 아무도 안 잡는다는 뜻이다 — 테스트를 보강해야 한다.
    """
    src = (REPO / GOLDEN_REL).read_text()
    sb = build_sandbox(tmp_path)
    (sb / GOLDEN_REL).write_text(src.replace(mutant.old, mutant.new, 1))

    ok, tail = run_oracle(sb)
    assert not ok, (
        f"뮤턴트 '{mutant.name}' 가 살아남았다 ({tail}).\n"
        f"  망가뜨린 것: {mutant.why}\n"
        f"  {mutant.old!r} → {mutant.new!r}\n"
        f"  tests/test_golden_deltarule.py 에 이 실수를 잡는 테스트가 없다."
    )
