"""도구가 **직접 실행**으로도 돌아가는가.

`tools/gemm_tiler.py` 는 패키지 임포트로는 멀쩡하면서
`python3 tools/gemm_tiler.py` 로는 `ModuleNotFoundError: No module named
'tools'` 로 죽었다. README 에 그 명령이 적혀 있었고, 테스트는 임포트만
했기 때문에 통과했다.

사용자가 README 를 따라 치는 명령이 동작하는지는 임포트 테스트가
증명하지 못한다. 그래서 실제로 서브프로세스로 돌린다.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# README 의 빠른 시작에 적힌 것들. 인자 없이 보고서를 내는 도구만.
RUNNABLE = [
    "tools/perf_model.py",
    "tools/gemm_tiler.py",
]


@pytest.mark.parametrize("script", RUNNABLE)
def test_script_runs_standalone(script):
    """리포지토리 루트에서 직접 실행했을 때 0 으로 끝나야 한다."""
    r = subprocess.run([sys.executable, script], cwd=ROOT,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (
        f"{script} 직접 실행 실패 (exit {r.returncode})\n"
        f"--- stderr ---\n{r.stderr[-2000:]}")
    assert r.stdout.strip(), f"{script} 가 아무것도 출력하지 않았다"


@pytest.mark.parametrize("script", RUNNABLE)
def test_script_runs_from_another_directory(script):
    """다른 디렉터리에서 절대 경로로 불러도 돌아야 한다.

    cwd 에 의존해 임포트가 되는 것은 운이다.
    """
    r = subprocess.run([sys.executable, str(ROOT / script)], cwd="/tmp",
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (
        f"{script} 를 /tmp 에서 실행 실패 (exit {r.returncode})\n"
        f"--- stderr ---\n{r.stderr[-2000:]}")


@pytest.mark.parametrize("script", RUNNABLE)
def test_script_is_importable_too(script):
    """임포트 경로도 깨지지 않아야 한다 (직접 실행을 고치다 깨기 쉽다)."""
    mod = script.replace("/", ".").removesuffix(".py")
    r = subprocess.run([sys.executable, "-c", f"import {mod}"], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, \
        f"import {mod} 실패\n--- stderr ---\n{r.stderr[-2000:]}"


def test_readme_quickstart_commands_exist():
    """README 가 가리키는 스크립트가 실제로 있어야 한다."""
    readme = (ROOT / "README.md").read_text()
    import re
    for m in re.findall(r"python3(?:\.\d+)? ((?:tools|sim)/[\w/]+\.py)", readme):
        assert (ROOT / m).exists(), f"README 가 없는 파일을 가리킨다: {m}"
