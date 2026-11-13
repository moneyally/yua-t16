#!/usr/bin/env python3
"""run_one.py — 테스트벤치 하나를 격리된 프로세스에서 실행한다.

인자: <testbench_stem> <top_module> <tb_dir>
run_all.py 가 서브프로세스로 호출한다. 한 프로세스에 여러 테스트벤치를
넣으면 sys.path 가 누적 오염되고, Verilator 메이크파일이 부르는 python3 가
그 경로를 읽어 "SRE module mismatch" 로 죽는다.
"""
import json
import os
import sys
from pathlib import Path

# Verilator 는 작은 설계에서 "fast build" 경로를 타며 verilator_includer 를
# `#!/usr/bin/env python3` 로 실행한다. cocotb 는 자기 site-packages 를
# PYTHONPATH 에 넣는데, 시스템 python3 가 cocotb 를 설치한 인터프리터와
# 다른 버전이면 그 경로의 re 모듈을 읽어 "SRE module mismatch" 로 죽는다.
# 빌드 동안 python3 가 지금 인터프리터를 가리키도록 shim 을 앞에 둔다.
_SHIM = Path(__file__).resolve().parent / ".pyshim"
_SHIM.mkdir(exist_ok=True)
_LINK = _SHIM / "python3"
if not _LINK.exists():
    _LINK.symlink_to(sys.executable)
os.environ["PATH"] = f"{_SHIM}{os.pathsep}{os.environ.get('PATH', '')}"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rtl_deps import deps, packages                          # noqa: E402
from cocotb_tools.runner import get_runner                   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    stem, top, tb_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    # 선택 4번째 인자: 파라미터 JSON. 같은 모듈의 다른 구성도 회귀에 넣는다.
    params = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    tag = stem if not params else stem

    # 빌드는 top 모듈별로 공유한다. 테스트벤치마다 따로 빌드하면
    # g2_ctrl_top 하나를 10번, mxu_bf16_128x128(PAR=8)을 2번 컴파일한다.
    suffix = "" if not params else "_" + "_".join(
        f"{k}{v}" for k, v in sorted(params.items()))
    build = ROOT / "sim_build" / "build" / f"{top}{suffix}"
    results = ROOT / "sim_build" / "results"
    results.mkdir(parents=True, exist_ok=True)

    # top 이 rtl/ 이 아니라 sim/cocotb/ 에 있는 검증용 래퍼일 수 있다.
    # 그 경우 래퍼가 감싸는 모듈의 의존성 + 래퍼 자신을 쓴다.
    wrapper = ROOT / "sim/cocotb" / f"{top}.sv"
    if wrapper.exists():
        inner = {"tb_mxu16_flat": "mxu_bf16_16x16"}.get(top)
        base = deps(inner) if inner else packages()
        src = base + [wrapper]
    else:
        src = deps(top)

    runner = get_runner("verilator")
    runner.build(
        sources=src,
        hdl_toplevel=top,
        parameters=params,
        build_args=["--timing", "-Wno-fatal", "--public-flat-rw",
                    "-Wno-WIDTHEXPAND", "-Wno-TIMESCALEMOD",
                    "--unroll-count", "4096"],
        build_dir=build,
        always=False,
    )
    sys.path.insert(0, str(tb_dir))
    runner.test(hdl_toplevel=top, test_module=stem,
                results_xml=str(results / f"{stem}.xml"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
