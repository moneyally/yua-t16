#!/usr/bin/env python3
"""run_tb.py — tb/ 의 cocotb 테스트벤치를 실행하는 러너.

docs/AUDIT.md §4 가 지적한 문제를 메운다: tb/ 에 Makefile 이 없어서
CLAUDE.md 4절의 `cd tb && make SIM=verilator ...` 가 실행 불가능했다.

사용:
    python3 tb/run_tb.py <toplevel> <module> [source.sv ...]

예:
    python3 tb/run_tb.py desc_fsm_v2 tb_desc_fsm_v2_done_pulse
    python3 tb/run_tb.py g2_ctrl_top tb_g2_ctrl_top_fault_irq

소스를 생략하면 rtl/<toplevel>.sv 하나만 쓴다.
SIM 환경변수로 시뮬레이터 선택 (기본 icarus). SV2V=0 이면 sv2v 전처리를 끈다.
파형은 build/tb/<toplevel>/ 에 남는다.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TB_DIRS = [Path(__file__).resolve().parent, Path(__file__).resolve().parent / "behavioral"]


def _verilator_ok() -> bool:
    """cocotb 2.x 는 verilator 5.022+ 의 VerilatedVpi API 를 요구한다."""
    import shutil
    import subprocess
    if not shutil.which("verilator"):
        return False
    try:
        out = subprocess.check_output(["verilator", "--version"], text=True).split()
        maj, minor = out[1].split(".")[:2]
        return (int(maj), int(minor)) >= (5, 22)
    except Exception:
        return False


def _find_test_dir(module: str) -> Path:
    """테스트 모듈이 tb/ 인지 tb/behavioral/ 인지 찾는다."""
    for d in TB_DIRS:
        if (d / f"{module}.py").exists():
            return d
    raise SystemExit(f"테스트 모듈을 찾을 수 없다: {module}.py (찾은 곳: {[str(d) for d in TB_DIRS]})")


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2

    toplevel = sys.argv[1]
    module = sys.argv[2]
    extra = sys.argv[3:]

    sources = [ROOT / s for s in extra] if extra else [ROOT / "rtl" / f"{toplevel}.sv"]
    missing = [str(p) for p in sources if not p.exists()]
    if missing:
        print("소스 없음:", *missing, sep="\n  ", file=sys.stderr)
        return 2

    # 파라미터 덮어쓰기: PARAM_D=64 처럼 환경변수로 준다.
    # (d=64 확장처럼 같은 RTL 을 다른 크기로 돌릴 때 쓴다 — 파일을 복제하지 않는다)
    parameters = {}
    for key, val in os.environ.items():
        if key.startswith("PARAM_"):
            parameters[key[len("PARAM_"):]] = int(val, 0)
    if parameters:
        print(f"[run_tb] 파라미터 덮어쓰기: {parameters}")

    # verilator 5.022+ 가 있으면 그쪽이 기본이다 (네이티브 SystemVerilog, sv2v 불필요).
    # 없거나 구버전이면 icarus + sv2v 로 내려간다.
    sim = os.environ.get("SIM") or ("verilator" if _verilator_ok() else "icarus")
    build_dir = ROOT / "build" / "tb" / toplevel
    build_dir.mkdir(parents=True, exist_ok=True)

    # iverilog 12 는 always_comb 안의 `automatic` 선언과 unpacked array 포트를
    # 지원하지 않는다 (docs/AUDIT.md §2). scripts/synth_gate.sh 와 같은 방식으로
    # sv2v 로 Verilog-2005 로 낮춰서 먹인다.
    # Debian verilator 5.020 은 cocotb 2.x 가 요구하는 VerilatedVpi API
    # (clearEvalNeeded / doInertialPuts) 가 없어서 현재 쓸 수 없다 — SIM=icarus 가 기본.
    if sim == "icarus" and os.environ.get("SV2V", "1") != "0":
        import subprocess
        flat = build_dir / f"{toplevel}.v"
        with open(flat, "w") as fh:
            # sv2v 는 `timescale 을 버린다. iverilog 기본 정밀도는 1s 라서
            # Clock(10, "ns") 이 "Unable to accurately represent" 로 죽는다.
            fh.write("`timescale 1ns/1ps\n")
            fh.flush()
            rc = subprocess.call(["sv2v", *[str(p) for p in sources]], stdout=fh)
        if rc != 0:
            print(f"sv2v 변환 실패 (rc={rc})", file=sys.stderr)
            return 2
        print(f"[run_tb] sv2v -> {flat}")
        sources = [flat]

    from cocotb_tools.runner import get_runner

    runner = get_runner(sim)
    build_args = []
    if sim == "verilator":
        build_args += ["-DCOCOTB_SIM=1", "-Wno-fatal", "--trace", "--trace-structs"]
    elif sim == "icarus":
        build_args += ["-g2012", "-DCOCOTB_SIM=1"]

    runner.build(
        verilog_sources=[str(p) for p in sources],
        hdl_toplevel=toplevel,
        parameters=parameters,
        build_dir=str(build_dir),
        build_args=build_args,
        always=True,
        # icarus 는 빌드 시점에 waves=True 여야 덤프 모듈을 생성한다.
        waves=True,
    )
    results = runner.test(
        hdl_toplevel=toplevel,
        test_module=module,
        test_dir=str(_find_test_dir(module)),
        build_dir=str(build_dir),
        waves=True,
        # tb/results.xml (git 추적 파일) 을 덮어쓰지 않도록 build 아래로 뺀다.
        results_xml=str(build_dir / "results.xml"),
    )
    print(f"\n[run_tb] results xml: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
