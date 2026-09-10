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

    sim = os.environ.get("SIM", "icarus")
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
        build_dir=str(build_dir),
        build_args=build_args,
        always=True,
        # icarus 는 빌드 시점에 waves=True 여야 덤프 모듈을 생성한다.
        waves=True,
    )
    results = runner.test(
        hdl_toplevel=toplevel,
        test_module=module,
        test_dir=str(Path(__file__).resolve().parent),
        build_dir=str(build_dir),
        waves=True,
        # tb/results.xml (git 추적 파일) 을 덮어쓰지 않도록 build 아래로 뺀다.
        results_xml=str(build_dir / "results.xml"),
    )
    print(f"\n[run_tb] results xml: {results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
