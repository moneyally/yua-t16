# sim/cocotb/run_gemm_top.py
from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path

from cocotb_test.simulator import run


def project_root() -> Path:
    # .../yua-t16/sim/cocotb/run_gemm_top.py 기준으로 루트 찾기
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root = project_root()
    sim_dir = root / "sim" / "cocotb"
    build_dir = root / "sim_build" / "gemm_top"

    verilog_sources = [
        root / "rtl" / "mac_pe.sv",
        root / "rtl" / "mac_array.sv",
        root / "rtl" / "act_sram.sv",
        root / "rtl" / "wgt_sram.sv",
        root / "rtl" / "ctrl_fsm.sv",
        root / "rtl" / "gemm_core.sv",
        root / "rtl" / "gemm_top.sv",
    ]

    missing = [p for p in verilog_sources if not p.exists()]
    if missing:
        print("[run_gemm_top] Missing verilog_sources:", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(2)

    # ---- FULL LOG SETTINGS (stdout flush + cocotb log level) ----
    # 환경변수로 덮어쓸 수 있게 기본값만 세팅
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("COCOTB_LOG_LEVEL", "DEBUG")          # INFO면 필요시 바꿔도 됨
    os.environ.setdefault("COCOTB_REDUCED_LOG_FMT", "0")        # 로그 포맷 축약 끔
    os.environ.setdefault("PYTHONWARNINGS", "default")          # warning도 다 보이게
    os.environ.setdefault("WAVES", "1")                         # cocotb-test waves 힌트

    # ---- CLEAN BUILD (기본: 매번 삭제) ----
    clean = os.environ.get("CLEAN", "1")
    if clean != "0" and build_dir.exists():
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)

    print("[run_gemm_top] root      :", root)
    print("[run_gemm_top] sim_dir   :", sim_dir)
    print("[run_gemm_top] build_dir :", build_dir)
    print("[run_gemm_top] log_level :", os.environ.get("COCOTB_LOG_LEVEL"))
    print("[run_gemm_top] waves     :", os.environ.get("WAVES"))
    print("[run_gemm_top] sources   :")
    for p in verilog_sources:
        print("  -", p)
    sys.stdout.flush()

    # ---- RUN ----
    run(
        simulator="icarus",
        toplevel="gemm_top",
        module="test_gemm_top",                 # sim/cocotb/test_gemm_top.py 의 @cocotb.test 들 실행
        python_search=[str(sim_dir)],
        verilog_sources=[str(p) for p in verilog_sources],
        compile_args=["-g2012", "-DCOCOTB_SIM=1"],
        sim_build=str(build_dir),
        waves=True,
    )


if __name__ == "__main__":
    main()
