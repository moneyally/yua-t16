"""
run_gemm_int4.py — cocotb-test runner for gemm_int4 module
YUA-T16 v2 INT4 GEMM simulation
"""
from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path

from cocotb_test.simulator import run


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root = project_root()
    sim_dir = root / "sim" / "cocotb"
    build_dir = root / "sim_build" / "gemm_int4"

    verilog_sources = [
        root / "rtl" / "gemm_int4.sv",
    ]

    missing = [p for p in verilog_sources if not p.exists()]
    if missing:
        print("[run_gemm_int4] Missing verilog_sources:", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(2)

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("COCOTB_LOG_LEVEL", "INFO")
    os.environ.setdefault("COCOTB_REDUCED_LOG_FMT", "0")
    os.environ.setdefault("PYTHONWARNINGS", "default")

    # Clean build
    clean = os.environ.get("CLEAN", "1")
    if clean != "0" and build_dir.exists():
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)

    print("[run_gemm_int4] root      :", root)
    print("[run_gemm_int4] sim_dir   :", sim_dir)
    print("[run_gemm_int4] build_dir :", build_dir)
    print("[run_gemm_int4] sources   :")
    for p in verilog_sources:
        print("  -", p)
    sys.stdout.flush()

    run(
        simulator="icarus",
        toplevel="gemm_int4",
        module="test_gemm_int4",
        python_search=[str(sim_dir)],
        verilog_sources=[str(p) for p in verilog_sources],
        compile_args=["-g2012", "-DCOCOTB_SIM=1"],
        sim_build=str(build_dir),
        waves=True,
    )


if __name__ == "__main__":
    main()
