"""run_kvc.py — cocotb runner for kvc_core"""
from __future__ import annotations
import os, sys, shutil
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root      = project_root()
    sim_dir   = root / "sim" / "cocotb"
    build_dir = root / "sim_build" / "kvc_core"

    verilog_sources = [str(root / "rtl" / "kvc_core.sv")]

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("COCOTB_LOG_LEVEL", "INFO")
    os.environ.setdefault("COCOTB_REDUCED_LOG_FMT", "0")

    clean = os.environ.get("CLEAN", "1")
    if clean != "0" and build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    from cocotb_test.simulator import run
    run(
        simulator="icarus",
        toplevel="kvc_core",
        module="test_kvc",
        python_search=[str(sim_dir)],
        verilog_sources=verilog_sources,
        compile_args=["-g2012", "-DCOCOTB_SIM=1"],
        sim_build=str(build_dir),
        waves=True,
    )


if __name__ == "__main__":
    main()
