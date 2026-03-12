"""run_vpu_synth.py — cocotb runner for vpu_core_synth (DEPTH=256)"""
from __future__ import annotations
import os
import shutil
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root      = project_root()
    sim_dir   = root / "sim" / "cocotb"
    build_dir = root / "sim_build" / "vpu_core_synth"

    verilog_sources = [str(root / "rtl" / "vpu_core_synth.sv")]

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
        toplevel="vpu_core_synth",
        module="test_vpu_synth",
        python_search=[str(sim_dir)],
        verilog_sources=verilog_sources,
        compile_args=["-g2012", "-DCOCOTB_SIM=1"],
        sim_build=str(build_dir),
        waves=True,
    )


if __name__ == "__main__":
    main()
