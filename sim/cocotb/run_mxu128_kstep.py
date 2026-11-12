"""mxu_bf16_128x128 K-step 러너.

환경변수:
  MXU128_RTL  new | legacy   (기본 new)
  MXU128_PAR  PAR 값 (new 전용, 기본 8)
"""
import os, sys
from pathlib import Path
from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parents[2]
WHICH = os.environ.get("MXU128_RTL", "new")
PAR   = int(os.environ.get("MXU128_PAR", "8"))

if WHICH == "legacy":
    sources = [ROOT/"sim/legacy/mxu_bf16_16x16.sv", ROOT/"sim/legacy/mxu_bf16_128x128.sv"]
    params  = {}
    tag     = "legacy"
else:
    sources = [ROOT/"rtl/bf16_pkg.sv", ROOT/"rtl/mxu_bf16_16x16.sv", ROOT/"rtl/mxu_bf16_128x128.sv"]
    params  = {"PAR": PAR}
    tag     = f"new_par{PAR}"

runner = get_runner("verilator")
runner.build(
    sources=sources,
    hdl_toplevel="mxu_bf16_128x128",
    parameters=params,
    build_args=["--timing", "-Wno-fatal", "--public-flat-rw", "-Wno-WIDTHEXPAND"],
    build_dir=ROOT/"sim_build"/f"mxu128_{tag}",
    always=True,
)
sys.path.insert(0, str(ROOT/"sim/cocotb"))
runner.test(hdl_toplevel="mxu_bf16_128x128", test_module="test_mxu128_kstep")
