import os, sys
from pathlib import Path
from cocotb_tools.runner import get_runner
ROOT = Path(__file__).resolve().parents[2]
SIM = os.environ.get("SIM", "verilator")
runner = get_runner(SIM)
args = ["-g2012"] if SIM == "icarus" else ["--timing", "-Wno-fatal", "--public-flat-rw", "-Wno-WIDTHEXPAND"]
runner.build(sources=[ROOT/"rtl/bf16_pkg.sv", ROOT/"rtl/mxu_bf16_16x16.sv"],
             hdl_toplevel="mxu_bf16_16x16", build_args=args,
             build_dir=ROOT/"sim_build"/f"tb_mxu16_{SIM}", always=True)
sys.path.insert(0, str(ROOT/"tb"))
runner.test(hdl_toplevel="mxu_bf16_16x16", test_module="tb_mxu_bf16_16x16")
