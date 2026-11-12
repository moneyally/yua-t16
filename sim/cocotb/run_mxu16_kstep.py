import sys
from pathlib import Path
from cocotb_tools.runner import get_runner
ROOT = Path(__file__).resolve().parents[2]
runner = get_runner("icarus")
runner.build(
    verilog_sources=[ROOT / "sim/cocotb/tb_mxu16_flat.sv", ROOT / "rtl/mxu_bf16_16x16.sv"],
    hdl_toplevel="tb_mxu16_flat", build_args=["-g2012"],
    build_dir=ROOT / "sim_build" / "mxu16_kstep", always=True)
sys.path.insert(0, str(ROOT / "sim/cocotb"))
runner.test(hdl_toplevel="tb_mxu16_flat", test_module="test_mxu16_kstep")
