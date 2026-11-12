"""기존 tb_mxu_bf16_128x128.py 회귀 — 이미 빌드된 PAR=8 산출물 재사용."""
import sys
from pathlib import Path
from cocotb_tools.runner import get_runner
ROOT = Path(__file__).resolve().parents[2]
runner = get_runner("verilator")
runner.build(
    sources=[ROOT/"rtl/bf16_pkg.sv", ROOT/"rtl/mxu_bf16_16x16.sv", ROOT/"rtl/mxu_bf16_128x128.sv"],
    hdl_toplevel="mxu_bf16_128x128",
    parameters={"PAR": 8},
    build_args=["--timing", "-Wno-fatal", "--public-flat-rw", "-Wno-WIDTHEXPAND"],
    build_dir=ROOT/"sim_build"/"mxu128_new_par8",
    always=False,          # 캐시 재사용
)
sys.path.insert(0, str(ROOT/"tb"))
runner.test(hdl_toplevel="mxu_bf16_128x128", test_module="tb_mxu_bf16_128x128")
