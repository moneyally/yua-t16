"""STREAM_RD 모드 (FP32_OUT=0, PAR=8) 러너."""
import os, sys
from pathlib import Path
_SHIM = Path(__file__).resolve().parent / ".pyshim"
_SHIM.mkdir(exist_ok=True)
if not (_SHIM / "python3").exists():
    (_SHIM / "python3").symlink_to(sys.executable)
os.environ["PATH"] = f"{_SHIM}{os.pathsep}{os.environ.get('PATH','')}"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rtl_deps import deps
from cocotb_tools.runner import get_runner

ROOT = Path(__file__).resolve().parents[2]
runner = get_runner("verilator")
runner.build(
    sources=deps("mxu_bf16_128x128"),
    hdl_toplevel="mxu_bf16_128x128",
    parameters={"PAR": 8, "FP32_OUT": 0, "STREAM_RD": 1},
    build_args=["--timing", "-Wno-fatal", "--public-flat-rw",
                "-Wno-WIDTHEXPAND", "-Wno-TIMESCALEMOD", "--unroll-count", "4096"],
    build_dir=ROOT / "sim_build" / "build" / "mxu128_stream_rd",
    always=False,
)
sys.path.insert(0, str(ROOT / "sim/cocotb"))
res = ROOT / "sim_build" / "results"
res.mkdir(parents=True, exist_ok=True)
runner.test(hdl_toplevel="mxu_bf16_128x128", test_module="test_mxu128_stream_rd",
            results_xml=str(res / "test_mxu128_stream_rd.xml"))
