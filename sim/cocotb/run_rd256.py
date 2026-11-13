import os, sys
from pathlib import Path
_S = Path(__file__).resolve().parent / ".pyshim"; _S.mkdir(exist_ok=True)
if not (_S/"python3").exists(): (_S/"python3").symlink_to(sys.executable)
os.environ["PATH"] = f"{_S}{os.pathsep}{os.environ.get('PATH','')}"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rtl_deps import deps
from cocotb_tools.runner import get_runner
ROOT = Path(__file__).resolve().parents[2]
r = get_runner("verilator")
r.build(sources=deps("gemm_top"), hdl_toplevel="gemm_top",
        parameters={"RD_DATA_W": int(os.environ.get("RD_DATA_W","256"))},
        build_args=["--timing","-Wno-fatal","--public-flat-rw",
                    "-Wno-WIDTHEXPAND","-Wno-TIMESCALEMOD","--unroll-count","4096"],
        build_dir=ROOT/"sim_build"/"build"/f"gemm_top_RD_DATA_W{os.environ.get('RD_DATA_W','256')}", always=False)
sys.path.insert(0, str(ROOT/"sim/cocotb"))
res = ROOT/"sim_build"/"results"; res.mkdir(parents=True, exist_ok=True)
r.test(hdl_toplevel="gemm_top", test_module="test_gemm_rd256",
       results_xml=str(res/"test_gemm_rd256.xml"))
