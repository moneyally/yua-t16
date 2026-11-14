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
W = int(os.environ.get("RD_DATA_W", "128"))
NB = int(os.environ.get("BANKS", "2"))
r.build(sources=deps("gemm_core"), hdl_toplevel="gemm_core",
        parameters={"RD_DATA_W": W, "BANKS": NB},
        build_args=["--timing","-Wno-fatal","--public-flat-rw",
                    "-Wno-WIDTHEXPAND","-Wno-TIMESCALEMOD","--unroll-count","4096"],
        build_dir=ROOT/"sim_build"/"build"/f"gemm_core_W{W}_B{NB}", always=False)
sys.path.insert(0, str(ROOT/"sim/cocotb"))
res = ROOT/"sim_build"/"results"; res.mkdir(parents=True, exist_ok=True)
r.test(hdl_toplevel="gemm_core", test_module="test_gemm_core_throughput",
       results_xml=str(res/"test_gemm_core_throughput.xml"))
