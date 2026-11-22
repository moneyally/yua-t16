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
D  = int(os.environ.get("DIM", "16"))
W  = int(os.environ.get("RD_DATA_W", "256"))
WW = int(os.environ.get("WR_DATA_W", "128"))
MK = int(os.environ.get("MAX_KT", "256"))
C  = int(os.environ.get("CORES", "2"))
TM = os.environ.get("TEST_MODULE", "test_mxu_cluster")
r.build(sources=deps("mxu_cluster"), hdl_toplevel="mxu_cluster",
        parameters={"DIM": D, "RD_DATA_W": W, "WR_DATA_W": WW,
                    "MAX_KT": MK, "CORES": C},
        build_args=["--timing","-Wno-fatal","--public-flat-rw",
                    "-Wno-WIDTHEXPAND","-Wno-TIMESCALEMOD","--unroll-count","65536"],
        build_dir=ROOT/"sim_build"/"build"/f"mxu_cluster_D{D}_W{W}_WW{WW}_K{MK}_C{C}",
        always=False)
sys.path.insert(0, str(ROOT/"sim/cocotb"))
res = ROOT/"sim_build"/"results"; res.mkdir(parents=True, exist_ok=True)
r.test(hdl_toplevel="mxu_cluster", test_module=TM, results_xml=str(res/f"{TM}.xml"))
