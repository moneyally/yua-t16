#!/usr/bin/env python3
"""area_pe.py — PE 벤치 모듈들의 합성 후 셀 수를 비교한다."""
import re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = [ROOT / "rtl/bf16_pkg.sv",
       ROOT / "sim/synth/pe_bench.sv", ROOT / "sim/synth/rd_bench.sv"]


def area(top: str) -> dict:
    srcs = " ".join(str(p) for p in SRC)
    script = (ROOT / "sim/synth/area.ys.in").read_text() \
        .replace("@SOURCES@", srcs).replace("@TOP@", top)
    sp = ROOT / "sim_build" / "synth"
    sp.mkdir(parents=True, exist_ok=True)
    f = sp / f"{top}.ys"
    f.write_text(script)
    r = subprocess.run(["yosys", "-s", str(f)],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        return {"error": (r.stderr or r.stdout).strip().splitlines()[-1][:200]}
    m = re.search(rf"===\s+{re.escape(top)}\s+===(.*?)(?=\n===|\Z)", r.stdout, re.S)
    body = m.group(1) if m else r.stdout
    def grab(k):
        mm = re.search(rf"Number of {k}:\s+(\d+)", body)
        return int(mm.group(1)) if mm else 0
    return {"cells": grab("cells"), "wires": grab("wires")}


if __name__ == "__main__":
    tops = sys.argv[1:] or ["pe_bench_fp32"]
    res = {}
    for t in tops:
        a = area(t)
        res[t] = a
        if "error" in a:
            print(f"{t:<22} 합성 실패: {a['error']}")
        else:
            per = "" if t.startswith(("tile_", "rd_", "conv_")) else f"   (타일 256개 기준 {a['cells']*256:>10,})"
            print(f"{t:<22} 셀 {a['cells']:>9,}{per}")
    if len(res) == 2 and all("cells" in v for v in res.values()):
        (n1, a1), (n2, a2) = res.items()
        print(f"\n{n1} → {n2}: 셀 {a1['cells']:,} → {a2['cells']:,} "
              f"({a1['cells']/a2['cells']:.2f}× 감소)")
