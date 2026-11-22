#!/usr/bin/env python3
"""area_pe.py — PE 벤치 모듈들의 합성 후 셀 수를 비교한다."""
import re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = [ROOT / "rtl/bf16_pkg.sv",
       ROOT / "rtl/mxu_bf16_16x16.sv",
       ROOT / "sim/synth/pe_bench.sv", ROOT / "sim/synth/rd_bench.sv",
       ROOT / "sim/synth/tile_bench.sv"]


def _why(r) -> str:
    """합성이 왜 실패했는지 고른다.

    전에는 출력의 **마지막 줄**을 그대로 썼다. Yosys 는 진행 상황을
    계속 찍으므로, 죽었을 때 마지막 줄이 "14.1. Executing OPT_EXPR
    pass" 같은 진행 로그가 되어 원인이 전혀 보이지 않았다. 실제 원인을
    찾는 데 그 메시지는 아무 도움이 되지 않았다. ERROR 줄을 먼저 찾고,
    없으면 신호로 죽었는지까지 적는다.
    """
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    for line in out.splitlines():
        if "ERROR" in line:
            return line.strip()[:200]
    lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
    last = lines[-1][:120] if lines else "(출력 없음)"
    if r.returncode < 0:
        return (f"신호 {-r.returncode} 로 죽었다 (메모리/디스크 부족일 수 "
                f"있다). 마지막 진행: {last}")
    return f"rc={r.returncode}: " + " | ".join(t[:70] for t in lines[-3:])


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
        return {"error": _why(r)}
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
