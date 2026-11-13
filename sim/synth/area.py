#!/usr/bin/env python3
"""area.py — 모듈의 합성 후 셀 수를 센다 (재설계 전/후 비교용).

사용: python3 sim/synth/area.py <top_module> [<top_module> ...]
"""
import re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim/cocotb"))
from rtl_deps import deps  # noqa: E402


# 합성에서 제외할 시뮬레이션 전용 소스 (real 타입 등 비합성 구문 사용)
SIM_ONLY = {"vpu_fp16_utils.sv"}


def area(top: str) -> dict:
    srcs = " ".join(str(p) for p in deps(top) if p.name not in SIM_ONLY)
    script = (ROOT / "sim/synth/area.ys.in").read_text() \
        .replace("@SOURCES@", srcs).replace("@TOP@", top)
    sp = ROOT / "sim_build" / "synth"
    sp.mkdir(parents=True, exist_ok=True)
    f = sp / f"{top}.ys"
    f.write_text(script)
    r = subprocess.run(["yosys", "-s", str(f)],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        return {"error": (r.stderr or r.stdout).strip().splitlines()[-1][:160]}

    out = r.stdout
    # 최상위 모듈의 통계 블록만 본다
    m = re.search(rf"=== {re.escape(top)} ===(.*?)(?==== |\Z)", out, re.S)
    body = m.group(1) if m else out
    def grab(key):
        mm = re.search(rf"{key}\s+(\d+)", body)
        return int(mm.group(1)) if mm else 0
    return {
        "cells": grab("Number of cells"),
        "wires": grab("Number of wires"),
        "memories": grab("Number of memory bits"),
    }


if __name__ == "__main__":
    for t in sys.argv[1:]:
        a = area(t)
        if "error" in a:
            print(f"{t:<24} 합성 실패: {a['error']}")
        else:
            print(f"{t:<24} 셀 {a['cells']:>9,}   와이어 {a['wires']:>9,}")
