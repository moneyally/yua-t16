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
        return {"error": _why(r)}

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
