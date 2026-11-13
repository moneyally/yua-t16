#!/usr/bin/env python3
"""run_all.py — tb/ 와 sim/cocotb/ 의 모든 cocotb 테스트벤치를 실행한다.

레포에는 테스트벤치가 36개 있었지만 러너는 2개뿐이었다. 나머지 34개는
실행될 수 없었다. 이 러너는 테스트벤치 이름에서 top 모듈을 추론하고
rtl_deps 로 의존 파일을 모아 전부 돌린다.

사용:
  python3.13 sim/cocotb/run_all.py              # 전체
  python3.13 sim/cocotb/run_all.py desc_queue   # 이름에 포함된 것만
"""
from __future__ import annotations
import os, re, sys, json, subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rtl_deps import module_table                            # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# 이름으로 추론되지 않는 것만 명시한다.
# G3 통합 테스트벤치는 OVERRIDES 에 적지 않는다 — 모듈과 테스트벤치 양쪽
# 헤더에 적힌 SSOT 이슈 ID(G3-INT-00N)로 자동 매칭한다. 이름으로 짐작하면
# tb_g3_int_single_layer(G3-INT-002)를 g3_int_top(G3-INT-001)에 잘못
# 붙이게 된다.
OVERRIDES: dict[str, str] = {
    "tb_g2_ctrl_top_host_e2e":   "g2_ctrl_top",
    "tb_cdc_fifo_async":         "cdc_fifo",
    "tb_cdc_fifo_reset":         "cdc_fifo",
    "test_mxu16_kstep":          "tb_mxu16_flat",
    "test_fxp_arith":            "fxp_dut",
    "test_mxu128_kstep":         "mxu_bf16_128x128",
    "test_mac_array":            "mac_array",
    "test_gemm_int4":            "gemm_int4",
    "test_gemm_top":             "gemm_top",
    "test_kvc":                  "kvc_core",
    "test_moe":                  "moe_router",
    "test_vpu":                  "vpu_core",
    "test_vpu_synth":            "vpu_core_synth",
}


_ID = re.compile(r"\b(G\d-[A-Z]+-\d+)\b")


def _ssot_index() -> dict[str, str]:
    """모듈 헤더의 SSOT 이슈 ID → 모듈 이름."""
    idx: dict[str, str] = {}
    for f in sorted(Path(ROOT / "rtl").glob("*.sv")):
        txt = f.read_text()
        m = re.search(r"^\s*module\s+([A-Za-z_]\w*)", txt, re.M)
        if not m:
            continue
        head = txt[: txt.find("module")] or txt[:800]
        for ident in _ID.findall(head):
            idx.setdefault(ident, m.group(1))
    return idx


_SSOT = None


def infer_top(stem: str, mods: set[str]) -> str | None:
    """SSOT 이슈 ID → 이름 접두사 순으로 top 을 찾는다."""
    global _SSOT
    if stem in OVERRIDES:
        return OVERRIDES[stem]

    if _SSOT is None:
        _SSOT = _ssot_index()
    for d in (ROOT / "tb", ROOT / "sim/cocotb"):
        f = d / f"{stem}.py"
        if f.exists():
            for ident in _ID.findall(f.read_text()[:600]):
                if ident in _SSOT:
                    return _SSOT[ident]
            break
    body = stem[3:] if stem.startswith("tb_") else stem
    parts = body.split("_")
    for n in range(len(parts), 0, -1):
        cand = "_".join(parts[:n])
        if cand in mods:
            return cand
    return None


# 같은 모듈의 다른 파라미터 구성도 회귀에 포함한다.
# (테스트벤치 이름 → top, 파라미터)
VARIANTS: dict[str, tuple[str, dict]] = {
    "test_mxu128_stream_rd": ("mxu_bf16_128x128",
                              {"PAR": 8, "FP32_OUT": 0, "STREAM_RD": 1}),
}


def discover() -> list[tuple[str, Path, str | None]]:
    mods = set(module_table())
    found = []
    for d, pat in ((ROOT / "tb", "tb_*.py"), (ROOT / "sim/cocotb", "test_*.py")):
        for f in sorted(d.glob(pat)):
            found.append((f.stem, f.parent, infer_top(f.stem, mods)))
    return found


def run_one(stem: str, tb_dir: Path, top: str) -> subprocess.CompletedProcess:
    """격리된 서브프로세스로 실행한다 (sys.path 오염 방지)."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    cmd = [sys.executable, str(ROOT / "sim/cocotb/run_one.py"), stem, top, str(tb_dir)]
    if stem in VARIANTS:
        cmd.append(json.dumps(VARIANTS[stem][1]))
    return subprocess.run(cmd, cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=1800)


def main() -> int:
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    report: list[dict] = []

    for stem, tb_dir, top in discover():
        if filt and filt not in stem:
            continue
        if stem in VARIANTS:
            top = VARIANTS[stem][0]
        if top is None:
            report.append({"tb": stem, "top": None, "status": "NO_TOP"})
            print(f"[SKIP] {stem:<34} top 모듈 추론 실패")
            continue
        print(f"[RUN ] {stem:<34} top={top}")
        try:
            cp = run_one(stem, tb_dir, top)
            if cp.returncode == 0:
                report.append({"tb": stem, "top": top, "status": "RAN"})
            else:
                err = ""
                for line in (cp.stdout + cp.stderr).splitlines():
                    if "%Error" in line or "error:" in line.lower():
                        err = line.strip()[:120]
                        break
                report.append({"tb": stem, "top": top, "status": "ERROR",
                               "err": err or f"returncode {cp.returncode}"})
                print(f"[FAIL] {stem:<34} {err or cp.returncode}")
        except subprocess.TimeoutExpired:
            report.append({"tb": stem, "top": top, "status": "TIMEOUT"})
            print(f"[FAIL] {stem:<34} 시간 초과")

    out = ROOT / "sim_build" / "run_all_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n리포트: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
