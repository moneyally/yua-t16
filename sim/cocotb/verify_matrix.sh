#!/usr/bin/env bash
# mxu_bf16_128x128 전체 검증 매트릭스.
#   - legacy  : 원본 RTL (K>1 버그 재현)
#   - PAR=1/2/4/8 : 새 RTL 전 구성
# 각 구성마다 K-step 테스트 + 기존 회귀 테스트를 돌린다.
set -u
export PATH=/opt/verilator/bin:$PATH
cd "$(dirname "$0")/../.."
OUT=sim_build/verify_report.txt
: > "$OUT"

run() {  # run <라벨> <환경변수들...>
  local label="$1"; shift
  echo "════════ $label ════════" | tee -a "$OUT"
  env "$@" PYTHONPATH=sim/cocotb python3.13 sim/cocotb/run_mxu128_kstep.py 2>&1 \
    | grep -E "PERF|TESTS=" | tee -a "$OUT"
}

run "원본 RTL (legacy)"      MXU128_RTL=legacy
for p in 1 2 4 8; do
  run "새 RTL PAR=$p"        MXU128_RTL=new MXU128_PAR=$p
done

echo "════════ 기존 테스트벤치 회귀 (PAR=8) ════════" | tee -a "$OUT"
PYTHONPATH=tb python3.13 sim/cocotb/run_mxu128_regress.py 2>&1 | grep -E "TESTS=" | tee -a "$OUT"

echo "════════ 16×16 타일 회귀 ════════" | tee -a "$OUT"
PYTHONPATH=tb SIM=verilator python3.13 sim/cocotb/run_tb_mxu16.py 2>&1 | grep -E "TESTS=" | tee -a "$OUT"

echo; echo "리포트: $OUT"
