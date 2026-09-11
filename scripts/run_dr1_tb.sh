#!/usr/bin/env bash
# =============================================================================
# run_dr1_tb.sh — ORBIT-DR1 RTL 테스트벤치 전부 실행
#
# 각 테스트벤치는 **골든 모델의 같은 이름 함수와 비트 비교**로만 통과한다
# (CLAUDE.md 규칙 2). 대응 관계:
#
#   rtl/dr1/requant_q15.sv   ↔  sim/golden/deltarule.py  requantize_q15()
#   rtl/dr1/matvec_unit.sv   ↔  sim/golden/deltarule.py  matvec()
#   rtl/dr1/update_unit.sv   ↔  sim/golden/deltarule.py  update_row()     (W6)
#   rtl/dr1/state_sram.sv    ↔  상태 배열 S + tools/orbit_pack.py 평탄화 규칙
#   rtl/dr1/vec_regs.sv      ↔  q/k/v 인자 + tools/orbit_pack.py
#   rtl/dr1/err_unit.sv      ↔  sim/golden/deltarule.py  compute_err()
#   rtl/dr1/dr1_top.sv       ↔  sim/golden/deltarule.py  step()  (전체 경로, W7)
#                               + tb/tb_dr1_top.py 하네스의 불변조건 I1
#
# 사용:  bash scripts/run_dr1_tb.sh ; echo $?     # 0 이어야 한다
# =============================================================================
set -u -o pipefail
cd "$(dirname "$0")/.."

DR1="rtl/dr1"
FAIL=0
RESULTS=""

run_one() {
  local top="$1" mod="$2"; shift 2
  local out
  out=$(python3 tb/run_tb.py "$top" "$mod" "$@" 2>&1)
  local line
  line=$(printf '%s\n' "$out" | grep -oE 'TESTS=[0-9]+ PASS=[0-9]+ FAIL=[0-9]+ SKIP=[0-9]+' | tail -1)
  if [ -z "$line" ]; then
    printf "  BUILDERR %-18s %s\n" "$mod" "$(printf '%s\n' "$out" | grep -m1 -E '%Error|Error:|AssertionError' | cut -c1-70)"
    FAIL=1; return
  fi
  local f
  f=$(printf '%s\n' "$line" | sed -nE 's/.*FAIL=([0-9]+).*/\1/p')
  if [ "$f" = "0" ]; then
    printf "  ok       %-18s %s\n" "$mod" "$line"
  else
    printf "  FAIL     %-18s %s\n" "$mod" "$line"
    printf '%s\n' "$out" | grep -m3 -E 'AssertionError' | sed 's/^/             /'
    FAIL=1
  fi
  RESULTS="$RESULTS$mod:$line
"
}

echo "=== ORBIT-DR1 RTL 테스트벤치 (골든 비트 비교) ==="
run_one requant_q15    tb_requant_q15   "$DR1/requant_q15.sv"
run_one state_sram     tb_state_sram    "$DR1/state_sram.sv"
run_one vec_regs       tb_vec_regs      "$DR1/vec_regs.sv"
run_one matvec_tb_wrap tb_matvec_unit   "$DR1/matvec_tb_wrap.sv" "$DR1/matvec_unit.sv" "$DR1/state_sram.sv" "$DR1/requant_q15.sv"
run_one update_unit    tb_update_unit   "$DR1/update_unit.sv" "$DR1/requant_q15.sv" rtl/mac_pe.sv
run_one err_unit       tb_err_unit      "$DR1/err_unit.sv" "$DR1/requant_q15.sv"
run_one dr1_scratch    tb_dr1_scratch   "$DR1/dr1_scratch.sv"

# DR1 최상위 — dr1_top + dr1_scratch 결선 (W7)
DR1_ALL="$DR1/dr1_tb_wrap.sv $DR1/dr1_top.sv $DR1/dr1_scratch.sv $DR1/state_sram.sv \
         $DR1/vec_regs.sv $DR1/matvec_unit.sv $DR1/update_unit.sv $DR1/err_unit.sv \
         $DR1/requant_q15.sv rtl/mac_pe.sv"
run_one dr1_tb_wrap    tb_dr1_top_fsm     $DR1_ALL
# 하네스(tb/tb_dr1_top.py)를 실 RTL 에 붙인 것 — I1 + 1/10/100토큰 비트 일치 (W7)
run_one dr1_tb_wrap    tb_dr1_harness_rtl $DR1_ALL

# 디스크립터·IRQ 경로 (BUG-001 회귀 확장)
run_one g2_ctrl_top    tb_g2_ctrl_top_dr1_fault $(ls rtl/*.sv rtl/dr1/*.sv)
# **호스트 스택 E2E** — OrbitDevice → CocotbBackend → g2_ctrl_top (W9·W10)
run_one g2_ctrl_top    tb_dr1_host_e2e          $(ls rtl/*.sv rtl/dr1/*.sv)
# 보드 경로의 앞단 — AXI4-Lite 브리지 (W11). 보드는 없지만 프로토콜은 지금 검증한다
run_one axil_reg_bridge tb_axil_reg_bridge      rtl/axil_reg_bridge.sv

echo ""
if [ "$FAIL" -eq 0 ]; then
  echo "=== 전부 통과 ==="
else
  echo "=== 실패 있음 ==="
fi
exit "$FAIL"
