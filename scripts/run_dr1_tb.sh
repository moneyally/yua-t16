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
#   rtl/dr1/dr1_top.sv       ↔  spec/deltarule.md 2·4절 (INIT/DUMP 만, W6)
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
run_one dr1_top        tb_dr1_top_fsm   "$DR1/dr1_top.sv" "$DR1/state_sram.sv"
# 하네스(tb/tb_dr1_top.py)를 실 RTL 에 붙인 것 — 불변조건 I1 + RTL 경로 오류 주입 (W6)
run_one dr1_top        tb_dr1_harness_rtl "$DR1/dr1_top.sv" "$DR1/state_sram.sv"
# 디스크립터·IRQ 경로에서의 DR1 fault (BUG-001 회귀 확장)
run_one g2_ctrl_top    tb_g2_ctrl_top_dr1_fault $(ls rtl/*.sv rtl/dr1/*.sv)

echo ""
if [ "$FAIL" -eq 0 ]; then
  echo "=== 전부 통과 ==="
else
  echo "=== 실패 있음 ==="
fi
exit "$FAIL"
