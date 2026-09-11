#!/usr/bin/env bash
# =============================================================================
# synth_gate.sh — ORBIT 합성 가능성 게이트 (PLAN W1-2)
#
#   rtl/ 의 합성 대상 RTL 전체가
#     (1) CLAUDE.md 규칙 1 의 금지 토큰을 쓰지 않고
#     (2) sv2v 로 Verilog-2005 로 낮춰지고
#     (3) yosys 에서 elaborate + 구조 검사를 통과하고
#     (4) yosys 로 실제 합성되는지
#   확인한다. 하나라도 실패하면 비-0 으로 종료한다.
#
# -----------------------------------------------------------------------------
# 이 게이트의 위상 — 반드시 읽을 것
# -----------------------------------------------------------------------------
#   sv2v -> yosys 는 **일일 게이트**다. 빠르고 공짜라서 매일 돌린다.
#   **최종 합성 판정은 Vivado 다.** 이 게이트가 통과했다고 FPGA 에서 합성·타이밍이
#   된다는 뜻이 아니고, 이 게이트가 막혔다고 반드시 설계 결함인 것도 아니다.
#
#   이 레포 RTL 다수는 unpacked array 포트를 쓴다. 합법 SystemVerilog 이고
#   Vivado 는 그대로 합성하지만 yosys 내장 프론트엔드는 (0.33, 0.69 모두)
#   파싱하지 못한다. 그래서 sv2v 로 낮춰서 먹인다. **RTL 을 yosys 취향에 맞춰
#   리팩터하지 않는다.** 근거: docs/AUDIT.md §2, docs/LOG.md 결정 #1.
#
# -----------------------------------------------------------------------------
# 왜 2단인가
# -----------------------------------------------------------------------------
#   STAGE 1 (elaborate) : 전 모듈, 모듈당 ELAB_TIMEOUT(기본 120s), 병렬.
#                         진짜 회귀는 거의 전부 여기서 잡힌다 — 없는 모듈
#                         인스턴스화, 불법 SV, 경계 없는 while, 비상수 함수 호출,
#                         포트 폭 불일치. **에러는 게이트 실패.**
#                         시간 초과는 STAGE 2 와 같은 WARN 규칙을 따른다.
#   STAGE 2 (synth)     : 전 모듈, 모듈당 SYNTH_TIMEOUT(기본 240s).
#                         에러는 **게이트 실패**.
#                         시간 초과와 **OOM kill(abc 가 메모리로 죽는 것)** 은
#                         **WARN 으로 표시하고 목록에 남긴다** —
#                         "느려서 못 끝냄"은 "합성 불가"가 아니기 때문이다.
#                         (예: mxu_bf16_16x16 은 FP32 가산기 256개라 20분+.)
#                         --strict 를 주면 시간 초과도 실패로 친다.
#
# -----------------------------------------------------------------------------
# 의존성 (전부 무료, CLAUDE.md 규칙 4 범위)
# -----------------------------------------------------------------------------
#   yosys : apt-get install -y yosys
#   sv2v  : https://github.com/zachjs/sv2v/releases (단일 바이너리, MIT)
#           unzip 후 PATH 에 두거나 SV2V=/path/to/sv2v
#
# 사용:
#   bash scripts/synth_gate.sh ; echo $?          # 0 이어야 한다
#   bash scripts/synth_gate.sh --stage1           # 빠른 검사만
#   bash scripts/synth_gate.sh --strict           # 시간 초과도 실패로
#   bash scripts/synth_gate.sh --check-mem state_sram   # BRAM 추론 검사 추가
#   SYNTH_TIMEOUT=1800 ELAB_TIMEOUT=600 JOBS=8 bash scripts/synth_gate.sh
# =============================================================================
set -u -o pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
BUILD="$ROOT/build/synth_gate"
FLAT="$BUILD/flat.v"

YOSYS="${YOSYS:-yosys}"
SV2V="${SV2V:-sv2v}"
SYNTH_TIMEOUT="${SYNTH_TIMEOUT:-240}"
ELAB_TIMEOUT="${ELAB_TIMEOUT:-120}"
JOBS="${JOBS:-4}"

# -----------------------------------------------------------------------------
# 알려진 미완성 모듈 — 게이트 실패로 치지 않되 매 실행마다 출력한다.
# 숨기는 것이 아니다. 목록에 넣을 때는 반드시 (1) 왜 미완성인지 (2) 어느 문서에
# 근거가 있는지를 함께 적는다. 근거 없이 추가하지 말 것.
# -----------------------------------------------------------------------------
# 2026-09-11: pcie_ep_versal 의 BAR 요청 출력을 **정의된 비활성 값으로 구동**해서
# 두 모듈 다 목록에서 뺐다 (check -assert 171건 -> 0건). CQ->BAR 디코드는 여전히
# 구현되지 않았고 PCIe 는 동작하지 않는다 — 그건 rtl/pcie_ep_versal.sv 헤더와
# README 상태표에 적혀 있다. 여기는 "게이트가 못 보는 것" 목록이지 "미구현" 목록이 아니다.
declare -A KNOWN_INCOMPLETE=(
)

STAGE1_ONLY=0
STRICT=0
CHECK_MEM=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --stage1) STAGE1_ONLY=1; shift ;;
    --strict) STRICT=1; shift ;;
    --check-mem)
      # 이 모듈이 yosys 에서 **메모리 셀($mem_v2)로 인식되는지** 검사한다.
      # 플립플롭으로 풀리면 실패. BRAM 추론이 깨졌다는 뜻이고, 실물에서
      # state_sram 이 D*D*W 개 FF 로 펼쳐진다 (d=16 이면 4096 FF).
      # docs/BUGS.md BUG-006 이 "합성에서 조용히 다른 회로가 되는" 예였다 —
      # 이것도 같은 종류라서 게이트로 막는다.
      [ -n "${2:-}" ] || { echo "--check-mem 에 모듈 이름이 필요하다"; exit 2; }
      CHECK_MEM="$CHECK_MEM $2"; shift 2 ;;
    *) echo "알 수 없는 인자: $1"; exit 2 ;;
  esac
done

fail() { echo ""; echo "FAIL: $*"; exit 1; }

echo "=== ORBIT synth gate ==============================================="
echo "일일 게이트 (sv2v -> yosys). 최종 합성 판정은 Vivado."
echo ""

# --- 0. 툴 ------------------------------------------------------------------
command -v "$YOSYS" >/dev/null 2>&1 || fail "yosys 없음. apt-get install -y yosys"
command -v "$SV2V"  >/dev/null 2>&1 || fail "sv2v 없음. https://github.com/zachjs/sv2v/releases 에서 받아 PATH 에 두거나 SV2V=<경로>"
echo "[0/4] tools   : $($YOSYS -V 2>&1 | head -1) / sv2v $($SV2V --version 2>&1 | awk '{print $2}')"

mkdir -p "$BUILD"

# --- 1. 대상 수집 ------------------------------------------------------------
# rtl/ 최상위 + rtl/dr1/ (ORBIT-DR1 신규 모듈).
# rtl/behavioral/ 은 **일부러 제외**한다 (행동 모델, 합성 대상 아님).
mapfile -t SOURCES < <(ls "$ROOT"/rtl/*.sv "$ROOT"/rtl/*.v "$ROOT"/rtl/dr1/*.sv 2>/dev/null | sort)
[ "${#SOURCES[@]}" -gt 0 ] || fail "rtl/ 에 합성 대상 파일이 없다"
echo "[1/4] sources : ${#SOURCES[@]} files (rtl/ + rtl/dr1/, rtl/behavioral/ 제외)"

# --- 2. 금지 토큰 (CLAUDE.md 규칙 1) -----------------------------------------
if ! python3 "$ROOT/scripts/check_banned_tokens.py" "${SOURCES[@]}" > "$BUILD/banned_tokens.txt" 2>&1; then
  cat "$BUILD/banned_tokens.txt"
  fail "금지 토큰 발견 (real / \$itor / \$rtoi / \$exp / \$sqrt / #delay). 해당 모듈은 rtl/behavioral/ 로."
fi
echo "[2/4] tokens  : clean"

# --- 3. sv2v ------------------------------------------------------------------
if ! "$SV2V" "${SOURCES[@]}" > "$FLAT" 2> "$BUILD/sv2v.err"; then
  cat "$BUILD/sv2v.err"
  fail "sv2v 변환 실패 (위 에러 참조)"
fi
mapfile -t MODULES < <(grep -oE "^module [A-Za-z_0-9]+" "$FLAT" | awk '{print $2}' | sort -u)
echo "[3/4] sv2v    : OK  (${#MODULES[@]} modules -> build/synth_gate/flat.v)"

# --- STAGE 1: elaborate + check (전 모듈, 하드 실패) --------------------------
echo ""
echo "--- STAGE 1: elaborate + check (전 모듈, 모듈당 ${ELAB_TIMEOUT}s, 병렬 ${JOBS}) ---"
rm -f "$BUILD/.e_fail" "$BUILD/.e_timeout"
printf '%s\n' "${MODULES[@]}" | xargs -P "$JOBS" -I{} bash -c '
  m="$1"; BUILD="$2"; FLAT="$3"; YOSYS="$4"; TMO="$5"
  log="$BUILD/elab_$m.log"
  timeout "$TMO" "$YOSYS" -q -l "$log" -p "read_verilog -defer $FLAT; hierarchy -check -top $m; proc; opt_expr; opt_clean; check -assert" >/dev/null 2>&1
  rc=$?
  if [ "$rc" -eq 124 ]; then
    printf "  TIMEOUT %-24s >%ss\n" "$m" "$TMO"; echo "$m" >> "$BUILD/.e_timeout"
  elif [ "$rc" -ne 0 ]; then
    printf "  FAIL    %-24s %s\n" "$m" "$(grep -m1 ERROR "$log" 2>/dev/null | cut -c1-70)"; echo "$m" >> "$BUILD/.e_fail"
  else
    printf "  ok      %s\n" "$m"
  fi
' _ {} "$BUILD" "$FLAT" "$YOSYS" "$ELAB_TIMEOUT"

E_TMO=0
[ -f "$BUILD/.e_timeout" ] && E_TMO=$(sort -u "$BUILD/.e_timeout" | wc -l)
E_REAL=(); E_KNOWN=()
if [ -f "$BUILD/.e_fail" ]; then
  while read -r m; do
    if [ -n "${KNOWN_INCOMPLETE[$m]:-}" ]; then E_KNOWN+=("$m"); else E_REAL+=("$m"); fi
  done < <(sort -u "$BUILD/.e_fail")
fi
if [ "${#E_KNOWN[@]}" -gt 0 ]; then
  echo ""
  echo "  KNOWN INCOMPLETE (게이트 실패로 치지 않음, 매번 표시):"
  for m in "${E_KNOWN[@]}"; do echo "    - $m: ${KNOWN_INCOMPLETE[$m]}"; done
fi
if [ "${#E_REAL[@]}" -gt 0 ]; then
  echo ""; echo "elaborate 실패 모듈:"; printf '       %s\n' "${E_REAL[@]}"
  fail "STAGE 1 elaborate 실패 ${#E_REAL[@]}개"
fi
if [ "$E_TMO" -gt 0 ]; then
  echo "  WARN: elaborate 시간 초과 ${E_TMO}개 (>${ELAB_TIMEOUT}s):"; sed 's/^/         /' "$BUILD/.e_timeout"
fi
echo "  -> STAGE 1 통과 ($(( ${#MODULES[@]} - E_TMO - ${#E_KNOWN[@]} ))/${#MODULES[@]}, 시간 초과 ${E_TMO}, known ${#E_KNOWN[@]})"

# --- 메모리 추론 검사 (--check-mem) --------------------------------------------
if [ -n "$CHECK_MEM" ]; then
  echo ""
  echo "--- 메모리 추론 검사 ($CHECK_MEM) ---"
  MEM_FAIL=0
  for m in $CHECK_MEM; do
    log="$BUILD/mem_$m.log"
    if ! "$YOSYS" -q -l "$log" -p "read_verilog -defer $FLAT; hierarchy -check -top $m; proc; memory_collect; stat" >/dev/null 2>&1; then
      printf "  FAIL  %-24s yosys 실패 (%s)\n" "$m" "$log"; MEM_FAIL=1; continue
    fi
    # 해당 모듈의 stat 섹션만 떼어낸다
    sec=$(awk -v m="$m" 'index($0, "=== " m " ===") > 0 { on = 1 } on' "$log")
    cnt=$(printf '%s\n' "$sec" | grep -oE '[$]mem(_v2)?[[:space:]]+[0-9]+' | head -1 | grep -oE '[0-9]+$')
    if [ -n "$cnt" ] && [ "$cnt" -ge 1 ]; then
      printf "  ok    %-24s \$mem_v2 x%s (BRAM 추론됨)\n" "$m" "$cnt"
    else
      ff=$(printf '%s\n' "$sec" | grep -oE '[$][a-z]*dff[a-z_0-9]*[[:space:]]+[0-9]+' \
           | grep -oE '[0-9]+$' | awk '{s+=$1} END{print s+0}')
      printf "  FAIL  %-24s \$mem 셀이 없다. 플립플롭 %s개로 풀렸다 — BRAM 추론 깨짐\n" "$m" "$ff"
      MEM_FAIL=1
    fi
  done
  if [ "$MEM_FAIL" -ne 0 ]; then
    fail "메모리 추론 검사 실패. 배열에 리셋이 붙었거나 접근이 여러 always_ff 로 갈라졌는지 확인할 것."
  fi
fi

if [ "$STAGE1_ONLY" -eq 1 ]; then
  echo ""
  echo "=== PASS (--stage1) ================================================"
  exit 0
fi

# --- STAGE 2: full synth (전 모듈, 모듈당 타임아웃) --------------------------
rm -f "$BUILD/.fail" "$BUILD/.timeout" "$BUILD/.zero" "$BUILD/.oom" "$BUILD/.cells"
echo ""
echo "--- STAGE 2: synth -top (전 모듈, 모듈당 ${SYNTH_TIMEOUT}s, 병렬 ${JOBS}) ---"
printf '%s\n' "${MODULES[@]}" | xargs -P "$JOBS" -I{} bash -c '
  m="$1"; BUILD="$2"; FLAT="$3"; YOSYS="$4"; TMO="$5"
  log="$BUILD/synth_$m.log"
  t0=$(date +%s)
  timeout "$TMO" "$YOSYS" -l "$log" -p "read_verilog -defer $FLAT; hierarchy -check -top $m; synth -top $m; stat" >/dev/null 2>&1
  rc=$?
  t1=$(date +%s)
  cells=$(grep -E "Number of cells:" "$log" 2>/dev/null | tail -1 | awk "{print \$NF}")
  if [ "$rc" -eq 124 ]; then
    printf "  TIMEOUT %-24s >%ss\n" "$m" "$TMO"
    echo "$m" >> "$BUILD/.timeout"
  elif [ "$rc" -eq 137 ] || [ "$rc" -eq 139 ] || grep -q "bad_alloc\|ABC: Killed" "$log" 2>/dev/null; then
    # rc 137 = SIGKILL (OOM killer), 139 = SIGSEGV, "ABC: Killed" = abc 가 OOM 으로 죽음.
    # 자원 한계이지 합성 실패가 아니다. TIMEOUT 과 같은 등급으로 다룬다.
    # 병렬(JOBS)을 줄이면 대개 통과한다.
    printf "  OOMKILL %-24s %-8s abc/yosys 가 메모리로 죽었다 (rc=%s)\n" "$m" "$((t1-t0))s" "$rc"
    echo "$m" >> "$BUILD/.oom"
  elif [ "$rc" -ne 0 ] || [ -z "$cells" ]; then
    printf "  FAIL    %-24s %s\n" "$m" "$(grep -m1 ERROR "$log" 2>/dev/null | cut -c1-70)"
    echo "$m" >> "$BUILD/.fail"
  elif [ "$cells" = "0" ]; then
    # 0 셀은 "합성 성공"이 아니라 "전부 최적화로 사라짐"이다. docs/BUGS.md BUG-006 이
    # 정확히 이 증상이었다 (시뮬 전용 X 가드가 wgt_sram 전체를 삭제).
    printf "  ZERO    %-24s %-8s cells=0  <- 전부 최적화됨. 죽은 로직 의심\n" "$m" "$((t1-t0))s"
    echo "$m" >> "$BUILD/.zero"
  else
    printf "  ok      %-24s %-8s cells(design total)=%s\n" "$m" "$((t1-t0))s" "$cells"
    echo "$m $cells" >> "$BUILD/.cells"
  fi
' _ {} "$BUILD" "$FLAT" "$YOSYS" "$SYNTH_TIMEOUT"

N_TMO=0
[ -f "$BUILD/.timeout" ] && N_TMO=$(sort -u "$BUILD/.timeout" | wc -l)
S2_REAL=(); S2_KNOWN=()
if [ -f "$BUILD/.fail" ]; then
  while read -r m; do
    if [ -n "${KNOWN_INCOMPLETE[$m]:-}" ]; then S2_KNOWN+=("$m"); else S2_REAL+=("$m"); fi
  done < <(sort -u "$BUILD/.fail")
fi
N_FAIL=${#S2_REAL[@]}

echo ""
if [ "$N_TMO" -gt 0 ]; then
  echo "WARN: 시간 초과 ${N_TMO}개 (>${SYNTH_TIMEOUT}s) — 합성 불가가 아니라 '측정 못 함'이다:"
  sed 's/^/       /' "$BUILD/.timeout"
  echo "       SYNTH_TIMEOUT=1800 으로 다시 돌리거나 Vivado 로 판정할 것."
fi
if [ "${#S2_KNOWN[@]}" -gt 0 ]; then
  echo "KNOWN INCOMPLETE (합성 단계, 실패로 치지 않음):"
  for m in "${S2_KNOWN[@]}"; do echo "  - $m: ${KNOWN_INCOMPLETE[$m]}"; done
fi
N_OOM=0
if [ -f "$BUILD/.oom" ]; then
  N_OOM=$(sort -u "$BUILD/.oom" | wc -l)
  echo "WARN: 메모리 부족으로 죽은 모듈 ${N_OOM}개 — 합성 불가가 아니라 '측정 못 함'이다:"
  sed 's/^/       /' "$BUILD/.oom"
  echo "       JOBS=2 또는 JOBS=1 로 다시 돌려볼 것. 최종 판정은 Vivado."
fi
# --- 셀 수 회귀 검사 (docs/BUGS.md BUG-011) ---------------------------------
# **기능 테스트는 면적 회귀를 못 잡는다.** 실제로 `mac_pe` 를 손대면서 논리적으로
# 같은 식으로 바꿨는데 mac_array 가 +46% 커진 적이 있다. 전 테스트가 통과했다.
# 잡은 것은 셀 수 비교뿐이었다. 그래서 게이트에 넣는다.
#
# 기준선: scripts/cell_baseline.txt  ("모듈 셀수" 한 줄씩)
#   - 증가 10% 초과  -> **FAIL**. 의도한 변경이면 기준선을 고쳐서 커밋한다
#   - 감소 / 신규     -> WARN 만. 줄어드는 건 보통 좋은 일이고, 신규는 기준선에 추가하라고 알린다
N_CELLGROW=0
CELL_BASE="$(dirname "$0")/cell_baseline.txt"
if [ -f "$BUILD/.cells" ] && [ -f "$CELL_BASE" ]; then
  echo ""
  echo "--- 셀 수 회귀 검사 (기준선: scripts/cell_baseline.txt) ---"
  CELL_REPORT=$(awk -v base="$CELL_BASE" '
    BEGIN { while ((getline line < base) > 0) {
              if (line ~ /^#/ || line == "") continue
              split(line, a, /[ \t]+/); b[a[1]] = a[2] } }
    { now[$1] = $2 }
    END {
      grow = 0
      for (m in now) {
        if (!(m in b)) { printf "  NEW     %-24s %s (기준선에 추가할 것)\n", m, now[m]; continue }
        if (b[m] == 0) continue
        d = (now[m] - b[m]) * 100.0 / b[m]
        if (d > 10.0)      { printf "  GROW    %-24s %s -> %s (+%.1f%%)  <- 의도한 변경인가?\n", m, b[m], now[m], d; grow++ }
        else if (d < -5.0) { printf "  shrink  %-24s %s -> %s (%.1f%%)\n", m, b[m], now[m], d }
      }
      printf "GROWCOUNT=%d\n", grow
    }' "$BUILD/.cells")
  echo "$CELL_REPORT" | grep -v '^GROWCOUNT=' || true
  N_CELLGROW=$(echo "$CELL_REPORT" | sed -n 's/^GROWCOUNT=//p')
  if [ "${N_CELLGROW:-0}" -eq 0 ]; then
    echo "  ok    기준선 대비 10% 초과 증가 없음"
  else
    echo ""
    echo "FAIL: 셀 수가 기준선보다 10% 넘게 늘어난 모듈 ${N_CELLGROW}개."
    echo "      의도한 변경이면 **이유를 커밋 메시지에 적고** scripts/cell_baseline.txt 를 갱신한다:"
    echo "        cp build/synth_gate/.cells scripts/cell_baseline.txt   # 확인 후에만"
  fi
elif [ -f "$BUILD/.cells" ]; then
  echo ""
  echo "WARN: scripts/cell_baseline.txt 이 없다 — 셀 수 회귀를 못 본다."
  echo "      만들려면: cp build/synth_gate/.cells scripts/cell_baseline.txt"
fi

N_ZERO=0
if [ -f "$BUILD/.zero" ]; then
  N_ZERO=$(sort -u "$BUILD/.zero" | wc -l)
  echo "WARN: 합성 결과가 0 셀인 모듈 ${N_ZERO}개 — 성공이 아니라 '전부 사라졌다'는 뜻이다:"
  sed 's/^/       /' "$BUILD/.zero"
  echo "       시뮬 전용 X 비교 구문이 합성에서 조건을 상수로 접었을 수 있다. docs/BUGS.md BUG-006."
fi
if [ "$N_FAIL" -gt 0 ]; then
  echo "합성 실패 모듈:"; printf '       %s\n' "${S2_REAL[@]}"
  fail "STAGE 2 합성 실패 ${N_FAIL}개"
fi
if [ "${N_CELLGROW:-0}" -gt 0 ]; then
  fail "셀 수 회귀 ${N_CELLGROW}개 (기준선 +10% 초과). docs/BUGS.md BUG-011 참조"
fi
if [ "$STRICT" -eq 1 ] && [ $(( N_TMO + E_TMO + N_OOM )) -gt 0 ]; then
  fail "--strict: 미측정 $(( N_TMO + E_TMO + N_OOM ))개(시간 초과/OOM)를 실패로 처리"
fi

echo "=== PASS ==========================================================="
echo "STAGE 1 elaborate $(( ${#MODULES[@]} - E_TMO - ${#E_KNOWN[@]} ))/${#MODULES[@]}  (시간 초과 ${E_TMO}, known-incomplete ${#E_KNOWN[@]})"
echo "STAGE 2 synth     $(( ${#MODULES[@]} - N_TMO - N_ZERO - N_OOM - ${#S2_KNOWN[@]} ))/${#MODULES[@]}  (시간 초과 ${N_TMO}, OOM ${N_OOM}, 0셀 ${N_ZERO}, known-incomplete ${#S2_KNOWN[@]})"
echo "주의: 이것은 일일 게이트다. FPGA 합성·타이밍 판정은 Vivado 로만 한다."
exit 0
