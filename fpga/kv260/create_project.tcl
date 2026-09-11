# create_project.tcl — ORBIT-DR1 KV260 Vivado 프로젝트 (PLAN W11)
#
# 사용:  vivado -mode batch -source fpga/kv260/create_project.tcl
#   또는: source fpga/kv260/create_project.tcl   (Vivado Tcl 콘솔에서)
#
# ──────────────────────────────────────────────────────────────────────────
# 이 스크립트가 하는 일과 **하지 않는 일**
# ──────────────────────────────────────────────────────────────────────────
# 한다:
#   - KV260 (xck26) 타깃 프로젝트 생성
#   - 합성 대상 RTL 추가 (rtl/*.sv + rtl/dr1/*.sv, behavioral 제외)
#   - top = dr1_soc_top (AXI4-Lite 슬레이브)
#   - **out-of-context 합성** — PS 블록 디자인 없이 PL 로직만 합성/구현해서
#     자원·타이밍을 먼저 본다. IO 버퍼를 안 만들므로 핀 배치가 필요 없다.
#
# 하지 않는다:
#   - Zynq MPSoC PS 블록 디자인 생성 (보드가 온 뒤에 한다)
#   - 비트스트림 생성
#
# ⚠ **이 스크립트는 한 번도 실행된 적이 없다.** 이 컨테이너에 Vivado 가 없다.
#   자원·타이밍 수치는 전부 미측정이다 — docs/FPGA.md 참조.
# ──────────────────────────────────────────────────────────────────────────

set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file normalize "${script_dir}/../.."]

set project_name "orbit_dr1_kv260"
set project_dir  "${repo_root}/build/vivado/${project_name}"

# KV260 Starter Kit: K26 SOM = XCK26-SFVC784-2LV-C
set part "xck26-sfvc784-2LV-c"

# 목표 주파수 (PLAN W11: 100MHz 목표, 안 나오면 낮추고 기록)
set target_period_ns 10.0

puts "INFO: project  = ${project_name}"
puts "INFO: part     = ${part}"
puts "INFO: 목표 주기 = ${target_period_ns} ns"

create_project ${project_name} ${project_dir} -part ${part} -force

# 보드 파일이 설치돼 있으면 붙인다 (없어도 part-only 로 진행)
set board_part "xilinx.com:kv260_som:part0:1.4"
if {[llength [get_board_parts -quiet ${board_part}]] > 0} {
  set_property board_part ${board_part} [current_project]
  puts "INFO: board_part = ${board_part}"
} else {
  puts "WARN: board_part ${board_part} 없음 — part 만으로 진행한다"
}

# ── RTL 추가 ───────────────────────────────────────────────────────────
# rtl/behavioral/ 은 **합성 대상이 아니다** (CLAUDE.md 2절: real 타입 격리).
set rtl_files [concat \
  [glob -nocomplain ${repo_root}/rtl/*.sv] \
  [glob -nocomplain ${repo_root}/rtl/*.v]  \
  [glob -nocomplain ${repo_root}/rtl/dr1/*.sv] \
]
if {[llength ${rtl_files}] == 0} {
  error "RTL 파일을 못 찾았다: ${repo_root}/rtl"
}
add_files -norecurse ${rtl_files}
set_property file_type "SystemVerilog" [get_files *.sv]
puts "INFO: RTL [llength ${rtl_files}] 개 추가"

# ── 제약 ───────────────────────────────────────────────────────────────
add_files -fileset constrs_1 -norecurse ${script_dir}/kv260_dr1.xdc

# ── top ────────────────────────────────────────────────────────────────
set_property top dr1_soc_top [current_fileset]
update_compile_order -fileset sources_1

# ── out-of-context 합성 ────────────────────────────────────────────────
# PS BD 없이 PL 만 본다. IO 버퍼를 만들지 않으므로 핀 제약이 필요 없다.
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MODE} -value {out_of_context} \
             -objects [get_runs synth_1]

# 면적을 먼저 보고 싶으므로 리소스 공유를 기본값으로 둔다 (튜닝은 나중)
puts "INFO: 프로젝트 생성 완료. 다음:"
puts "  launch_runs synth_1 -jobs 4; wait_on_run synth_1"
puts "  open_run synth_1; report_utilization; report_timing_summary"
puts "  launch_runs impl_1 -jobs 4; wait_on_run impl_1"
puts ""
puts "리포트는 docs/FPGA.md 에 **실제 출력 그대로** 붙인다 (CLAUDE.md 규칙 3)."
