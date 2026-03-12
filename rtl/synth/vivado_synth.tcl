# vivado_synth.tcl — Vivado synthesis script for ORBIT-G1 components
# Target: Arty A7-100T (xc7a100tcsg324-1)
#
# Usage:
#   cd rtl/synth
#   vivado -mode batch -source vivado_synth.tcl
#
# Outputs:
#   vivado_project/   — Vivado project directory
#   vpu_utilization.rpt
#   vpu_timing.rpt
#   gemm_utilization.rpt
#   gemm_timing.rpt

set project_name "orbit_g1_synth"
set device       "xc7a100tcsg324-1"

# ── Create project ────────────────────────────────────────────────────────────
create_project $project_name ./vivado_project -part $device -force

set_property target_language SystemVerilog [current_project]

# ── Add RTL sources ───────────────────────────────────────────────────────────
add_files [list \
  ../../rtl/vpu_core_synth.sv \
  ../../rtl/gemm_int4_synth.sv \
]
set_property file_type {SystemVerilog} [get_files *.sv]

# ── Add constraints ───────────────────────────────────────────────────────────
add_files -fileset constrs_1 constraints.xdc

# ═══════════════════════════════════════════════════════════════════════════════
# Synthesize vpu_core_synth (out-of-context)
# ═══════════════════════════════════════════════════════════════════════════════
set_property top vpu_core_synth [current_fileset]
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} \
             -value {-mode out_of_context} \
             -objects [get_runs synth_1]

puts "INFO: Launching synthesis for vpu_core_synth..."
launch_runs synth_1 -jobs 4
wait_on_run synth_1

if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
  puts "ERROR: vpu_core_synth synthesis failed"
  exit 1
}

open_run synth_1 -name synth_1
report_utilization   -file vpu_utilization.rpt  -hierarchical
report_timing_summary -file vpu_timing.rpt       -max_paths 10
puts "INFO: vpu_core_synth synthesis complete."
puts "INFO:   Utilization -> vpu_utilization.rpt"
puts "INFO:   Timing      -> vpu_timing.rpt"

# ═══════════════════════════════════════════════════════════════════════════════
# Synthesize gemm_int4_synth (new run)
# ═══════════════════════════════════════════════════════════════════════════════
create_run synth_gemm -flow {Vivado Synthesis 2024} -parent_run impl_1
set_property top gemm_int4_synth [get_runs synth_gemm]
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} \
             -value {-mode out_of_context} \
             -objects [get_runs synth_gemm]

puts "INFO: Launching synthesis for gemm_int4_synth..."
launch_runs synth_gemm -jobs 4
wait_on_run synth_gemm

if {[get_property PROGRESS [get_runs synth_gemm]] != "100%"} {
  puts "ERROR: gemm_int4_synth synthesis failed"
  exit 1
}

open_run synth_gemm -name synth_gemm
report_utilization   -file gemm_utilization.rpt  -hierarchical
report_timing_summary -file gemm_timing.rpt       -max_paths 10
puts "INFO: gemm_int4_synth synthesis complete."
puts "INFO:   Utilization -> gemm_utilization.rpt"
puts "INFO:   Timing      -> gemm_timing.rpt"

puts ""
puts "═══════════════════════════════════════════════════════"
puts " ORBIT-G1 Synthesis Complete"
puts " Device  : $device"
puts " Reports : vpu_utilization.rpt  vpu_timing.rpt"
puts "           gemm_utilization.rpt gemm_timing.rpt"
puts "═══════════════════════════════════════════════════════"
