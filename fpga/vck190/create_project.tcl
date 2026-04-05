# create_project.tcl — ORBIT-G2 Proto-B VCK190 Vivado Project
# SSOT: ORBIT_G2_VCK190_PCIE_BRINGUP.md
#
# Usage: vivado -mode batch -source create_project.tcl
#    or: source create_project.tcl  (from Vivado Tcl console)
#
# Prerequisites:
#   - Vivado 2024.1+ (Versal support)
#   - VCK190 board files installed
#
# This script:
#   1. Creates project targeting VCK190 / VC1902
#   2. Sources CPM IP configuration
#   3. Adds RTL sources
#   4. Adds constraints
#   5. Sets up synthesis/implementation runs

set project_name "orbit_g2_protob"
set project_dir  "[file dirname [info script]]/../../build/vivado/${project_name}"
set rtl_dir      "[file dirname [info script]]/../../rtl"
set xdc_dir      "[file dirname [info script]]"

# ── Target board/part ─────────────────────────────────────────
# VCK190 board: xcvc1902-vsva2197-2MP-e-S
set part "xcvc1902-vsva2197-2MP-e-S"
set board_part "xilinx.com:vck190:part0:3.2"

puts "INFO: Creating project ${project_name} at ${project_dir}"
puts "INFO: Part: ${part}"

# Create project
create_project ${project_name} ${project_dir} -part ${part} -force
# Board part optional — skip if not installed
if {[llength [get_board_parts -quiet ${board_part}]] > 0} {
  set_property board_part ${board_part} [current_project]
  puts "INFO: Board part set to ${board_part}"
} else {
  puts "WARN: Board part ${board_part} not found. Using part-only mode."
}

# ── Add RTL sources ───────────────────────────────────────────
set rtl_files [list \
  "${rtl_dir}/g2_protob_top.sv" \
  "${rtl_dir}/pcie_ep_versal.sv" \
  "${rtl_dir}/dma_bridge.sv" \
  "${rtl_dir}/g2_ctrl_top.sv" \
  "${rtl_dir}/reg_top.sv" \
  "${rtl_dir}/reset_seq.sv" \
  "${rtl_dir}/desc_queue.sv" \
  "${rtl_dir}/desc_fsm_v2.sv" \
  "${rtl_dir}/gemm_top.sv" \
  "${rtl_dir}/ctrl_fsm.sv" \
  "${rtl_dir}/gemm_core.sv" \
  "${rtl_dir}/act_sram.sv" \
  "${rtl_dir}/wgt_sram.sv" \
  "${rtl_dir}/mac_array.sv" \
  "${rtl_dir}/mac_pe.sv" \
  "${rtl_dir}/oom_guard.sv" \
  "${rtl_dir}/trace_ring.sv" \
  "${rtl_dir}/irq_ctrl.sv" \
  "${rtl_dir}/cdc_fifo.sv" \
]

foreach f ${rtl_files} {
  if {[file exists ${f}]} {
    add_files -norecurse ${f}
    puts "INFO: Added ${f}"
  } else {
    puts "WARNING: RTL file not found: ${f}"
  }
}

# Set top module
set_property top g2_protob_top [current_fileset]

# ── Add constraints ───────────────────────────────────────────
set xdc_file "${xdc_dir}/vck190_pcie.xdc"
if {[file exists ${xdc_file}]} {
  add_files -fileset constrs_1 -norecurse ${xdc_file}
  puts "INFO: Added constraints: ${xdc_file}"
} else {
  puts "WARNING: Constraints file not found: ${xdc_file}"
}

# ── CPM IP creation ───────────────────────────────────────────
puts "INFO: Sourcing CPM IP configuration..."
source "[file dirname [info script]]/create_cpm_ip.tcl"

# ── Synthesis settings ────────────────────────────────────────
set_property strategy Flow_PerfOptimized_high [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY rebuilt [get_runs synth_1]

# ── Implementation settings ───────────────────────────────────
set_property strategy Performance_ExplorePostRoutePhysOpt [get_runs impl_1]

puts "INFO: Project creation complete."
puts "INFO: Next steps:"
puts "  1. Open project in Vivado GUI to verify CPM IP"
puts "  2. Generate output products for CPM IP"
puts "  3. Run synthesis:     launch_runs synth_1 -jobs 8"
puts "  4. Run implementation: launch_runs impl_1 -to_step write_bitstream -jobs 8"
