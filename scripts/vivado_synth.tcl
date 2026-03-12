# scripts/vivado_synth.tcl — Vivado synthesis for ORBIT-G1
# Target : Arty A7-100T (xc7a100tcsg324-1)
# Clock  : 150 MHz (6.667 ns)
#
# Usage:
#   cd /path/to/yua-t16
#   vivado -mode batch -source scripts/vivado_synth.tcl
#
# Outputs (written to reports/):
#   vpu_utilization.rpt   vpu_timing.rpt
#   gemm_utilization.rpt  gemm_timing.rpt
#   orbit_g1_full.rpt     orbit_g1_timing.rpt

set DEVICE  "xc7a100tcsg324-1"
set CLK_NS  6.667
set RTL_DIR [file normalize [file dirname [info script]]/../rtl]
set RPT_DIR [file normalize [file dirname [info script]]/../reports]

file mkdir $RPT_DIR

puts "============================================================"
puts " ORBIT-G1 Vivado Synthesis"
puts " Device  : $DEVICE"
puts " Clock   : 150 MHz ($CLK_NS ns)"
puts " RTL dir : $RTL_DIR"
puts "============================================================"

# ── Helper: run synthesis + report ─────────────────────────────────────────
proc synth_and_report {top src_files rpt_prefix rpt_dir} {
    set proj_name "proj_${top}"
    create_project $proj_name ./vivado_work/$proj_name -part $::DEVICE -force
    set_property target_language SystemVerilog [current_project]

    foreach f $src_files {
        add_files $f
        set_property file_type {SystemVerilog} [get_files [file tail $f]]
    }

    # Constraints: 150 MHz clock
    set xdc_file [file normalize [file dirname [info script]]/../scripts/constraints.xdc]
    if {[file exists $xdc_file]} {
        add_files -fileset constrs_1 $xdc_file
    } else {
        # Inline constraint if file not found
        set fd [open "/tmp/orbit_g1_clk.xdc" w]
        puts $fd "create_clock -period $::CLK_NS -name clk \[get_ports clk\]"
        puts $fd "set_false_path -from \[get_ports rst_n\]"
        close $fd
        add_files -fileset constrs_1 /tmp/orbit_g1_clk.xdc
    }

    set_property top $top [current_fileset]
    set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} \
                 -value {-mode out_of_context} \
                 -objects [get_runs synth_1]

    puts "INFO: Synthesizing $top..."
    launch_runs synth_1 -jobs 4
    wait_on_run synth_1

    if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
        puts "ERROR: $top synthesis failed"
        return
    }

    open_run synth_1 -name synth_1
    report_utilization   -file ${rpt_dir}/${rpt_prefix}_utilization.rpt -hierarchical
    report_timing_summary -file ${rpt_dir}/${rpt_prefix}_timing.rpt     -max_paths 20
    report_clock_interaction -file ${rpt_dir}/${rpt_prefix}_clk.rpt

    # Print quick summary to console
    puts ""
    puts "=== $top Utilization Summary ==="
    report_utilization -return_string -quiet
    puts ""
    close_project
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. VPU core synth (Q8.8 fixed-point, 256-wide SIMD)
# ══════════════════════════════════════════════════════════════════════════════
synth_and_report \
    vpu_core_synth \
    [list ${RTL_DIR}/vpu_core_synth.sv] \
    vpu \
    $RPT_DIR

# ══════════════════════════════════════════════════════════════════════════════
# 2. GEMM INT4 FPGA (LUT-based multiplier, DSP-minimal)
# ══════════════════════════════════════════════════════════════════════════════
synth_and_report \
    gemm_int4_fpga \
    [list ${RTL_DIR}/gemm_int4_fpga.sv] \
    gemm_fpga \
    $RPT_DIR

# ══════════════════════════════════════════════════════════════════════════════
# 3. GEMM INT4 synth (original, ASIC reference — expected DSP overflow on A7)
# ══════════════════════════════════════════════════════════════════════════════
synth_and_report \
    gemm_int4_synth \
    [list ${RTL_DIR}/gemm_int4_synth.sv] \
    gemm_orig \
    $RPT_DIR

puts ""
puts "============================================================"
puts " Synthesis Complete"
puts " Reports in: $RPT_DIR"
puts "   vpu_utilization.rpt      vpu_timing.rpt"
puts "   gemm_fpga_utilization.rpt gemm_fpga_timing.rpt"
puts "   gemm_orig_utilization.rpt gemm_orig_timing.rpt"
puts "============================================================"
