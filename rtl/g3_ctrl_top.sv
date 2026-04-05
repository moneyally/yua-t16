// g3_ctrl_top.sv — ORBIT-G3 Control Plane Top (Phase 1 Skeleton)
// SSOT: ORBIT_G3_RTL_PLAN.md (G3-RTL-006), ORBIT_G3_REG_SPEC.md
//
// Minimal skeleton: g3_reg_top instantiated, MXU status wired,
// backward/optimizer/collective/fabric tied off.
// G2 control plane is NOT instantiated here (separate module).
// This top is the G3 register+compute shell only.
`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off PINCONNECTEMPTY */

module g3_ctrl_top (
  input  logic        clk,
  input  logic        rst_n,

  // Register bus
  input  logic [19:0] reg_addr,
  input  logic        reg_wr_en,
  input  logic [31:0] reg_wr_data,
  output logic [31:0] reg_rd_data,

  // MXU status (from mxu_bf16_16x16 or future mxu path)
  input  logic [31:0] mxu_status_in,
  input  logic [31:0] mxu_err_code_in,
  input  logic [31:0] mxu_tile_cnt_in,

  // MXU command (to mxu path)
  output logic        mxu_cmd_valid,
  output logic [31:0] mxu_cmd,
  output logic [31:0] mxu_cfg0,
  output logic [31:0] mxu_cfg1,
  output logic [63:0] mxu_act_addr,
  output logic [63:0] mxu_wgt_addr,
  output logic [63:0] mxu_out_addr,

  // Status
  output logic        irq_out
);

  // ═══════════════════════════════════════════════════════════
  // G3 Register Bank
  // ═══════════════════════════════════════════════════════════

  // G2-compatible ports (stub for standalone G3 skeleton)
  logic [3:0]  boot_cause_stub;
  logic        sw_reset_stub, sw_cause_clr_stub, wdog_test_stub;
  logic [31:0] desc_stage_stub [0:15];
  logic [3:0]  doorbell_stub;
  logic [15:0] q_head_stub [0:3];
  logic [15:0] q_tail_stub [0:3];
  logic [3:0]  overflow_stub;
  logic [3:0]  overflow_clr_stub;
  logic        tc0_enable_stub, tc0_halt_stub, tc0_fault_clr_stub;
  logic        perf_freeze_stub;
  logic        irq_pending_w1c_en_stub;
  logic [31:0] irq_pending_w1c_data_stub;
  logic        irq_mask_wr_en_stub;
  logic [31:0] irq_mask_wr_data_stub;
  logic        irq_force_wr_en_stub;
  logic [31:0] irq_force_wr_data_stub;
  logic        trace_enable_stub, trace_freeze_stub, trace_fatal_only_stub;
  logic [9:0]  trace_rd_addr_stub;

  // G3 MXU ports
  logic        mxu_cmd_valid_r;
  logic [31:0] mxu_cmd_r, mxu_cfg0_r, mxu_cfg1_r;
  logic [63:0] mxu_act_addr_r, mxu_wgt_addr_r, mxu_out_addr_r;

  // G3 backward/optimizer/collective/fabric stubs
  logic        bkwd_cmd_valid_stub;
  logic [31:0] bkwd_cmd_stub;
  logic [63:0] bkwd_act_addr_stub, bkwd_grad_addr_stub;
  logic [31:0] bkwd_loss_scale_stub;
  logic        opt_cmd_valid_stub;
  logic [31:0] opt_cmd_stub, opt_lr_stub, opt_beta1_stub, opt_beta2_stub, opt_epsilon_stub;
  logic [63:0] opt_param_addr_stub, opt_state_addr_stub;
  logic        coll_cmd_valid_stub;
  logic [31:0] coll_cmd_stub, coll_buffer_size_stub, coll_timeout_stub;
  logic [63:0] coll_peer_mask_stub, coll_buffer_addr_stub, coll_result_addr_stub;
  logic [31:0] fabric_chip_id_stub;
  logic        fabric_loopback_stub;

  g3_reg_top u_g3_reg (
    .clk(clk), .rst_n(rst_n),
    .addr(reg_addr), .wr_en(reg_wr_en), .wr_data(reg_wr_data), .rd_data(reg_rd_data),

    // G2 stubs
    .boot_cause(4'b0001),  // POR
    .sw_reset_pulse(sw_reset_stub), .sw_cause_clr(sw_cause_clr_stub),
    .wdog_test_pulse(wdog_test_stub),
    .desc_stage(desc_stage_stub), .doorbell_pulse(doorbell_stub),
    .q_head(q_head_stub), .q_tail(q_tail_stub),
    .overflow_flags(4'd0), .overflow_clr(overflow_clr_stub),
    .oom_state(2'd0), .oom_admission_stop(1'b0), .oom_prefetch_clamp(1'b0),
    .oom_usage_lo(32'd0), .oom_reserved_lo(32'd0), .oom_effective_lo(32'd0),
    .tc0_runstate(32'd0), .tc0_fault_status(32'd0),
    .tc0_perf_cycles(64'd0), .tc0_desc_ptr(64'd0),
    .tc0_enable(tc0_enable_stub), .tc0_halt(tc0_halt_stub), .tc0_fault_clr(tc0_fault_clr_stub),
    .dma_status(32'd0), .dma_err_code(32'd0),
    .mxu_busy_cycles(64'd0), .mxu_tile_count(32'd0),
    .desc_done_count(32'd0), .perf_freeze(perf_freeze_stub),
    .irq_pending(32'd0), .irq_mask_rd(32'hFFFFFFFF), .irq_cause_last(32'd0),
    .irq_pending_w1c_en(irq_pending_w1c_en_stub),
    .irq_pending_w1c_data(irq_pending_w1c_data_stub),
    .irq_mask_wr_en(irq_mask_wr_en_stub), .irq_mask_wr_data(irq_mask_wr_data_stub),
    .irq_force_wr_en(irq_force_wr_en_stub), .irq_force_wr_data(irq_force_wr_data_stub),
    .trace_head(16'd0), .trace_tail(16'd0), .trace_drop_count(32'd0),
    .trace_enable(trace_enable_stub), .trace_freeze(trace_freeze_stub),
    .trace_fatal_only(trace_fatal_only_stub),
    .trace_rd_addr(trace_rd_addr_stub),
    .trace_rd_data(64'd0), .trace_rd_type(4'd0), .trace_rd_fatal(1'b0),

    // G3 MXU — real connection
    .mxu_cmd_valid(mxu_cmd_valid_r),
    .mxu_cmd(mxu_cmd_r), .mxu_cfg0(mxu_cfg0_r), .mxu_cfg1(mxu_cfg1_r),
    .mxu_act_addr(mxu_act_addr_r), .mxu_wgt_addr(mxu_wgt_addr_r),
    .mxu_out_addr(mxu_out_addr_r),
    .mxu_status(mxu_status_in),
    .mxu_err_code(mxu_err_code_in),
    .mxu_tile_cnt_g3(mxu_tile_cnt_in),

    // G3 backward — stub
    .bkwd_cmd_valid(bkwd_cmd_valid_stub), .bkwd_cmd(bkwd_cmd_stub),
    .bkwd_act_addr(bkwd_act_addr_stub), .bkwd_grad_addr(bkwd_grad_addr_stub),
    .bkwd_loss_scale(bkwd_loss_scale_stub),
    .bkwd_status(32'd0), .bkwd_err_code(32'd0),

    // G3 optimizer — stub
    .opt_cmd_valid(opt_cmd_valid_stub), .opt_cmd(opt_cmd_stub),
    .opt_lr(opt_lr_stub), .opt_beta1(opt_beta1_stub),
    .opt_beta2(opt_beta2_stub), .opt_epsilon(opt_epsilon_stub),
    .opt_param_addr(opt_param_addr_stub), .opt_state_addr(opt_state_addr_stub),
    .opt_status(32'd0), .opt_err_code(32'd0),

    // G3 collective — stub
    .coll_cmd_valid(coll_cmd_valid_stub), .coll_cmd(coll_cmd_stub),
    .coll_peer_mask(coll_peer_mask_stub), .coll_buffer_addr(coll_buffer_addr_stub),
    .coll_buffer_size(coll_buffer_size_stub), .coll_result_addr(coll_result_addr_stub),
    .coll_timeout(coll_timeout_stub),
    .coll_status(32'd0), .coll_err_code(32'd0), .coll_bytes(64'd0),

    // G3 fabric — stub
    .fabric_link_status(32'd0), .fabric_err_code(32'd0),
    .fabric_peer_id_cfg(), .fabric_topology_cfg(),
    .fabric_routing_sel(), .fabric_routing_data()
  );

  // ═══════════════════════════════════════════════════════════
  // MXU command output
  // ═══════════════════════════════════════════════════════════
  assign mxu_cmd_valid = mxu_cmd_valid_r;
  assign mxu_cmd       = mxu_cmd_r;
  assign mxu_cfg0      = mxu_cfg0_r;
  assign mxu_cfg1      = mxu_cfg1_r;
  assign mxu_act_addr  = mxu_act_addr_r;
  assign mxu_wgt_addr  = mxu_wgt_addr_r;
  assign mxu_out_addr  = mxu_out_addr_r;

  // ═══════════════════════════════════════════════════════════
  // IRQ stub (no real IRQ sources yet in G3 skeleton)
  // ═══════════════════════════════════════════════════════════
  assign irq_out = 1'b0;

  // TODO: future desc_fsm ingress (G3-RTL-005)
  // TODO: future backward/optimizer/collective engine connections

endmodule

`default_nettype wire
