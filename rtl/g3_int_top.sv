// g3_int_top.sv — ORBIT-G3 Integration Top for G3-INT-001
// SSOT: ORBIT_G3_RTL_ISSUES.md (G3-INT-001)
//
// Wires: g3_desc_fsm + mxu_bf16_16x16 into a single testable unit.
// Descriptor ingress via integration_desc_* test hook ports.
// NOT for production — production uses g3_ctrl_top with queue path.
//
// E2E path:
//   integration_desc_valid/bytes → g3_desc_fsm → mxu adapter → mxu_bf16_16x16
//   mxu done → g3_desc_fsm.core_done → done_pulse
`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off UNUSEDSIGNAL */

module g3_int_top #(
  parameter int DESC_SIZE = 64,
  parameter int MXU_ROWS  = 16,
  parameter int MXU_COLS  = 16
)(
  input  logic        clk,
  input  logic        rst_n,

  // ── Integration test hook: descriptor ingress ──────────
  input  logic        int_desc_valid,
  input  logic [7:0]  int_desc_bytes [0:DESC_SIZE-1],
  output logic        int_desc_ready,
  input  logic [1:0]  int_queue_class,

  // ── MXU data input (from test memory / TB) ─────────────
  input  logic [15:0] mxu_a_row [0:MXU_ROWS-1],  // BF16
  input  logic [15:0] mxu_b_col [0:MXU_COLS-1],  // BF16
  input  logic        mxu_data_valid,             // TB drives data

  // ── Status / observability ─────────────────────────────
  output logic        done_pulse,
  output logic        fault_valid,
  output logic [7:0]  fault_code,
  output logic        busy,
  output logic [7:0]  current_opcode,

  // ── MXU accumulator output (for result verification) ───
  output logic [31:0] mxu_acc [0:MXU_ROWS-1][0:MXU_COLS-1]
);

  // ═══════════════════════════════════════════════════════════
  // g3_desc_fsm
  // ═══════════════════════════════════════════════════════════
  logic        fsm_mxu_cmd_valid, fsm_mxu_cmd_ready;
  logic [31:0] fsm_mxu_cfg0, fsm_mxu_cfg1;
  logic [63:0] fsm_mxu_act_addr, fsm_mxu_wgt_addr, fsm_mxu_out_addr;
  logic        fsm_gemm_cmd_valid, fsm_gemm_cmd_ready;
  logic [63:0] fsm_gemm_act, fsm_gemm_wgt, fsm_gemm_out;
  logic [31:0] fsm_gemm_kt;
  logic        fsm_done, fsm_fault_valid, fsm_busy;
  logic [7:0]  fsm_fault_code, fsm_opcode;
  logic [1:0]  fsm_qclass;

  // MXU completion feedback
  logic        mxu_done;

  g3_desc_fsm #(.DESC_SIZE(DESC_SIZE), .TIMEOUT_DEFAULT(32'd1000)) u_fsm (
    .clk(clk), .rst_n(rst_n),
    .desc_valid(int_desc_valid),
    .desc_bytes(int_desc_bytes),
    .desc_ready(int_desc_ready),
    .queue_class(int_queue_class),

    .mxu_cmd_valid(fsm_mxu_cmd_valid),
    .mxu_cmd_ready(fsm_mxu_cmd_ready),
    .mxu_cfg0(fsm_mxu_cfg0), .mxu_cfg1(fsm_mxu_cfg1),
    .mxu_act_addr(fsm_mxu_act_addr),
    .mxu_wgt_addr(fsm_mxu_wgt_addr),
    .mxu_out_addr(fsm_mxu_out_addr),

    .gemm_cmd_valid(fsm_gemm_cmd_valid),
    .gemm_cmd_ready(1'b1),  // not used this turn
    .gemm_act_addr(fsm_gemm_act), .gemm_wgt_addr(fsm_gemm_wgt),
    .gemm_out_addr(fsm_gemm_out), .gemm_Kt(fsm_gemm_kt),

    .bkwd_cmd_valid(), .opt_cmd_valid(), .coll_cmd_valid(),

    .core_done(mxu_done),
    .timeout_cycles(32'd1000),  // short for integration test

    .fault_valid(fsm_fault_valid), .fault_code(fsm_fault_code),
    .busy(fsm_busy), .done_pulse(fsm_done),
    .current_opcode(fsm_opcode), .current_qclass(fsm_qclass)
  );

  // ═══════════════════════════════════════════════════════════
  // MXU adapter: FSM dispatch → MXU control
  // ═══════════════════════════════════════════════════════════
  // g3_desc_fsm issues mxu_cmd_valid pulse.
  // mxu_bf16_16x16 needs: en, acc_clr, a_row[], b_col[].
  //
  // Integration approach:
  //   1. FSM mxu_cmd_valid → latch "MXU active" state
  //   2. TB drives a_row/b_col via mxu_data_valid
  //   3. When TB asserts mxu_data_valid, MXU en=1
  //   4. TB drives data for K steps
  //   5. TB deasserts mxu_data_valid → MXU done pulse generated

  typedef enum logic [1:0] {
    MXU_IDLE,
    MXU_ACTIVE,
    MXU_DONE
  } mxu_state_t;

  mxu_state_t mxu_st;

  // Accept FSM dispatch immediately
  assign fsm_mxu_cmd_ready = (mxu_st == MXU_IDLE);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      mxu_st <= MXU_IDLE;
    end else begin
      case (mxu_st)
        MXU_IDLE: begin
          if (fsm_mxu_cmd_valid && fsm_mxu_cmd_ready)
            mxu_st <= MXU_ACTIVE;
        end
        MXU_ACTIVE: begin
          // TB drives data. When TB stops (data_valid drops after at least 1 cycle),
          // transition to DONE.
          if (!mxu_data_valid && mxu_was_active)
            mxu_st <= MXU_DONE;
        end
        MXU_DONE: begin
          mxu_st <= MXU_IDLE;
        end
        default: mxu_st <= MXU_IDLE;
      endcase
    end
  end

  // Track if MXU was ever active (to detect falling edge of data_valid)
  logic mxu_was_active;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      mxu_was_active <= 1'b0;
    else if (mxu_st == MXU_IDLE)
      mxu_was_active <= 1'b0;
    else if (mxu_data_valid)
      mxu_was_active <= 1'b1;
  end

  assign mxu_done = (mxu_st == MXU_DONE);

  // ═══════════════════════════════════════════════════════════
  // mxu_bf16_16x16
  // ═══════════════════════════════════════════════════════════
  logic mxu_en, mxu_clr;

  assign mxu_en  = (mxu_st == MXU_ACTIVE) && mxu_data_valid;
  assign mxu_clr = fsm_mxu_cmd_valid && fsm_mxu_cmd_ready; // clear on new dispatch

  mxu_bf16_16x16 u_mxu (
    .clk(clk), .rst_n(rst_n),
    .en(mxu_en), .acc_clr(mxu_clr),
    .a_row(mxu_a_row), .b_col(mxu_b_col),
    .acc_out(mxu_acc),
    .busy()
  );

  // ═══════════════════════════════════════════════════════════
  // Output
  // ═══════════════════════════════════════════════════════════
  assign done_pulse     = fsm_done;
  assign fault_valid    = fsm_fault_valid;
  assign fault_code     = fsm_fault_code;
  assign busy           = fsm_busy;
  assign current_opcode = fsm_opcode;

endmodule

`default_nettype wire
