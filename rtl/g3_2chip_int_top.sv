// g3_2chip_int_top.sv — ORBIT-G3 2-Chip Distributed Training Step
// SSOT: ORBIT_G3_RTL_ISSUES.md (G3-INT-004)
//
// Path per chip: forward(MXU) → backward(dW)
// Then: collective all-reduce SUM on dW0 + dW1
// Then: optimizer per chip with reduced_dW as gradient
//
// Active region: 16×16. SUM reduction (not average).
// Separate from g3_train_int_top: optimizer is post-collective here.
`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off UNUSEDSIGNAL */

module g3_2chip_int_top #(
  parameter int DIM = 16
)(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,

  // Hyperparameters (shared)
  input  logic [31:0] lr_fp32,
  input  logic [31:0] beta1_fp32,
  input  logic [31:0] beta2_fp32,
  input  logic [31:0] epsilon_fp32,
  input  logic [31:0] weight_decay_fp32,
  input  logic        adamw_enable,

  // Chip 0 inputs
  input  logic [15:0] x0_bf16  [0:DIM-1][0:DIM-1],
  input  logic [15:0] w0_bf16  [0:DIM-1][0:DIM-1],
  input  logic [15:0] dy0_bf16 [0:DIM-1][0:DIM-1],
  input  logic [31:0] param0_in [0:DIM-1][0:DIM-1],
  input  logic [31:0] m0_in     [0:DIM-1][0:DIM-1],
  input  logic [31:0] v0_in     [0:DIM-1][0:DIM-1],

  // Chip 1 inputs
  input  logic [15:0] x1_bf16  [0:DIM-1][0:DIM-1],
  input  logic [15:0] w1_bf16  [0:DIM-1][0:DIM-1],
  input  logic [15:0] dy1_bf16 [0:DIM-1][0:DIM-1],
  input  logic [31:0] param1_in [0:DIM-1][0:DIM-1],
  input  logic [31:0] m1_in     [0:DIM-1][0:DIM-1],
  input  logic [31:0] v1_in     [0:DIM-1][0:DIM-1],

  // Chip 0 outputs
  output logic [31:0] param0_out [0:DIM-1][0:DIM-1],
  output logic [31:0] m0_out     [0:DIM-1][0:DIM-1],
  output logic [31:0] v0_out     [0:DIM-1][0:DIM-1],

  // Chip 1 outputs
  output logic [31:0] param1_out [0:DIM-1][0:DIM-1],
  output logic [31:0] m1_out     [0:DIM-1][0:DIM-1],
  output logic [31:0] v1_out     [0:DIM-1][0:DIM-1],

  // Observable
  output logic [31:0] dw0_out [0:DIM-1][0:DIM-1],
  output logic [31:0] dw1_out [0:DIM-1][0:DIM-1],
  output logic [31:0] reduced_dw [0:DIM-1][0:DIM-1],

  // Status
  output logic        busy,
  output logic        done_pulse,
  output logic [7:0]  err_code
);

  // ═══════════════════════════════════════════════════════════
  // Backward engines (chip0, chip1) — dW = X^T * dY
  // ═══════════════════════════════════════════════════════════
  logic bkwd0_start, bkwd0_done, bkwd0_busy;
  logic [7:0] bkwd0_err;

  backward_engine #(.DIM(DIM)) u_bkwd0 (
    .clk(clk), .rst_n(rst_n),
    .start(bkwd0_start), .mode(2'd1), .acc_clr(1'b0),
    .x_in(x0_bf16), .w_in(w0_bf16), .dy_in(dy0_bf16),
    .result(dw0_out),
    .busy(bkwd0_busy), .done_pulse(bkwd0_done), .err_code(bkwd0_err)
  );

  logic bkwd1_start, bkwd1_done, bkwd1_busy;
  logic [7:0] bkwd1_err;

  backward_engine #(.DIM(DIM)) u_bkwd1 (
    .clk(clk), .rst_n(rst_n),
    .start(bkwd1_start), .mode(2'd1), .acc_clr(1'b0),
    .x_in(x1_bf16), .w_in(w1_bf16), .dy_in(dy1_bf16),
    .result(dw1_out),
    .busy(bkwd1_busy), .done_pulse(bkwd1_done), .err_code(bkwd1_err)
  );

  // ═══════════════════════════════════════════════════════════
  // Collective engine — all-reduce SUM(dW0, dW1)
  // ═══════════════════════════════════════════════════════════
  logic coll_start, coll_done, coll_busy;
  logic [7:0] coll_err;

  collective_engine #(.DIM(DIM)) u_coll (
    .clk(clk), .rst_n(rst_n),
    .start(coll_start),
    .op_type(8'h01),        // ALL_REDUCE_SUM
    .peer_mask(8'h03),      // 2-peer valid
    .local_in(dw0_out),     // chip0 gradient
    .peer_in(dw1_out),      // chip1 gradient
    .result_out(reduced_dw),
    .busy(coll_busy), .done_pulse(coll_done), .err_code(coll_err)
  );

  // ═══════════════════════════════════════════════════════════
  // Optimizer units (chip0, chip1) — both get reduced_dW
  // ═══════════════════════════════════════════════════════════
  logic opt0_start, opt0_done, opt0_busy;
  logic [7:0] opt0_err;

  optimizer_unit #(.DIM(DIM)) u_opt0 (
    .clk(clk), .rst_n(rst_n),
    .start(opt0_start), .adamw_enable(adamw_enable),
    .lr_fp32(lr_fp32), .beta1_fp32(beta1_fp32),
    .beta2_fp32(beta2_fp32), .epsilon_fp32(epsilon_fp32),
    .weight_decay_fp32(weight_decay_fp32),
    .param_in(param0_in), .grad_in(reduced_dw),
    .m_in(m0_in), .v_in(v0_in),
    .param_out(param0_out), .m_out(m0_out), .v_out(v0_out),
    .busy(opt0_busy), .done_pulse(opt0_done), .err_code(opt0_err)
  );

  logic opt1_start, opt1_done, opt1_busy;
  logic [7:0] opt1_err;

  optimizer_unit #(.DIM(DIM)) u_opt1 (
    .clk(clk), .rst_n(rst_n),
    .start(opt1_start), .adamw_enable(adamw_enable),
    .lr_fp32(lr_fp32), .beta1_fp32(beta1_fp32),
    .beta2_fp32(beta2_fp32), .epsilon_fp32(epsilon_fp32),
    .weight_decay_fp32(weight_decay_fp32),
    .param_in(param1_in), .grad_in(reduced_dw),
    .m_in(m1_in), .v_in(v1_in),
    .param_out(param1_out), .m_out(m1_out), .v_out(v1_out),
    .busy(opt1_busy), .done_pulse(opt1_done), .err_code(opt1_err)
  );

  // ═══════════════════════════════════════════════════════════
  // Done pulse capture (registered, edge-safe)
  // ═══════════════════════════════════════════════════════════
  logic bkwd0_done_r, bkwd1_done_r, coll_done_r, opt0_done_r, opt1_done_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      bkwd0_done_r <= 0; bkwd1_done_r <= 0;
      coll_done_r  <= 0;
      opt0_done_r  <= 0; opt1_done_r  <= 0;
    end else begin
      // Clear on stage entry, set on subordinate done
      if (tst == ST_BKWD) begin bkwd0_done_r <= 0; bkwd1_done_r <= 0; end
      else begin
        if (bkwd0_done) bkwd0_done_r <= 1;
        if (bkwd1_done) bkwd1_done_r <= 1;
      end

      if (tst == ST_COLL) coll_done_r <= 0;
      else if (coll_done) coll_done_r <= 1;

      if (tst == ST_OPT) begin opt0_done_r <= 0; opt1_done_r <= 0; end
      else begin
        if (opt0_done) opt0_done_r <= 1;
        if (opt1_done) opt1_done_r <= 1;
      end
    end
  end

  // ═══════════════════════════════════════════════════════════
  // Top FSM
  // ═══════════════════════════════════════════════════════════
  typedef enum logic [3:0] {
    ST_IDLE,
    ST_BKWD,         // launch both backward engines
    ST_WAIT_BKWD,    // wait for both done
    ST_COLL,         // launch collective
    ST_WAIT_COLL,    // wait for collective done
    ST_OPT,          // launch both optimizers
    ST_WAIT_OPT,     // wait for both done
    ST_DONE,
    ST_FAULT
  } state_t;

  state_t tst;

  assign busy = (tst != ST_IDLE && tst != ST_DONE && tst != ST_FAULT);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      tst        <= ST_IDLE;
      bkwd0_start <= 0; bkwd1_start <= 0;
      coll_start  <= 0;
      opt0_start  <= 0; opt1_start  <= 0;
      done_pulse  <= 0;
      err_code    <= 8'd0;
    end else begin
      done_pulse  <= 0;
      bkwd0_start <= 0; bkwd1_start <= 0;
      coll_start  <= 0;
      opt0_start  <= 0; opt1_start  <= 0;

      case (tst)
        ST_IDLE: begin
          err_code <= 0;
          if (start) tst <= ST_BKWD;
        end

        ST_BKWD: begin
          bkwd0_start <= 1;
          bkwd1_start <= 1;
          tst <= ST_WAIT_BKWD;
        end

        ST_WAIT_BKWD: begin
          if (bkwd0_done_r && bkwd1_done_r) begin
            if (bkwd0_err != 0) begin err_code <= 8'h10; tst <= ST_FAULT; end
            else if (bkwd1_err != 0) begin err_code <= 8'h11; tst <= ST_FAULT; end
            else tst <= ST_COLL;
          end
        end

        ST_COLL: begin
          coll_start <= 1;
          tst <= ST_WAIT_COLL;
        end

        ST_WAIT_COLL: begin
          if (coll_done_r) begin
            if (coll_err != 0) begin err_code <= 8'h20; tst <= ST_FAULT; end
            else tst <= ST_OPT;
          end
        end

        ST_OPT: begin
          opt0_start <= 1;
          opt1_start <= 1;
          tst <= ST_WAIT_OPT;
        end

        ST_WAIT_OPT: begin
          if (opt0_done_r && opt1_done_r) begin
            if (opt0_err != 0) begin err_code <= 8'h10; tst <= ST_FAULT; end
            else if (opt1_err != 0) begin err_code <= 8'h11; tst <= ST_FAULT; end
            else tst <= ST_DONE;
          end
        end

        ST_DONE: begin
          done_pulse <= 1;
          tst <= ST_IDLE;
        end

        ST_FAULT: begin
          done_pulse <= 1;
          tst <= ST_IDLE;
        end

        default: tst <= ST_IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
