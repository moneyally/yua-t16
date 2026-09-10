// g3_train_int_top.sv — ORBIT-G3 Single-Layer Training Step Integration
// SSOT: ORBIT_G3_RTL_ISSUES.md (G3-INT-002)
//
// Integrates: mxu_bf16_16x16 (forward) + backward_engine (dW) + optimizer_unit
// Path: forward(Y=X*W) → backward(dW=X^T*dY) → optimizer(W'=Adam(W,dW))
//
// Active region: 16×16 (correctness-first, control FSM focus)
// dW path only this turn. dX deferred.
//
// Top FSM: IDLE→FWD→WAIT_FWD→BKWD→WAIT_BKWD→OPT→WAIT_OPT→DONE
`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off UNUSEDSIGNAL */

module g3_train_int_top #(
  parameter int DIM = 16
)(
  input  logic        clk,
  input  logic        rst_n,

  // Control
  input  logic        start,

  // Hyperparameters (FP32 bits)
  input  logic [31:0] lr_fp32,
  input  logic [31:0] beta1_fp32,
  input  logic [31:0] beta2_fp32,
  input  logic [31:0] epsilon_fp32,
  input  logic [31:0] weight_decay_fp32,
  input  logic        adamw_enable,

  // Input tensors (BF16 for forward, FP32 for optimizer state)
  input  logic [15:0] x_bf16  [0:DIM-1][0:DIM-1],  // activation
  input  logic [15:0] w_bf16  [0:DIM-1][0:DIM-1],  // weight
  input  logic [15:0] dy_bf16 [0:DIM-1][0:DIM-1],  // upstream gradient
  input  logic [31:0] param_in [0:DIM-1][0:DIM-1],  // FP32 param (= W in FP32)
  input  logic [31:0] m_in     [0:DIM-1][0:DIM-1],  // Adam m state
  input  logic [31:0] v_in     [0:DIM-1][0:DIM-1],  // Adam v state

  // Outputs
  output logic [31:0] param_out [0:DIM-1][0:DIM-1],  // updated param
  output logic [31:0] m_out     [0:DIM-1][0:DIM-1],  // updated m
  output logic [31:0] v_out     [0:DIM-1][0:DIM-1],  // updated v

  // Observable intermediate
  output logic [31:0] fwd_acc  [0:DIM-1][0:DIM-1],   // forward result (FP32)
  output logic [31:0] dw_result [0:DIM-1][0:DIM-1],  // backward dW (FP32)

  // Status
  output logic        busy,
  output logic        train_done_pulse,
  output logic [7:0]  err_code   // 0=ok, [1:0]=stage (1=fwd,2=bkwd,3=opt)
);

  // ═══════════════════════════════════════════════════════════
  // Top FSM
  // ═══════════════════════════════════════════════════════════
  typedef enum logic [3:0] {
    ST_IDLE,
    ST_FWD,
    ST_WAIT_FWD,
    ST_BKWD,
    ST_WAIT_BKWD,
    ST_OPT,
    ST_WAIT_OPT,
    ST_DONE,
    ST_FAULT
  } train_state_t;

  train_state_t tst;

  assign busy = (tst != ST_IDLE && tst != ST_DONE && tst != ST_FAULT);

  // ═══════════════════════════════════════════════════════════
  // Forward: mxu_bf16_16x16 (Y = X * W, K=DIM steps)
  // ═══════════════════════════════════════════════════════════
  logic [15:0] fwd_a [0:DIM-1];
  logic [15:0] fwd_b [0:DIM-1];
  wire         fwd_en  = (tst == ST_WAIT_FWD);
  wire         fwd_clr = (tst == ST_FWD);
  logic [4:0]  fwd_k;

  mxu_bf16_16x16 u_fwd_mxu (
    .clk(clk), .rst_n(rst_n),
    .en(fwd_en), .acc_clr(fwd_clr),
    .a_row(fwd_a), .b_col(fwd_b),
    .acc_out(fwd_acc),
    .busy()
  );

  // Slice X column k and W column k for forward
  integer fi;
  always_comb begin
    for (fi = 0; fi < DIM; fi++) begin
      fwd_a[fi] = x_bf16[fi][fwd_k[3:0]];   // X column k = X[:][k]
      fwd_b[fi] = w_bf16[fwd_k[3:0]][fi];    // W row k = W[k][:]
    end
  end

  // ═══════════════════════════════════════════════════════════
  // Backward: backward_engine (dW = X^T * dY)
  // ═══════════════════════════════════════════════════════════
  logic        bkwd_start;
  logic        bkwd_busy, bkwd_done;
  logic [7:0]  bkwd_err;

  backward_engine #(.DIM(DIM)) u_bkwd (
    .clk(clk), .rst_n(rst_n),
    .start(bkwd_start),
    .mode(2'd1),          // MODE_DW
    .acc_clr(1'b0),
    .x_in(x_bf16),
    .w_in(w_bf16),
    .dy_in(dy_bf16),
    .result(dw_result),
    .busy(bkwd_busy),
    .done_pulse(bkwd_done),
    .err_code(bkwd_err)
  );

  // ═══════════════════════════════════════════════════════════
  // Optimizer: optimizer_unit (W' = Adam(W, dW))
  // ═══════════════════════════════════════════════════════════
  logic        opt_start;
  logic        opt_busy, opt_done;
  logic [7:0]  opt_err;

  optimizer_unit #(.DIM(DIM)) u_opt (
    .clk(clk), .rst_n(rst_n),
    .start(opt_start),
    .adamw_enable(adamw_enable),
    .lr_fp32(lr_fp32),
    .beta1_fp32(beta1_fp32),
    .beta2_fp32(beta2_fp32),
    .epsilon_fp32(epsilon_fp32),
    .weight_decay_fp32(weight_decay_fp32),
    .param_in(param_in),
    .grad_in(dw_result),     // dW from backward
    .m_in(m_in),
    .v_in(v_in),
    .param_out(param_out),
    .m_out(m_out),
    .v_out(v_out),
    .busy(opt_busy),
    .done_pulse(opt_done),
    .err_code(opt_err)
  );

  // ═══════════════════════════════════════════════════════════
  // Done pulse capture (registered, edge-safe)
  // ═══════════════════════════════════════════════════════════
  logic bkwd_done_r, opt_done_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      bkwd_done_r <= 1'b0;
      opt_done_r  <= 1'b0;
    end else begin
      if (tst == ST_BKWD) bkwd_done_r <= 1'b0;
      else if (bkwd_done) bkwd_done_r <= 1'b1;

      if (tst == ST_OPT) opt_done_r <= 1'b0;
      else if (opt_done) opt_done_r <= 1'b1;
    end
  end

  // ═══════════════════════════════════════════════════════════
  // Top FSM
  // ═══════════════════════════════════════════════════════════
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      tst              <= ST_IDLE;
      fwd_k            <= 5'd0;
      bkwd_start       <= 1'b0;
      opt_start        <= 1'b0;
      train_done_pulse <= 1'b0;
      err_code         <= 8'd0;
    end else begin
      train_done_pulse <= 1'b0;
      bkwd_start       <= 1'b0;
      opt_start        <= 1'b0;

      case (tst)
        ST_IDLE: begin
          err_code <= 8'd0;
          if (start) begin
            fwd_k <= 5'd0;
            tst   <= ST_FWD;
          end
        end

        // ── Forward ────────────────────────────────
        ST_FWD: begin
          // acc_clr is combinational (fwd_clr = tst==ST_FWD)
          tst <= ST_WAIT_FWD;
        end

        ST_WAIT_FWD: begin
          // en is combinational (fwd_en = tst==ST_WAIT_FWD)
          // Run DIM K-steps
          if (fwd_k == DIM - 1) begin
            tst <= ST_BKWD;
          end
          fwd_k <= fwd_k + 1;
        end

        // ── Backward (dW = X^T * dY) ──────────────
        ST_BKWD: begin
          bkwd_start <= 1'b1;
          tst        <= ST_WAIT_BKWD;
        end

        ST_WAIT_BKWD: begin
          if (bkwd_done_r) begin
            if (bkwd_err != 0) begin
              err_code <= {6'd0, 2'd2};  // stage 2 = backward
              tst      <= ST_FAULT;
            end else begin
              tst <= ST_OPT;
            end
          end
        end

        // ── Optimizer ─────────────────────────────
        ST_OPT: begin
          opt_start <= 1'b1;
          tst       <= ST_WAIT_OPT;
        end

        ST_WAIT_OPT: begin
          if (opt_done_r) begin
            if (opt_err != 0) begin
              err_code <= {6'd0, 2'd3};  // stage 3 = optimizer
              tst      <= ST_FAULT;
            end else begin
              tst <= ST_DONE;
            end
          end
        end

        // ── Done / Fault ──────────────────────────
        ST_DONE: begin
          train_done_pulse <= 1'b1;
          tst              <= ST_IDLE;
        end

        ST_FAULT: begin
          train_done_pulse <= 1'b1;
          tst              <= ST_IDLE;
        end

        default: tst <= ST_IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
