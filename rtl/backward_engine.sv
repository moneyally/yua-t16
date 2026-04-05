// backward_engine.sv — ORBIT-G3 GEMM Backward Primitive
// SSOT: ORBIT_G3_RTL_ISSUES.md (G3-RTL-010), ORBIT_G3_ARCHITECTURE.md
//
// Computes single-layer GEMM backward:
//   MODE_DW: dW = X^T * dY   (weight gradient)
//   MODE_DX: dX = dY * W^T   (input gradient)
//
// Uses one mxu_bf16_16x16 instance internally (iterative, 16×16 scope).
// Full 128×128 version deferred — this is correctness-first 16×16 backward.
//
// Interface matches forward MXU style: start/busy/done_pulse.
// BF16 input, FP32 accumulate output.
//
// For MODE_DW: internally transposes X (reads column-major).
// For MODE_DX: internally transposes W (reads column-major).
`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off UNUSEDSIGNAL */

module backward_engine #(
  parameter int DIM = 16   // 16×16 for correctness-first
)(
  input  logic        clk,
  input  logic        rst_n,

  // Control
  input  logic        start,
  input  logic [1:0]  mode,       // 0=invalid, 1=DW, 2=DX
  input  logic        acc_clr,

  // Input matrices (BF16, row-major DIM×DIM)
  input  logic [15:0] x_in  [0:DIM-1][0:DIM-1],  // activation
  input  logic [15:0] w_in  [0:DIM-1][0:DIM-1],  // weight
  input  logic [15:0] dy_in [0:DIM-1][0:DIM-1],  // upstream gradient

  // Output (FP32, DIM×DIM)
  output logic [31:0] result [0:DIM-1][0:DIM-1],

  // Status
  output logic        busy,
  output logic        done_pulse,
  output logic [7:0]  err_code    // 0=ok, 1=invalid mode
);

  localparam logic [1:0] MODE_INVALID = 2'd0;
  localparam logic [1:0] MODE_DW      = 2'd1;
  localparam logic [1:0] MODE_DX      = 2'd2;

  // ═══════════════════════════════════════════════════════════
  // State machine
  // ═══════════════════════════════════════════════════════════
  typedef enum logic [3:0] {
    ST_IDLE,
    ST_CHECK_MODE,
    ST_CLEAR,
    ST_CLEAR_WAIT,  // settle after clear
    ST_COMPUTE,     // drive MXU for K steps
    ST_WAIT,        // wait for MXU result settle
    ST_STORE,       // copy MXU acc to result
    ST_DONE,
    ST_ERROR
  } state_t;

  state_t st;

  logic [1:0]  mode_r;
  logic [4:0]  k_cnt;   // K-step counter (0 to DIM-1)

  // ═══════════════════════════════════════════════════════════
  // MXU 16×16 instance
  // ═══════════════════════════════════════════════════════════
  logic [15:0] mxu_a [0:DIM-1];
  logic [15:0] mxu_b [0:DIM-1];
  logic        mxu_en, mxu_clr;
  logic [31:0] mxu_acc [0:DIM-1][0:DIM-1];

  // mxu_en/mxu_clr as combinational for same-cycle effect
  wire mxu_en_comb = (st == ST_COMPUTE);
  wire mxu_clr_comb = (st == ST_CLEAR);

  mxu_bf16_16x16 u_mxu (
    .clk(clk), .rst_n(rst_n),
    .en(mxu_en_comb), .acc_clr(mxu_clr_comb),
    .a_row(mxu_a), .b_col(mxu_b),
    .acc_out(mxu_acc),
    .busy()
  );

  // ═══════════════════════════════════════════════════════════
  // Input routing based on mode
  // ═══════════════════════════════════════════════════════════
  // MODE_DW: dW = X^T * dY
  //   For each k-step (k = 0..DIM-1):
  //     a_row[i] = X[k][i]  (column k of X = row k, transposed view)
  //     b_col[j] = dY[k][j] (row k of dY)
  //
  // MODE_DX: dX = dY * W^T
  //   For each k-step (k = 0..DIM-1):
  //     a_row[i] = dY[i][k]  (column k of dY... but we need row-of-dY)
  //     Hmm, let's think again:
  //     dX[i][j] = sum_k dY[i][k] * W[j][k]  (W^T means W transposed)
  //   So per k-step:
  //     a_row[i] = dY[i][k]  (column k of dY)
  //     b_col[j] = W[j][k]   (column k of W = W^T row k)

  integer ri;
  always_comb begin
    for (ri = 0; ri < DIM; ri++) begin
      mxu_a[ri] = 16'd0;
      mxu_b[ri] = 16'd0;
    end

    if (st == ST_COMPUTE) begin
      case (mode_r)
        MODE_DW: begin
          // a = X^T column k = X[k][:]
          // b = dY row k = dY[k][:]
          for (ri = 0; ri < DIM; ri++) begin
            mxu_a[ri] = x_in[k_cnt[3:0]][ri];
            mxu_b[ri] = dy_in[k_cnt[3:0]][ri];
          end
        end
        MODE_DX: begin
          // a = dY column k = dY[:][k]
          // b = W column k = W[:][k]  (= W^T row k)
          for (ri = 0; ri < DIM; ri++) begin
            mxu_a[ri] = dy_in[ri][k_cnt[3:0]];
            mxu_b[ri] = w_in[ri][k_cnt[3:0]];
          end
        end
        default: ;
      endcase
    end
  end

  // ═══════════════════════════════════════════════════════════
  // Result storage
  // ═══════════════════════════════════════════════════════════
  integer si, sj;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (si = 0; si < DIM; si++)
        for (sj = 0; sj < DIM; sj++)
          result[si][sj] <= 32'd0;
    end else if (acc_clr) begin
      for (si = 0; si < DIM; si++)
        for (sj = 0; sj < DIM; sj++)
          result[si][sj] <= 32'd0;
    end else if (st == ST_STORE) begin
      for (si = 0; si < DIM; si++)
        for (sj = 0; sj < DIM; sj++)
          result[si][sj] <= mxu_acc[si][sj];
    end
  end

  // ═══════════════════════════════════════════════════════════
  // FSM
  // ═══════════════════════════════════════════════════════════
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st         <= ST_IDLE;
      mode_r     <= MODE_INVALID;
      k_cnt      <= 5'd0;
      done_pulse <= 1'b0;
      err_code   <= 8'd0;
    end else begin
      done_pulse <= 1'b0;

      case (st)
        ST_IDLE: begin
          err_code <= 8'd0;
          if (start) begin
            mode_r <= mode;
            st     <= ST_CHECK_MODE;
          end
        end

        ST_CHECK_MODE: begin
          if (mode_r == MODE_DW || mode_r == MODE_DX) begin
            st <= ST_CLEAR;
          end else begin
            err_code <= 8'd1;  // invalid mode
            st       <= ST_ERROR;
          end
        end

        ST_CLEAR: begin
          // mxu_clr is combinational (mxu_clr_comb = st==ST_CLEAR)
          k_cnt <= 5'd0;
          st    <= ST_CLEAR_WAIT;
        end

        ST_CLEAR_WAIT: begin
          // 1-cycle settle after mxu_clr for acc to zero
          st <= ST_COMPUTE;
        end

        ST_COMPUTE: begin
          // en is combinational (mxu_en_comb), data routed by k_cnt
          if (k_cnt == DIM - 1) begin
            st <= ST_WAIT;  // done after this cycle's MAC
          end
          k_cnt <= k_cnt + 1;
        end

        ST_WAIT: begin
          // Wait 1 cycle for final MAC result to settle
          st <= ST_STORE;
        end

        ST_STORE: begin
          // result[] latched from mxu_acc in separate always_ff
          st <= ST_DONE;
        end

        ST_DONE: begin
          done_pulse <= 1'b1;
          st         <= ST_IDLE;
        end

        ST_ERROR: begin
          done_pulse <= 1'b1;  // signal completion even on error
          st         <= ST_IDLE;
        end

        default: st <= ST_IDLE;
      endcase
    end
  end

  assign busy = (st != ST_IDLE && st != ST_DONE && st != ST_ERROR);

endmodule

`default_nettype wire
