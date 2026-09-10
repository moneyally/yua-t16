// optimizer_unit.sv — ORBIT-G3 Adam/AdamW Optimizer Primitive
// SSOT: ORBIT_G3_RTL_ISSUES.md (G3-RTL-011), ORBIT_G3_ARCHITECTURE.md
//
// Single-tensor Adam/AdamW parameter update:
//   m_t = beta1 * m + (1-beta1) * g
//   v_t = beta2 * v + (1-beta2) * g^2
//   p   = p - lr * m_t / (sqrt(v_t) + eps)       [Adam]
//   p   = p - lr * (m_t / (sqrt(v_t) + eps) + wd * p)  [AdamW]
//
// All FP32. 16×16 tensor, iterative element-by-element update.
// Correctness-first: uses behavioral real arithmetic internally.
`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off UNUSEDSIGNAL */

module optimizer_unit #(
  parameter int DIM = 16
)(
  input  logic        clk,
  input  logic        rst_n,

  // Control
  input  logic        start,
  input  logic        adamw_enable,   // 0=Adam, 1=AdamW

  // Hyperparameters (FP32 bit patterns)
  input  logic [31:0] lr_fp32,
  input  logic [31:0] beta1_fp32,
  input  logic [31:0] beta2_fp32,
  input  logic [31:0] epsilon_fp32,
  input  logic [31:0] weight_decay_fp32,

  // Tensor inputs (FP32, 16×16)
  input  logic [31:0] param_in [0:DIM-1][0:DIM-1],
  input  logic [31:0] grad_in  [0:DIM-1][0:DIM-1],
  input  logic [31:0] m_in     [0:DIM-1][0:DIM-1],
  input  logic [31:0] v_in     [0:DIM-1][0:DIM-1],

  // Tensor outputs (FP32, 16×16)
  output logic [31:0] param_out [0:DIM-1][0:DIM-1],
  output logic [31:0] m_out     [0:DIM-1][0:DIM-1],
  output logic [31:0] v_out     [0:DIM-1][0:DIM-1],

  // Status
  output logic        busy,
  output logic        done_pulse,
  output logic [7:0]  err_code    // 0=ok
);

  // ═══════════════════════════════════════════════════════════
  // FP32 <-> real conversion (behavioral, correctness-first)
  // ═══════════════════════════════════════════════════════════
  function automatic real fp32_to_real(input logic [31:0] bits);
    logic sign_bit;
    logic [7:0] exp_bits;
    logic [22:0] man_bits;
    real result;
    integer exp_val;

    sign_bit = bits[31];
    exp_bits = bits[30:23];
    man_bits = bits[22:0];

    if (exp_bits == 0 && man_bits == 0) begin
      fp32_to_real = 0.0;
    end else if (exp_bits == 8'hFF) begin
      fp32_to_real = sign_bit ? -3.4e38 : 3.4e38;
    end else begin
      exp_val = exp_bits - 127;
      result = 1.0 + $itor(man_bits) / 8388608.0;
      if (exp_val >= 0)
        result = result * (2.0 ** exp_val);
      else
        result = result / (2.0 ** (-exp_val));
      if (sign_bit) result = -result;
      fp32_to_real = result;
    end
  endfunction

  function automatic logic [31:0] real_to_fp32(input real v);
    logic s;
    real a, mf;
    integer ei, iter;
    logic [22:0] mb;
    logic [7:0] eb;
    integer mb_raw;

    s = (v < 0.0) ? 1'b1 : 1'b0;
    a = s ? -v : v;

    if (a == 0.0) begin
      real_to_fp32 = 32'd0;
    end else if (a >= 3.4e38) begin
      real_to_fp32 = {s, 8'hFE, 23'h7FFFFF};
    end else begin
      ei = 0; mf = a; iter = 0;
      while (mf >= 2.0 && iter < 200) begin mf = mf / 2.0; ei = ei + 1; iter = iter + 1; end
      while (mf < 1.0 && iter < 200) begin mf = mf * 2.0; ei = ei - 1; iter = iter + 1; end
      mb_raw = $rtoi((mf - 1.0) * 8388608.0 + 0.5);
      if (mb_raw >= 8388608) begin mb = 23'd0; ei = ei + 1; end
      else mb = mb_raw[22:0];
      ei = ei + 127;
      if (ei <= 0) real_to_fp32 = 32'd0;
      else if (ei >= 255) real_to_fp32 = {s, 8'hFE, 23'h7FFFFF};
      else begin eb = ei[7:0]; real_to_fp32 = {s, eb, mb}; end
    end
  endfunction

  // ═══════════════════════════════════════════════════════════
  // State machine
  // ═══════════════════════════════════════════════════════════
  typedef enum logic [2:0] {
    ST_IDLE,
    ST_LATCH_HP,   // latch hyperparams
    ST_COMPUTE,    // iterate over DIM×DIM elements
    ST_DONE,
    ST_ERROR
  } state_t;

  state_t st;
  logic [4:0] row_cnt, col_cnt;  // 0..DIM-1
  logic       adamw_r;

  // Latched hyperparams as real
  real lr_r, beta1_r, beta2_r, eps_r, wd_r;

  assign busy = (st != ST_IDLE && st != ST_DONE && st != ST_ERROR);

  // ═══════════════════════════════════════════════════════════
  // Element-wise Adam/AdamW update (1 element per cycle)
  // ═══════════════════════════════════════════════════════════

  integer oi, oj;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st         <= ST_IDLE;
      row_cnt    <= 5'd0;
      col_cnt    <= 5'd0;
      adamw_r    <= 1'b0;
      done_pulse <= 1'b0;
      err_code   <= 8'd0;
      lr_r       <= 0.0;
      beta1_r    <= 0.0;
      beta2_r    <= 0.0;
      eps_r      <= 0.0;
      wd_r       <= 0.0;
      for (oi = 0; oi < DIM; oi++)
        for (oj = 0; oj < DIM; oj++) begin
          param_out[oi][oj] <= 32'd0;
          m_out[oi][oj]     <= 32'd0;
          v_out[oi][oj]     <= 32'd0;
        end
    end else begin
      done_pulse <= 1'b0;

      case (st)
        ST_IDLE: begin
          err_code <= 8'd0;
          if (start) begin
            adamw_r <= adamw_enable;
            st      <= ST_LATCH_HP;
          end
        end

        ST_LATCH_HP: begin
          lr_r    <= fp32_to_real(lr_fp32);
          beta1_r <= fp32_to_real(beta1_fp32);
          beta2_r <= fp32_to_real(beta2_fp32);
          eps_r   <= fp32_to_real(epsilon_fp32);
          wd_r    <= fp32_to_real(weight_decay_fp32);
          row_cnt <= 5'd0;
          col_cnt <= 5'd0;
          st      <= ST_COMPUTE;
        end

        ST_COMPUTE: begin
          // Adam/AdamW update for element [row_cnt][col_cnt]
          automatic real g, p, m, v;
          automatic real m_new, v_new, p_new;
          automatic real update;

          g = fp32_to_real(grad_in[row_cnt[3:0]][col_cnt[3:0]]);
          p = fp32_to_real(param_in[row_cnt[3:0]][col_cnt[3:0]]);
          m = fp32_to_real(m_in[row_cnt[3:0]][col_cnt[3:0]]);
          v = fp32_to_real(v_in[row_cnt[3:0]][col_cnt[3:0]]);

          // m_t = beta1 * m + (1-beta1) * g
          m_new = beta1_r * m + (1.0 - beta1_r) * g;

          // v_t = beta2 * v + (1-beta2) * g^2
          v_new = beta2_r * v + (1.0 - beta2_r) * g * g;

          // update = lr * m_new / (sqrt(v_new) + eps)
          update = lr_r * m_new / ($sqrt(v_new > 0.0 ? v_new : 0.0) + eps_r);

          // p_new = p - update [Adam]
          // p_new = p - update - lr * wd * p [AdamW]
          if (adamw_r)
            p_new = p - update - lr_r * wd_r * p;
          else
            p_new = p - update;

          param_out[row_cnt[3:0]][col_cnt[3:0]] <= real_to_fp32(p_new);
          m_out[row_cnt[3:0]][col_cnt[3:0]]     <= real_to_fp32(m_new);
          v_out[row_cnt[3:0]][col_cnt[3:0]]     <= real_to_fp32(v_new);

          // Advance
          if (col_cnt == DIM - 1) begin
            col_cnt <= 5'd0;
            if (row_cnt == DIM - 1) begin
              st <= ST_DONE;
            end else begin
              row_cnt <= row_cnt + 1;
            end
          end else begin
            col_cnt <= col_cnt + 1;
          end
        end

        ST_DONE: begin
          done_pulse <= 1'b1;
          st         <= ST_IDLE;
        end

        ST_ERROR: begin
          done_pulse <= 1'b1;
          st         <= ST_IDLE;
        end

        default: st <= ST_IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
