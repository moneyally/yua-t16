// collective_engine.sv — ORBIT-G3 Collective Communication Primitive
// SSOT: ORBIT_G3_RTL_ISSUES.md (G3-RTL-020), ORBIT_G3_ARCHITECTURE.md
//
// 2-peer all-reduce SUM primitive.
// result[i][j] = local_in[i][j] + peer_in[i][j]
//
// Op types:
//   0x1 = ALL_REDUCE_SUM (implemented)
//   0x2 = ALL_GATHER (stub → unsupported error)
//   0x3 = REDUCE_SCATTER (stub → unsupported error)
//
// Peer data supplied via peer_in port (TB or future fabric link).
// FP32 throughout. Iterative element-by-element reduction.
`timescale 1ns/1ps
`default_nettype none

module collective_engine #(
  parameter int DIM = 16
)(
  input  logic        clk,
  input  logic        rst_n,

  // Control
  input  logic        start,
  input  logic [7:0]  op_type,       // 0x1=ALL_REDUCE_SUM, 0x2=ALL_GATHER, 0x3=REDUCE_SCATTER
  input  logic [7:0]  peer_mask,     // bit0=self, bit1=peer. Valid: 0x03 (2-peer)

  // Data
  input  logic [31:0] local_in [0:DIM-1][0:DIM-1],  // FP32
  input  logic [31:0] peer_in  [0:DIM-1][0:DIM-1],  // FP32
  output logic [31:0] result_out [0:DIM-1][0:DIM-1], // FP32

  // Status
  output logic        busy,
  output logic        done_pulse,
  output logic [7:0]  err_code   // 0=ok, 1=unsupported op, 2=invalid peer_mask
);

  // ═══════════════════════════════════════════════════════════
  // Op type constants
  // ═══════════════════════════════════════════════════════════
  localparam logic [7:0] OP_ALL_REDUCE_SUM = 8'h01;
  localparam logic [7:0] OP_ALL_GATHER     = 8'h02;
  localparam logic [7:0] OP_REDUCE_SCATTER = 8'h03;

  // Valid peer mask for 2-peer: self + 1 peer
  localparam logic [7:0] VALID_PEER_MASK = 8'h03;

  // ═══════════════════════════════════════════════════════════
  // FP32 behavioral add (same as used in MXU/optimizer)
  // ═══════════════════════════════════════════════════════════
  function automatic real fp32_to_real(input logic [31:0] bits);
    logic [7:0] exp_bits;
    logic [22:0] man_bits;
    real result;
    integer exp_val;

    exp_bits = bits[30:23];
    man_bits = bits[22:0];

    if (exp_bits == 0 && man_bits == 0) begin
      fp32_to_real = 0.0;
    end else if (exp_bits == 8'hFF) begin
      fp32_to_real = bits[31] ? -3.4e38 : 3.4e38;
    end else begin
      exp_val = exp_bits - 127;
      result = 1.0 + $itor(man_bits) / 8388608.0;
      if (exp_val >= 0)
        result = result * (2.0 ** exp_val);
      else
        result = result / (2.0 ** (-exp_val));
      if (bits[31]) result = -result;
      fp32_to_real = result;
    end
  endfunction

  function automatic logic [31:0] real_to_fp32(input real v);
    logic s;
    real a, mf;
    integer ei, iter, mb_raw;
    logic [22:0] mb;
    logic [7:0] eb;

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
    ST_VALIDATE,
    ST_REDUCE,
    ST_DONE,
    ST_ERROR
  } state_t;

  state_t st;
  logic [4:0] row_cnt, col_cnt;
  logic [7:0] op_r;
  logic [7:0] pmask_r;

  assign busy = (st != ST_IDLE && st != ST_DONE && st != ST_ERROR);

  // ═══════════════════════════════════════════════════════════
  // FSM
  // ═══════════════════════════════════════════════════════════
  integer ri, ci;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st         <= ST_IDLE;
      row_cnt    <= 5'd0;
      col_cnt    <= 5'd0;
      op_r       <= 8'd0;
      pmask_r    <= 8'd0;
      done_pulse <= 1'b0;
      err_code   <= 8'd0;
      for (ri = 0; ri < DIM; ri++)
        for (ci = 0; ci < DIM; ci++)
          result_out[ri][ci] <= 32'd0;
    end else begin
      done_pulse <= 1'b0;

      case (st)
        ST_IDLE: begin
          err_code <= 8'd0;
          if (start) begin
            op_r    <= op_type;
            pmask_r <= peer_mask;
            st      <= ST_VALIDATE;
          end
        end

        ST_VALIDATE: begin
          // Check op_type and peer_mask
          if (op_r != OP_ALL_REDUCE_SUM) begin
            err_code <= 8'd1;  // unsupported op
            st       <= ST_ERROR;
          end else if (pmask_r != VALID_PEER_MASK) begin
            err_code <= 8'd2;  // invalid peer mask
            st       <= ST_ERROR;
          end else begin
            row_cnt <= 5'd0;
            col_cnt <= 5'd0;
            st      <= ST_REDUCE;
          end
        end

        ST_REDUCE: begin
          // All-reduce SUM: result = local + peer
          automatic real l_val, p_val, sum_val;
          l_val = fp32_to_real(local_in[row_cnt[3:0]][col_cnt[3:0]]);
          p_val = fp32_to_real(peer_in[row_cnt[3:0]][col_cnt[3:0]]);
          sum_val = l_val + p_val;
          result_out[row_cnt[3:0]][col_cnt[3:0]] <= real_to_fp32(sum_val);

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
          done_pulse <= 1'b1;  // signal completion even on error
          st         <= ST_IDLE;
        end

        default: st <= ST_IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
