// vpu_core.sv — Behavioral VPU for Icarus Verilog 12
// Uses blocking assignments throughout (real types + logic arrays).
// src_r/aux_r unpacked via generate/assign outside always blocks.
`timescale 1ns/1ps
`default_nettype none

module vpu_core #(
  parameter int DEPTH = 256
)(
  input  logic        clk,
  input  logic        rst_n,

  input  logic [3:0]  op_type,
  input  logic [15:0] vec_len,
  input  logic [15:0] imm_fp16_0,
  input  logic [15:0] imm_fp16_1,

  input  logic        start,
  output logic        busy,
  output logic        done,

  input  logic [16*DEPTH-1:0] src_flat,
  input  logic [16*DEPTH-1:0] aux_flat,
  output logic [16*DEPTH-1:0] dst_flat
);

  // ── Unpack src/aux via continuous assigns (outside always) ───────────────
  wire [15:0] src_r [0:DEPTH-1];
  wire [15:0] aux_r [0:DEPTH-1];
  reg  [15:0] dst_r [0:DEPTH-1];

  genvar gi;
  generate
    for (gi = 0; gi < DEPTH; gi = gi+1) begin : UNPACK
      assign src_r[gi]             = src_flat[gi*16 +: 16];
      assign aux_r[gi]             = aux_flat[gi*16 +: 16];
      assign dst_flat[gi*16 +: 16] = dst_r[gi];
    end
  endgenerate

  // ── FP helpers ──────────────────────────────────────────────────────────
  function automatic real f16r;
    input [15:0] b;
    reg [4:0] e5; reg [9:0] m; reg s;
    real r; integer ei;
    begin
      s = b[15]; e5 = b[14:10]; m = b[9:0];
      if (e5 == 5'h1F) begin
        r = s ? -65504.0 : 65504.0;
      end else if (e5 == 5'h00) begin
        r = $itor(m) * 5.960464e-8;
        if (s) r = -r;
      end else begin
        ei = e5 - 15;
        r  = 1.0 + $itor(m) / 1024.0;
        if (ei >= 0) r = r * (2.0 ** ei);
        else         r = r / (2.0 ** (-ei));
        if (s) r = -r;
      end
      f16r = r;
    end
  endfunction

  function automatic [15:0] rf16;
    input real v;
    reg s; real a, mf; integer ei, iter;
    reg [9:0] mb; reg [4:0] eb;
    begin
      s = (v < 0.0) ? 1'b1 : 1'b0;
      a = s ? -v : v;
      if (a == 0.0)     begin rf16 = {s, 15'h0};      end
      else if (a >= 65504.0) begin rf16 = {s, 5'h1E, 10'h3FF}; end
      else if (a < 5.96e-8)  begin rf16 = {s, 15'h0};      end
      else begin
        if (a < 6.104e-5) begin
          mb = $rtoi(a / 5.960464e-8 + 0.5);
          if (mb > 1023) mb = 1023;
          rf16 = {s, 5'h00, mb};
        end else begin
          ei = 0; mf = a; iter = 0;
          while (mf >= 2.0 && iter < 64) begin mf = mf/2.0; ei = ei+1; iter = iter+1; end
          while (mf <  1.0 && iter < 64) begin mf = mf*2.0; ei = ei-1; iter = iter+1; end
          begin : RF16_NORM
            integer mb_raw;
            mb_raw = $rtoi((mf - 1.0) * 1024.0 + 0.5);
            // Check overflow BEFORE truncating to 10 bits
            if (mb_raw >= 1024) begin mb = 10'h0; ei = ei+1; end
            else                begin mb = mb_raw; end
          end
          ei = ei + 15;
          if (ei <= 0)  rf16 = {s, 15'h0};
          else if (ei >= 31) rf16 = {s, 5'h1E, 10'h3FF};
          else begin eb = ei; rf16 = {s, eb, mb}; end
        end
      end
    end
  endfunction

  function automatic real sigmoid_r;
    input real x;
    begin
      if (x >  16.0) sigmoid_r = 1.0;
      else if (x < -16.0) sigmoid_r = 0.0;
      else sigmoid_r = 1.0 / (1.0 + $exp(-x));
    end
  endfunction

  // ── State registers ─────────────────────────────────────────────────────
  localparam ST_IDLE  = 3'd0;
  localparam ST_PASS1 = 3'd1;
  localparam ST_PASS2 = 3'd2;
  localparam ST_PASS3 = 3'd3;
  localparam ST_DONE  = 3'd4;

  localparam OP_ELEM_ADD    = 4'h0;
  localparam OP_ELEM_MUL    = 4'h1;
  localparam OP_SCALE       = 4'h2;
  localparam OP_RESIDUAL    = 4'h3;
  localparam OP_RMSNORM     = 4'h4;
  localparam OP_SILU        = 4'h5;
  localparam OP_ROPE        = 4'h6;
  localparam OP_SOFTMAX     = 4'h7;
  localparam OP_CLAMP       = 4'h8;
  localparam OP_GELU_APPROX = 4'h9;

  reg [2:0]  state;
  reg [3:0]  op_r;
  reg [15:0] vlen_r;
  reg [15:0] imm0_r, imm1_r;
  reg [15:0] idx;

  real r_sum_sq, r_max, r_sum_exp, r_scale, r_sum;

  integer ii;

  // ── Main FSM (blocking assignments — behavioral style) ───────────────────
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state     = ST_IDLE;
      busy      = 1'b0;
      done      = 1'b0;
      idx       = 16'd0;
      op_r      = 4'd0;
      vlen_r    = 16'd0;
      imm0_r    = 16'd0;
      imm1_r    = 16'd0;
      r_sum_sq  = 0.0;
      r_max     = -65504.0;
      r_sum_exp = 0.0;
      r_scale   = 0.0;
      r_sum     = 0.0;
      for (ii = 0; ii < DEPTH; ii = ii+1) dst_r[ii] = 16'h0;
    end else begin
      done = 1'b0;  // default

      case (state)

        ST_IDLE: begin
          if (start) begin
            op_r      = op_type;
            vlen_r    = vec_len;
            imm0_r    = imm_fp16_0;
            imm1_r    = imm_fp16_1;
            idx       = 16'd0;
            busy      = 1'b1;
            r_sum_sq  = 0.0;
            r_max     = -65504.0;
            r_sum_exp = 0.0;
            state     = ST_PASS1;
          end
        end

        ST_PASS1: begin
          if (idx < vlen_r) begin
            // Process element idx
            begin : P1_ELEM
              real vs, va, imm0, imm1, res, xe, xo, cv, sv;
              vs   = f16r(src_r[idx]);
              va   = f16r(aux_r[idx]);
              imm0 = f16r(imm0_r);
              imm1 = f16r(imm1_r);
              case (op_r)
                OP_ELEM_ADD, OP_RESIDUAL: dst_r[idx] = rf16(vs + va);
                OP_ELEM_MUL:              dst_r[idx] = rf16(vs * va);
                OP_SCALE:                 dst_r[idx] = rf16(vs * imm0);
                OP_SILU:                  dst_r[idx] = rf16(vs * sigmoid_r(vs));
                OP_GELU_APPROX:           dst_r[idx] = rf16(vs * sigmoid_r(1.702 * vs));
                OP_CLAMP: begin
                  res = vs;
                  if (res < imm0) res = imm0;
                  if (res > imm1) res = imm1;
                  dst_r[idx] = rf16(res);
                end
                OP_RMSNORM:
                  r_sum_sq = r_sum_sq + vs * vs;
                OP_SOFTMAX:
                  if (vs > r_max) r_max = vs;
                OP_ROPE: begin
                  if (idx[0] == 1'b1) begin
                    xe = f16r(src_r[idx-1]);
                    xo = vs;
                    cv = f16r(aux_r[idx-1]);
                    sv = va;
                    dst_r[idx-1] = rf16(xe*cv - xo*sv);
                    dst_r[idx]   = rf16(xo*cv + xe*sv);
                  end
                end
                default: ;
              endcase
            end
            idx = idx + 16'd1;
          end else begin
            // Pass 1 complete
            begin : P1_END
              real ep, ms;
              case (op_r)
                OP_RMSNORM: begin
                  ep = f16r(imm0_r);
                  if (ep <= 0.0) ep = 1.0e-5;
                  ms = r_sum_sq / $itor(vlen_r) + ep;
                  r_scale = 1.0 / $sqrt(ms);
                  idx   = 16'd0;
                  state = ST_PASS2;
                end
                OP_SOFTMAX: begin
                  r_scale   = r_max;
                  r_sum_exp = 0.0;
                  idx       = 16'd0;
                  state     = ST_PASS2;
                end
                default: begin
                  state = ST_DONE;
                  busy  = 1'b0;
                  done  = 1'b1;
                end
              endcase
            end
          end
        end

        ST_PASS2: begin
          if (idx < vlen_r) begin
            begin : P2_ELEM
              real vs, va, shifted, ev;
              vs = f16r(src_r[idx]);
              va = f16r(aux_r[idx]);
              case (op_r)
                OP_RMSNORM:
                  dst_r[idx] = rf16(vs * va * r_scale);
                OP_SOFTMAX: begin
                  shifted = vs - r_scale;
                  if (shifted < -87.3) shifted = -87.3;
                  ev = $exp(shifted);
                  r_sum_exp = r_sum_exp + ev;
                  dst_r[idx] = rf16(ev);
                end
                default: ;
              endcase
            end
            idx = idx + 16'd1;
          end else begin
            case (op_r)
              OP_SOFTMAX: begin
                r_sum = r_sum_exp;
                idx   = 16'd0;
                state = ST_PASS3;
              end
              default: begin
                state = ST_DONE;
                busy  = 1'b0;
                done  = 1'b1;
              end
            endcase
          end
        end

        ST_PASS3: begin
          if (idx < vlen_r) begin
            begin : P3_ELEM
              real ev, norm;
              ev   = f16r(dst_r[idx]);
              norm = (r_sum > 0.0) ? ev / r_sum : 0.0;
              dst_r[idx] = rf16(norm);
            end
            idx = idx + 16'd1;
          end else begin
            state = ST_DONE;
            busy  = 1'b0;
            done  = 1'b1;
          end
        end

        ST_DONE:
          state = ST_IDLE;

        default:
          state = ST_IDLE;

      endcase
    end
  end

endmodule
`default_nettype wire
