// gemm_int4_synth.sv — Synthesizable INT4 GEMM for Arty A7-100T
// Identical port interface to gemm_int4.sv.
//
// Strategy:
//   - INT4 weights (4-bit two's complement) sign-extended to 8-bit
//   - INT8 activations
//   - INT32 accumulation: a[i][k] * b[k][j] accumulates directly in INT32
//   - FP16 scale converted to Q8.8 fixed-point: scale_q88[j]
//     FP16 -> Q8.8:  extract sign/exp/mantissa, shift mantissa to Q8.8 position
//   - Final output: c[i][j] = acc_int32[i][j] * scale_q88[j] (Q8.8 result stored in INT32 field)
//
// No `real`, $itor, $rtoi, $exp, $sqrt — fully synthesizable.
//
`timescale 1ns/1ps
`default_nettype none

module gemm_int4_synth #(
  parameter int TILE = 16
)(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  output logic        busy,
  output logic        done,

  // A matrix: INT8 activations [TILE][TILE], flat packed (8-bit each)
  input  logic [TILE*TILE*8-1:0]  a_flat,

  // B matrix: INT4 weights [TILE][TILE], flat packed (4-bit each)
  input  logic [TILE*TILE*4-1:0]  b_flat,

  // Scale factors: one FP16 scale per column j
  input  logic [TILE*16-1:0]      scale_flat,

  // Output: INT32 accumulator [TILE][TILE], flat packed (32-bit each)
  output logic [TILE*TILE*32-1:0] c_flat
);

  // ─── FP16 -> Q8.8 fixed-point conversion (combinational, integer-only) ───
  //
  // FP16 layout: [15]=sign, [14:10]=exp(bias 15), [9:0]=mantissa
  //
  // Algorithm:
  //   1. Extract sign, biased_exp, mantissa
  //   2. If biased_exp==0: denormal -> value ~= 0 (Q8.8 = 0)
  //   3. If biased_exp==31: Inf/NaN -> saturate to max Q8.8 (0x7FFF)
  //   4. Normal: unbiased_exp = biased_exp - 15
  //      full_mantissa (11-bit) = {1, mantissa[9:0]}  (* implied leading 1)
  //      Q8.8 value = full_mantissa shifted by (unbiased_exp - 2)
  //        (mantissa is 1.MMMM -> in Q8.8 that's value*256/1024 with exp adjustment)
  //      More precisely:
  //        value = 1.mantissa * 2^(exp-15)
  //        Q8.8  = value * 256 = 1.mantissa * 2^(exp-15+8)
  //               = full_mantissa_11bit * 2^(exp-15+8-10)
  //               = full_mantissa_11bit * 2^(exp - 17)
  //      Let shift = exp - 17:
  //        if shift >= 0: Q8.8 = full_mantissa << shift   (cap at +16 bits = 26)
  //        if shift <  0: Q8.8 = full_mantissa >> (-shift)
  //      Saturate result to 15-bit magnitude, apply sign.
  //
  function automatic signed [15:0] fp16_to_q88;
    input logic [15:0] fp16;
    logic        s;
    logic [4:0]  be;
    logic [9:0]  mant;
    logic [10:0] full_mant; // 1.mantissa in 11-bit integer
    integer      exp_unbiased;
    integer      shift;
    logic [31:0] shifted_val;
    logic signed [15:0] result;
    begin
      s    = fp16[15];
      be   = fp16[14:10];
      mant = fp16[9:0];

      if (be == 5'b11111) begin
        // Inf or NaN -> saturate
        result = s ? 16'sh8000 : 16'sh7FFF;
      end else if (be == 5'b00000) begin
        // Denormal or zero -> 0
        result = 16'sh0000;
      end else begin
        full_mant    = {1'b1, mant};          // 11-bit: 1.mantissa
        exp_unbiased = int'(be) - 15;
        shift        = exp_unbiased - 17;     // see derivation above

        if (shift >= 0) begin
          if (shift > 20) shift = 20;         // prevent overflow beyond 32-bit
          shifted_val = {21'd0, full_mant} << shift;
        end else begin
          if (-shift >= 11) begin
            shifted_val = 32'd0;              // fully shifted out
          end else begin
            shifted_val = {21'd0, full_mant} >> (-shift);
          end
        end

        // Saturate to signed 15-bit magnitude (Q8.8 max positive = 0x7FFF = 32767)
        if (shifted_val > 32'd32767)
          result = 16'sh7FFF;
        else
          result = signed'(shifted_val[15:0]);

        if (s) result = -result;
      end

      fp16_to_q88 = result;
    end
  endfunction

  // ─── Unpack A: INT8 signed activations ───────────────────────────────────
  logic signed [7:0] a_raw [0:TILE-1][0:TILE-1];

  genvar gi, gj;
  generate
    for (gi = 0; gi < TILE; gi = gi+1) begin : GEN_A_I
      for (gj = 0; gj < TILE; gj = gj+1) begin : GEN_A_J
        assign a_raw[gi][gj] = signed'(a_flat[(gi*TILE+gj)*8 +: 8]);
      end
    end
  endgenerate

  // ─── Unpack B: INT4 sign-extended to 8-bit signed ────────────────────────
  logic signed [7:0] b_raw [0:TILE-1][0:TILE-1];

  genvar bi, bj;
  generate
    for (bi = 0; bi < TILE; bi = bi+1) begin : GEN_B_I
      for (bj = 0; bj < TILE; bj = bj+1) begin : GEN_B_J
        logic [3:0] nibble;
        assign nibble        = b_flat[(bi*TILE+bj)*4 +: 4];
        assign b_raw[bi][bj] = {{4{nibble[3]}}, nibble};
      end
    end
  endgenerate

  // ─── Unpack scale: FP16 -> Q8.8 (combinational) ──────────────────────────
  logic signed [15:0] scale_q88 [0:TILE-1];

  genvar sj;
  generate
    for (sj = 0; sj < TILE; sj = sj+1) begin : GEN_SCALE
      assign scale_q88[sj] = fp16_to_q88(scale_flat[sj*16 +: 16]);
    end
  endgenerate

  // ─── Accumulators: INT32 (no real) ───────────────────────────────────────
  logic signed [31:0] acc     [0:TILE-1][0:TILE-1]; // raw integer dot-product
  logic signed [31:0] c_out   [0:TILE-1][0:TILE-1]; // final scaled output

  // ─── State machine ────────────────────────────────────────────────────────
  typedef enum logic [1:0] {
    S_IDLE    = 2'd0,
    S_COMPUTE = 2'd1,
    S_SCALE   = 2'd2,
    S_DONE    = 2'd3
  } state_t;

  state_t st;

  logic [$clog2(TILE)-1:0] ci, cj;

  integer ii, jj, kk;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st   <= S_IDLE;
      ci   <= '0;
      cj   <= '0;
      busy <= 1'b0;
      done <= 1'b0;
      for (ii = 0; ii < TILE; ii = ii+1) begin
        for (jj = 0; jj < TILE; jj = jj+1) begin
          acc[ii][jj]   <= 32'sd0;
          c_out[ii][jj] <= 32'sd0;
        end
      end
    end else begin
      case (st)

        S_IDLE: begin
          done <= 1'b0;
          if (start) begin
            busy <= 1'b1;
            for (ii = 0; ii < TILE; ii = ii+1)
              for (jj = 0; jj < TILE; jj = jj+1)
                acc[ii][jj] <= 32'sd0;
            ci <= '0;
            cj <= '0;
            st <= S_COMPUTE;
          end
        end

        S_COMPUTE: begin
          // One output element per clock: compute acc[ci][cj] = sum_k( a[ci][k] * b[k][cj] )
          // All integer arithmetic — INT8 * INT8 -> INT16, accumulated to INT32
          begin : COMPUTE_ELEM
            logic signed [31:0] dot;
            dot = 32'sd0;
            for (kk = 0; kk < TILE; kk = kk+1) begin
              // 8-bit * 8-bit signed = 16-bit, extended to 32-bit for accumulation
              dot = dot + (32'(signed'(a_raw[ci][kk])) * 32'(signed'(b_raw[kk][cj])));
            end
            acc[ci][cj] <= dot;
          end

          // Advance (ci, cj)
          if (cj == TILE-1) begin
            cj <= '0;
            if (ci == TILE-1) begin
              ci <= '0;
              st <= S_SCALE;
            end else begin
              ci <= ci + 1'b1;
            end
          end else begin
            cj <= cj + 1'b1;
          end
        end

        S_SCALE: begin
          // Apply per-column Q8.8 scale: c_out[i][j] = acc[i][j] * scale_q88[j] >> 8
          // acc[i][j] is INT32, scale_q88[j] is Q8.8 (signed 16-bit)
          // product is 48-bit, we keep the upper 32 bits (>> 8)
          for (ii = 0; ii < TILE; ii = ii+1) begin
            for (jj = 0; jj < TILE; jj = jj+1) begin
              begin : SCALE_ELEM
                logic signed [47:0] product48;
                product48      = acc[ii][jj] * {{16{scale_q88[jj][15]}}, scale_q88[jj]};
                c_out[ii][jj] <= product48[39:8]; // shift right 8, take 32 bits
              end
            end
          end
          st <= S_DONE;
        end

        S_DONE: begin
          busy <= 1'b0;
          done <= 1'b1;
          st   <= S_IDLE;
        end

        default: begin
          st <= S_IDLE;
        end

      endcase
    end
  end

  // ─── Pack c_out -> c_flat ────────────────────────────────────────────────
  genvar oi, oj;
  generate
    for (oi = 0; oi < TILE; oi = oi+1) begin : GEN_C_I
      for (oj = 0; oj < TILE; oj = oj+1) begin : GEN_C_J
        assign c_flat[(oi*TILE+oj)*32 +: 32] = c_out[oi][oj];
      end
    end
  endgenerate

endmodule
`default_nettype wire
