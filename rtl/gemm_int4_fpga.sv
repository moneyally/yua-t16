// gemm_int4_fpga.sv — FPGA-optimized INT4 GEMM for Arty A7-100T
// Identical port interface to gemm_int4_synth.sv.
//
// Key differences from gemm_int4_synth.sv (DSP-heavy):
//   1. COMPUTE: INT8 × INT4 via shift-and-add (zero DSP) instead of * operator
//   2. SCALE:   Serialized one element/clock → only 2 DSP48E1 (32×16→48)
//
// Resource target (Arty A7-100T, xc7a100tcsg324-1):
//   DSP48E1  ≤ 4    (budget: 90 total, VPU uses 11 → 79 remain → 80% = 72)
//   LUT      ≤ ~8K  (COMPUTE logic + SCALE mux)
//   FF       ≤ ~4K  (accumulators + state)
//
// Latency (TILE=16):
//   COMPUTE: 16×16 = 256 cycles  (one output element per cycle)
//   SCALE:   16×16 = 256 cycles  (one element per cycle)
//   Total:   ~514 cycles @ 150 MHz ≈ 3.4 µs per 16×16 tile
//
`timescale 1ns/1ps
`default_nettype none

module gemm_int4_fpga #(
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

  // ─── FP16 -> Q8.8 conversion (same as gemm_int4_synth) ──────────────────
  function automatic signed [15:0] fp16_to_q88;
    input logic [15:0] fp16;
    logic        s;
    logic [4:0]  be;
    logic [9:0]  mant;
    logic [10:0] full_mant;
    integer      shift;
    logic [31:0] sv;
    logic signed [15:0] result;
    begin
      s    = fp16[15];
      be   = fp16[14:10];
      mant = fp16[9:0];
      if (be == 5'b11111) begin
        result = s ? 16'sh8000 : 16'sh7FFF;
      end else if (be == 5'b00000) begin
        result = 16'sh0000;
      end else begin
        full_mant = {1'b1, mant};
        shift     = int'(be) - 17;
        if (shift >= 0) begin
          if (shift > 20) shift = 20;
          sv = {21'd0, full_mant} << shift;
        end else begin
          if (-shift >= 11) sv = 32'd0;
          else              sv = {21'd0, full_mant} >> (-shift);
        end
        result = (sv > 32'd32767) ? 16'sh7FFF : signed'(sv[15:0]);
        if (s) result = -result;
      end
      fp16_to_q88 = result;
    end
  endfunction

  // ─── INT8 × INT4 shift-and-add (NO DSP) ──────────────────────────────────
  // b_nibble is 4-bit two's complement: value = -8*b[3] + 4*b[2] + 2*b[1] + b[0]
  // a × b = -a*8*b[3] + a*4*b[2] + a*2*b[1] + a*b[0]
  //       = conditional shifts + signed add  → synthesizes to LUT+CARRY chain
  // Result: 12-bit signed (INT8 × [-8..7] → [-1024..889])
  function automatic signed [11:0] mul_i8_i4;
    input signed [7:0]  a;
    input        [3:0]  b_nibble;
    logic signed [11:0] p0, p1, p2, p3;
    begin
      p0 = b_nibble[0] ? {{4{a[7]}}, a}       : 12'sd0;  // a * 1
      p1 = b_nibble[1] ? {{3{a[7]}}, a, 1'b0} : 12'sd0;  // a * 2
      p2 = b_nibble[2] ? {{2{a[7]}}, a, 2'b0} : 12'sd0;  // a * 4
      p3 = b_nibble[3] ? {{1{a[7]}}, a, 3'b0} : 12'sd0;  // a * 8 (sign bit)
      mul_i8_i4 = p0 + p1 + p2 - p3;  // subtract p3 (b[3] is sign bit, weight -8)
    end
  endfunction

  // ─── Unpack A (INT8) ──────────────────────────────────────────────────────
  wire signed [7:0] a_raw [0:TILE-1][0:TILE-1];
  genvar gi, gj;
  generate
    for (gi = 0; gi < TILE; gi = gi+1) begin : UA_I
      for (gj = 0; gj < TILE; gj = gj+1) begin : UA_J
        assign a_raw[gi][gj] = signed'(a_flat[(gi*TILE+gj)*8 +: 8]);
      end
    end
  endgenerate

  // ─── Unpack B (INT4 nibbles) ──────────────────────────────────────────────
  wire [3:0] b_nib [0:TILE-1][0:TILE-1];
  genvar bi, bj;
  generate
    for (bi = 0; bi < TILE; bi = bi+1) begin : UB_I
      for (bj = 0; bj < TILE; bj = bj+1) begin : UB_J
        assign b_nib[bi][bj] = b_flat[(bi*TILE+bj)*4 +: 4];
      end
    end
  endgenerate

  // ─── Unpack scale FP16 -> Q8.8 ────────────────────────────────────────────
  wire signed [15:0] scale_q88 [0:TILE-1];
  genvar sj;
  generate
    for (sj = 0; sj < TILE; sj = sj+1) begin : GS
      assign scale_q88[sj] = fp16_to_q88(scale_flat[sj*16 +: 16]);
    end
  endgenerate

  // ─── Accumulators ─────────────────────────────────────────────────────────
  logic signed [31:0] acc   [0:TILE-1][0:TILE-1];
  logic signed [31:0] c_out [0:TILE-1][0:TILE-1];

  // ─── State machine ────────────────────────────────────────────────────────
  localparam logic [1:0] S_IDLE    = 2'd0;
  localparam logic [1:0] S_COMPUTE = 2'd1;
  localparam logic [1:0] S_SCALE   = 2'd2;
  localparam logic [1:0] S_DONE    = 2'd3;

  logic [1:0]              st;
  logic [$clog2(TILE)-1:0] ci, cj;

  integer ii, jj, kk;

  // ── COMPUTE: dot product for (ci,cj) using shift-and-add muls (no DSP) ───
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st   <= S_IDLE;
      ci   <= '0;
      cj   <= '0;
      busy <= 1'b0;
      done <= 1'b0;
      for (ii = 0; ii < TILE; ii = ii+1)
        for (jj = 0; jj < TILE; jj = jj+1) begin
          acc[ii][jj]   <= 32'sd0;
          c_out[ii][jj] <= 32'sd0;
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
          // Compute acc[ci][cj] = sum_k( a[ci][k] * b[k][cj] )
          // Uses shift-and-add mul_i8_i4 — no DSP48E1
          begin : COMPUTE_ELEM
            logic signed [31:0] dot;
            dot = 32'sd0;
            for (kk = 0; kk < TILE; kk = kk+1) begin
              dot = dot + 32'(mul_i8_i4(a_raw[ci][kk], b_nib[kk][cj]));
            end
            acc[ci][cj] <= dot;
          end

          if (cj == TILE[($clog2(TILE))-1:0] - 1'b1) begin
            cj <= '0;
            if (ci == TILE[($clog2(TILE))-1:0] - 1'b1) begin
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
          // Serialized: one element per clock.
          // Uses a single 32×16→48 multiply which maps to ≤2 DSP48E1.
          begin : SCALE_ELEM
            logic signed [47:0] product48;
            product48       = acc[ci][cj] * {{16{scale_q88[cj][15]}}, scale_q88[cj]};
            c_out[ci][cj]  <= product48[39:8];
          end

          if (cj == TILE[($clog2(TILE))-1:0] - 1'b1) begin
            cj <= '0;
            if (ci == TILE[($clog2(TILE))-1:0] - 1'b1) begin
              ci <= '0;
              st <= S_DONE;
            end else begin
              ci <= ci + 1'b1;
            end
          end else begin
            cj <= cj + 1'b1;
          end
        end

        S_DONE: begin
          busy <= 1'b0;
          done <= 1'b1;
          st   <= S_IDLE;
        end

        default: st <= S_IDLE;

      endcase
    end
  end

  // ─── Pack output ──────────────────────────────────────────────────────────
  genvar oi, oj;
  generate
    for (oi = 0; oi < TILE; oi = oi+1) begin : CO_I
      for (oj = 0; oj < TILE; oj = oj+1) begin : CO_J
        assign c_flat[(oi*TILE+oj)*32 +: 32] = c_out[oi][oj];
      end
    end
  endgenerate

endmodule
`default_nettype wire
