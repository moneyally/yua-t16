// mxu_bf16_16x16.sv — BF16 16×16 Matrix Multiply Unit for ORBIT-G3
// SSOT: ORBIT_G3_RTL_PLAN.md (G3-RTL-003), ORBIT_G3_ARCHITECTURE.md
//
// Computes C[16×16] += A[16×K] × B[K×16] in BF16, accumulated in FP32.
// One K-step per cycle: 256 BF16 MACs/cycle.
//
// Interface:
//   - a_row[0:15]: 16 BF16 values (one row of A, column k)
//   - b_col[0:15]: 16 BF16 values (one column of B, row k)
//   - acc_out[0:15][0:15]: 16×16 FP32 accumulator (256 values)
//
// Pipeline: 2-stage
//   Stage 1: BF16 × BF16 → FP32 product (combinational multiply)
//   Stage 2: FP32 + FP32 → FP32 accumulate (registered add)
//
// BF16 format: {sign[15], exponent[14:7], mantissa[6:0]}
//   Multiply: sign XOR, exponent add-127, mantissa 8×8 → 16-bit product
//   Then convert to FP32 for accumulation.
`timescale 1ns/1ps
`default_nettype none

module mxu_bf16_16x16 #(
  parameter int ROWS = 16,
  parameter int COLS = 16
)(
  input  logic        clk,
  input  logic        rst_n,

  // Control
  input  logic        en,           // MAC enable (1 = accumulate this cycle)
  input  logic        acc_clr,      // clear accumulators to 0

  // Input: one column-step of A and B
  input  logic [15:0] a_row [0:ROWS-1],   // 16 BF16 values
  input  logic [15:0] b_col [0:COLS-1],   // 16 BF16 values

  // Output: 16×16 FP32 accumulator
  output logic [31:0] acc_out [0:ROWS-1][0:COLS-1],

  // Status
  output logic        busy
);

  assign busy = en;

  // ═══════════════════════════════════════════════════════════
  // BF16 multiply → FP32 product (combinational)
  // ═══════════════════════════════════════════════════════════
  //
  // BF16: {s[15], e[14:7], m[6:0]}
  //   implied 1.m[6:0] = 8-bit mantissa with hidden bit
  //   exponent bias = 127
  //
  // Product:
  //   sign = a.s ^ b.s
  //   exp  = a.e + b.e - 127
  //   mant = (1.a.m) × (1.b.m) = 16-bit, normalize to 1.xxx...
  //   Result as FP32: {sign, exp[7:0], mant[22:0]}

  function automatic logic [31:0] bf16_mul(input logic [15:0] a, input logic [15:0] b);
    logic        a_sign, b_sign, r_sign;
    logic [7:0]  a_exp, b_exp;
    logic [6:0]  a_man, b_man;
    logic [8:0]  r_exp_raw;
    logic [15:0] man_prod;  // 8×8 = 16 bits
    logic [22:0] r_man;
    logic [7:0]  r_exp;

    a_sign = a[15];
    b_sign = b[15];
    a_exp  = a[14:7];
    b_exp  = b[14:7];
    a_man  = a[6:0];
    b_man  = b[6:0];

    r_sign = a_sign ^ b_sign;

    // Zero check
    if (a_exp == 8'd0 || b_exp == 8'd0) begin
      bf16_mul = 32'd0;
      // end case
    end

    // Inf/NaN check (simplified: treat as max finite)
    if (a_exp == 8'hFF || b_exp == 8'hFF) begin
      bf16_mul = {r_sign, 8'hFE, 23'h7FFFFF};
      // end case
    end

    // Mantissa multiply: (1.a_man) × (1.b_man)
    man_prod = {1'b1, a_man} * {1'b1, b_man};
    // man_prod is 16 bits: [15] is always 1 if both inputs nonzero
    // Format: xx.xxxxxxxxxxxxxx (2 integer bits, 14 fraction)

    // Exponent
    r_exp_raw = {1'b0, a_exp} + {1'b0, b_exp} - 9'd127;

    // Normalize: if man_prod[15]==1, shift right by 1, exp+1
    if (man_prod[15]) begin
      r_man = {man_prod[14:0], 8'd0}; // 15 bits → 23 bits (pad low)
      r_exp_raw = r_exp_raw + 1;
    end else begin
      r_man = {man_prod[13:0], 9'd0}; // 14 bits → 23 bits (pad low)
    end

    // Clamp exponent
    if (r_exp_raw[8] || r_exp_raw == 9'd0) begin
      // Underflow
      bf16_mul = 32'd0;
      // end case
    end
    if (r_exp_raw >= 9'd255) begin
      // Overflow → max finite
      bf16_mul = {r_sign, 8'hFE, 23'h7FFFFF};
      // end case
    end

    r_exp = r_exp_raw[7:0];
    bf16_mul = {r_sign, r_exp, r_man};
  endfunction

  // ═══════════════════════════════════════════════════════════
  // FP32 add (simplified, for accumulation)
  // ═══════════════════════════════════════════════════════════

  function automatic logic [31:0] fp32_add(input logic [31:0] a, input logic [31:0] b);
    // All declarations at top (verilator requirement)
    logic a_s, b_s, r_s;
    logic [7:0] a_e, b_e, r_e, exp_diff;
    logic [22:0] a_m, b_m, r_m;
    logic [24:0] a_full, b_full;
    logic [25:0] sum;

    a_s = a[31]; a_e = a[30:23]; a_m = a[22:0];
    b_s = b[31]; b_e = b[30:23]; b_m = b[22:0];

    if (a_e == 0 && a_m == 0) begin
      fp32_add = b;
      // end case
    end
    if (b_e == 0 && b_m == 0) begin
      fp32_add = a;
      // end case
    end

    a_full = {1'b1, a_m, 1'b0};
    b_full = {1'b1, b_m, 1'b0};

    if (a_e >= b_e) begin
      exp_diff = a_e - b_e;
      if (exp_diff > 25) b_full = 0;
      else b_full = b_full >> exp_diff;
      r_e = a_e;

      if (a_s == b_s) begin
        sum = {1'b0, a_full} + {1'b0, b_full};
        r_s = a_s;
      end else begin
        if (a_full >= b_full) begin
          sum = {1'b0, a_full} - {1'b0, b_full};
          r_s = a_s;
        end else begin
          sum = {1'b0, b_full} - {1'b0, a_full};
          r_s = b_s;
        end
      end
    end else begin
      exp_diff = b_e - a_e;
      if (exp_diff > 25) a_full = 0;
      else a_full = a_full >> exp_diff;
      r_e = b_e;

      if (a_s == b_s) begin
        sum = {1'b0, b_full} + {1'b0, a_full};
        r_s = a_s;
      end else begin
        if (b_full >= a_full) begin
          sum = {1'b0, b_full} - {1'b0, a_full};
          r_s = b_s;
        end else begin
          sum = {1'b0, a_full} - {1'b0, b_full};
          r_s = a_s;
        end
      end
    end

    // Normalize
    if (sum == 0) begin
      fp32_add = 32'd0;
      // end case
    end

    if (sum[25]) begin
      sum = sum >> 1;
      r_e = r_e + 1;
    end else begin
      while (!sum[24] && r_e > 0) begin
        sum = sum << 1;
        r_e = r_e - 1;
      end
    end

    r_m = sum[23:1]; // drop hidden bit and guard
    if (r_e >= 8'hFF) begin
      fp32_add = {r_s, 8'hFE, 23'h7FFFFF}; // overflow clamp
    end else begin
      fp32_add = {r_s, r_e, r_m};
    end
  endfunction

  // ═══════════════════════════════════════════════════════════
  // 16×16 MAC array
  // ═══════════════════════════════════════════════════════════

  genvar r, c;
  generate
    for (r = 0; r < ROWS; r = r + 1) begin : GEN_ROW
      for (c = 0; c < COLS; c = c + 1) begin : GEN_COL

        // Product: BF16 × BF16 → FP32 (combinational)
        logic [31:0] product;
        assign product = bf16_mul(a_row[r], b_col[c]);

        // Accumulator: FP32 register
        always_ff @(posedge clk or negedge rst_n) begin
          if (!rst_n) begin
            acc_out[r][c] <= 32'd0;
          end else if (acc_clr) begin
            acc_out[r][c] <= 32'd0;
          end else if (en) begin
            acc_out[r][c] <= fp32_add(acc_out[r][c], product);
          end
        end

      end
    end
  endgenerate

endmodule

`default_nettype wire
