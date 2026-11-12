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
//
// 산술(bf16_mul / fp32_add)은 rtl/bf16_pkg.sv 로 옮겼다. 이 모듈 안에만
// 있던 탓에 mxu_bf16_128x128 이 누적 덧셈을 따로 구현했고 거기서 버그가
// 났다. 이제 두 모듈이 같은 함수를 쓴다.
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

  import bf16_pkg::bf16_mul;
  import bf16_pkg::fp32_add;

  assign busy = en;

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
