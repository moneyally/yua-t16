// pe_bench.sv — PE 1개의 누산 경로 면적 측정용 벤치 (합성 전용, 시뮬 아님)
//
// mxu_bf16_16x16 은 동일한 PE 를 256개 인스턴스화한다. PE 하나의 비용이
// 타일 비용을 지배하므로, 재설계 전/후 비교는 여기서 한다.
// Yosys 0.33 이 unpacked 배열 포트를 파싱하지 못해 타일 자체는 합성할 수
// 없다. 포트를 전부 packed 로 둔 이유다.
`timescale 1ns/1ps
`default_nettype none

// ── 기존 PE: BF16 곱 → FP32 부동소수점 누산 ──────────────────────
module pe_bench_fp32 (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        en,
  input  logic        acc_clr,
  input  logic [15:0] a,
  input  logic [15:0] b,
  output logic [31:0] acc
);
  logic [31:0] product;
  assign product = bf16_pkg::bf16_mul(a, b);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)        acc <= 32'd0;
    else if (acc_clr)  acc <= 32'd0;
    else if (en)       acc <= bf16_pkg::fp32_add(acc, product);
  end
endmodule

`default_nettype wire

// ── 재설계 PE: BF16 곱 → 고정소수점 누산 → 출력에서 FP32 변환 ──────
module pe_bench_fxp (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        en,
  input  logic        acc_clr,
  input  logic [15:0] a,
  input  logic [15:0] b,
  output logic [31:0] acc
);
  logic signed [bf16_pkg::ACC_W-1:0] incr, acc_fxp;
  assign incr = bf16_pkg::bf16_mul_fxp(a, b);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)       acc_fxp <= '0;
    else if (acc_clr) acc_fxp <= '0;
    else if (en)      acc_fxp <= bf16_pkg::fxp_add_sat(acc_fxp, incr);
  end

  assign acc = bf16_pkg::fxp_to_fp32(acc_fxp);
endmodule

// 누산 루프만 (출력 변환 제외) — 임계 경로에 실제로 들어가는 부분
module pe_bench_fxp_loop (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        en,
  input  logic        acc_clr,
  input  logic [15:0] a,
  input  logic [15:0] b,
  output logic signed [bf16_pkg::ACC_W-1:0] acc_fxp
);
  logic signed [bf16_pkg::ACC_W-1:0] incr;
  assign incr = bf16_pkg::bf16_mul_fxp(a, b);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)       acc_fxp <= '0;
    else if (acc_clr) acc_fxp <= '0;
    else if (en)      acc_fxp <= bf16_pkg::fxp_add_sat(acc_fxp, incr);
  end
endmodule
