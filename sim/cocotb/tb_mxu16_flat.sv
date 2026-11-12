// tb_mxu16_flat.sv — 검증 전용 래퍼.
// Icarus 의 VPI 가 2차원 unpacked 배열을 노출하지 못해서,
// mxu_bf16_16x16 의 acc_out[16][16] 을 1차원 acc_flat[256] 으로 평탄화한다.
// 합성 대상이 아니다.
`timescale 1ns/1ps
`default_nettype none

module tb_mxu16_flat #(
  parameter int N = 16
)(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        en,
  input  logic        acc_clr,
  input  logic [15:0] a_row [0:N-1],
  input  logic [15:0] b_col [0:N-1],
  output logic [31:0] acc_flat [0:N*N-1],
  output logic        busy
);
  logic [31:0] acc2d [0:N-1][0:N-1];

  mxu_bf16_16x16 #(.ROWS(N), .COLS(N)) u_dut (
    .clk(clk), .rst_n(rst_n), .en(en), .acc_clr(acc_clr),
    .a_row(a_row), .b_col(b_col), .acc_out(acc2d), .busy(busy)
  );

  genvar i, j;
  generate
    for (i = 0; i < N; i++) begin : FR
      for (j = 0; j < N; j++) begin : FC
        assign acc_flat[i*N + j] = acc2d[i][j];
      end
    end
  endgenerate
endmodule

`default_nettype wire
