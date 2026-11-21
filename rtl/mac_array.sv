`timescale 1ns/1ps
`default_nettype none

// mac_array.sv — N×N INT8 MAC 배열 (외적 누산)
//
// 사이클마다 a_row (N개) 와 b_col (N개) 의 외적을 받아 N² 누산기에 더한다.
// Kt 사이클을 돌면 A(N×Kt) · B(Kt×N) 의 N×N 결과가 나온다.
//
// N 을 키우면 MAC 은 N² 로, 피연산자 바이트는 N 으로 늘어난다. 즉 산술
// 강도가 N 에 비례한다 — 대역폭 대비 연산을 늘리는 정석 경로다.
// spec/memory-bandwidth-analysis.md 참조.
module mac_array #(
  parameter int N = 16
)(
  input  logic clk,
  input  logic rst_n,
  input  logic en,
  input  logic acc_clr,
  input  logic signed [7:0] a_row [0:N-1],
  input  logic signed [7:0] b_col [0:N-1],
  // Packed flat output: acc_out_flat[IDX*32 +: 32] = result for
  // row=IDX/N, col=IDX%N.
  // Exported as a packed vector to work around the Icarus 12
  // unpacked-array-output limitation.
  output logic [32*N*N-1:0] acc_out_flat
);

  genvar i, j;
  generate
    for (i = 0; i < N; i++) begin : ROW
      for (j = 0; j < N; j++) begin : COL
        localparam int IDX = (i * N) + j;
        localparam int LSB = IDX * 32;

        mac_pe u_pe (
          .clk     (clk),
          .rst_n   (rst_n),
          .en      (en),
          .acc_clr (acc_clr),
          .a       (a_row[i]),
          .b       (b_col[j]),
          .acc     (acc_out_flat[LSB +: 32])
        );
      end
    end
  endgenerate

endmodule

`default_nettype wire
