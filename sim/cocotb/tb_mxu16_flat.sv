// tb_mxu16_flat.sv — 검증 전용 래퍼.
// Icarus 의 VPI 가 2차원 unpacked 배열을 노출하지 못해서, 테스트가
// 인덱싱할 수 있는 1차원 acc_flat[256] 으로 바꿔 준다. DUT 쪽 포트는
// packed 평탄 벡터이고(그 이유는 mxu_bf16_16x16.sv 주석 참고), 여기서
// 테스트가 보기 편한 unpacked 배열로 풀어 준다. 합성 대상이 아니다.
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
  logic [N*16-1:0]    a_flat, b_flat;
  logic [N*N*32-1:0]  acc_packed;

  genvar i;
  generate
    for (i = 0; i < N; i++) begin : FIN
      assign a_flat[16*i +: 16] = a_row[i];
      assign b_flat[16*i +: 16] = b_col[i];
    end
    // acc_out 은 행 우선 평탄화라 acc_flat[i*N + j] 와 같은 순서다.
    for (i = 0; i < N*N; i++) begin : FOUT
      assign acc_flat[i] = acc_packed[32*i +: 32];
    end
  endgenerate

  mxu_bf16_16x16 #(.ROWS(N), .COLS(N)) u_dut (
    .clk(clk), .rst_n(rst_n), .en(en), .acc_clr(acc_clr),
    .a_row(a_flat), .b_col(b_flat), .acc_out(acc_packed), .busy(busy)
  );
endmodule

`default_nettype wire
