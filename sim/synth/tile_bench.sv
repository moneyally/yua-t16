// tile_bench.sv — 타일 하나를 통째로 합성해 두 읽기 모드의 면적을 잰다.
//                 (합성 전용)
//
// Yosys 0.33 이 `mxu_bf16_16x16` 의 unpacked 배열 포트를 파싱하지
// 못한다. 그래서 이전에는 타일을 통째로 재지 못하고 `pe_bench`
// (변환기 255개 제거) 와 `rd_bench` (먹스 + 변환기 1개) 를 따로 재어
// 더하고 뺐다. 그 추정이 타일 전체에서도 성립하는지 확인하려면
// 타일을 합성해야 한다.
//
// 포트만 평탄화한 래퍼를 씌우면 Yosys 가 통과한다. 래퍼 자체는 배선
// 뿐이므로 셀을 더하지 않는다 (abc -g simple 이 정리한다).
//
//   tile_fp32   : FP32_OUT=1, STREAM_RD=0  — PE 마다 변환기
//   tile_stream : FP32_OUT=0, STREAM_RD=1  — 타일당 변환기 하나
//
// 두 수의 차이가 "공유 변환기로 바꿔서 실제로 아끼는 면적" 이다.
`timescale 1ns/1ps
`default_nettype none

module tile_fp32 #(
  parameter int ROWS = 16,
  parameter int COLS = 16
)(
  input  logic clk,
  input  logic rst_n,
  input  logic en,
  input  logic acc_clr,
  input  logic [ROWS*16-1:0] a_row_flat,
  input  logic [COLS*16-1:0] b_col_flat,
  output logic [ROWS*COLS*32-1:0] acc_out_flat,
  output logic busy
);
  logic [15:0] a_row [0:ROWS-1];
  logic [15:0] b_col [0:COLS-1];
  logic [31:0] acc_out [0:ROWS-1][0:COLS-1];
  logic [31:0] rd_unused;

  genvar i, j;
  generate
    for (i = 0; i < ROWS; i++) begin : G_A
      assign a_row[i] = a_row_flat[i*16 +: 16];
    end
    for (i = 0; i < COLS; i++) begin : G_B
      assign b_col[i] = b_col_flat[i*16 +: 16];
    end
    for (i = 0; i < ROWS; i++) begin : G_OR
      for (j = 0; j < COLS; j++) begin : G_OC
        assign acc_out_flat[(i*COLS + j)*32 +: 32] = acc_out[i][j];
      end
    end
  endgenerate

  mxu_bf16_16x16 #(.ROWS(ROWS), .COLS(COLS),
                   .FP32_OUT(1), .STREAM_RD(0)) u_tile (
    .clk(clk), .rst_n(rst_n), .en(en), .acc_clr(acc_clr),
    .a_row(a_row), .b_col(b_col), .acc_out(acc_out),
    .rd_idx('0), .rd_fp32(rd_unused), .busy(busy));
endmodule

module tile_stream #(
  parameter int ROWS = 16,
  parameter int COLS = 16
)(
  input  logic clk,
  input  logic rst_n,
  input  logic en,
  input  logic acc_clr,
  input  logic [ROWS*16-1:0] a_row_flat,
  input  logic [COLS*16-1:0] b_col_flat,
  input  logic [$clog2(ROWS*COLS)-1:0] rd_idx,
  output logic [31:0] rd_fp32,
  output logic busy
);
  logic [15:0] a_row [0:ROWS-1];
  logic [15:0] b_col [0:COLS-1];
  logic [31:0] acc_out [0:ROWS-1][0:COLS-1];

  genvar i;
  generate
    for (i = 0; i < ROWS; i++) begin : G_A
      assign a_row[i] = a_row_flat[i*16 +: 16];
    end
    for (i = 0; i < COLS; i++) begin : G_B
      assign b_col[i] = b_col_flat[i*16 +: 16];
    end
  endgenerate

  mxu_bf16_16x16 #(.ROWS(ROWS), .COLS(COLS),
                   .FP32_OUT(0), .STREAM_RD(1)) u_tile (
    .clk(clk), .rst_n(rst_n), .en(en), .acc_clr(acc_clr),
    .a_row(a_row), .b_col(b_col), .acc_out(acc_out),
    .rd_idx(rd_idx), .rd_fp32(rd_fp32), .busy(busy));
endmodule

`default_nettype wire
