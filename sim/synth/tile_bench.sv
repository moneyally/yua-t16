// tile_bench.sv — 타일 하나를 두 읽기 모드로 합성해 면적을 비교한다.
//                 (합성 전용)
//
// 타일의 포트가 packed 로 바뀌어 Yosys 가 파싱한다. 여기서 하는 일은
// 파라미터를 고정해 **합성 top 이름을 두 개 만드는 것**뿐이다
// (`sim/synth/area.py` 는 top 이름으로만 돌고 파라미터를 넘기지 않는다).
//
//   tile_fp32     : FP32_OUT=1, STREAM_RD=0            — PE 마다 변환기
//   tile_stream   : FP32_OUT=0, STREAM_RD=1, LANES=1    — 타일당 1개
//   tile_stream4  : 같음, LANES=4   — WR_DATA_W=128 에 맞는 레인 수
//   tile_stream16 : 같음, LANES=16  — WR_DATA_W=512 에 맞는 레인 수
//
// 레인을 WPB 개 두면 비트 하나를 한 사이클에 채워 비트당 사이클 손실이
// 0 이다. 그 구성의 면적이 얼마인지가 여기서 나온다.
//
// 두 수의 차이가 공유 변환기로 실제로 아끼는 면적이다. 전에는 타일을
// 합성할 수 없어 `pe_bench` 와 `rd_bench` 를 더하고 뺀 추정값을 썼다.
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
  input  logic [ROWS*16-1:0] a_row,
  input  logic [COLS*16-1:0] b_col,
  output logic [ROWS*COLS*32-1:0] acc_out,
  output logic busy
);
  logic [31:0] rd_unused;

  mxu_bf16_16x16 #(.ROWS(ROWS), .COLS(COLS),
                   .FP32_OUT(1), .STREAM_RD(0)) u_tile (
    .clk(clk), .rst_n(rst_n), .en(en), .acc_clr(acc_clr),
    .a_row(a_row), .b_col(b_col), .acc_out(acc_out),
    .rd_idx('0), .rd_fp32(rd_unused), .busy(busy));
endmodule

module tile_stream #(
  parameter int ROWS = 16,
  parameter int COLS = 16,
  parameter int LANES = 1
)(
  input  logic clk,
  input  logic rst_n,
  input  logic en,
  input  logic acc_clr,
  input  logic [ROWS*16-1:0] a_row,
  input  logic [COLS*16-1:0] b_col,
  input  logic [$clog2(ROWS*COLS)-1:0] rd_idx,
  output logic [LANES*32-1:0] rd_fp32,
  output logic busy
);
  // FP32_OUT=0 이면 acc_out 은 전부 0 이다 (변환기가 제거됐다).
  logic [ROWS*COLS*32-1:0] acc_unused;

  mxu_bf16_16x16 #(.ROWS(ROWS), .COLS(COLS),
                   .FP32_OUT(0), .STREAM_RD(1),
                   .RD_LANES(LANES)) u_tile (
    .clk(clk), .rst_n(rst_n), .en(en), .acc_clr(acc_clr),
    .a_row(a_row), .b_col(b_col), .acc_out(acc_unused),
    .rd_idx(rd_idx), .rd_fp32(rd_fp32), .busy(busy));
endmodule

// 레인 수를 고정한 top 들. area.py 는 파라미터를 넘기지 않으므로
// 이름을 따로 둬야 비교할 수 있다.
module tile_stream4 (
  input  logic clk, rst_n, en, acc_clr,
  input  logic [16*16-1:0] a_row,
  input  logic [16*16-1:0] b_col,
  input  logic [7:0]       rd_idx,
  output logic [4*32-1:0]  rd_fp32,
  output logic busy
);
  tile_stream #(.ROWS(16), .COLS(16), .LANES(4)) u (
    .clk(clk), .rst_n(rst_n), .en(en), .acc_clr(acc_clr),
    .a_row(a_row), .b_col(b_col), .rd_idx(rd_idx),
    .rd_fp32(rd_fp32), .busy(busy));
endmodule

module tile_stream16 (
  input  logic clk, rst_n, en, acc_clr,
  input  logic [16*16-1:0] a_row,
  input  logic [16*16-1:0] b_col,
  input  logic [7:0]       rd_idx,
  output logic [16*32-1:0] rd_fp32,
  output logic busy
);
  tile_stream #(.ROWS(16), .COLS(16), .LANES(16)) u (
    .clk(clk), .rst_n(rst_n), .en(en), .acc_clr(acc_clr),
    .a_row(a_row), .b_col(b_col), .rd_idx(rd_idx),
    .rd_fp32(rd_fp32), .busy(busy));
endmodule

`default_nettype wire
