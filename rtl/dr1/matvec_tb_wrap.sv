// ============================================================================
// matvec_tb_wrap.sv — matvec_unit + state_sram 결선 (테스트벤치 전용 top)
//
// tb/tb_matvec_unit.py 가 "state_sram 실제 인스턴스"에 연결된 matvec_unit 을
// 검사하도록 두 모듈을 묶는다. dr1_top(W6) 이 같은 방식으로 연결한다 —
// 여기 결선이 dr1_top 의 참고 배선이다.
//
// 합성 대상이다 (게이트가 본다). 로직은 없고 결선만 있다.
// ============================================================================
`timescale 1ns/1ps
`default_nettype none

module matvec_tb_wrap #(
  parameter int D     = 16,
  parameter int W     = 16,
  parameter int ACC_W = 40
)(
  input  wire                  clk,
  input  wire                  rst_n,

  // state_sram 쓰기 포트 (테스트벤치가 S 를 채운다)
  input  wire                  wr_en,
  input  wire [$clog2(D)-1:0]  wr_row,
  input  wire [D*W-1:0]        wr_data,
  input  wire                  clr_start,
  output wire                  clr_busy,

  // matvec 제어
  input  wire                  start,
  input  wire [D*W-1:0]        x_flat,
  output wire [D*W-1:0]        y_flat,
  output wire [31:0]           sat_count,
  output wire [31:0]           cycles,
  output wire                  busy,
  output wire                  done
);

  wire                 rd_en;
  wire [$clog2(D)-1:0] rd_row;
  wire [D*W-1:0]       rd_data;

  state_sram #(.D(D), .W(W)) u_sram (
    .clk(clk), .rst_n(rst_n),
    .rd_en(rd_en), .rd_row(rd_row), .rd_data(rd_data),
    .wr_en(wr_en), .wr_row(wr_row), .wr_data(wr_data),
    .clr_start(clr_start), .clr_busy(clr_busy)
  );

  matvec_unit #(.D(D), .W(W), .ACC_W(ACC_W)) u_mv (
    .clk(clk), .rst_n(rst_n),
    .start(start), .x_flat(x_flat),
    .sram_rd_en(rd_en), .sram_rd_row(rd_row), .sram_rd_data(rd_data),
    .y_flat(y_flat), .sat_count(sat_count), .cycles(cycles),
    .busy(busy), .done(done)
  );

endmodule

`default_nettype wire
