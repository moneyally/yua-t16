// cdc_fifo.sv — Asynchronous FIFO with Gray-code pointers for ORBIT-G2
// SSOT: ORBIT_G2_RTL_SKELETONS.md section 2.7
//
// Standard async FIFO:
//   - Independent wr_clk / rd_clk
//   - Gray-coded write/read pointers for safe CDC
//   - Registered full/empty flags (conservative, no false negatives)
//   - Parameterized depth (must be power of 2) and data width
`timescale 1ns/1ps
`default_nettype none

module cdc_fifo #(
  parameter int DATA_W = 32,
  parameter int DEPTH  = 16    // must be power of 2, minimum 4
)(
  // Write domain
  input  logic              wr_clk,
  input  logic              wr_rst_n,
  input  logic              wr_valid,
  output logic              wr_ready,
  input  logic [DATA_W-1:0] wr_data,
  output logic              full,

  // Read domain
  input  logic              rd_clk,
  input  logic              rd_rst_n,
  input  logic              rd_valid,   // read request
  output logic              rd_ready,   // data available
  output logic [DATA_W-1:0] rd_data,   // FWFT: rd_ready 동안 유효
  output logic              empty
);

  // 파라미터 정합성 검사.
  // 원래 이 initial 블록이 포트 리스트 한가운데에 있어서 모듈이 아예
  // 파싱되지 않았다 (syntax error: unexpected initial, expecting '[').
  // 본문으로 옮겼다.
  //
  // DEPTH=2 면 그레이 코드 full 비교에서 rd_gray_sync2[PTR_W-3:0] 의
  // PTR_W 가 2 라 인덱스가 -1 이 된다.
  initial begin
    if (DEPTH < 4)
      $fatal(1, "cdc_fifo: DEPTH must be >= 4 (got %0d)", DEPTH);
    if ((DEPTH & (DEPTH - 1)) != 0)
      $fatal(1, "cdc_fifo: DEPTH must be a power of 2 (got %0d)", DEPTH);
  end

  localparam int ADDR_W = $clog2(DEPTH);
  localparam int PTR_W  = ADDR_W + 1;  // extra MSB for wrap detection

  // ---------------------------------------------------------------
  // Memory
  // ---------------------------------------------------------------
  logic [DATA_W-1:0] mem [0:DEPTH-1];

  // ---------------------------------------------------------------
  // Binary <-> Gray conversion
  // ---------------------------------------------------------------
  function automatic logic [PTR_W-1:0] bin2gray(input logic [PTR_W-1:0] b);
    bin2gray = b ^ (b >> 1);
  endfunction

  function automatic logic [PTR_W-1:0] gray2bin(input logic [PTR_W-1:0] g);
    logic [PTR_W-1:0] b;
    integer i;
    begin
      b[PTR_W-1] = g[PTR_W-1];
      for (i = PTR_W-2; i >= 0; i = i - 1)
        b[i] = b[i+1] ^ g[i];
      gray2bin = b;
    end
  endfunction

  // ---------------------------------------------------------------
  // Write domain
  // ---------------------------------------------------------------
  logic [PTR_W-1:0] wr_bin, wr_bin_next;
  logic [PTR_W-1:0] wr_gray, wr_gray_next;
  logic [PTR_W-1:0] rd_gray_sync1, rd_gray_sync2;  // rd gray -> wr domain

  assign wr_bin_next  = wr_bin + PTR_W'(wr_valid & wr_ready);
  assign wr_gray_next = bin2gray(wr_bin_next);

  always_ff @(posedge wr_clk or negedge wr_rst_n) begin
    if (!wr_rst_n) begin
      wr_bin  <= '0;
      wr_gray <= '0;
    end else begin
      wr_bin  <= wr_bin_next;
      wr_gray <= wr_gray_next;
    end
  end

  // Write to memory
  always_ff @(posedge wr_clk) begin
    if (wr_valid & wr_ready)
      mem[wr_bin[ADDR_W-1:0]] <= wr_data;
  end

  // Synchronize rd_gray into wr domain (2-FF)
  always_ff @(posedge wr_clk or negedge wr_rst_n) begin
    if (!wr_rst_n) begin
      rd_gray_sync1 <= '0;
      rd_gray_sync2 <= '0;
    end else begin
      rd_gray_sync1 <= rd_gray;   // from rd domain
      rd_gray_sync2 <= rd_gray_sync1;
    end
  end

  // Full: gray pointers match with top 2 MSBs inverted, rest identical.
  // Proof (Cummings SNUG-2002):
  //   Binary full = {wr[MSB] != rd[MSB], wr[ADDR-1:0] == rd[ADDR-1:0]}
  //   In gray code, this maps to the top 2 gray bits being inverted
  //   because bin2gray flips MSB-1 when MSB differs.
  //   Verified: DEPTH=4 (PTR_W=3), all 4 full pairs match this pattern.
  logic full_val;
  assign full_val = (wr_gray_next == {~rd_gray_sync2[PTR_W-1:PTR_W-2],
                                       rd_gray_sync2[PTR_W-3:0]});

  always_ff @(posedge wr_clk or negedge wr_rst_n) begin
    if (!wr_rst_n)
      full <= 1'b0;
    else
      full <= full_val;
  end

  assign wr_ready = ~full;

  // ---------------------------------------------------------------
  // Read domain
  // ---------------------------------------------------------------
  logic [PTR_W-1:0] rd_bin, rd_bin_next;
  logic [PTR_W-1:0] rd_gray, rd_gray_next;
  logic [PTR_W-1:0] wr_gray_sync1, wr_gray_sync2;  // wr gray -> rd domain

  wire rd_handshake = rd_valid & rd_ready;

  assign rd_bin_next  = rd_bin + PTR_W'(rd_handshake);
  assign rd_gray_next = bin2gray(rd_bin_next);

  always_ff @(posedge rd_clk or negedge rd_rst_n) begin
    if (!rd_rst_n) begin
      rd_bin  <= '0;
      rd_gray <= '0;
    end else begin
      rd_bin  <= rd_bin_next;
      rd_gray <= rd_gray_next;
    end
  end

  // Read data — first-word-fall-through (FWFT).
  //
  // 포트 선언이 rd_ready 를 "data available" 로 정의한다. 즉 rd_ready 가 1 인
  // 동안 rd_data 는 이미 유효해야 한다. 원래 구현은 rd_handshake 가 난 "다음"
  // 사이클에 데이터를 실어서 그 계약을 어겼고, 첫 읽기는 리셋값 0 이 나왔다.
  // (이 모듈은 포트 리스트 안에 initial 블록이 있어 파싱 자체가 안 됐던 탓에
  //  한 번도 검증된 적이 없다.)
  //
  // 트레이드오프: 비동기 메모리 읽기라 rd_clk 경로가 길어진다. 타이밍이
  // 문제가 되면 prefetch 레지스터를 둔 FWFT 로 바꿔야 한다 — 출력만 등록하면
  // 계약이 다시 깨진다.
  assign rd_data = mem[rd_bin[ADDR_W-1:0]];

  // Synchronize wr_gray into rd domain (2-FF)
  always_ff @(posedge rd_clk or negedge rd_rst_n) begin
    if (!rd_rst_n) begin
      wr_gray_sync1 <= '0;
      wr_gray_sync2 <= '0;
    end else begin
      wr_gray_sync1 <= wr_gray;   // from wr domain
      wr_gray_sync2 <= wr_gray_sync1;
    end
  end

  // Empty: gray pointers match exactly
  logic empty_val;
  assign empty_val = (rd_gray_next == wr_gray_sync2);

  always_ff @(posedge rd_clk or negedge rd_rst_n) begin
    if (!rd_rst_n)
      empty <= 1'b1;   // empty after reset
    else
      empty <= empty_val;
  end

  assign rd_ready = ~empty;

endmodule

`default_nettype wire
