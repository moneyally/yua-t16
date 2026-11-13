`timescale 1ns/1ps
`default_nettype none

// act_sram.sv — 피연산자 스테이징 SRAM
//
// DATA_W 는 DMA 읽기 폭과 같다. 128 이면 워드 하나가 K-step 하나
// (INT8 16개), 256 이면 K-step 두 개가 한 워드에 들어온다.
// 넓힌 쪽은 DMA 비트 수가 줄어 읽기 대역폭 병목이 완화된다.
// DEPTH 는 워드 수다 (= MAX_KT / (DATA_W/128)).
module act_sram #(
  parameter int MAX_KT = 256,
  parameter int DATA_W = 128,
  parameter int DEPTH  = MAX_KT
)(
  input  logic clk,
  input  logic rst_n,

  input  logic we,
  input  logic re,

  input  logic [$clog2(DEPTH)-1:0]  waddr,
  input  logic [DATA_W-1:0]         wdata,

  input  logic [$clog2(DEPTH)-1:0]  raddr,
  output logic [DATA_W-1:0]         rdata
);

  localparam int AW = $clog2(DEPTH);

  logic [DATA_W-1:0] mem [0:DEPTH-1];
  integer i;

  // --------------------------------------------
  // X-safe address check (Icarus friendly)
  // --------------------------------------------
  function automatic logic addr_is_valid(input logic [AW-1:0] a);
    integer k;
    begin
      addr_is_valid = 1'b1;
      for (k = 0; k < AW; k = k + 1) begin
        if (a[k] !== 1'b0 && a[k] !== 1'b1)
          addr_is_valid = 1'b0;
      end
    end
  endfunction

`ifdef COCOTB_SIM
  initial begin
    rdata = '0;
    for (i = 0; i < DEPTH; i = i + 1)
      mem[i] = '0;
  end
`endif

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rdata <= '0;
`ifdef COCOTB_SIM
      for (i = 0; i < DEPTH; i = i + 1)
        mem[i] <= '0;
`endif
    end else begin
      // ---------------- write ----------------
      if (we && addr_is_valid(waddr))
        mem[waddr] <= wdata;

      // ---------------- read (X-safe) ----------------
      if (re && addr_is_valid(raddr))
        rdata <= mem[raddr];
      else
        rdata <= rdata; // ✅ HOLD (reset에서 0이므로 X-safe)
    end
  end

endmodule

`default_nettype wire
