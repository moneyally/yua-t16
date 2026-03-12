`timescale 1ns/1ps
`default_nettype none

module wgt_sram #(
  parameter int MAX_KT = 256
)(
  input  logic clk,
  input  logic rst_n,

  input  logic we,
  input  logic re,

  input  logic [$clog2(MAX_KT)-1:0] waddr,
  input  logic [127:0]              wdata,

  input  logic [$clog2(MAX_KT)-1:0] raddr,
  output logic [127:0]              rdata
);

  localparam int AW = $clog2(MAX_KT);

  logic [127:0] mem [0:MAX_KT-1];
  integer i;

  function automatic logic has_x_addr(input logic [AW-1:0] a);
    has_x_addr = (^a === 1'bx);
  endfunction

`ifdef COCOTB_SIM
  initial begin
    rdata = '0;
    for (i = 0; i < MAX_KT; i = i + 1)
      mem[i] = '0;
  end
`endif

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rdata <= '0;
`ifdef COCOTB_SIM
      for (i = 0; i < MAX_KT; i = i + 1)
        mem[i] <= '0;
`endif
    end else begin
      if (we && !has_x_addr(waddr))
        mem[waddr] <= wdata;

      if (re && !has_x_addr(raddr))
        rdata <= mem[raddr];
      else
        rdata <= rdata; // ✅ HOLD (reset에서 0이므로 X-safe)
    end
  end

endmodule

`default_nettype wire
