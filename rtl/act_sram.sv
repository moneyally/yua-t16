`timescale 1ns/1ps
`default_nettype none

module act_sram #(
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
