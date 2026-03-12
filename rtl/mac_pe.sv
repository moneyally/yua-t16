`timescale 1ns/1ps
`default_nettype none

module mac_pe (
  input  logic              clk,
  input  logic              rst_n,
  input  logic              en,
  input  logic              acc_clr,
  input  logic signed [7:0]  a,
  input  logic signed [7:0]  b,
  output logic signed [31:0] acc
);

  // ----------------------------
  // multiply (explicit width)
  // ----------------------------
  logic signed [15:0] prod;
  always_comb begin
    prod = $signed(a) * $signed(b);
  end

`ifdef COCOTB_SIM
  initial acc = 32'sd0;
`endif

  // ----------------------------
  // accumulator
  // ----------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      acc <= 32'sd0;
    end
    else if (acc_clr) begin
      acc <= 32'sd0;
    end
 else if (en) begin
   acc <= acc + {{16{prod[15]}}, prod};
 end
  end
endmodule

`default_nettype wire
