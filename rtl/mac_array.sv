`timescale 1ns/1ps
`default_nettype none

module mac_array (
  input  logic clk,
  input  logic rst_n,
  input  logic en,
  input  logic acc_clr,
  input  logic signed [7:0] a_row [0:15],
  input  logic signed [7:0] b_col [0:15],
  // Packed flat output: acc_out_flat[IDX*32 +: 32] = result for row=IDX/16, col=IDX%16
  // Exported as packed vector to work around Icarus 12 unpacked-array-output limitation.
  output logic [32*256-1:0] acc_out_flat
);

  // --------------------------------------------------
  // MAC PE instantiation — each pe drives its 32-bit slice
  // --------------------------------------------------
  genvar i, j;
  generate
    for (i = 0; i < 16; i++) begin : ROW
      for (j = 0; j < 16; j++) begin : COL
        localparam int IDX = (i * 16) + j;
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
