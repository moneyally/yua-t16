module mac_array (
  input  logic clk,
  input  logic rst_n,

  input  logic        en,
  input  logic        acc_clr,

  input  logic signed [7:0] a_row [0:15],
  input  logic signed [7:0] b_col [0:15],

  output logic signed [31:0] acc_out [0:15][0:15]
);

  genvar i, j;
  generate
    for (i = 0; i < 16; i++) begin : ROW
      for (j = 0; j < 16; j++) begin : COL
        mac_pe u_pe (
          .clk     (clk),
          .en      (en),
          .acc_clr (acc_clr),
          .a       (a_row[i]),
          .b       (b_col[j]),
          .acc     (acc_out[i][j])
        );
      end
    end
  endgenerate

endmodule
