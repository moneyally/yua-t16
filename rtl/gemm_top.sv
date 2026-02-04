module gemm_top (
  input  logic        clk,
  input  logic        rst_n,

  input  logic        desc_valid,
  input  logic [7:0]  desc_bytes [0:63],
  output logic        desc_ready
);

  logic        gemm_valid;
  logic        gemm_done;
  logic [63:0] act_addr, wgt_addr, out_addr;
  logic [31:0] Kt;

  ctrl_fsm u_ctrl (
    .clk(clk),
    .rst_n(rst_n),

    .desc_valid(desc_valid),
    .desc_bytes(desc_bytes),
    .desc_ready(desc_ready),

    .gemm_valid(gemm_valid),
    .act_addr(act_addr),
    .wgt_addr(wgt_addr),
    .out_addr(out_addr),
    .Kt(Kt),
    .gemm_done(gemm_done)
  );

  gemm_stub u_stub (
    .clk(clk),
    .rst_n(rst_n),
    .valid(gemm_valid),
    .Kt(Kt),
    .done(gemm_done)
  );

endmodule
