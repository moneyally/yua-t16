// fxp_dut.sv — bf16_pkg 고정소수점 함수 검증용 DUT (시뮬 전용)
`timescale 1ns/1ps
`default_nettype none

module fxp_dut (
  input  logic [15:0] a,
  input  logic [15:0] b,
  input  logic [47:0] acc_in,
  input  logic [47:0] add_x,
  input  logic [47:0] add_y,
  output logic [47:0] mul_fxp,     // bf16_mul_fxp(a, b)
  output logic [47:0] add_sat,     // fxp_add_sat(add_x, add_y)
  output logic [31:0] to_fp32      // fxp_to_fp32(acc_in)
);
  assign mul_fxp = bf16_pkg::bf16_mul_fxp(a, b);
  assign add_sat = bf16_pkg::fxp_add_sat($signed(add_x), $signed(add_y));
  assign to_fp32 = bf16_pkg::fxp_to_fp32($signed(acc_in));
endmodule

`default_nettype wire
