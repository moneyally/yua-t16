// vpu_top.sv
// Top-level VPU wrapper — instantiates vpu_core and provides the
// same memory interface for testbench connectivity.

`timescale 1ns/1ps

module vpu_top (
  input  logic        clk,
  input  logic        rst_n,

  // Descriptor-level interface
  input  logic [3:0]  op_type,
  input  logic        data_type,
  input  logic [19:0] vec_len,
  input  logic [15:0] imm_fp16_0,
  input  logic [15:0] imm_fp16_1,

  input  logic        start,
  output logic        busy,
  output logic        done,

  // Memory ports (pass-through)
  output logic [19:0] src_addr,
  output logic        src_re,
  input  logic [15:0] src_rdata,

  output logic [19:0] aux_addr,
  output logic        aux_re,
  input  logic [15:0] aux_rdata,

  output logic [19:0] dst_addr,
  output logic        dst_we,
  output logic [15:0] dst_wdata
);

  vpu_core u_core (
    .clk        (clk),
    .rst_n      (rst_n),
    .op_type    (op_type),
    .data_type  (data_type),
    .vec_len    (vec_len),
    .imm_fp16_0 (imm_fp16_0),
    .imm_fp16_1 (imm_fp16_1),
    .start      (start),
    .busy       (busy),
    .done       (done),
    .src_addr   (src_addr),
    .src_re     (src_re),
    .src_rdata  (src_rdata),
    .aux_addr   (aux_addr),
    .aux_re     (aux_re),
    .aux_rdata  (aux_rdata),
    .dst_addr   (dst_addr),
    .dst_we     (dst_we),
    .dst_wdata  (dst_wdata)
  );

endmodule
