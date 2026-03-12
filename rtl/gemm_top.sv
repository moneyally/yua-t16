`timescale 1ns/1ps
`default_nettype none

module gemm_top #(
  parameter int MAX_KT = 256
)(
  input  logic        clk,
  input  logic        rst_n,

  input  logic        desc_valid,
  input  logic [7:0]  desc_bytes [0:63],
  output logic        desc_ready,

  // memory read
  output logic        rd_req_valid,
  input  logic        rd_req_ready,
  output logic [63:0] rd_req_addr,
  output logic [15:0] rd_req_len_bytes,
  input  logic        rd_done,

  input  logic        rd_data_valid,
  input  logic [127:0] rd_data,
  output logic        rd_data_ready,
  input  logic        rd_data_last,

  // memory write
  output logic        wr_req_valid,
  input  logic        wr_req_ready,
  output logic [63:0] wr_req_addr,
  output logic [15:0] wr_req_len_bytes,
  input  logic        wr_done,

  output logic        wr_data_valid,
  input  logic        wr_data_ready,
  output logic [127:0] wr_data,
  output logic        wr_data_last,

  output logic        busy,
  output logic        done_pulse,
  output logic [31:0] perf_cycles,
  output logic [31:0] perf_bytes
);

  // -------------------------
  // internal signals
  // -------------------------
  logic        cmd_valid_c;     // ctrl_fsm asserts while ST_DISPATCH
  logic        cmd_valid_r;     // held until core accepts
  logic        cmd_ready_core;  // from gemm_core
  logic        cmd_ready_ctrl;  // to ctrl_fsm (delayed/qualified)

  logic [63:0] act_addr_c, wgt_addr_c, out_addr_c;
  logic [31:0] Kt_c;

  logic [63:0] act_addr_r, wgt_addr_r, out_addr_r;
  logic [31:0] Kt_r;

  logic        core_done;
  logic        core_done_r;

  logic        ctrl_busy;
  logic        core_busy;

  // -------------------------
  // CTRL FSM
  // -------------------------
  ctrl_fsm u_ctrl (
    .clk(clk),
    .rst_n(rst_n),

    .desc_valid(desc_valid),
    .desc_bytes(desc_bytes),
    .desc_ready(desc_ready),

    .cmd_valid(cmd_valid_c),
    .cmd_ready(cmd_ready_ctrl),   // ✅ IMPORTANT: ctrl sees "accept-ready", not raw core ready

    .act_addr(act_addr_c),
    .wgt_addr(wgt_addr_c),
    .out_addr(out_addr_c),
    .Kt(Kt_c),

    .core_done(core_done_r),
    .busy(ctrl_busy),
    .done_pulse(done_pulse)
  );

  // -------------------------------------------------
  // Command bridge (robust)
  // - latch cmd + payload once
  // - hold valid until core accepts (cmd_ready_core)
  // - ctrl_fsm only proceeds when cmd_valid_r is asserted (so it cannot "run ahead")
  // -------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cmd_valid_r <= 1'b0;
      act_addr_r  <= 64'd0;
      wgt_addr_r  <= 64'd0;
      out_addr_r  <= 64'd0;
      Kt_r        <= 32'd0;
    end else begin
      // latch new command if we don't already have one pending
      if (!cmd_valid_r && cmd_valid_c) begin
        cmd_valid_r <= 1'b1;
        act_addr_r  <= act_addr_c;
        wgt_addr_r  <= wgt_addr_c;
        out_addr_r  <= out_addr_c;
        Kt_r        <= Kt_c;
      end
      // drop when core accepts
      else if (cmd_valid_r && cmd_ready_core) begin
        cmd_valid_r <= 1'b0;
      end
    end
  end

  // ctrl should only move DISPATCH->WAIT when we're actually presenting a pending cmd to core
  // (this intentionally adds a 1-cycle "bridge" so core never misses the command)
  assign cmd_ready_ctrl = cmd_valid_r && cmd_ready_core;

  // -------------------------
  // GEMM CORE
  // -------------------------
  gemm_core #(.MAX_KT(MAX_KT)) u_core (
    .clk(clk),
    .rst_n(rst_n),

    .cmd_valid(cmd_valid_r),
    .cmd_ready(cmd_ready_core),

    .act_addr(act_addr_r),
    .wgt_addr(wgt_addr_r),
    .out_addr(out_addr_r),
    .Kt(Kt_r),

    .done(core_done),

    .rd_req_valid(rd_req_valid),
    .rd_req_ready(rd_req_ready),
    .rd_req_addr(rd_req_addr),
    .rd_req_len_bytes(rd_req_len_bytes),
    .rd_done(rd_done),

    .rd_data_valid(rd_data_valid),
    .rd_data(rd_data),
    .rd_data_ready(rd_data_ready),
    .rd_data_last(rd_data_last),

    .wr_req_valid(wr_req_valid),
    .wr_req_ready(wr_req_ready),
    .wr_req_addr(wr_req_addr),
    .wr_req_len_bytes(wr_req_len_bytes),
    .wr_done(wr_done),

    .wr_data_valid(wr_data_valid),
    .wr_data_ready(wr_data_ready),
    .wr_data(wr_data),
    .wr_data_last(wr_data_last),

    .perf_cycles(perf_cycles),
    .perf_bytes(perf_bytes),

    .busy(core_busy)
  );

  // -------------------------
  // core_done register (level)
  // -------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      core_done_r <= 1'b0;
    else
      core_done_r <= core_done;
  end

  // -------------------------
  // top busy
  // -------------------------
  assign busy = ctrl_busy | core_busy;

endmodule

`default_nettype wire
