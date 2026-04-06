// g3_2chip_fabric_int_top.sv — ORBIT-G3 Fabric-Connected 2-Chip Training
// SSOT: ORBIT_G3_RTL_ISSUES.md (G3-INT-005)
//
// Replaces direct peer_in feed with fabric send/recv path:
//   chip0 backward → dW0 → fabric_ctrl0 tx → fabric_ctrl1 rx → peer_dW for chip1
//   chip1 backward → dW1 → fabric_ctrl1 tx → fabric_ctrl0 rx → peer_dW for chip0
//   collective per chip: local_dW + peer_dW → reduced_dW
//   optimizer per chip: Adam(param, reduced_dW)
//
// Half-duplex fabric: send both, then recv both (cross-connected internally).
// Active region: 16×16. SUM reduction.
`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off UNUSEDSIGNAL */

module g3_2chip_fabric_int_top #(
  parameter int DIM = 16
)(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,

  // Hyperparams
  input  logic [31:0] lr_fp32, beta1_fp32, beta2_fp32, epsilon_fp32, weight_decay_fp32,
  input  logic        adamw_enable,

  // Chip 0
  input  logic [15:0] x0_bf16[0:DIM-1][0:DIM-1],
  input  logic [15:0] w0_bf16[0:DIM-1][0:DIM-1],
  input  logic [15:0] dy0_bf16[0:DIM-1][0:DIM-1],
  input  logic [31:0] param0_in[0:DIM-1][0:DIM-1],
  input  logic [31:0] m0_in[0:DIM-1][0:DIM-1],
  input  logic [31:0] v0_in[0:DIM-1][0:DIM-1],
  output logic [31:0] param0_out[0:DIM-1][0:DIM-1],
  output logic [31:0] m0_out[0:DIM-1][0:DIM-1],
  output logic [31:0] v0_out[0:DIM-1][0:DIM-1],

  // Chip 1
  input  logic [15:0] x1_bf16[0:DIM-1][0:DIM-1],
  input  logic [15:0] w1_bf16[0:DIM-1][0:DIM-1],
  input  logic [15:0] dy1_bf16[0:DIM-1][0:DIM-1],
  input  logic [31:0] param1_in[0:DIM-1][0:DIM-1],
  input  logic [31:0] m1_in[0:DIM-1][0:DIM-1],
  input  logic [31:0] v1_in[0:DIM-1][0:DIM-1],
  output logic [31:0] param1_out[0:DIM-1][0:DIM-1],
  output logic [31:0] m1_out[0:DIM-1][0:DIM-1],
  output logic [31:0] v1_out[0:DIM-1][0:DIM-1],

  // Observable
  output logic [31:0] dw0_out[0:DIM-1][0:DIM-1],
  output logic [31:0] dw1_out[0:DIM-1][0:DIM-1],
  output logic [31:0] reduced_dw0[0:DIM-1][0:DIM-1],
  output logic [31:0] reduced_dw1[0:DIM-1][0:DIM-1],

  output logic        busy,
  output logic        done_pulse,
  output logic [7:0]  err_code
);

  // ═══════════════════════════════════════════════════════════
  // Backward engines
  // ═══════════════════════════════════════════════════════════
  logic bk0_start, bk0_done, bk1_start, bk1_done;
  logic [7:0] bk0_err, bk1_err;

  backward_engine #(.DIM(DIM)) u_bk0 (
    .clk(clk),.rst_n(rst_n),.start(bk0_start),.mode(2'd1),.acc_clr(1'b0),
    .x_in(x0_bf16),.w_in(w0_bf16),.dy_in(dy0_bf16),
    .result(dw0_out),.busy(),.done_pulse(bk0_done),.err_code(bk0_err));

  backward_engine #(.DIM(DIM)) u_bk1 (
    .clk(clk),.rst_n(rst_n),.start(bk1_start),.mode(2'd1),.acc_clr(1'b0),
    .x_in(x1_bf16),.w_in(w1_bf16),.dy_in(dy1_bf16),
    .result(dw1_out),.busy(),.done_pulse(bk1_done),.err_code(bk1_err));

  // ═══════════════════════════════════════════════════════════
  // Fabric controllers (chip0, chip1)
  // ═══════════════════════════════════════════════════════════
  // Cross-connected: chip0 tx → chip1 rx, chip1 tx → chip0 rx
  logic fc0_send_start, fc0_recv_start, fc0_done, fc0_busy;
  logic [7:0] fc0_err;
  logic fc0_ltx_valid, fc0_ltx_ready, fc0_ltx_last;
  logic [31:0] fc0_ltx_data;
  logic fc0_prx_valid, fc0_prx_ready, fc0_prx_last;
  logic [31:0] fc0_prx_data;
  logic fc0_ftx_valid, fc0_ftx_ready, fc0_ftx_last;
  logic [31:0] fc0_ftx_data;
  logic fc0_frx_valid, fc0_frx_ready, fc0_frx_last;
  logic [31:0] fc0_frx_data;

  scale_fabric_ctrl u_fc0 (
    .clk(clk),.rst_n(rst_n),
    .send_start(fc0_send_start),.recv_start(fc0_recv_start),
    .busy(fc0_busy),.done_pulse(fc0_done),.err_code(fc0_err),
    .local_tx_valid(fc0_ltx_valid),.local_tx_ready(fc0_ltx_ready),
    .local_tx_data(fc0_ltx_data),.local_tx_last(fc0_ltx_last),
    .peer_rx_valid(fc0_prx_valid),.peer_rx_ready(fc0_prx_ready),
    .peer_rx_data(fc0_prx_data),.peer_rx_last(fc0_prx_last),
    .fabric_tx_valid(fc0_ftx_valid),.fabric_tx_ready(fc0_ftx_ready),
    .fabric_tx_data(fc0_ftx_data),.fabric_tx_last(fc0_ftx_last),
    .fabric_rx_valid(fc0_frx_valid),.fabric_rx_ready(fc0_frx_ready),
    .fabric_rx_data(fc0_frx_data),.fabric_rx_last(fc0_frx_last));

  logic fc1_send_start, fc1_recv_start, fc1_done, fc1_busy;
  logic [7:0] fc1_err;
  logic fc1_ltx_valid, fc1_ltx_ready, fc1_ltx_last;
  logic [31:0] fc1_ltx_data;
  logic fc1_prx_valid, fc1_prx_ready, fc1_prx_last;
  logic [31:0] fc1_prx_data;
  logic fc1_ftx_valid, fc1_ftx_ready, fc1_ftx_last;
  logic [31:0] fc1_ftx_data;
  logic fc1_frx_valid, fc1_frx_ready, fc1_frx_last;
  logic [31:0] fc1_frx_data;

  scale_fabric_ctrl u_fc1 (
    .clk(clk),.rst_n(rst_n),
    .send_start(fc1_send_start),.recv_start(fc1_recv_start),
    .busy(fc1_busy),.done_pulse(fc1_done),.err_code(fc1_err),
    .local_tx_valid(fc1_ltx_valid),.local_tx_ready(fc1_ltx_ready),
    .local_tx_data(fc1_ltx_data),.local_tx_last(fc1_ltx_last),
    .peer_rx_valid(fc1_prx_valid),.peer_rx_ready(fc1_prx_ready),
    .peer_rx_data(fc1_prx_data),.peer_rx_last(fc1_prx_last),
    .fabric_tx_valid(fc1_ftx_valid),.fabric_tx_ready(fc1_ftx_ready),
    .fabric_tx_data(fc1_ftx_data),.fabric_tx_last(fc1_ftx_last),
    .fabric_rx_valid(fc1_frx_valid),.fabric_rx_ready(fc1_frx_ready),
    .fabric_rx_data(fc1_frx_data),.fabric_rx_last(fc1_frx_last));

  // Cross-connect: chip0 tx → chip1 rx, chip1 tx → chip0 rx
  assign fc1_frx_valid = fc0_ftx_valid;
  assign fc1_frx_data  = fc0_ftx_data;
  assign fc1_frx_last  = fc0_ftx_last;
  assign fc0_ftx_ready = fc1_frx_ready;

  assign fc0_frx_valid = fc1_ftx_valid;
  assign fc0_frx_data  = fc1_ftx_data;
  assign fc0_frx_last  = fc1_ftx_last;
  assign fc1_ftx_ready = fc0_frx_ready;

  // ═══════════════════════════════════════════════════════════
  // Gradient serializer/deserializer: 16×16 FP32 ↔ 256-beat stream
  // ═══════════════════════════════════════════════════════════
  // TX: flatten dW row-major, feed local_tx
  // RX: capture peer_rx beats into peer_dW buffer

  logic [31:0] peer_dw0 [0:DIM-1][0:DIM-1];  // chip0 receives chip1's dW
  logic [31:0] peer_dw1 [0:DIM-1][0:DIM-1];  // chip1 receives chip0's dW

  // TX beat counter per chip
  logic [8:0] tx0_idx, tx1_idx;  // 0..255
  logic [8:0] rx0_idx, rx1_idx;

  // TX0: serialize dw0_out (active in PASS1: chip0 sends)
  assign fc0_ltx_valid = (tst == ST_WAIT_FAB_PASS1) && (tx0_idx < DIM*DIM);
  assign fc0_ltx_data  = dw0_out[tx0_idx[7:4]][tx0_idx[3:0]];
  assign fc0_ltx_last  = (tx0_idx == DIM*DIM - 1);

  // TX1: serialize dw1_out (active in PASS2: chip1 sends)
  assign fc1_ltx_valid = (tst == ST_WAIT_FAB_PASS2) && (tx1_idx < DIM*DIM);
  assign fc1_ltx_data  = dw1_out[tx1_idx[7:4]][tx1_idx[3:0]];
  assign fc1_ltx_last  = (tx1_idx == DIM*DIM - 1);

  // RX: deserialize into peer_dw buffers
  assign fc0_prx_ready = 1'b1;
  assign fc1_prx_ready = 1'b1;

  integer pi, pj;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (pi=0;pi<DIM;pi++) for(pj=0;pj<DIM;pj++) begin
        peer_dw0[pi][pj] <= 32'd0;
        peer_dw1[pi][pj] <= 32'd0;
      end
      rx0_idx <= 0; rx1_idx <= 0;
    end else begin
      // PASS1: chip1 receives chip0's dW (via fc1 peer_rx)
      if (tst == ST_WAIT_FAB_PASS1) begin
        if (fc1_prx_valid) begin
          peer_dw1[rx1_idx[7:4]][rx1_idx[3:0]] <= fc1_prx_data;
          rx1_idx <= rx1_idx + 1;
        end
      end
      // PASS2: chip0 receives chip1's dW (via fc0 peer_rx)
      if (tst == ST_WAIT_FAB_PASS2) begin
        if (fc0_prx_valid) begin
          peer_dw0[rx0_idx[7:4]][rx0_idx[3:0]] <= fc0_prx_data;
          rx0_idx <= rx0_idx + 1;
        end
      end
      if (tst == ST_FAB_PASS1) begin rx0_idx <= 0; rx1_idx <= 0; end
    end
  end

  // TX index advance
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin tx0_idx <= 0; tx1_idx <= 0; end
    else begin
      if (tst == ST_FAB_PASS1) begin tx0_idx <= 0; end
      else if (tst == ST_FAB_PASS2) begin tx1_idx <= 0; end
      else begin
        if (fc0_ltx_valid && fc0_ltx_ready) tx0_idx <= tx0_idx + 1;
        if (fc1_ltx_valid && fc1_ltx_ready) tx1_idx <= tx1_idx + 1;
      end
    end
  end

  // ═══════════════════════════════════════════════════════════
  // Collective engines (per chip)
  // ═══════════════════════════════════════════════════════════
  logic coll0_start, coll0_done, coll1_start, coll1_done;
  logic [7:0] coll0_err, coll1_err;

  collective_engine #(.DIM(DIM)) u_coll0 (
    .clk(clk),.rst_n(rst_n),.start(coll0_start),
    .op_type(8'h01),.peer_mask(8'h03),
    .local_in(dw0_out),.peer_in(peer_dw0),.result_out(reduced_dw0),
    .busy(),.done_pulse(coll0_done),.err_code(coll0_err));

  collective_engine #(.DIM(DIM)) u_coll1 (
    .clk(clk),.rst_n(rst_n),.start(coll1_start),
    .op_type(8'h01),.peer_mask(8'h03),
    .local_in(dw1_out),.peer_in(peer_dw1),.result_out(reduced_dw1),
    .busy(),.done_pulse(coll1_done),.err_code(coll1_err));

  // ═══════════════════════════════════════════════════════════
  // Optimizers
  // ═══════════════════════════════════════════════════════════
  logic opt0_start, opt0_done, opt1_start, opt1_done;
  logic [7:0] opt0_err, opt1_err;

  optimizer_unit #(.DIM(DIM)) u_opt0 (
    .clk(clk),.rst_n(rst_n),.start(opt0_start),.adamw_enable(adamw_enable),
    .lr_fp32(lr_fp32),.beta1_fp32(beta1_fp32),.beta2_fp32(beta2_fp32),
    .epsilon_fp32(epsilon_fp32),.weight_decay_fp32(weight_decay_fp32),
    .param_in(param0_in),.grad_in(reduced_dw0),.m_in(m0_in),.v_in(v0_in),
    .param_out(param0_out),.m_out(m0_out),.v_out(v0_out),
    .busy(),.done_pulse(opt0_done),.err_code(opt0_err));

  optimizer_unit #(.DIM(DIM)) u_opt1 (
    .clk(clk),.rst_n(rst_n),.start(opt1_start),.adamw_enable(adamw_enable),
    .lr_fp32(lr_fp32),.beta1_fp32(beta1_fp32),.beta2_fp32(beta2_fp32),
    .epsilon_fp32(epsilon_fp32),.weight_decay_fp32(weight_decay_fp32),
    .param_in(param1_in),.grad_in(reduced_dw1),.m_in(m1_in),.v_in(v1_in),
    .param_out(param1_out),.m_out(m1_out),.v_out(v1_out),
    .busy(),.done_pulse(opt1_done),.err_code(opt1_err));

  // ═══════════════════════════════════════════════════════════
  // Done capture
  // ═══════════════════════════════════════════════════════════
  logic bk0_d,bk1_d,fc0_sd,fc1_sd,fc0_rd,fc1_rd,coll0_d,coll1_d,opt0_d,opt1_d;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      {bk0_d,bk1_d,fc0_sd,fc1_sd,fc0_rd,fc1_rd,coll0_d,coll1_d,opt0_d,opt1_d} <= '0;
    end else begin
      if (tst==ST_BKWD) begin bk0_d<=0;bk1_d<=0; end else begin
        if(bk0_done) bk0_d<=1; if(bk1_done) bk1_d<=1; end
      if (tst==ST_FAB_PASS1) begin fc0_sd<=0; fc1_rd<=0; end
      else if (tst==ST_FAB_PASS2) begin fc1_sd<=0; fc0_rd<=0; end
      else begin
        if(fc0_done) begin if(fc0_busy==0) begin fc0_sd<=1; fc0_rd<=1; end end
        if(fc1_done) begin if(fc1_busy==0) begin fc1_sd<=1; fc1_rd<=1; end end
      end
      if (tst==ST_COLL) begin coll0_d<=0;coll1_d<=0; end else begin
        if(coll0_done) coll0_d<=1; if(coll1_done) coll1_d<=1; end
      if (tst==ST_OPT) begin opt0_d<=0;opt1_d<=0; end else begin
        if(opt0_done) opt0_d<=1; if(opt1_done) opt1_d<=1; end
    end
  end

  // ═══════════════════════════════════════════════════════════
  // Top FSM
  // ═══════════════════════════════════════════════════════════
  // Half-duplex requires sequential: chip0 send + chip1 recv, then swap.
  typedef enum logic [3:0] {
    ST_IDLE,
    ST_BKWD,
    ST_WAIT_BKWD,
    ST_FAB_PASS1,       // chip0 send, chip1 recv (chip1 gets chip0's dW)
    ST_WAIT_FAB_PASS1,
    ST_FAB_PASS2,       // chip1 send, chip0 recv (chip0 gets chip1's dW)
    ST_WAIT_FAB_PASS2,
    ST_COLL,
    ST_WAIT_COLL,
    ST_OPT,
    ST_WAIT_OPT,
    ST_DONE,
    ST_FAULT
  } state_t;

  state_t tst;
  assign busy = (tst != ST_IDLE && tst != ST_DONE && tst != ST_FAULT);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      tst <= ST_IDLE;
      {bk0_start,bk1_start} <= '0;
      {fc0_send_start,fc1_send_start,fc0_recv_start,fc1_recv_start} <= '0;
      {coll0_start,coll1_start,opt0_start,opt1_start} <= '0;
      done_pulse <= 0; err_code <= 0;
    end else begin
      done_pulse <= 0;
      {bk0_start,bk1_start} <= '0;
      {fc0_send_start,fc1_send_start,fc0_recv_start,fc1_recv_start} <= '0;
      {coll0_start,coll1_start,opt0_start,opt1_start} <= '0;

      case (tst)
        ST_IDLE: begin
          err_code <= 0;
          if (start) tst <= ST_BKWD;
        end

        ST_BKWD: begin bk0_start<=1; bk1_start<=1; tst<=ST_WAIT_BKWD; end
        ST_WAIT_BKWD: if (bk0_d && bk1_d) begin
          if(bk0_err) begin err_code<=8'h10; tst<=ST_FAULT; end
          else if(bk1_err) begin err_code<=8'h11; tst<=ST_FAULT; end
          else tst<=ST_FAB_PASS1;
        end

        // PASS1: chip0 sends, chip1 receives
        ST_FAB_PASS1: begin
          fc0_send_start<=1; fc1_recv_start<=1;
          tst<=ST_WAIT_FAB_PASS1;
        end
        ST_WAIT_FAB_PASS1: begin
          if (fc0_sd && fc1_rd) tst <= ST_FAB_PASS2;
        end

        // PASS2: chip1 sends, chip0 receives
        ST_FAB_PASS2: begin
          fc1_send_start<=1; fc0_recv_start<=1;
          tst<=ST_WAIT_FAB_PASS2;
        end
        ST_WAIT_FAB_PASS2: begin
          if (fc1_sd && fc0_rd) tst <= ST_COLL;
        end

        ST_COLL: begin coll0_start<=1; coll1_start<=1; tst<=ST_WAIT_COLL; end
        ST_WAIT_COLL: if (coll0_d && coll1_d) begin
          if(coll0_err) begin err_code<=8'h30; tst<=ST_FAULT; end
          else tst<=ST_OPT;
        end

        ST_OPT: begin opt0_start<=1; opt1_start<=1; tst<=ST_WAIT_OPT; end
        ST_WAIT_OPT: if (opt0_d && opt1_d) begin
          if(opt0_err) begin err_code<=8'h40; tst<=ST_FAULT; end
          else tst<=ST_DONE;
        end

        ST_DONE: begin done_pulse<=1; tst<=ST_IDLE; end
        ST_FAULT: begin done_pulse<=1; tst<=ST_IDLE; end
        default: tst<=ST_IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
