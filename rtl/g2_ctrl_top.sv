// g2_ctrl_top.sv — ORBIT-G2 Proto-A Control-Plane Top
// SSOT: ORBIT_G2_DETAIL_BLOCKDIAG.md, ORBIT_G2_REG_SPEC.md, ORBIT_G2_RTL_SKELETONS.md
//
// Proto-A device contract: all REG_SPEC blocks that can be implemented without
// PCIe/HBM/ICI are wired here. Includes TC0 status, DMA shim, perf counters,
// watchdog stub, and full multi-queue control plane.
`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off PINCONNECTEMPTY */

module g2_ctrl_top #(
  parameter int MAX_KT      = 256,
  parameter int QUEUE_DEPTH = 16,
  parameter int DESC_BYTES  = 64,
  parameter int DESC_COST   = 4096,
  parameter int DMA_COST    = 16384
)(
  input  logic        clk,
  input  logic        por_n,
  input  logic [19:0] reg_addr,
  input  logic        reg_wr_en,
  input  logic [31:0] reg_wr_data,
  output logic [31:0] reg_rd_data,

  output logic         rd_req_valid,
  input  logic         rd_req_ready,
  output logic [63:0]  rd_req_addr,
  output logic [15:0]  rd_req_len_bytes,
  input  logic         rd_done,
  input  logic         rd_data_valid,
  input  logic [127:0] rd_data,
  output logic         rd_data_ready,
  input  logic         rd_data_last,

  output logic         wr_req_valid,
  input  logic         wr_req_ready,
  output logic [63:0]  wr_req_addr,
  output logic [15:0]  wr_req_len_bytes,
  input  logic         wr_done,
  output logic         wr_data_valid,
  input  logic         wr_data_ready,
  output logic [127:0] wr_data,
  output logic         wr_data_last,

  output logic         irq_out,
  output logic         reset_active
);

  // ===============================================================
  // Reset (with watchdog stub)
  // ===============================================================
  logic rst_io_n, rst_mem_n, rst_core_n;
  logic [3:0] boot_cause;
  logic sw_reset_pulse, sw_cause_clr, wdog_test_pulse;

  reset_seq u_reset (
    .clk(clk), .por_n(por_n),
    .sw_reset(sw_reset_pulse),
    .wdog_reset(wdog_test_pulse),  // watchdog stub: test inject only
    .pcie_flr(1'b0),
    .rst_io_n(rst_io_n), .rst_mem_n(rst_mem_n), .rst_core_n(rst_core_n),
    .boot_cause(boot_cause), .sw_cause_clr(sw_cause_clr),
    .reset_active(reset_active)
  );

  wire logic rst_n = rst_core_n;

  // ===============================================================
  // Forward declarations for reg_top connections
  // ===============================================================
  localparam int NUM_QUEUES = 4;
  localparam int DESC_WORDS = DESC_BYTES / 4;
  localparam int DESC_W     = DESC_BYTES * 8;

  logic [31:0] desc_stage [0:DESC_WORDS-1];
  logic [NUM_QUEUES-1:0] doorbell_pulse;
  logic [15:0] q_head [0:NUM_QUEUES-1];
  logic [15:0] q_tail [0:NUM_QUEUES-1];
  logic [15:0] q_depth [0:NUM_QUEUES-1];
  logic [NUM_QUEUES-1:0] overflow_flags, overflow_clr;

  logic [1:0] oom_state;
  logic oom_admission_stop, oom_prefetch_clamp;
  logic [39:0] oom_allocated, oom_reserved, oom_effective;

  logic [31:0] irq_pending, irq_mask_rd, irq_cause_last;
  logic irq_pending_w1c_en, irq_mask_wr_en, irq_force_wr_en;
  logic [31:0] irq_pending_w1c_data, irq_mask_wr_data, irq_force_wr_data;

  logic [15:0] trace_head, trace_tail;
  logic [31:0] trace_drop_count;
  logic trace_enable, trace_freeze, trace_fatal_only;
  logic [9:0] trace_rd_addr;
  logic [63:0] trace_rd_data;
  logic [3:0] trace_rd_type;
  logic trace_rd_fatal;

  // TC0 status signals (computed below)
  logic [31:0] tc0_runstate_val, tc0_fault_status_val;
  logic [63:0] tc0_perf_cycles_val, tc0_desc_ptr_val;
  logic tc0_enable, tc0_halt, tc0_fault_clr;

  // DMA shim
  logic [31:0] dma_status_val, dma_err_code_val;

  // Perf
  logic [63:0] mxu_busy_cycles_val;
  logic [31:0] mxu_tile_count_val, desc_done_count_val;
  logic perf_freeze;

  // ===============================================================
  // Register Bank
  // ===============================================================
  reg_top #(
    .NUM_QUEUES(NUM_QUEUES), .DESC_WORDS(DESC_WORDS), .TRACE_ADDR_W(10)
  ) u_reg (
    .clk(clk), .rst_n(rst_n),
    .addr(reg_addr), .wr_en(reg_wr_en), .wr_data(reg_wr_data), .rd_data(reg_rd_data),
    .boot_cause(boot_cause), .sw_reset_pulse(sw_reset_pulse), .sw_cause_clr(sw_cause_clr),
    .wdog_test_pulse(wdog_test_pulse),
    .desc_stage(desc_stage), .doorbell_pulse(doorbell_pulse),
    .q_head(q_head), .q_tail(q_tail),
    .overflow_flags(overflow_flags), .overflow_clr(overflow_clr),
    .oom_state(oom_state), .oom_admission_stop(oom_admission_stop),
    .oom_prefetch_clamp(oom_prefetch_clamp),
    .oom_usage_lo(oom_allocated[31:0]), .oom_reserved_lo(oom_reserved[31:0]),
    .oom_effective_lo(oom_effective[31:0]),
    .tc0_runstate(tc0_runstate_val), .tc0_fault_status(tc0_fault_status_val),
    .tc0_perf_cycles(tc0_perf_cycles_val), .tc0_desc_ptr(tc0_desc_ptr_val),
    .tc0_enable(tc0_enable), .tc0_halt(tc0_halt), .tc0_fault_clr(tc0_fault_clr),
    .dma_status(dma_status_val), .dma_err_code(dma_err_code_val),
    .mxu_busy_cycles(mxu_busy_cycles_val), .mxu_tile_count(mxu_tile_count_val),
    .desc_done_count(desc_done_count_val), .perf_freeze(perf_freeze),
    .irq_pending(irq_pending), .irq_mask_rd(irq_mask_rd), .irq_cause_last(irq_cause_last),
    .irq_pending_w1c_en(irq_pending_w1c_en), .irq_pending_w1c_data(irq_pending_w1c_data),
    .irq_mask_wr_en(irq_mask_wr_en), .irq_mask_wr_data(irq_mask_wr_data),
    .irq_force_wr_en(irq_force_wr_en), .irq_force_wr_data(irq_force_wr_data),
    .trace_head(trace_head), .trace_tail(trace_tail), .trace_drop_count(trace_drop_count),
    .trace_enable(trace_enable), .trace_freeze(trace_freeze), .trace_fatal_only(trace_fatal_only),
    .trace_rd_addr(trace_rd_addr), .trace_rd_data(trace_rd_data),
    .trace_rd_type(trace_rd_type), .trace_rd_fatal(trace_rd_fatal)
  );

  // ===============================================================
  // Descriptor Queue
  // ===============================================================
  logic [NUM_QUEUES-1:0] q_push_valid, q_push_ready;
  logic [DESC_W-1:0] q_push_data [0:NUM_QUEUES-1];
  logic [NUM_QUEUES-1:0] q_pop_valid, q_pop_ready;
  logic [DESC_W-1:0] q_pop_data [0:NUM_QUEUES-1];
  logic any_overflow;

  desc_queue #(.NUM_QUEUES(NUM_QUEUES), .QUEUE_DEPTH(QUEUE_DEPTH), .DESC_W(DESC_W)) u_desc_queue (
    .clk(clk), .rst_n(rst_n),
    .push_valid(q_push_valid), .push_ready(q_push_ready), .push_data(q_push_data),
    .pop_valid(q_pop_valid), .pop_ready(q_pop_ready), .pop_data(q_pop_data),
    .q_head(q_head), .q_tail(q_tail), .q_depth(q_depth),
    .overflow_flags(overflow_flags), .overflow_clr(overflow_clr),
    .any_overflow(any_overflow)
  );

  logic [DESC_W-1:0] staged_desc_packed;
  genvar gi;
  generate for (gi = 0; gi < DESC_WORDS; gi++) begin : PACK
    assign staged_desc_packed[gi*32 +: 32] = desc_stage[gi];
  end endgenerate

  genvar pqi;
  generate for (pqi = 0; pqi < NUM_QUEUES; pqi++) begin : PUSH
    assign q_push_valid[pqi] = doorbell_pulse[pqi] & ~oom_admission_stop;
    assign q_push_data[pqi]  = staged_desc_packed;
  end endgenerate

  // ===============================================================
  // Priority Arbiter Q3>Q0>Q1>Q2 (gated by tc0_enable, blocked by tc0_halt)
  // ===============================================================
  logic [1:0] sel_queue;
  logic arb_valid, arb_ready;

  always_comb begin
    sel_queue = 2'd0; arb_valid = 1'b0;
    if (tc0_enable && !tc0_halt) begin
      if      (q_pop_ready[3]) begin sel_queue = 2'd3; arb_valid = 1'b1; end
      else if (q_pop_ready[0]) begin sel_queue = 2'd0; arb_valid = 1'b1; end
      else if (q_pop_ready[1]) begin sel_queue = 2'd1; arb_valid = 1'b1; end
      else if (q_pop_ready[2]) begin sel_queue = 2'd2; arb_valid = 1'b1; end
    end
  end

  assign q_pop_valid[0] = arb_ready & arb_valid & (sel_queue == 2'd0);
  assign q_pop_valid[1] = arb_ready & arb_valid & (sel_queue == 2'd1);
  assign q_pop_valid[2] = arb_ready & arb_valid & (sel_queue == 2'd2);
  assign q_pop_valid[3] = arb_ready & arb_valid & (sel_queue == 2'd3);

  // ===============================================================
  // Descriptor Hold
  // ===============================================================
  logic [7:0] desc_hold [0:DESC_BYTES-1];
  logic [1:0] desc_hold_qclass;

  logic [DESC_W-1:0] sel_pop_data;
  always_comb begin
    case (sel_queue)
      2'd0: sel_pop_data = q_pop_data[0];
      2'd1: sel_pop_data = q_pop_data[1];
      2'd2: sel_pop_data = q_pop_data[2];
      2'd3: sel_pop_data = q_pop_data[3];
    endcase
  end

  integer dhi;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (dhi = 0; dhi < DESC_BYTES; dhi++) desc_hold[dhi] <= 8'd0;
      desc_hold_qclass <= 2'd0;
    end else if (arb_valid && arb_ready) begin
      for (dhi = 0; dhi < DESC_BYTES; dhi++) desc_hold[dhi] <= sel_pop_data[dhi*8 +: 8];
      desc_hold_qclass <= sel_queue;
    end
  end

  // ===============================================================
  // desc_fsm_v2
  // ===============================================================
  logic fsm_desc_valid, fsm_desc_ready;
  logic fsm_cmd_valid, fsm_cmd_ready;
  logic [7:0] fsm_cmd_opcode;
  logic [63:0] fsm_act_addr, fsm_wgt_addr, fsm_out_addr;
  logic [31:0] fsm_Kt;
  logic fsm_fault_valid;
  logic [7:0] fsm_fault_code;
  logic fsm_busy, fsm_done_pulse;

  assign fsm_desc_valid = arb_valid;
  assign arb_ready      = fsm_desc_ready;

  desc_fsm_v2 #(.DESC_SIZE(DESC_BYTES)) u_desc_fsm (
    .clk(clk), .rst_n(rst_n),
    .desc_valid(fsm_desc_valid), .desc_bytes(desc_hold), .desc_ready(fsm_desc_ready),
    .queue_class(desc_hold_qclass),
    .cmd_valid(fsm_cmd_valid), .cmd_ready(fsm_cmd_ready),
    .cmd_opcode(fsm_cmd_opcode),
    .act_addr(fsm_act_addr), .wgt_addr(fsm_wgt_addr), .out_addr(fsm_out_addr),
    .Kt(fsm_Kt),
    .core_done(gemm_done_pulse),
    .timeout_cycles(32'd100_000),
    .fault_valid(fsm_fault_valid), .fault_code(fsm_fault_code),
    .busy(fsm_busy), .done_pulse(fsm_done_pulse)
  );

  // ===============================================================
  // GEMM Top
  // ===============================================================
  logic gemm_desc_valid, gemm_desc_ready;
  logic gemm_busy, gemm_done_pulse;
  logic [31:0] gemm_perf_cycles, gemm_perf_bytes;

  assign gemm_desc_valid = fsm_cmd_valid && (fsm_cmd_opcode == 8'h02);
  assign fsm_cmd_ready   = gemm_desc_ready;

  gemm_top #(.MAX_KT(MAX_KT)) u_gemm (
    .clk(clk), .rst_n(rst_n),
    .desc_valid(gemm_desc_valid), .desc_bytes(desc_hold), .desc_ready(gemm_desc_ready),
    .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready),
    .rd_req_addr(rd_req_addr), .rd_req_len_bytes(rd_req_len_bytes), .rd_done(rd_done),
    .rd_data_valid(rd_data_valid), .rd_data(rd_data),
    .rd_data_ready(rd_data_ready), .rd_data_last(rd_data_last),
    .wr_req_valid(wr_req_valid), .wr_req_ready(wr_req_ready),
    .wr_req_addr(wr_req_addr), .wr_req_len_bytes(wr_req_len_bytes), .wr_done(wr_done),
    .wr_data_valid(wr_data_valid), .wr_data_ready(wr_data_ready),
    .wr_data(wr_data), .wr_data_last(wr_data_last),
    .busy(gemm_busy), .done_pulse(gemm_done_pulse),
    .perf_cycles(gemm_perf_cycles), .perf_bytes(gemm_perf_bytes)
  );

  // ===============================================================
  // TC0 RUNSTATE / FAULT_STATUS (REG_SPEC section 7)
  // ===============================================================
  // RUNSTATE: 0=IDLE,1=FETCH,2=RUN,3=STALL,4=FAULT
  always_comb begin
    tc0_runstate_val = 32'd0;
    if (fsm_fault_valid)
      tc0_runstate_val[2:0] = 3'd4;       // FAULT
    else if (gemm_busy) begin
      tc0_runstate_val[2:0] = 3'd2;       // RUN
      tc0_runstate_val[8] = rd_req_valid;  // WAIT_DMA
    end else if (fsm_busy)
      tc0_runstate_val[2:0] = 3'd1;       // FETCH (CRC/decode/dispatch)
    // else IDLE = 0
  end

  // FAULT_STATUS: latched fault code, W1C via tc0_fault_clr
  logic [31:0] fault_status_r;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      fault_status_r <= 32'd0;
    else if (tc0_fault_clr)
      fault_status_r <= 32'd0;
    else if (fsm_fault_valid)
      fault_status_r <= {24'd0, fsm_fault_code};
  end
  assign tc0_fault_status_val = fault_status_r;

  // DESC_PTR: latch act_addr on dispatch as proxy
  logic [63:0] desc_ptr_r;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) desc_ptr_r <= 64'd0;
    else if (fsm_cmd_valid && fsm_cmd_ready) desc_ptr_r <= fsm_act_addr;
  end
  assign tc0_desc_ptr_val = desc_ptr_r;

  // PERF_CYCLES: real cycles from gemm_core (activity-based, freezable)
  logic [63:0] perf_cyc_r;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) perf_cyc_r <= 64'd0;
    else if (!perf_freeze && (fsm_busy || gemm_busy))
      perf_cyc_r <= perf_cyc_r + 1'b1;
  end
  assign tc0_perf_cycles_val = perf_cyc_r;

  // ===============================================================
  // DMA Status Shim (Proto-A: derived from gemm/FSM state)
  // ===============================================================
  // Proto-A shim semantics: no real DMA bridge.
  // BUSY/DONE/ERR/TIMEOUT derived from gemm_top and desc_fsm_v2.
  // Proto-B replaces with real DMA bridge outputs.
  logic dma_done_latch, dma_err_latch, dma_timeout_latch;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      dma_done_latch    <= 1'b0;
      dma_err_latch     <= 1'b0;
      dma_timeout_latch <= 1'b0;
    end else begin
      if (gemm_done_pulse) dma_done_latch <= 1'b1;
      if (fsm_fault_valid && fsm_fault_code != 8'h03)
        dma_err_latch <= 1'b1;
      if (fsm_fault_valid && fsm_fault_code == 8'h03)
        dma_timeout_latch <= 1'b1;
      // Clear on new dispatch
      if (fsm_cmd_valid && fsm_cmd_ready) begin
        dma_done_latch    <= 1'b0;
        dma_err_latch     <= 1'b0;
        dma_timeout_latch <= 1'b0;
      end
    end
  end

  assign dma_status_val = {16'd0,
                           gemm_busy ? 8'd1 : 8'd0,  // INFLIGHT[15:8]
                           4'd0,
                           dma_timeout_latch,          // [3]
                           dma_err_latch,              // [2]
                           dma_done_latch,             // [1]
                           gemm_busy};                 // [0] BUSY
  assign dma_err_code_val = fault_status_r;

  // ===============================================================
  // Perf Counters (activity-based)
  // ===============================================================
  // MXU busy cycles = gemm_perf_cycles (from gemm_core, real)
  logic [63:0] mxu_bcyc_r;
  logic [31:0] tile_count_r, done_count_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      mxu_bcyc_r  <= 64'd0;
      tile_count_r <= 32'd0;
      done_count_r <= 32'd0;
    end else if (!perf_freeze) begin
      if (gemm_busy) mxu_bcyc_r <= mxu_bcyc_r + 1'b1;
      if (gemm_done_pulse) tile_count_r <= tile_count_r + 1'b1;
      if (fsm_done_pulse)  done_count_r <= done_count_r + 1'b1;
    end
  end

  assign mxu_busy_cycles_val = mxu_bcyc_r;
  assign mxu_tile_count_val  = tile_count_r;
  assign desc_done_count_val = done_count_r;

  // ===============================================================
  // OOM Guard
  // ===============================================================
  logic oom_alloc_inc, oom_alloc_dec, oom_dma_inc, oom_dma_dec, oom_underflow;
  assign oom_alloc_inc = |doorbell_pulse & ~oom_admission_stop;
  assign oom_alloc_dec = fsm_done_pulse;
  assign oom_dma_inc   = fsm_cmd_valid & fsm_cmd_ready & (fsm_cmd_opcode == 8'h02);
  assign oom_dma_dec   = gemm_done_pulse | (fsm_fault_valid & (fsm_cmd_opcode == 8'h02));

  oom_guard u_oom (
    .clk(clk), .rst_n(rst_n),
    .alloc_inc(oom_alloc_inc), .alloc_dec(oom_alloc_dec), .alloc_bytes(32'(DESC_COST)),
    .dma_reserve_inc(oom_dma_inc), .dma_reserve_dec(oom_dma_dec), .dma_rsv_bytes(32'(DMA_COST)),
    .prefetch_reserve_inc(1'b0), .prefetch_reserve_dec(1'b0), .prefetch_rsv_bytes(32'd0),
    .thresh_pressure(40'd100_000), .thresh_critical(40'd200_000), .thresh_emergency(40'd300_000),
    .pressure_state(oom_state), .admission_stop(oom_admission_stop), .prefetch_clamp(oom_prefetch_clamp),
    .allocated_bytes(oom_allocated), .reserved_bytes(oom_reserved),
    .effective_usage(oom_effective), .underflow_error(oom_underflow)
  );

  // ===============================================================
  // Trace Ring
  // ===============================================================
  logic trace_valid;
  logic [3:0] trace_type;
  logic trace_fatal_flag;
  logic [63:0] trace_payload;

  localparam logic [3:0] TEVT_DISPATCH = 4'd1;
  localparam logic [3:0] TEVT_DONE     = 4'd2;
  localparam logic [3:0] TEVT_FAULT    = 4'd3;
  localparam logic [3:0] TEVT_OVERFLOW = 4'd4;

  always_comb begin
    trace_valid = 1'b0; trace_type = 4'd0; trace_fatal_flag = 1'b0; trace_payload = 64'd0;
    if (fsm_fault_valid) begin
      trace_valid = 1'b1; trace_type = TEVT_FAULT; trace_fatal_flag = 1'b1;
      trace_payload = {46'd0, desc_hold_qclass, fsm_fault_code, 8'd0};
    end else if (fsm_done_pulse) begin
      trace_valid = 1'b1; trace_type = TEVT_DONE;
      trace_payload = {46'd0, desc_hold_qclass, fsm_cmd_opcode, 8'd0};
    end else if (fsm_cmd_valid && fsm_cmd_ready) begin
      trace_valid = 1'b1; trace_type = TEVT_DISPATCH;
      trace_payload = {46'd0, desc_hold_qclass, fsm_cmd_opcode, 8'd0};
    end else if (any_overflow) begin
      trace_valid = 1'b1; trace_type = TEVT_OVERFLOW; trace_fatal_flag = 1'b1;
      trace_payload = {60'd0, overflow_flags};
    end
  end

  logic trace_wrap_irq;

  trace_ring #(.DEPTH(1024), .ENTRY_W(64), .TYPE_W(4)) u_trace (
    .clk(clk), .rst_n(rst_n),
    .trace_valid(trace_valid), .trace_type(trace_type),
    .trace_fatal(trace_fatal_flag), .trace_payload(trace_payload),
    .ctrl_enable(trace_enable), .ctrl_freeze(trace_freeze), .ctrl_fatal_only(trace_fatal_only),
    .ring_head(trace_head), .ring_tail(trace_tail), .drop_count(trace_drop_count),
    .rd_addr(trace_rd_addr), .rd_data(trace_rd_data),
    .rd_type(trace_rd_type), .rd_fatal(trace_rd_fatal),
    .wrap_irq_pulse(trace_wrap_irq)
  );

  // ===============================================================
  // IRQ Controller
  // ===============================================================
  logic [1:0] oom_state_d;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) oom_state_d <= 2'd0; else oom_state_d <= oom_state;
  end
  wire oom_press_edge = (oom_state >= 2'd1) & (oom_state_d < 2'd1);
  wire oom_emerg_edge = (oom_state == 2'd3) & (oom_state_d != 2'd3);

  logic [31:0] irq_sources;
  assign irq_sources = {20'd0, trace_wrap_irq, 1'b0/*wdog*/, 1'b0/*ici*/,
                         1'b0/*hbm_uncorr*/, 1'b0/*hbm_corr*/, 1'b0/*tc1*/,
                         fsm_fault_valid, oom_emerg_edge, oom_press_edge,
                         1'b0/*dma_err*/, 1'b0/*dma_done*/, fsm_done_pulse};

  logic [31:0] irq_mask_out;
  logic msix_req;
  logic [4:0] msix_vector;

  irq_ctrl u_irq (
    .clk(clk), .rst_n(rst_n),
    .irq_sources(irq_sources),
    .pending_w1c_en(irq_pending_w1c_en), .pending_w1c_data(irq_pending_w1c_data),
    .mask_wr_en(irq_mask_wr_en), .mask_wr_data(irq_mask_wr_data),
    .force_wr_en(irq_force_wr_en), .force_wr_data(irq_force_wr_data),
    .irq_pending(irq_pending), .irq_mask(irq_mask_out), .irq_cause_last(irq_cause_last),
    .irq_out(irq_out), .msix_req(msix_req), .msix_vector(msix_vector)
  );
  assign irq_mask_rd = irq_mask_out;

endmodule
`default_nettype wire
