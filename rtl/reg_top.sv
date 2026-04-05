// reg_top.sv — ORBIT-G2 Proto-A Register Bank
// SSOT: ORBIT_G2_REG_SPEC.md
//
// Blocks implemented:
//   0x0_0xxx  Global/Version (RO)
//   0x0_1xxx  Reset/Boot/Watchdog
//   0x0_2xxx  Queue Doorbell (WO) + Descriptor staging (RW)
//   0x0_3xxx  Queue Status (RO) + Q_OVERFLOW (W1C)
//   0x1_0xxx  DMA Engine shim (Proto-A: derived from gemm state)
//   0x2_0xxx  OOM Guard
//   0x4_0xxx  TC0 Control/Status
//   0x5_0xxx  TC1 (read-as-zero, HAS_TC1=0)
//   0x6_0xxx  VPU/MXU Perf
//   0x9_0xxx  IRQ ctrl
//   0xA_0xxx  Trace Ring + read window
`timescale 1ns/1ps
`default_nettype none

module reg_top #(
  parameter int NUM_QUEUES    = 4,
  parameter int DESC_WORDS    = 16,
  parameter int TRACE_ADDR_W  = 10
)(
  input  logic        clk,
  input  logic        rst_n,

  // Register bus
  input  logic [19:0] addr,
  input  logic        wr_en,
  input  logic [31:0] wr_data,
  output logic [31:0] rd_data,

  // ── reset_seq ──
  input  logic [3:0]  boot_cause,
  output logic        sw_reset_pulse,
  output logic        sw_cause_clr,
  output logic        wdog_test_pulse,   // watchdog test inject

  // ── desc_queue ──
  output logic [31:0]              desc_stage [0:DESC_WORDS-1],
  output logic [NUM_QUEUES-1:0]    doorbell_pulse,
  input  logic [15:0] q_head [0:NUM_QUEUES-1],
  input  logic [15:0] q_tail [0:NUM_QUEUES-1],
  input  logic [NUM_QUEUES-1:0] overflow_flags,
  output logic [NUM_QUEUES-1:0] overflow_clr,

  // ── OOM guard ──
  input  logic [1:0]   oom_state,
  input  logic         oom_admission_stop,
  input  logic         oom_prefetch_clamp,
  input  logic [31:0]  oom_usage_lo,
  input  logic [31:0]  oom_reserved_lo,
  input  logic [31:0]  oom_effective_lo,

  // ── TC0 status (from g2_ctrl_top) ──
  input  logic [31:0]  tc0_runstate,
  input  logic [31:0]  tc0_fault_status,
  input  logic [63:0]  tc0_perf_cycles,
  input  logic [63:0]  tc0_desc_ptr,
  output logic         tc0_enable,
  output logic         tc0_halt,
  output logic         tc0_fault_clr,

  // ── DMA shim status ──
  input  logic [31:0]  dma_status,
  input  logic [31:0]  dma_err_code,

  // ── Perf counters ──
  input  logic [63:0]  mxu_busy_cycles,
  input  logic [31:0]  mxu_tile_count,
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [31:0]  desc_done_count,  // descriptor completion count (no REG_SPEC address yet)
  /* verilator lint_on UNUSEDSIGNAL */
  output logic         perf_freeze,

  // ── IRQ ctrl ──
  input  logic [31:0] irq_pending,
  input  logic [31:0] irq_mask_rd,
  input  logic [31:0] irq_cause_last,
  output logic        irq_pending_w1c_en,
  output logic [31:0] irq_pending_w1c_data,
  output logic        irq_mask_wr_en,
  output logic [31:0] irq_mask_wr_data,
  output logic        irq_force_wr_en,
  output logic [31:0] irq_force_wr_data,

  // ── Trace ring ──
  input  logic [15:0] trace_head,
  input  logic [15:0] trace_tail,
  input  logic [31:0] trace_drop_count,
  output logic        trace_enable,
  output logic        trace_freeze,
  output logic        trace_fatal_only,
  output logic [TRACE_ADDR_W-1:0] trace_rd_addr,
  input  logic [63:0]             trace_rd_data,
  input  logic [3:0]              trace_rd_type,
  input  logic                    trace_rd_fatal
);

  // ===============================================================
  // Address map (offset from 0x8030_0000)
  // ===============================================================
  // Global 0x0_0000
  localparam logic [19:0] A_G2_ID       = 20'h0_0000;
  localparam logic [19:0] A_G2_VERSION  = 20'h0_0004;
  localparam logic [19:0] A_G2_CAP0     = 20'h0_0008;
  localparam logic [19:0] A_G2_CAP1     = 20'h0_000C;
  localparam logic [19:0] A_BUILD_LO    = 20'h0_0010;
  localparam logic [19:0] A_BUILD_HI    = 20'h0_0014;

  // Reset 0x0_1000
  localparam logic [19:0] A_BOOT_CAUSE  = 20'h0_1000;
  localparam logic [19:0] A_SW_RESET    = 20'h0_1004;
  localparam logic [19:0] A_BOOT_VEC_LO = 20'h0_1008;
  localparam logic [19:0] A_BOOT_VEC_HI = 20'h0_100C;
  localparam logic [19:0] A_STRAP       = 20'h0_1010;
  localparam logic [19:0] A_WDOG_CTRL   = 20'h0_1014;

  // Queue 0x0_2000 / 0x0_3000
  localparam logic [19:0] A_Q0_DOORBELL = 20'h0_2000;
  localparam logic [19:0] A_DESC_STAGE_BASE = 20'h0_2100;
  localparam logic [19:0] A_DESC_STAGE_END  = 20'h0_213C;
  localparam logic [19:0] A_Q0_STATUS   = 20'h0_3000;
  localparam logic [19:0] A_Q_OVERFLOW  = 20'h0_3010;

  // DMA 0x1_0000
  localparam logic [19:0] A_DMA_STATUS  = 20'h1_0010;
  localparam logic [19:0] A_DMA_ERR     = 20'h1_0014;

  // OOM 0x2_0000
  localparam logic [19:0] A_OOM_USAGE_LO = 20'h2_0000;
  localparam logic [19:0] A_OOM_RESV_LO  = 20'h2_0008;
  localparam logic [19:0] A_OOM_EFF_LO   = 20'h2_0010;
  localparam logic [19:0] A_OOM_STATE    = 20'h2_001C;

  // TC0 0x4_0000 (0x8034_0000 - 0x8030_0000 = 0x40000)
  localparam logic [19:0] A_TC0_RUNSTATE = 20'h4_0000;
  localparam logic [19:0] A_TC0_CTRL     = 20'h4_0004;
  localparam logic [19:0] A_TC0_DPTR_LO  = 20'h4_0008;
  localparam logic [19:0] A_TC0_DPTR_HI  = 20'h4_000C;
  localparam logic [19:0] A_TC0_PCYC_LO  = 20'h4_0010;
  localparam logic [19:0] A_TC0_PCYC_HI  = 20'h4_0014;
  localparam logic [19:0] A_TC0_FAULT    = 20'h4_0018;

  // Perf 0x6_0000 (0x8036_0000)
  localparam logic [19:0] A_MXU_BCYC_LO = 20'h6_0000;
  localparam logic [19:0] A_MXU_BCYC_HI = 20'h6_0004;
  localparam logic [19:0] A_VPU_BCYC_LO = 20'h6_0008;  // read-as-zero in Proto-A
  localparam logic [19:0] A_VPU_BCYC_HI = 20'h6_000C;
  localparam logic [19:0] A_MXU_TILE    = 20'h6_0010;
  localparam logic [19:0] A_VPU_OP      = 20'h6_0014;   // read-as-zero
  localparam logic [19:0] A_PERF_FREEZE = 20'h6_0018;

  // IRQ 0x9_0000
  localparam logic [19:0] A_IRQ_PENDING = 20'h9_0000;
  localparam logic [19:0] A_IRQ_MASK    = 20'h9_0004;
  localparam logic [19:0] A_IRQ_FORCE   = 20'h9_0008;
  localparam logic [19:0] A_IRQ_CAUSE   = 20'h9_0010;

  // Trace 0xA_0000
  localparam logic [19:0] A_TRACE_HEAD  = 20'hA_0000;
  localparam logic [19:0] A_TRACE_TAIL  = 20'hA_0004;
  localparam logic [19:0] A_TRACE_CTRL  = 20'hA_0010;
  localparam logic [19:0] A_TRACE_DROP  = 20'hA_0014;
  localparam logic [19:0] A_TRACE_WIN_BASE = 20'hA_0100;
  localparam logic [19:0] A_TRACE_WIN_END  = 20'hA_20FF;
  localparam logic [19:0] A_TRACE_META_BASE = 20'hA_3000;
  localparam logic [19:0] A_TRACE_META_END  = 20'hA_3FFC;

  // ===============================================================
  // Fixed values
  // ===============================================================
  localparam logic [31:0] G2_ID_VAL      = 32'h4732_0001;
  localparam logic [31:0] G2_VERSION_VAL = 32'h0001_0000;
  localparam logic [31:0] G2_CAP0_VAL    = 32'h0000_0060; // HAS_TRACE_RING + HAS_OOM_GUARD

  // ===============================================================
  // RW registers
  // ===============================================================
  // Descriptor staging
  integer si;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (si = 0; si < DESC_WORDS; si = si + 1)
        desc_stage[si] <= 32'd0;
    end else if (wr_en && addr >= A_DESC_STAGE_BASE && addr <= A_DESC_STAGE_END) begin
      automatic int idx = (32'(addr) - 32'(A_DESC_STAGE_BASE)) >> 2;
      if (idx < DESC_WORDS) desc_stage[idx] <= wr_data;
    end
  end

  // Trace control
  logic [31:0] trace_ctrl_r;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) trace_ctrl_r <= 32'd0;
    else if (wr_en && addr == A_TRACE_CTRL) trace_ctrl_r <= wr_data;
  end
  assign trace_enable     = trace_ctrl_r[0];
  assign trace_freeze     = trace_ctrl_r[1];
  assign trace_fatal_only = trace_ctrl_r[2];

  // TC0 CTRL register
  logic [31:0] tc0_ctrl_r;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) tc0_ctrl_r <= 32'h0000_0001; // ENABLE=1 by default
    else if (wr_en && addr == A_TC0_CTRL) tc0_ctrl_r <= wr_data;
  end
  assign tc0_enable    = tc0_ctrl_r[0];
  assign tc0_halt      = tc0_ctrl_r[1];
  assign tc0_fault_clr = wr_en && (addr == A_TC0_FAULT); // W1C: any write clears

  // Perf freeze
  logic [31:0] perf_freeze_r;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) perf_freeze_r <= 32'd0;
    else if (wr_en && addr == A_PERF_FREEZE) perf_freeze_r <= wr_data;
  end
  assign perf_freeze = perf_freeze_r[0];

  // Watchdog control (Proto-A stub: register only, no actual timer)
  logic [31:0] wdog_ctrl_r;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) wdog_ctrl_r <= 32'd0;
    else if (wr_en && addr == A_WDOG_CTRL) wdog_ctrl_r <= wr_data;
  end
  // Watchdog test inject: write bit[31]=1 fires wdog_reset pulse
  assign wdog_test_pulse = wr_en && (addr == A_WDOG_CTRL) && wr_data[31];

  // Boot vector (stored but not consumed in Proto-A)
  logic [31:0] boot_vec_lo_r, boot_vec_hi_r;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin boot_vec_lo_r <= 32'd0; boot_vec_hi_r <= 32'd0; end
    else begin
      if (wr_en && addr == A_BOOT_VEC_LO) boot_vec_lo_r <= wr_data;
      if (wr_en && addr == A_BOOT_VEC_HI) boot_vec_hi_r <= wr_data;
    end
  end

  // ===============================================================
  // Trace read window decode
  // ===============================================================
  logic trace_win_hit, trace_meta_hit;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [31:0] trace_entry_idx;
  /* verilator lint_on UNUSEDSIGNAL */

  always_comb begin
    trace_rd_addr = '0; trace_win_hit = 1'b0; trace_meta_hit = 1'b0; trace_entry_idx = 0;
    if (addr >= A_TRACE_WIN_BASE && addr <= A_TRACE_WIN_END) begin
      trace_win_hit = 1'b1;
      trace_entry_idx = (32'(addr) - 32'(A_TRACE_WIN_BASE)) >> 3;
      trace_rd_addr = trace_entry_idx[TRACE_ADDR_W-1:0];
    end else if (addr >= A_TRACE_META_BASE && addr <= A_TRACE_META_END) begin
      trace_meta_hit = 1'b1;
      trace_entry_idx = (32'(addr) - 32'(A_TRACE_META_BASE)) >> 2;
      trace_rd_addr = trace_entry_idx[TRACE_ADDR_W-1:0];
    end
  end

  // ===============================================================
  // Write strobes
  // ===============================================================
  always_comb begin
    sw_reset_pulse = 1'b0; sw_cause_clr = 1'b0;
    doorbell_pulse = '0; overflow_clr = '0;
    irq_pending_w1c_en = 1'b0; irq_pending_w1c_data = 32'd0;
    irq_mask_wr_en = 1'b0; irq_mask_wr_data = 32'd0;
    irq_force_wr_en = 1'b0; irq_force_wr_data = 32'd0;

    if (wr_en) begin
      case (addr)
        A_SW_RESET:    sw_reset_pulse = wr_data[0];
        A_BOOT_CAUSE:  sw_cause_clr = 1'b1;
        A_Q0_DOORBELL:     doorbell_pulse[0] = 1'b1;
        A_Q0_DOORBELL + 4: doorbell_pulse[1] = 1'b1;
        A_Q0_DOORBELL + 8: doorbell_pulse[2] = 1'b1;
        A_Q0_DOORBELL + 12: doorbell_pulse[3] = 1'b1;
        A_Q_OVERFLOW:  overflow_clr = wr_data[NUM_QUEUES-1:0];
        A_IRQ_PENDING: begin irq_pending_w1c_en = 1'b1; irq_pending_w1c_data = wr_data; end
        A_IRQ_MASK:    begin irq_mask_wr_en = 1'b1; irq_mask_wr_data = wr_data; end
        A_IRQ_FORCE:   begin irq_force_wr_en = 1'b1; irq_force_wr_data = wr_data; end
        default: ;
      endcase
    end
  end

  // ===============================================================
  // Read mux
  // ===============================================================
  always_comb begin
    rd_data = 32'd0;

    if (trace_win_hit) begin
      rd_data = addr[2] ? trace_rd_data[63:32] : trace_rd_data[31:0];
    end else if (trace_meta_hit) begin
      rd_data = {24'd0, trace_rd_type, 3'd0, trace_rd_fatal};
    end else begin
      case (addr)
        // Global
        A_G2_ID:       rd_data = G2_ID_VAL;
        A_G2_VERSION:  rd_data = G2_VERSION_VAL;
        A_G2_CAP0:     rd_data = G2_CAP0_VAL;
        A_G2_CAP1:     rd_data = 32'd0;
        A_BUILD_LO:    rd_data = 32'd0;
        A_BUILD_HI:    rd_data = 32'd0;

        // Reset/Boot
        A_BOOT_CAUSE:  rd_data = {28'd0, boot_cause};
        A_BOOT_VEC_LO: rd_data = boot_vec_lo_r;
        A_BOOT_VEC_HI: rd_data = boot_vec_hi_r;
        A_STRAP:       rd_data = 32'd0;  // read-as-zero in Proto-A
        A_WDOG_CTRL:   rd_data = wdog_ctrl_r;

        // Queue status
        A_Q0_STATUS:      rd_data = {q_tail[0], q_head[0]};
        A_Q0_STATUS + 4:  rd_data = {q_tail[1], q_head[1]};
        A_Q0_STATUS + 8:  rd_data = {q_tail[2], q_head[2]};
        A_Q0_STATUS + 12: rd_data = {q_tail[3], q_head[3]};
        A_Q_OVERFLOW:     rd_data = {{(32-NUM_QUEUES){1'b0}}, overflow_flags};

        // DMA (Proto-A shim)
        A_DMA_STATUS:  rd_data = dma_status;
        A_DMA_ERR:     rd_data = dma_err_code;

        // OOM
        A_OOM_USAGE_LO: rd_data = oom_usage_lo;
        A_OOM_RESV_LO:  rd_data = oom_reserved_lo;
        A_OOM_EFF_LO:   rd_data = oom_effective_lo;
        A_OOM_STATE:     rd_data = {22'd0, oom_prefetch_clamp, oom_admission_stop, 6'd0, oom_state};

        // TC0
        A_TC0_RUNSTATE: rd_data = tc0_runstate;
        A_TC0_CTRL:     rd_data = tc0_ctrl_r;
        A_TC0_DPTR_LO:  rd_data = tc0_desc_ptr[31:0];
        A_TC0_DPTR_HI:  rd_data = tc0_desc_ptr[63:32];
        A_TC0_PCYC_LO:  rd_data = tc0_perf_cycles[31:0];
        A_TC0_PCYC_HI:  rd_data = tc0_perf_cycles[63:32];
        A_TC0_FAULT:     rd_data = tc0_fault_status;

        // Perf
        A_MXU_BCYC_LO: rd_data = mxu_busy_cycles[31:0];
        A_MXU_BCYC_HI: rd_data = mxu_busy_cycles[63:32];
        A_VPU_BCYC_LO: rd_data = 32'd0;  // no VPU in Proto-A
        A_VPU_BCYC_HI: rd_data = 32'd0;
        A_MXU_TILE:     rd_data = mxu_tile_count;
        A_VPU_OP:       rd_data = 32'd0;
        A_PERF_FREEZE:  rd_data = perf_freeze_r;

        // IRQ
        A_IRQ_PENDING: rd_data = irq_pending;
        A_IRQ_MASK:    rd_data = irq_mask_rd;
        A_IRQ_CAUSE:   rd_data = irq_cause_last;

        // Trace
        A_TRACE_HEAD:  rd_data = {16'd0, trace_head};
        A_TRACE_TAIL:  rd_data = {16'd0, trace_tail};
        A_TRACE_CTRL:  rd_data = trace_ctrl_r;
        A_TRACE_DROP:  rd_data = trace_drop_count;

        default: begin
          if (addr >= A_DESC_STAGE_BASE && addr <= A_DESC_STAGE_END) begin
            automatic int ridx = (32'(addr) - 32'(A_DESC_STAGE_BASE)) >> 2;
            if (ridx < DESC_WORDS) rd_data = desc_stage[ridx];
          end
        end
      endcase
    end
  end

endmodule

`default_nettype wire
