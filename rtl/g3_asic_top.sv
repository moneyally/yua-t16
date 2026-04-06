// g3_asic_top.sv — ORBIT-G3 ASIC Top-Level Skeleton
// SSOT: ORBIT_G3_ARCHITECTURE.md, ORBIT_G3_RTL_PLAN.md, ORBIT_G3_REG_SPEC.md
//
// Top-level chip boundary: all module ownership, ports, and clock domains fixed.
// Real active paths: register, compute dispatch, fabric boundary.
// Stubs: HBM controller, NoC router.
//
// Clock domains:
//   core_clk   — compute, training, control logic
//   mem_clk    — HBM controller (stub)
//   pcie_clk   — host MMIO (future PCIe endpoint)
//   fabric_clk — chip-to-chip fabric (stub)
//
// Sections:
//   1. Control Plane     — g3_reg_top, g3_desc_fsm
//   2. Compute Cluster   — mxu_bf16_128x128
//   3. Training          — backward_engine, optimizer_unit, loss_scaler
//   4. Distributed       — collective_engine, scale_fabric_ctrl
//   5. Memory            — hbm_ctrl stub, noc placeholder
`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off PINCONNECTEMPTY */
/* verilator lint_off UNDRIVEN */

module g3_asic_top #(
  parameter int DIM = 16,
  parameter int MXU_DIM = 128,
  parameter int NUM_HBM_CH = 4
)(
  // ═══════════════════════════════════════════════════════════
  // Clocks and Resets
  // ═══════════════════════════════════════════════════════════
  input  logic        core_clk,
  input  logic        mem_clk,       // HBM domain (stub)
  input  logic        pcie_clk,      // Host MMIO domain (stub, future CDC)
  input  logic        fabric_clk,    // Chip-to-chip domain (stub, future CDC)
  input  logic        por_n,         // Power-on reset (async, active-low)

  // ═══════════════════════════════════════════════════════════
  // Section 1: Host / MMIO Interface
  // ═══════════════════════════════════════════════════════════
  // Proto-A/B style register bus. Future: PCIe BAR0 via pcie_clk + CDC.
  // For ASIC skeleton: directly on core_clk (CDC boundary marked).
  input  logic [19:0] reg_addr,
  input  logic        reg_wr_en,
  input  logic [31:0] reg_wr_data,
  output logic [31:0] reg_rd_data,
  output logic        irq_out,
  output logic        reset_active,

  // ═══════════════════════════════════════════════════════════
  // Section 5: Memory Interface (HBM stub)
  // ═══════════════════════════════════════════════════════════
  // Per-channel HBM request/response. Stub: tied off internally.
  // Future: connect to HBM PHY or FPGA DDR bridge.
  output logic [NUM_HBM_CH-1:0]  hbm_req_valid,
  input  logic [NUM_HBM_CH-1:0]  hbm_req_ready,
  output logic [63:0]            hbm_req_addr   [0:NUM_HBM_CH-1],
  output logic                   hbm_req_write  [0:NUM_HBM_CH-1],
  output logic [255:0]           hbm_req_wdata  [0:NUM_HBM_CH-1],
  input  logic [NUM_HBM_CH-1:0]  hbm_rsp_valid,
  input  logic [255:0]           hbm_rsp_rdata  [0:NUM_HBM_CH-1],

  // ═══════════════════════════════════════════════════════════
  // Section 4: Fabric Interface (chip-to-chip)
  // ═══════════════════════════════════════════════════════════
  output logic        fabric_tx_valid,
  input  logic        fabric_tx_ready,
  output logic [31:0] fabric_tx_data,
  output logic        fabric_tx_last,
  input  logic        fabric_rx_valid,
  output logic        fabric_rx_ready,
  input  logic [31:0] fabric_rx_data,
  input  logic        fabric_rx_last,

  // ═══════════════════════════════════════════════════════════
  // Debug / Status
  // ═══════════════════════════════════════════════════════════
  output logic [7:0]  chip_status     // boot/training/idle indicator
);

  // ═══════════════════════════════════════════════════════════
  // Internal reset (single domain for skeleton — future: per-domain)
  // ═══════════════════════════════════════════════════════════
  // TODO: multi-domain reset sequencer (G3-RTL-023)
  // TODO: CDC crossings between core_clk/mem_clk/pcie_clk/fabric_clk
  wire rst_n = por_n;  // simplified for skeleton
  assign reset_active = ~por_n;

  // ═══════════════════════════════════════════════════════════
  // SECTION 1: Control Plane
  // ═══════════════════════════════════════════════════════════
  // g3_reg_top: G2-compatible + G3 extension registers
  // g3_desc_fsm: descriptor decode/dispatch

  // --- g3_reg_top wires (selected subset for skeleton) ---
  logic        mxu_cmd_valid;
  logic [31:0] mxu_cmd, mxu_cfg0, mxu_cfg1;
  logic [63:0] mxu_act_addr, mxu_wgt_addr, mxu_out_addr;
  logic [31:0] mxu_status;

  logic        bkwd_cmd_valid;
  logic [31:0] bkwd_status;

  logic        opt_cmd_valid;
  logic [31:0] opt_status;

  logic        coll_cmd_valid;
  logic [31:0] coll_status;

  // g3_reg_top instantiation
  g3_reg_top u_reg (
    .clk(core_clk), .rst_n(rst_n),
    .addr(reg_addr), .wr_en(reg_wr_en), .wr_data(reg_wr_data), .rd_data(reg_rd_data),
    // G2 ports — stubbed for ASIC skeleton
    .boot_cause(4'b0001), .sw_reset_pulse(), .sw_cause_clr(), .wdog_test_pulse(),
    .desc_stage(), .doorbell_pulse(),
    .q_head('{default:16'd0}), .q_tail('{default:16'd0}),
    .overflow_flags(4'd0), .overflow_clr(),
    .oom_state(2'd0), .oom_admission_stop(1'b0), .oom_prefetch_clamp(1'b0),
    .oom_usage_lo(32'd0), .oom_reserved_lo(32'd0), .oom_effective_lo(32'd0),
    .tc0_runstate(32'd0), .tc0_fault_status(32'd0),
    .tc0_perf_cycles(64'd0), .tc0_desc_ptr(64'd0),
    .tc0_enable(), .tc0_halt(), .tc0_fault_clr(),
    .dma_status(32'd0), .dma_err_code(32'd0),
    .mxu_busy_cycles(64'd0), .mxu_tile_count(32'd0),
    .desc_done_count(32'd0), .perf_freeze(),
    .irq_pending(32'd0), .irq_mask_rd(32'hFFFFFFFF), .irq_cause_last(32'd0),
    .irq_pending_w1c_en(), .irq_pending_w1c_data(),
    .irq_mask_wr_en(), .irq_mask_wr_data(),
    .irq_force_wr_en(), .irq_force_wr_data(),
    .trace_head(16'd0), .trace_tail(16'd0), .trace_drop_count(32'd0),
    .trace_enable(), .trace_freeze(), .trace_fatal_only(),
    .trace_rd_addr(), .trace_rd_data(64'd0), .trace_rd_type(4'd0), .trace_rd_fatal(1'b0),
    // G3 MXU
    .mxu_cmd_valid(mxu_cmd_valid), .mxu_cmd(mxu_cmd),
    .mxu_cfg0(mxu_cfg0), .mxu_cfg1(mxu_cfg1),
    .mxu_act_addr(mxu_act_addr), .mxu_wgt_addr(mxu_wgt_addr), .mxu_out_addr(mxu_out_addr),
    .mxu_status(mxu_status), .mxu_err_code(32'd0), .mxu_tile_cnt_g3(32'd0),
    // G3 Backward
    .bkwd_cmd_valid(bkwd_cmd_valid), .bkwd_cmd(),
    .bkwd_act_addr(), .bkwd_grad_addr(), .bkwd_loss_scale(),
    .bkwd_status(bkwd_status), .bkwd_err_code(32'd0),
    // G3 Optimizer
    .opt_cmd_valid(opt_cmd_valid), .opt_cmd(),
    .opt_lr(), .opt_beta1(), .opt_beta2(), .opt_epsilon(),
    .opt_param_addr(), .opt_state_addr(),
    .opt_status(opt_status), .opt_err_code(32'd0),
    // G3 Collective
    .coll_cmd_valid(coll_cmd_valid), .coll_cmd(),
    .coll_peer_mask(), .coll_buffer_addr(), .coll_buffer_size(),
    .coll_result_addr(), .coll_timeout(),
    .coll_status(coll_status), .coll_err_code(32'd0), .coll_bytes(64'd0),
    // G3 Fabric
    .fabric_link_status(32'd0), .fabric_err_code(32'd0),
    .fabric_peer_id_cfg(), .fabric_topology_cfg(),
    .fabric_routing_sel(), .fabric_routing_data()
  );

  // ═══════════════════════════════════════════════════════════
  // SECTION 2: Compute Cluster
  // ═══════════════════════════════════════════════════════════
  // MXU 128×128 (future: multiple instances for tensor parallelism)
  // Placeholder: not connected to full datapath in skeleton.
  // Ownership: receives dispatch from g3_desc_fsm, reads/writes via NoC/HBM.
  assign mxu_status = 32'd0;  // idle stub

  // ═══════════════════════════════════════════════════════════
  // SECTION 3: Training
  // ═══════════════════════════════════════════════════════════
  // backward_engine, optimizer_unit, loss_scaler
  // Ownership: sequential pipeline after forward MXU.
  // Not instantiated individually in skeleton — present via integration tops.
  assign bkwd_status = 32'd0;
  assign opt_status  = 32'd0;

  // ═══════════════════════════════════════════════════════════
  // SECTION 4: Distributed / Fabric
  // ═══════════════════════════════════════════════════════════
  // scale_fabric_ctrl: connects to external fabric_tx/rx ports
  logic fc_send_start, fc_recv_start, fc_done, fc_busy;
  logic [7:0] fc_err;

  scale_fabric_ctrl u_fabric (
    .clk(core_clk), .rst_n(rst_n),
    .send_start(fc_send_start), .recv_start(fc_recv_start),
    .busy(fc_busy), .done_pulse(fc_done), .err_code(fc_err),
    .local_tx_valid(1'b0), .local_tx_ready(), .local_tx_data(32'd0), .local_tx_last(1'b0),
    .peer_rx_valid(), .peer_rx_ready(1'b1), .peer_rx_data(), .peer_rx_last(),
    .fabric_tx_valid(fabric_tx_valid), .fabric_tx_ready(fabric_tx_ready),
    .fabric_tx_data(fabric_tx_data), .fabric_tx_last(fabric_tx_last),
    .fabric_rx_valid(fabric_rx_valid), .fabric_rx_ready(fabric_rx_ready),
    .fabric_rx_data(fabric_rx_data), .fabric_rx_last(fabric_rx_last)
  );

  assign fc_send_start = 1'b0;  // controlled by future training FSM
  assign fc_recv_start = 1'b0;

  // collective_engine: local + peer → reduced (future integration)
  assign coll_status = 32'd0;

  // ═══════════════════════════════════════════════════════════
  // SECTION 5: Memory (HBM stub)
  // ═══════════════════════════════════════════════════════════
  // HBM channels: stub — no requests issued, responses ignored.
  // Future: hbm_ctrl.sv with channel arbiter, ECC, temp throttle.
  genvar ch;
  generate
    for (ch = 0; ch < NUM_HBM_CH; ch++) begin : HBM_STUB
      assign hbm_req_valid[ch] = 1'b0;
      assign hbm_req_addr[ch]  = 64'd0;
      assign hbm_req_write[ch] = 1'b0;
      assign hbm_req_wdata[ch] = 256'd0;
    end
  endgenerate

  // ═══════════════════════════════════════════════════════════
  // IRQ / Status
  // ═══════════════════════════════════════════════════════════
  assign irq_out = 1'b0;  // future: from irq_ctrl
  assign chip_status = {4'd0, fc_busy, 1'b0, ~por_n, por_n};  // [0]=alive, [1]=reset

  // ═══════════════════════════════════════════════════════════
  // TODO markers for future integration
  // ═══════════════════════════════════════════════════════════
  // TODO: G3-RTL-022 noc_router — on-chip mesh interconnect
  // TODO: G3-RTL-023 multi_chip_reset — multi-domain reset sequencer
  // TODO: G3-RTL-033 hbm_ctrl_real — HBM3e controller with channel arbiter
  // TODO: G3-RTL-032 xla_desc_decoder — XLA HLO → descriptor HW assist
  // TODO: CDC crossings: core_clk ↔ pcie_clk, core_clk ↔ mem_clk, core_clk ↔ fabric_clk

endmodule

`default_nettype wire
