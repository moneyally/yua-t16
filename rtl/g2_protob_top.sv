// g2_protob_top.sv — ORBIT-G2 Proto-B Top Shell
// SSOT: ORBIT_G2_PCIE_BAR_SPEC.md, ORBIT_G2_DETAIL_BLOCKDIAG.md section 3
//
// Wraps:
//   pcie_ep_versal  — PCIe endpoint BAR interface
//   dma_bridge      — DMA submit/status state machine
//   g2_ctrl_top     — Proto-A control plane (reused)
//
// Wiring:
//   BAR0 → reg_top (via g2_ctrl_top reg bus)
//   BAR4 → dma_bridge register interface
//   dma_bridge internal → g2_ctrl_top DMA ports
//   irq_ctrl → pcie_ep_versal MSI-X
//   FLR → reset_seq
//   BAR2 → CMEM placeholder (address decoded but data path stub)
`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off PINCONNECTEMPTY */

module g2_protob_top #(
  parameter int MAX_KT     = 256,
  parameter int QUEUE_DEPTH = 16,
  parameter int DESC_BYTES  = 64
)(
  // ── PCIe / Board ───────────────────────────────────────────
  input  logic        pcie_clk,
  input  logic        pcie_rst_n,

  // Vendor PCIe hard IP stubs (directly driven by TB for now)
  input  logic        cfg_link_up,
  input  logic        cfg_flr_active,

  // ── Board-level ────────────────────────────────────────────
  output logic        link_up,
  output logic        irq_out
);

  // ═══════════════════════════════════════════════════════════
  // PCIe Endpoint
  // ═══════════════════════════════════════════════════════════
  logic        ep_user_reset;
  logic        ep_flr;

  // BAR0 (1 MiB → 20-bit addr)
  logic        bar0_req_valid, bar0_req_ready;
  logic [19:0] bar0_req_addr;
  logic        bar0_req_wr;
  logic [31:0] bar0_req_wdata;
  logic [3:0]  bar0_req_be;
  logic        bar0_rsp_valid;
  logic [31:0] bar0_rsp_rdata;

  // BAR2 (stub)
  logic        bar2_req_valid, bar2_req_ready;
  logic [20:0] bar2_req_addr;
  logic        bar2_req_wr;
  logic [31:0] bar2_req_wdata;
  logic [3:0]  bar2_req_be;
  logic        bar2_rsp_valid;
  logic [31:0] bar2_rsp_rdata;

  // BAR4
  logic        bar4_req_valid, bar4_req_ready;
  logic [15:0] bar4_req_addr;
  logic        bar4_req_wr;
  logic [31:0] bar4_req_wdata;
  logic [3:0]  bar4_req_be;
  logic        bar4_rsp_valid;
  logic [31:0] bar4_rsp_rdata;

  // MSI-X
  logic        msix_req, msix_ack;
  logic [4:0]  msix_vector;

  pcie_ep_versal u_pcie_ep (
    // CPM clock/reset
    .user_clk(pcie_clk), .user_reset(~pcie_rst_n),
    .user_lnk_up(cfg_link_up), .cfg_phy_link_down(1'b0),
    .cfg_flr_active(cfg_flr_active), .cfg_flr_done(),
    // CQ/CC/RQ/RC stubs (tied off — skeleton)
    .m_axis_cq_tdata('0), .m_axis_cq_tvalid(1'b0), .m_axis_cq_tready(),
    .m_axis_cq_tlast(1'b0), .m_axis_cq_tkeep('0), .m_axis_cq_tuser('0),
    .s_axis_cc_tdata(), .s_axis_cc_tvalid(), .s_axis_cc_tready(1'b1),
    .s_axis_cc_tlast(), .s_axis_cc_tkeep(),
    .s_axis_rq_tdata(), .s_axis_rq_tvalid(), .s_axis_rq_tready(1'b1),
    .s_axis_rq_tlast(), .s_axis_rq_tkeep(),
    .m_axis_rc_tdata('0), .m_axis_rc_tvalid(1'b0), .m_axis_rc_tready(),
    .m_axis_rc_tlast(1'b0),
    // MSI-X
    .cfg_interrupt_msix_int(), .cfg_interrupt_msix_address(),
    .cfg_interrupt_msix_data(), .cfg_interrupt_msix_sent(1'b0),
    .cfg_interrupt_msix_fail(1'b0),
    // Internal BAR interfaces
    .link_up(link_up), .user_reset_out(ep_user_reset), .flr_active(ep_flr),
    .bar0_req_valid(bar0_req_valid), .bar0_req_ready(bar0_req_ready),
    .bar0_req_addr(bar0_req_addr), .bar0_req_wr(bar0_req_wr),
    .bar0_req_wdata(bar0_req_wdata), .bar0_req_be(bar0_req_be),
    .bar0_rsp_valid(bar0_rsp_valid), .bar0_rsp_rdata(bar0_rsp_rdata),
    .bar2_req_valid(bar2_req_valid), .bar2_req_ready(bar2_req_ready),
    .bar2_req_addr(bar2_req_addr), .bar2_req_wr(bar2_req_wr),
    .bar2_req_wdata(bar2_req_wdata), .bar2_req_be(bar2_req_be),
    .bar2_rsp_valid(bar2_rsp_valid), .bar2_rsp_rdata(bar2_rsp_rdata),
    .bar4_req_valid(bar4_req_valid), .bar4_req_ready(bar4_req_ready),
    .bar4_req_addr(bar4_req_addr), .bar4_req_wr(bar4_req_wr),
    .bar4_req_wdata(bar4_req_wdata), .bar4_req_be(bar4_req_be),
    .bar4_rsp_valid(bar4_rsp_valid), .bar4_rsp_rdata(bar4_rsp_rdata),
    .msix_req(msix_req), .msix_vector(msix_vector), .msix_ack(msix_ack)
  );

  // ═══════════════════════════════════════════════════════════
  // BAR0 → g2_ctrl_top register bus bridge
  // ═══════════════════════════════════════════════════════════
  // BAR0 req/rsp ↔ g2_ctrl_top reg_addr/wr_en/wr_data/rd_data
  // BAR0 addr is 17-bit (128 KiB). reg_top expects 20-bit offset.
  // For BAR0: offset[19:0] = {3'b0, bar0_addr[16:0]}
  logic [19:0] ctrl_reg_addr;
  logic        ctrl_reg_wr_en;
  logic [31:0] ctrl_reg_wr_data;
  logic [31:0] ctrl_reg_rd_data;

  assign ctrl_reg_addr    = bar0_req_addr;  // 20-bit direct (1 MiB = reg_top full range)
  assign ctrl_reg_wr_en   = bar0_req_valid & bar0_req_wr;
  assign ctrl_reg_wr_data = bar0_req_wdata;
  assign bar0_req_ready   = 1'b1;  // single-cycle, always ready

  // Read response (1-cycle latency)
  always_ff @(posedge pcie_clk or negedge pcie_rst_n) begin
    if (!pcie_rst_n) begin
      bar0_rsp_valid <= 1'b0;
      bar0_rsp_rdata <= 32'd0;
    end else begin
      bar0_rsp_valid <= bar0_req_valid & ~bar0_req_wr;
      bar0_rsp_rdata <= ctrl_reg_rd_data;
    end
  end

  // ═══════════════════════════════════════════════════════════
  // BAR2 → CMEM placeholder (stub: reads zero)
  // ═══════════════════════════════════════════════════════════
  assign bar2_req_ready = 1'b1;
  assign bar2_rsp_valid = 1'b0;  // no response yet
  assign bar2_rsp_rdata = 32'd0;

  // ═══════════════════════════════════════════════════════════
  // BAR4 → DMA Bridge
  // ═══════════════════════════════════════════════════════════
  logic [31:0] dma_reg_rdata;
  logic        dma_reg_rvalid;

  // Internal DMA request (from dma_bridge to memory subsystem)
  logic        dma_int_req_valid, dma_int_req_ready;
  logic [63:0] dma_int_req_addr;
  logic [31:0] dma_int_req_len;
  logic        dma_int_req_dir;
  logic [1:0]  dma_int_req_queue;
  logic [3:0]  dma_int_req_qos;

  // Completion feedback (from memory subsystem / gemm_core path)
  logic        dma_int_done, dma_int_err, dma_int_timeout;
  logic [7:0]  dma_int_err_code;

  logic        irq_dma_done, irq_dma_error;

  dma_bridge u_dma_bridge (
    .clk(pcie_clk), .rst_n(pcie_rst_n & ~ep_user_reset),
    .reg_valid(bar4_req_valid), .reg_addr(bar4_req_addr),
    .reg_wr(bar4_req_wr), .reg_wdata(bar4_req_wdata),
    .reg_rdata(dma_reg_rdata), .reg_rvalid(dma_reg_rvalid),
    .dma_req_valid(dma_int_req_valid), .dma_req_ready(dma_int_req_ready),
    .dma_req_addr(dma_int_req_addr), .dma_req_len(dma_int_req_len),
    .dma_req_dir(dma_int_req_dir), .dma_req_queue(dma_int_req_queue),
    .dma_req_qos(dma_int_req_qos),
    .dma_done(dma_int_done), .dma_err(dma_int_err),
    .dma_err_code_in(dma_int_err_code), .dma_timeout_in(dma_int_timeout),
    .irq_dma_done(irq_dma_done), .irq_dma_error(irq_dma_error)
  );

  assign bar4_req_ready = 1'b1;
  assign bar4_rsp_valid = dma_reg_rvalid;
  assign bar4_rsp_rdata = dma_reg_rdata;

  // ═══════════════════════════════════════════════════════════
  // g2_ctrl_top (Proto-A control plane, reused)
  // ═══════════════════════════════════════════════════════════
  // POR comes from PCIe reset + FLR
  wire ctrl_por_n = pcie_rst_n & ~ep_user_reset & ~ep_flr;

  logic         ctrl_rd_req_valid, ctrl_rd_req_ready;
  logic [63:0]  ctrl_rd_req_addr;
  logic [15:0]  ctrl_rd_req_len;
  logic         ctrl_rd_done;
  logic         ctrl_rd_data_valid;
  logic [127:0] ctrl_rd_data;
  logic         ctrl_rd_data_ready;
  logic         ctrl_rd_data_last;
  logic         ctrl_wr_req_valid, ctrl_wr_req_ready;
  logic [63:0]  ctrl_wr_req_addr;
  logic [15:0]  ctrl_wr_req_len;
  logic         ctrl_wr_done;
  logic         ctrl_wr_data_valid, ctrl_wr_data_ready;
  logic [127:0] ctrl_wr_data;
  logic         ctrl_wr_data_last;
  logic         ctrl_irq_out;
  logic         ctrl_reset_active;

  g2_ctrl_top #(.MAX_KT(MAX_KT), .QUEUE_DEPTH(QUEUE_DEPTH), .DESC_BYTES(DESC_BYTES)) u_ctrl (
    .clk(pcie_clk), .por_n(ctrl_por_n),
    .reg_addr(ctrl_reg_addr), .reg_wr_en(ctrl_reg_wr_en),
    .reg_wr_data(ctrl_reg_wr_data), .reg_rd_data(ctrl_reg_rd_data),
    .rd_req_valid(ctrl_rd_req_valid), .rd_req_ready(ctrl_rd_req_ready),
    .rd_req_addr(ctrl_rd_req_addr), .rd_req_len_bytes(ctrl_rd_req_len),
    .rd_done(ctrl_rd_done),
    .rd_data_valid(ctrl_rd_data_valid), .rd_data(ctrl_rd_data),
    .rd_data_ready(ctrl_rd_data_ready), .rd_data_last(ctrl_rd_data_last),
    .wr_req_valid(ctrl_wr_req_valid), .wr_req_ready(ctrl_wr_req_ready),
    .wr_req_addr(ctrl_wr_req_addr), .wr_req_len_bytes(ctrl_wr_req_len),
    .wr_done(ctrl_wr_done),
    .wr_data_valid(ctrl_wr_data_valid), .wr_data_ready(ctrl_wr_data_ready),
    .wr_data(ctrl_wr_data), .wr_data_last(ctrl_wr_data_last),
    .irq_out(ctrl_irq_out), .reset_active(ctrl_reset_active)
  );

  // ═══════════════════════════════════════════════════════════
  // DMA internal ↔ g2_ctrl_top DMA ports
  // ═══════════════════════════════════════════════════════════
  // Proto-B: dma_bridge issues requests that feed into g2_ctrl_top's
  // gemm_core DMA path. For now, g2_ctrl_top's DMA ports are driven
  // by the descriptor path (gemm_top). The dma_bridge is a separate
  // "host-initiated DMA" path. These will merge in the full DRAM scheduler.
  //
  // For skeleton: g2_ctrl_top DMA ports are tied to external (TB/memory)
  // and dma_bridge completion is driven by external feedback.
  assign ctrl_rd_req_ready = 1'b1;  // always accept (TB provides data)
  assign ctrl_rd_done      = 1'b0;
  assign ctrl_rd_data_valid = 1'b0;
  assign ctrl_rd_data       = 128'd0;
  assign ctrl_rd_data_last  = 1'b0;
  assign ctrl_wr_req_ready  = 1'b1;
  assign ctrl_wr_done       = 1'b0;
  assign ctrl_wr_data_ready = 1'b1;

  // DMA bridge internal: not connected to memory subsystem yet
  // (completion comes from TB or future DRAM scheduler)
  assign dma_int_req_ready = 1'b1;  // instant accept (skeleton)
  assign dma_int_done      = 1'b0;  // driven by TB
  assign dma_int_err       = 1'b0;
  assign dma_int_err_code  = 8'd0;
  assign dma_int_timeout   = 1'b0;

  // ═══════════════════════════════════════════════════════════
  // IRQ → MSI-X
  // ═══════════════════════════════════════════════════════════
  assign irq_out    = ctrl_irq_out;
  assign msix_req   = ctrl_irq_out;  // simplified: any IRQ → MSI-X
  assign msix_vector = 5'd0;          // single vector for now

endmodule

`default_nettype wire
