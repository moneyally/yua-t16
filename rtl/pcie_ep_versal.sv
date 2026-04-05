// pcie_ep_versal.sv — PCIe Endpoint Adapter for ORBIT-G2 Proto-B
// SSOT: ORBIT_G2_VCK190_PCIE_BRINGUP.md, PG343, PG344
//
// Target: VCK190 / VC1902 CPM (hardened PCIe Gen4 x8 block).
// NOT PL PCIE — CPM is the correct IP for VCK190.
//
// This adapter sits between the CPM AXI-Stream TLP interface and
// the internal BAR req/rsp ports used by g2_protob_top.
//
// CPM side (upstream, stub ports for now):
//   CQ: Completer Request  (host write/read TLPs arrive here)
//   CC: Completer Completion (device sends read responses here)
//   RQ: Requester Request  (device DMA read/write → host memory)
//   RC: Requester Completion (host DMA read response arrives here)
//
// Internal side (downstream, to g2_protob_top):
//   BAR0/BAR2/BAR4 simple req/rsp interfaces
//   MSI-X request/ack
//   Link/reset status
//
// For Proto-B skeleton: CPM AXI-Stream ports are stubs.
// The actual CQ→BAR decode + CC generation will be implemented
// when Vivado CPM IP is generated and ports are finalized.
`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off UNDRIVEN */
/* verilator lint_off UNUSEDPARAM */

module pcie_ep_versal #(
  parameter int BAR0_SIZE = 1048576,  // 1 MiB
  parameter int BAR2_SIZE = 2097152,  // 2 MiB
  parameter int BAR4_SIZE = 65536,    // 64 KiB
  parameter int AXI_DATA_W = 256     // CQ/CC/RQ/RC AXI-Stream width
)(
  // ═══════════════════════════════════════════════════════════
  // CPM-facing ports (upstream — from Versal CPM hard IP)
  // ═══════════════════════════════════════════════════════════

  // Clock / Reset from CPM
  input  logic        user_clk,       // 250 MHz (Gen4 x8)
  input  logic        user_reset,     // active-high, sync to user_clk

  // Link status from CPM
  input  logic        user_lnk_up,    // link trained
  input  logic        cfg_phy_link_down,

  // FLR from CPM
  input  logic        cfg_flr_active, // Function Level Reset
  output logic        cfg_flr_done,   // FLR acknowledgement

  // CQ: Completer Request (host → device TLPs)
  input  logic [AXI_DATA_W-1:0]  m_axis_cq_tdata,
  input  logic                    m_axis_cq_tvalid,
  output logic                    m_axis_cq_tready,
  input  logic                    m_axis_cq_tlast,
  input  logic [7:0]             m_axis_cq_tkeep,
  input  logic [87:0]            m_axis_cq_tuser,  // sideband (BAR ID, etc.)

  // CC: Completer Completion (device → host read responses)
  output logic [AXI_DATA_W-1:0]  s_axis_cc_tdata,
  output logic                    s_axis_cc_tvalid,
  input  logic                    s_axis_cc_tready,
  output logic                    s_axis_cc_tlast,
  output logic [7:0]             s_axis_cc_tkeep,

  // RQ: Requester Request (device DMA → host memory)
  output logic [AXI_DATA_W-1:0]  s_axis_rq_tdata,
  output logic                    s_axis_rq_tvalid,
  input  logic                    s_axis_rq_tready,
  output logic                    s_axis_rq_tlast,
  output logic [7:0]             s_axis_rq_tkeep,

  // RC: Requester Completion (host → device DMA responses)
  input  logic [AXI_DATA_W-1:0]  m_axis_rc_tdata,
  input  logic                    m_axis_rc_tvalid,
  output logic                    m_axis_rc_tready,
  input  logic                    m_axis_rc_tlast,

  // MSI-X from CPM
  output logic        cfg_interrupt_msix_int,
  output logic [31:0] cfg_interrupt_msix_address,
  output logic [31:0] cfg_interrupt_msix_data,
  input  logic        cfg_interrupt_msix_sent,
  input  logic        cfg_interrupt_msix_fail,

  // ═══════════════════════════════════════════════════════════
  // Internal-facing ports (downstream — to g2_protob_top)
  // ═══════════════════════════════════════════════════════════

  // Status
  output logic        link_up,
  output logic        user_reset_out,
  output logic        flr_active,

  // BAR0: Register access (1 MiB, UC)
  output logic        bar0_req_valid,
  input  logic        bar0_req_ready,
  output logic [19:0] bar0_req_addr,  // 20-bit: covers 1 MiB
  output logic        bar0_req_wr,
  output logic [31:0] bar0_req_wdata,
  output logic [3:0]  bar0_req_be,
  input  logic        bar0_rsp_valid,
  input  logic [31:0] bar0_rsp_rdata,

  // BAR2: CMEM aperture (2 MiB, WC)
  output logic        bar2_req_valid,
  input  logic        bar2_req_ready,
  output logic [20:0] bar2_req_addr,
  output logic        bar2_req_wr,
  output logic [31:0] bar2_req_wdata,
  output logic [3:0]  bar2_req_be,
  input  logic        bar2_rsp_valid,
  input  logic [31:0] bar2_rsp_rdata,

  // BAR4: DMA engine (64 KiB, UC)
  output logic        bar4_req_valid,
  input  logic        bar4_req_ready,
  output logic [15:0] bar4_req_addr,
  output logic        bar4_req_wr,
  output logic [31:0] bar4_req_wdata,
  output logic [3:0]  bar4_req_be,
  input  logic        bar4_rsp_valid,
  input  logic [31:0] bar4_rsp_rdata,

  // MSI-X request (from irq_ctrl)
  input  logic        msix_req,
  input  logic [4:0]  msix_vector,
  output logic        msix_ack
);

  // ═══════════════════════════════════════════════════════════
  // Status passthrough
  // ═══════════════════════════════════════════════════════════
  assign link_up        = user_lnk_up;
  assign user_reset_out = user_reset;
  assign flr_active     = cfg_flr_active;

  // FLR ack: 1-cycle delay
  logic flr_d;
  always_ff @(posedge user_clk) begin
    if (user_reset) flr_d <= 1'b0;
    else            flr_d <= cfg_flr_active;
  end
  assign cfg_flr_done = cfg_flr_active & ~flr_d;

  // ═══════════════════════════════════════════════════════════
  // CQ → BAR decode (STUB for skeleton)
  // ═══════════════════════════════════════════════════════════
  // In full implementation:
  //   1. Parse CQ TLP header (address, BAR ID, req type, dword count)
  //   2. Route to BAR0/BAR2/BAR4 based on BAR ID field in tuser
  //   3. For reads: capture tag+requester ID, issue CC after rsp_valid
  //   4. For writes: forward data to BAR write interface
  //
  // Skeleton: CQ not connected (testbench/direct BAR access for now)
  assign m_axis_cq_tready = 1'b1;  // always accept (sink)

  // CC: no completions generated in skeleton
  assign s_axis_cc_tdata  = '0;
  assign s_axis_cc_tvalid = 1'b0;
  assign s_axis_cc_tlast  = 1'b0;
  assign s_axis_cc_tkeep  = '0;

  // RQ: no DMA requests in skeleton
  assign s_axis_rq_tdata  = '0;
  assign s_axis_rq_tvalid = 1'b0;
  assign s_axis_rq_tlast  = 1'b0;
  assign s_axis_rq_tkeep  = '0;

  // RC: accept and discard
  assign m_axis_rc_tready = 1'b1;

  // ═══════════════════════════════════════════════════════════
  // MSI-X generation (simplified)
  // ═══════════════════════════════════════════════════════════
  // Full implementation: read MSI-X table entry for vector,
  // write address+data to cfg_interrupt_msix ports.
  // Skeleton: simple ack
  logic msix_req_d;
  always_ff @(posedge user_clk) begin
    if (user_reset) msix_req_d <= 1'b0;
    else            msix_req_d <= msix_req;
  end
  assign msix_ack = msix_req & ~msix_req_d;

  // Stub MSI-X CPM interface
  assign cfg_interrupt_msix_int     = msix_req & ~msix_req_d;
  assign cfg_interrupt_msix_address = 32'd0;  // from MSI-X table (stub)
  assign cfg_interrupt_msix_data    = {27'd0, msix_vector};

endmodule

`default_nettype wire
