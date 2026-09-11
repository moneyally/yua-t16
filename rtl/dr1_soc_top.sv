// ============================================================================
// dr1_soc_top.sv — 보드용 최상위 (AXI4-Lite ↔ g2_ctrl_top)
//
// PLAN W11: "`dr1_top` + AXI-Lite 래퍼 (기존 reg_top 재사용)".
//
// ----------------------------------------------------------------------------
// 구조
// ----------------------------------------------------------------------------
//   PS / AXI 마스터
//        │  AXI4-Lite (32비트)
//        ▼
//   axil_reg_bridge ──► g2_ctrl_top (reg_top + desc_queue + desc_fsm_v2
//                                    + dr1_top + dr1_scratch + trace + irq)
//
// **시뮬레이션에서 쓰는 경로와 같은 레지스터 구현을 본다.** cocotb 는 reg_addr/
// reg_wr_en 을 직접 흔들고, 보드는 이 브리지를 지난다 — 그 아래는 동일하다.
//
// ----------------------------------------------------------------------------
// DMA 포트 (rd_*/wr_*) → AXI4 마스터
// ----------------------------------------------------------------------------
// 전에는 묶어 뒀다 (요청 ready=1, 응답 0 → GEMM 디스크립터는 타임아웃 fault).
// 지금은 `axi4_master_adapter` 가 받는다 — `CLAUDE.md` 2절 "외부 메모리(DDR/HBM)
// 경로 없음" 을 닫는다. DR1 경로 자체는 여전히 온칩 스크래치만 쓴다
// (spec/deltarule.md 3.6절); 이 포트는 GEMM 경로와 앞으로의 상태 스필용이다.
//
// **미검증:** 이 AXI4 마스터는 시뮬레이션(tb/tb_axi4_master_adapter.py, 8/8)에서만
// 돌았다. 보드의 PS DDR 에 붙여 본 적은 없다 (docs/FPGA.md).
// ============================================================================
`timescale 1ns/1ps
`default_nettype none

module dr1_soc_top #(
  parameter int AW = 20
)(
  input  wire         clk,
  input  wire         resetn,          // AXI 규약: active-low

  // ── AXI4-Lite 슬레이브 ──
  input  wire [AW-1:0] s_axil_awaddr,
  input  wire          s_axil_awvalid,
  output wire          s_axil_awready,
  input  wire [31:0]   s_axil_wdata,
  input  wire [3:0]    s_axil_wstrb,
  input  wire          s_axil_wvalid,
  output wire          s_axil_wready,
  output wire [1:0]    s_axil_bresp,
  output wire          s_axil_bvalid,
  input  wire          s_axil_bready,
  input  wire [AW-1:0] s_axil_araddr,
  input  wire          s_axil_arvalid,
  output wire          s_axil_arready,
  output wire [31:0]   s_axil_rdata,
  output wire [1:0]    s_axil_rresp,
  output wire          s_axil_rvalid,
  input  wire          s_axil_rready,

  // ── AXI4 마스터 (PS DDR) ──
  output wire [3:0]    m_axi_arid,
  output wire [63:0]   m_axi_araddr,
  output wire [7:0]    m_axi_arlen,
  output wire [2:0]    m_axi_arsize,
  output wire [1:0]    m_axi_arburst,
  output wire          m_axi_arvalid,
  input  wire          m_axi_arready,
  input  wire [3:0]    m_axi_rid,
  input  wire [127:0]  m_axi_rdata,
  input  wire [1:0]    m_axi_rresp,
  input  wire          m_axi_rlast,
  input  wire          m_axi_rvalid,
  output wire          m_axi_rready,
  output wire [3:0]    m_axi_awid,
  output wire [63:0]   m_axi_awaddr,
  output wire [7:0]    m_axi_awlen,
  output wire [2:0]    m_axi_awsize,
  output wire [1:0]    m_axi_awburst,
  output wire          m_axi_awvalid,
  input  wire          m_axi_awready,
  output wire [127:0]  m_axi_wdata,
  output wire [15:0]   m_axi_wstrb,
  output wire          m_axi_wlast,
  output wire          m_axi_wvalid,
  input  wire          m_axi_wready,
  input  wire [3:0]    m_axi_bid,
  input  wire [1:0]    m_axi_bresp,
  input  wire          m_axi_bvalid,
  output wire          m_axi_bready,

  // ── 보드 신호 ──
  output wire          irq_out,
  output wire          reset_active,
  // AXI 응답 오류(RRESP/BRESP != OKAY) 1사이클 펄스. **삼키지 않는다** — 보드에서
  // LED/ILA 로 볼 수 있게 밖으로 낸다. 레지스터로 잡는 것은 별도 작업이다.
  output wire          mem_err
);

  wire [AW-1:0] reg_addr;
  wire          reg_wr_en;
  wire [31:0]   reg_wr_data;
  wire [31:0]   reg_rd_data;

  axil_reg_bridge #(.AW(AW), .RD_LAT(2)) u_axil (
    .clk(clk), .rst_n(resetn),
    .s_axil_awaddr(s_axil_awaddr), .s_axil_awvalid(s_axil_awvalid),
    .s_axil_awready(s_axil_awready),
    .s_axil_wdata(s_axil_wdata), .s_axil_wstrb(s_axil_wstrb),
    .s_axil_wvalid(s_axil_wvalid), .s_axil_wready(s_axil_wready),
    .s_axil_bresp(s_axil_bresp), .s_axil_bvalid(s_axil_bvalid),
    .s_axil_bready(s_axil_bready),
    .s_axil_araddr(s_axil_araddr), .s_axil_arvalid(s_axil_arvalid),
    .s_axil_arready(s_axil_arready),
    .s_axil_rdata(s_axil_rdata), .s_axil_rresp(s_axil_rresp),
    .s_axil_rvalid(s_axil_rvalid), .s_axil_rready(s_axil_rready),
    .reg_addr(reg_addr), .reg_wr_en(reg_wr_en),
    .reg_wr_data(reg_wr_data), .reg_rd_data(reg_rd_data)
  );

  // ── g2_ctrl_top ↔ axi4_master_adapter 사이 내부 요청 채널 ──
  wire          rd_req_valid, rd_req_ready;
  wire [63:0]   rd_req_addr;
  wire [15:0]   rd_req_len_bytes;
  wire          rd_done, rd_data_valid, rd_data_ready, rd_data_last;
  wire [127:0]  rd_data;

  wire          wr_req_valid, wr_req_ready;
  wire [63:0]   wr_req_addr;
  wire [15:0]   wr_req_len_bytes;
  wire          wr_done, wr_data_valid, wr_data_ready, wr_data_last;
  wire [127:0]  wr_data;

  g2_ctrl_top u_core (
    .clk(clk), .por_n(resetn),
    .reg_addr(reg_addr), .reg_wr_en(reg_wr_en),
    .reg_wr_data(reg_wr_data), .reg_rd_data(reg_rd_data),

    .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready),
    .rd_req_addr(rd_req_addr), .rd_req_len_bytes(rd_req_len_bytes),
    .rd_done(rd_done), .rd_data_valid(rd_data_valid), .rd_data(rd_data),
    .rd_data_ready(rd_data_ready), .rd_data_last(rd_data_last),

    .wr_req_valid(wr_req_valid), .wr_req_ready(wr_req_ready),
    .wr_req_addr(wr_req_addr), .wr_req_len_bytes(wr_req_len_bytes),
    .wr_done(wr_done), .wr_data_valid(wr_data_valid),
    .wr_data_ready(wr_data_ready),
    .wr_data(wr_data), .wr_data_last(wr_data_last),

    .irq_out(irq_out), .reset_active(reset_active)
  );

  axi4_master_adapter #(
    .DATA_W(128), .ADDR_W(64), .ID_W(4), .MAX_BURST(256)
  ) u_axi_m (
    .clk(clk), .rst_n(resetn),

    .rd_req_valid(rd_req_valid), .rd_req_ready(rd_req_ready),
    .rd_req_addr(rd_req_addr), .rd_req_len_bytes(rd_req_len_bytes),
    .rd_done(rd_done), .rd_data_valid(rd_data_valid), .rd_data(rd_data),
    .rd_data_ready(rd_data_ready), .rd_data_last(rd_data_last),

    .wr_req_valid(wr_req_valid), .wr_req_ready(wr_req_ready),
    .wr_req_addr(wr_req_addr), .wr_req_len_bytes(wr_req_len_bytes),
    .wr_done(wr_done), .wr_data_valid(wr_data_valid),
    .wr_data_ready(wr_data_ready),
    .wr_data(wr_data), .wr_data_last(wr_data_last),

    .err(mem_err),

    .m_axi_arid(m_axi_arid), .m_axi_araddr(m_axi_araddr),
    .m_axi_arlen(m_axi_arlen), .m_axi_arsize(m_axi_arsize),
    .m_axi_arburst(m_axi_arburst), .m_axi_arvalid(m_axi_arvalid),
    .m_axi_arready(m_axi_arready),
    .m_axi_rid(m_axi_rid), .m_axi_rdata(m_axi_rdata),
    .m_axi_rresp(m_axi_rresp), .m_axi_rlast(m_axi_rlast),
    .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready),

    .m_axi_awid(m_axi_awid), .m_axi_awaddr(m_axi_awaddr),
    .m_axi_awlen(m_axi_awlen), .m_axi_awsize(m_axi_awsize),
    .m_axi_awburst(m_axi_awburst), .m_axi_awvalid(m_axi_awvalid),
    .m_axi_awready(m_axi_awready),
    .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb),
    .m_axi_wlast(m_axi_wlast), .m_axi_wvalid(m_axi_wvalid),
    .m_axi_wready(m_axi_wready),
    .m_axi_bid(m_axi_bid), .m_axi_bresp(m_axi_bresp),
    .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready)
  );

endmodule

`default_nettype wire
