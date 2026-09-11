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
// DMA 포트 (rd_*/wr_*) 는 **묶어 둔다**
// ----------------------------------------------------------------------------
// DR1 경로는 온칩 스크래치만 쓴다 (spec/deltarule.md 3.6절). 외부 메모리가 필요한
// 것은 GEMM 경로뿐이고, 그것은 이 보드 빌드의 범위가 아니다.
//   - 요청 채널은 ready=1 로 묶어 hang 을 막는다
//   - 응답 채널은 0 으로 묶는다 → GEMM 디스크립터를 보내면 **타임아웃 fault** 가 난다
// 조용히 멈추는 것보다 fault 로 끝나는 편이 낫다. AXI4 마스터 포트를 붙이는 것은
// 보드가 실제로 온 뒤의 일이다 (docs/FPGA.md).
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

  // ── 보드 신호 ──
  output wire          irq_out,
  output wire          reset_active
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

  /* verilator lint_off PINCONNECTEMPTY */
  g2_ctrl_top u_core (
    .clk(clk), .por_n(resetn),
    .reg_addr(reg_addr), .reg_wr_en(reg_wr_en),
    .reg_wr_data(reg_wr_data), .reg_rd_data(reg_rd_data),

    // 외부 메모리 없음 — 요청은 즉시 수락하고 응답은 오지 않는다 (위 주석 참조)
    .rd_req_valid(), .rd_req_ready(1'b1),
    .rd_req_addr(), .rd_req_len_bytes(),
    .rd_done(1'b0), .rd_data_valid(1'b0), .rd_data(128'd0),
    .rd_data_ready(), .rd_data_last(1'b0),

    .wr_req_valid(), .wr_req_ready(1'b1),
    .wr_req_addr(), .wr_req_len_bytes(),
    .wr_done(1'b0), .wr_data_valid(), .wr_data_ready(1'b1),
    .wr_data(), .wr_data_last(),

    .irq_out(irq_out), .reset_active(reset_active)
  );
  /* verilator lint_on PINCONNECTEMPTY */

endmodule

`default_nettype wire
