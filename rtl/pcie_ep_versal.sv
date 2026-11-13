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
  // CQ → BAR 디코드 / CC 완료 생성
  // ═══════════════════════════════════════════════════════════
  //
  // 지원 범위: 1 DW (32비트) 메모리 읽기/쓰기.
  //   호스트 스택(tools/orbit_mmap_backend.py)이 BAR 를 mmap 해서
  //   32비트 단위로만 접근하므로 레지스터 경로에 필요한 전부다.
  //   2 DW 이상 요청은 Unsupported Request 로 완료한다.
  //
  // ── 비트 위치에 대한 경고 ──────────────────────────────────
  // 아래 CQ_* / CC_* 상수는 Xilinx PCIe 완결자 디스크립터 형식이다.
  // 이 저장소에는 PG343 이 없고 Vivado 도 없어서 "확인되지 않았다".
  // 형식이 다르면 이 상수들만 고치면 된다 — 아래 FSM 과 라우팅은
  // 상수를 통해서만 디스크립터를 보므로 손댈 필요가 없다.
  // tb/tb_pcie_ep_completer.py 도 같은 상수를 써서 자극을 만들기 때문에,
  // 테스트가 검증하는 것은 "형식" 이 아니라 "로직" 이다.
  //
  // 반드시 PG343 의 Completer Request/Completion Descriptor 표와
  // 대조한 뒤 합성하라.
  // ───────────────────────────────────────────────────────────

  // Completer Request 디스크립터 (첫 비트 128비트)
  localparam int CQ_ADDR_LSB   = 2;    // [63:2]  주소
  localparam int CQ_ADDR_W     = 62;
  localparam int CQ_DWCNT_LSB  = 64;   // [74:64] DW 수
  localparam int CQ_DWCNT_W    = 11;
  localparam int CQ_REQTYPE_LSB= 75;   // [78:75] 요청 종류
  localparam int CQ_REQTYPE_W  = 4;
  localparam int CQ_REQID_LSB  = 80;   // [95:80] Requester ID
  localparam int CQ_TAG_LSB    = 96;   // [103:96] Tag
  localparam int CQ_BARID_LSB  = 112;  // [114:112] BAR 번호
  localparam int CQ_BARID_W    = 3;
  localparam int CQ_PAYLOAD_LSB= 128;  // 첫 비트의 payload DW0

  localparam logic [3:0] REQ_MEM_READ  = 4'b0000;
  localparam logic [3:0] REQ_MEM_WRITE = 4'b0001;

  // Completion 디스크립터 (첫 96비트)
  localparam int CC_LOWADDR_LSB = 0;    // [6:0]
  localparam int CC_BYTECNT_LSB = 16;   // [28:16]
  localparam int CC_DWCNT_LSB   = 32;   // [42:32]
  localparam int CC_STATUS_LSB  = 43;   // [45:43]
  localparam int CC_REQID_LSB   = 48;   // [63:48]
  localparam int CC_TAG_LSB     = 64;   // [71:64]
  localparam int CC_CPLID_LSB   = 72;   // [87:72]
  localparam int CC_PAYLOAD_LSB = 96;   // 완료 payload DW0

  localparam logic [2:0] CPL_SUCCESS     = 3'b000;
  localparam logic [2:0] CPL_UNSUPPORTED = 3'b001;

  // first BE 는 tuser 하위 4비트
  localparam int CQ_TUSER_FBE_LSB = 0;

  // ── 디스크립터 분해 ────────────────────────────────────────
  logic [CQ_ADDR_W-1:0]    cq_addr_dw;   // DW 주소 (하위 2비트 제외)
  logic [CQ_DWCNT_W-1:0]   cq_dwcnt;
  logic [CQ_REQTYPE_W-1:0] cq_reqtype;
  logic [15:0]             cq_reqid;
  logic [7:0]              cq_tag;
  logic [CQ_BARID_W-1:0]   cq_barid;
  logic [31:0]             cq_wdata;
  logic [3:0]              cq_fbe;

  assign cq_addr_dw = m_axis_cq_tdata[CQ_ADDR_LSB    +: CQ_ADDR_W];
  assign cq_dwcnt   = m_axis_cq_tdata[CQ_DWCNT_LSB   +: CQ_DWCNT_W];
  assign cq_reqtype = m_axis_cq_tdata[CQ_REQTYPE_LSB +: CQ_REQTYPE_W];
  assign cq_reqid   = m_axis_cq_tdata[CQ_REQID_LSB   +: 16];
  assign cq_tag     = m_axis_cq_tdata[CQ_TAG_LSB     +: 8];
  assign cq_barid   = m_axis_cq_tdata[CQ_BARID_LSB   +: CQ_BARID_W];
  assign cq_wdata   = m_axis_cq_tdata[CQ_PAYLOAD_LSB +: 32];
  assign cq_fbe     = m_axis_cq_tuser[CQ_TUSER_FBE_LSB +: 4];

  // ── 래치 ───────────────────────────────────────────────────
  logic [CQ_ADDR_W-1:0]  addr_r;
  logic [15:0]           reqid_r;
  logic [7:0]            tag_r;
  logic [CQ_BARID_W-1:0] barid_r;
  logic [31:0]           wdata_r, rdata_r;
  logic [3:0]            be_r;
  logic                  is_write_r;
  logic                  unsupported_r;

  // ── FSM ────────────────────────────────────────────────────
  typedef enum logic [2:0] {
    P_IDLE,      // CQ 수신 대기
    P_REQ,       // BAR 요청 구동
    P_RD_WAIT,   // BAR 읽기 응답 대기
    P_CC,        // 완료 전송
    P_DRAIN      // 남은 비트 흡수 (멀티 비트 요청)
  } pstate_t;

  pstate_t pst;

  // BAR 선택. Xilinx BAR ID 는 BAR 번호 그대로다.
  localparam logic [2:0] BARID_0 = 3'd0;
  localparam logic [2:0] BARID_2 = 3'd2;
  localparam logic [2:0] BARID_4 = 3'd4;

  wire sel0 = (barid_r == BARID_0);
  wire sel2 = (barid_r == BARID_2);
  wire sel4 = (barid_r == BARID_4);

  wire bar_ready = (sel0 & bar0_req_ready)
                 | (sel2 & bar2_req_ready)
                 | (sel4 & bar4_req_ready);
  wire bar_rsp_valid = (sel0 & bar0_rsp_valid)
                     | (sel2 & bar2_rsp_valid)
                     | (sel4 & bar4_rsp_valid);
  wire [31:0] bar_rsp_rdata = sel0 ? bar0_rsp_rdata
                            : sel2 ? bar2_rsp_rdata
                            : sel4 ? bar4_rsp_rdata : 32'd0;

  // 알 수 없는 BAR 은 요청을 내보내지 않고 UR 로 완료한다.
  wire bar_known = sel0 | sel2 | sel4;

  assign m_axis_cq_tready = (pst == P_IDLE) || (pst == P_DRAIN);

  // 바이트 주소 → 각 BAR 폭으로 자른다 (하위 2비트는 DW 정렬이라 0)
  wire [63:0] byte_addr = {addr_r, 2'b00};

  assign bar0_req_valid = (pst == P_REQ) && sel0;
  assign bar0_req_addr  = byte_addr[19:0];
  assign bar0_req_wr    = is_write_r;
  assign bar0_req_wdata = wdata_r;
  assign bar0_req_be    = be_r;

  assign bar2_req_valid = (pst == P_REQ) && sel2;
  assign bar2_req_addr  = byte_addr[20:0];
  assign bar2_req_wr    = is_write_r;
  assign bar2_req_wdata = wdata_r;
  assign bar2_req_be    = be_r;

  assign bar4_req_valid = (pst == P_REQ) && sel4;
  assign bar4_req_addr  = byte_addr[15:0];
  assign bar4_req_wr    = is_write_r;
  assign bar4_req_wdata = wdata_r;
  assign bar4_req_be    = be_r;

  // ── 완료 디스크립터 ────────────────────────────────────────
  logic [AXI_DATA_W-1:0] cc_data;
  always_comb begin
    cc_data = '0;
    cc_data[CC_LOWADDR_LSB +: 7]  = {byte_addr[6:2], 2'b00};
    cc_data[CC_BYTECNT_LSB +: 13] = unsupported_r ? 13'd0 : 13'd4;
    cc_data[CC_DWCNT_LSB   +: 11] = unsupported_r ? 11'd0 : 11'd1;
    cc_data[CC_STATUS_LSB  +: 3]  = unsupported_r ? CPL_UNSUPPORTED : CPL_SUCCESS;
    cc_data[CC_REQID_LSB   +: 16] = reqid_r;
    cc_data[CC_TAG_LSB     +: 8]  = tag_r;
    cc_data[CC_CPLID_LSB   +: 16] = 16'd0;   // CPM 이 채운다
    cc_data[CC_PAYLOAD_LSB +: 32] = rdata_r;
  end

  assign s_axis_cc_tdata  = cc_data;
  assign s_axis_cc_tvalid = (pst == P_CC);
  assign s_axis_cc_tlast  = (pst == P_CC);
  // 완료는 디스크립터 3 DW + payload 1 DW = 4 DW
  assign s_axis_cc_tkeep  = (pst == P_CC) ? 8'h0F : 8'h00;

  always_ff @(posedge user_clk) begin
    if (user_reset) begin
      pst           <= P_IDLE;
      addr_r        <= '0;
      reqid_r       <= '0;
      tag_r         <= '0;
      barid_r       <= '0;
      wdata_r       <= '0;
      rdata_r       <= '0;
      be_r          <= '0;
      is_write_r    <= 1'b0;
      unsupported_r <= 1'b0;
    end else begin
      case (pst)
        P_IDLE: begin
          if (m_axis_cq_tvalid) begin
            addr_r     <= cq_addr_dw;
            reqid_r    <= cq_reqid;
            tag_r      <= cq_tag;
            barid_r    <= cq_barid;
            wdata_r    <= cq_wdata;
            be_r       <= cq_fbe;
            rdata_r    <= 32'd0;
            is_write_r <= (cq_reqtype == REQ_MEM_WRITE);

            if (cq_dwcnt != 11'd1) begin
              // 1 DW 외에는 지원하지 않는다.
              unsupported_r <= 1'b1;
              // 읽기면 UR 완료를 보내야 하고, 쓰기는 완료가 없다.
              pst <= (cq_reqtype == REQ_MEM_WRITE)
                   ? (m_axis_cq_tlast ? P_IDLE : P_DRAIN)
                   : (m_axis_cq_tlast ? P_CC   : P_DRAIN);
            end else begin
              unsupported_r <= 1'b0;
              pst <= P_REQ;
            end
          end
        end

        P_REQ: begin
          if (!bar_known) begin
            // 모르는 BAR — 요청을 내보내지 않는다
            unsupported_r <= 1'b1;
            pst <= is_write_r ? P_IDLE : P_CC;
          end else if (bar_ready) begin
            pst <= is_write_r ? P_IDLE : P_RD_WAIT;
          end
        end

        P_RD_WAIT: begin
          if (bar_rsp_valid) begin
            rdata_r <= bar_rsp_rdata;
            pst     <= P_CC;
          end
        end

        P_CC: begin
          if (s_axis_cc_tready) pst <= P_IDLE;
        end

        P_DRAIN: begin
          // 멀티 비트 요청의 남은 비트를 흘려보낸다
          if (m_axis_cq_tvalid && m_axis_cq_tlast)
            pst <= is_write_r ? P_IDLE : P_CC;
        end

        default: pst <= P_IDLE;
      endcase
    end
  end

  // RQ: DMA 요청은 dma_bridge 가 담당한다. 이 모듈은 완결자만 맡는다.
  assign s_axis_rq_tdata  = '0;
  assign s_axis_rq_tvalid = 1'b0;
  assign s_axis_rq_tlast  = 1'b0;
  assign s_axis_rq_tkeep  = '0;

  // RC: 요청자 완료도 마찬가지로 아직 흡수만 한다.
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
