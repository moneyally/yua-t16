// ============================================================================
// axi4_master_adapter.sv — g2_ctrl_top 의 메모리 요청 인터페이스 → AXI4 마스터
//
// SSOT: AMBA AXI4 (버스트 규칙 2개가 계약이다 — 아래 참조)
//       내부 인터페이스는 이 레포의 기존 규약이다 (rtl/g2_ctrl_top.sv 포트,
//       tb/dma_responder.py 가 시뮬레이션에서 흉내 내던 바로 그것).
//
// ----------------------------------------------------------------------------
// 왜 필요한가
// ----------------------------------------------------------------------------
// `CLAUDE.md` 2절: "**외부 메모리(DDR/HBM) 경로 없음.** `dma_bridge` 는 상태머신이고
// 실제 메모리 인터페이스가 아님." — 지금까지 `rd_req_*`/`wr_req_*` 를 받아 주는 것은
// 테스트벤치(`tb/dma_responder.py`)뿐이었다. 보드에서는 아무도 안 받는다.
//
// 이 모듈이 그 구멍을 메운다. PS 의 DDR 을 AXI4 로 읽고 쓴다.
// **PCIe 와는 무관하다** — PCIe 는 여전히 동작하지 않는다 (rtl/pcie_ep_versal.sv).
//
// ----------------------------------------------------------------------------
// AXI4 버스트 규칙 — 이 두 개가 이 모듈의 존재 이유다
// ----------------------------------------------------------------------------
//   1. 한 버스트는 **최대 256 비트** (AxLEN = beats-1, 8비트)
//   2. 한 버스트는 **4KB 경계를 넘을 수 없다**
//
// 내부 요청은 길이 제한이 없으므로 **쪼개야 한다**. 안 쪼개면 슬레이브가 조용히
// 엉뚱한 주소에 쓰거나 프로토콜 위반으로 멈춘다. 테스트벤치는 그것을 봐준다 —
// 실제 DDR 컨트롤러는 안 봐준다. 그래서 여기서 쪼갠다.
//
//   beats_to_4k = (4096 - (addr & 0xFFF)) / BYTES_PER_BEAT
//   n = min(남은 beat, 256, beats_to_4k)
//
// ----------------------------------------------------------------------------
// 계약
// ----------------------------------------------------------------------------
//   - 읽기/쓰기 각각 한 번에 하나씩만 진행한다 (outstanding 1). 단순함이 먼저다
//   - `rd_data_last` 는 **요청 전체의 마지막 beat** 에만 뜬다 (버스트마다가 아니다)
//   - `rd_done`/`wr_done` 은 요청 전체가 끝난 뒤 1사이클
//   - 응답 코드(RRESP/BRESP)가 OKAY 가 아니면 `err` 를 올리고 그 요청을 끝낸다
// ============================================================================
`timescale 1ns/1ps
`default_nettype none

module axi4_master_adapter #(
  parameter int DATA_W  = 128,             // 내부 데이터 폭 = AXI 데이터 폭
  parameter int ADDR_W  = 64,
  parameter int ID_W    = 4,
  parameter int MAX_BURST = 256            // AXI4 상한. 줄이면 더 잘게 쪼갠다
)(
  input  wire                 clk,
  input  wire                 rst_n,

  // ── 내부 읽기 요청 (g2_ctrl_top 쪽) ──
  input  wire                 rd_req_valid,
  output logic                rd_req_ready,
  input  wire [ADDR_W-1:0]    rd_req_addr,
  input  wire [15:0]          rd_req_len_bytes,
  output logic                rd_done,
  output logic                rd_data_valid,
  output logic [DATA_W-1:0]   rd_data,
  input  wire                 rd_data_ready,
  output logic                rd_data_last,

  // ── 내부 쓰기 요청 ──
  input  wire                 wr_req_valid,
  output logic                wr_req_ready,
  input  wire [ADDR_W-1:0]    wr_req_addr,
  input  wire [15:0]          wr_req_len_bytes,
  output logic                wr_done,
  input  wire                 wr_data_valid,
  output logic                wr_data_ready,
  input  wire [DATA_W-1:0]    wr_data,
  /* verilator lint_off UNUSEDSIGNAL */
  input  wire                 wr_data_last,   // 길이로 이미 안다. 신뢰하지 않는다
  /* verilator lint_on UNUSEDSIGNAL */

  // 응답 오류 (RRESP/BRESP != OKAY). 1사이클 펄스
  output logic                err,

  // ── AXI4 마스터 ──
  output logic [ID_W-1:0]     m_axi_arid,
  output logic [ADDR_W-1:0]   m_axi_araddr,
  output logic [7:0]          m_axi_arlen,
  output logic [2:0]          m_axi_arsize,
  output logic [1:0]          m_axi_arburst,
  output logic                m_axi_arvalid,
  input  wire                 m_axi_arready,

  /* verilator lint_off UNUSEDSIGNAL */
  input  wire [ID_W-1:0]      m_axi_rid,      // outstanding 1 이라 쓰지 않는다
  /* verilator lint_on UNUSEDSIGNAL */
  input  wire [DATA_W-1:0]    m_axi_rdata,
  input  wire [1:0]           m_axi_rresp,
  // RLAST 는 **신뢰하지 않는다** — 버스트 길이를 우리가 계산해서 냈으므로
  // 우리 카운터가 기준이다. 슬레이브가 RLAST 를 잘못 내도 우리는 안 흔들린다.
  /* verilator lint_off UNUSEDSIGNAL */
  input  wire                 m_axi_rlast,
  /* verilator lint_on UNUSEDSIGNAL */
  input  wire                 m_axi_rvalid,
  output logic                m_axi_rready,

  output logic [ID_W-1:0]     m_axi_awid,
  output logic [ADDR_W-1:0]   m_axi_awaddr,
  output logic [7:0]          m_axi_awlen,
  output logic [2:0]          m_axi_awsize,
  output logic [1:0]          m_axi_awburst,
  output logic                m_axi_awvalid,
  input  wire                 m_axi_awready,

  output logic [DATA_W-1:0]   m_axi_wdata,
  output logic [DATA_W/8-1:0] m_axi_wstrb,
  output logic                m_axi_wlast,
  output logic                m_axi_wvalid,
  input  wire                 m_axi_wready,

  /* verilator lint_off UNUSEDSIGNAL */
  input  wire [ID_W-1:0]      m_axi_bid,
  /* verilator lint_on UNUSEDSIGNAL */
  input  wire [1:0]           m_axi_bresp,
  input  wire                 m_axi_bvalid,
  output logic                m_axi_bready
);

  localparam int BYTES_PER_BEAT = DATA_W / 8;
  localparam int BEAT_SHIFT     = $clog2(BYTES_PER_BEAT);   // 128b -> 4
  localparam logic [2:0] AXSIZE = 3'(BEAT_SHIFT);           // 2^SIZE 바이트/beat
  localparam logic [1:0] BURST_INCR = 2'b01;
  localparam logic [1:0] RESP_OKAY  = 2'b00;
  localparam int PAGE = 4096;

  // ==========================================================================
  // 읽기 채널
  // ==========================================================================
  typedef enum logic [1:0] {
    RD_IDLE = 2'd0,
    RD_AR   = 2'd1,   // AR 핸드셰이크 대기
    RD_DATA = 2'd2,   // R beat 수신
    RD_DONE = 2'd3
  } rd_state_t;

  rd_state_t       rd_state;
  logic [ADDR_W-1:0] rd_addr;
  logic [16:0]     rd_left;        // 남은 beat (요청 전체)
  logic [8:0]      rd_burst_left;  // 이번 버스트의 남은 beat
  logic            rd_err_r;

  // 이번에 낼 버스트 길이 = min(남은 beat, MAX_BURST, 4KB 경계까지)
  // **두 제한을 모두 본다.** 하나만 보면 조용히 프로토콜을 어긴다.
  wire [16:0] rd_page_off = {5'b0, rd_addr[11:0]};
  wire [16:0] rd_to_page  = (17'(PAGE) - rd_page_off) >> BEAT_SHIFT;
  wire [16:0] rd_cap     = (rd_left < 17'(MAX_BURST)) ? rd_left : 17'(MAX_BURST);
  wire [16:0] rd_n       = (rd_cap < rd_to_page) ? rd_cap : rd_to_page;

  assign rd_req_ready  = (rd_state == RD_IDLE);
  assign m_axi_arid    = '0;
  assign m_axi_araddr  = rd_addr;
  assign m_axi_arlen   = 8'(rd_n - 17'd1);
  assign m_axi_arsize  = AXSIZE;
  assign m_axi_arburst = BURST_INCR;
  assign m_axi_arvalid = (rd_state == RD_AR);

  assign m_axi_rready  = (rd_state == RD_DATA) && rd_data_ready;
  assign rd_data       = m_axi_rdata;
  assign rd_data_valid = (rd_state == RD_DATA) && m_axi_rvalid;
  // 요청 전체의 마지막 beat 에만 last 를 올린다 (버스트 경계가 아니다)
  assign rd_data_last  = rd_data_valid && (rd_left == 17'd1);
  assign rd_done       = (rd_state == RD_DONE);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_state      <= RD_IDLE;
      rd_addr       <= '0;
      rd_left       <= '0;
      rd_burst_left <= '0;
      rd_err_r      <= 1'b0;
    end else begin
      case (rd_state)
        RD_IDLE: if (rd_req_valid) begin
          rd_addr  <= rd_req_addr;
          rd_left  <= 17'(rd_req_len_bytes >> BEAT_SHIFT);
          rd_err_r <= 1'b0;
          // 길이가 0 이면 할 일이 없다 — 바로 완료로 간다 (멈추지 않는다)
          rd_state <= (rd_req_len_bytes >> BEAT_SHIFT) == 0 ? RD_DONE : RD_AR;
        end

        RD_AR: if (m_axi_arready) begin
          rd_burst_left <= 9'(rd_n);
          rd_state      <= RD_DATA;
        end

        RD_DATA: if (m_axi_rvalid && m_axi_rready) begin
          if (m_axi_rresp != RESP_OKAY) rd_err_r <= 1'b1;
          rd_addr       <= rd_addr + ADDR_W'(BYTES_PER_BEAT);
          rd_left       <= rd_left - 17'd1;
          rd_burst_left <= rd_burst_left - 9'd1;
          if (rd_left == 17'd1)              rd_state <= RD_DONE;   // 전체 끝
          else if (rd_burst_left == 9'd1)    rd_state <= RD_AR;     // 버스트만 끝
        end

        RD_DONE: rd_state <= RD_IDLE;
        default: rd_state <= RD_IDLE;
      endcase
    end
  end

  // ==========================================================================
  // 쓰기 채널
  // ==========================================================================
  typedef enum logic [2:0] {
    WR_IDLE = 3'd0,
    WR_AW   = 3'd1,
    WR_DATA = 3'd2,
    WR_B    = 3'd3,   // 버스트마다 B 응답을 받는다
    WR_DONE = 3'd4
  } wr_state_t;

  wr_state_t       wr_state;
  logic [ADDR_W-1:0] wr_addr;
  logic [16:0]     wr_left;
  logic [8:0]      wr_burst_left;
  logic            wr_err_r;

  wire [16:0] wr_page_off = {5'b0, wr_addr[11:0]};
  wire [16:0] wr_to_page  = (17'(PAGE) - wr_page_off) >> BEAT_SHIFT;
  wire [16:0] wr_cap     = (wr_left < 17'(MAX_BURST)) ? wr_left : 17'(MAX_BURST);
  wire [16:0] wr_n       = (wr_cap < wr_to_page) ? wr_cap : wr_to_page;

  assign wr_req_ready  = (wr_state == WR_IDLE);
  assign m_axi_awid    = '0;
  assign m_axi_awaddr  = wr_addr;
  assign m_axi_awlen   = 8'(wr_n - 17'd1);
  assign m_axi_awsize  = AXSIZE;
  assign m_axi_awburst = BURST_INCR;
  assign m_axi_awvalid = (wr_state == WR_AW);

  assign m_axi_wdata   = wr_data;
  assign m_axi_wstrb   = '1;                       // 항상 전 바이트 (부분 쓰기 없음)
  assign m_axi_wvalid  = (wr_state == WR_DATA) && wr_data_valid;
  assign m_axi_wlast   = m_axi_wvalid && (wr_burst_left == 9'd1);
  assign wr_data_ready = (wr_state == WR_DATA) && m_axi_wready;

  assign m_axi_bready  = (wr_state == WR_B);
  assign wr_done       = (wr_state == WR_DONE);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_state      <= WR_IDLE;
      wr_addr       <= '0;
      wr_left       <= '0;
      wr_burst_left <= '0;
      wr_err_r      <= 1'b0;
    end else begin
      case (wr_state)
        WR_IDLE: if (wr_req_valid) begin
          wr_addr  <= wr_req_addr;
          wr_left  <= 17'(wr_req_len_bytes >> BEAT_SHIFT);
          wr_err_r <= 1'b0;
          wr_state <= (wr_req_len_bytes >> BEAT_SHIFT) == 0 ? WR_DONE : WR_AW;
        end

        WR_AW: if (m_axi_awready) begin
          wr_burst_left <= 9'(wr_n);
          wr_state      <= WR_DATA;
        end

        WR_DATA: if (m_axi_wvalid && m_axi_wready) begin
          wr_addr       <= wr_addr + ADDR_W'(BYTES_PER_BEAT);
          wr_left       <= wr_left - 17'd1;
          wr_burst_left <= wr_burst_left - 9'd1;
          if (wr_burst_left == 9'd1) wr_state <= WR_B;   // 버스트 끝 -> B 대기
        end

        WR_B: if (m_axi_bvalid) begin
          if (m_axi_bresp != RESP_OKAY) wr_err_r <= 1'b1;
          wr_state <= (wr_left == 17'd0) ? WR_DONE : WR_AW;
        end

        WR_DONE: wr_state <= WR_IDLE;
        default: wr_state <= WR_IDLE;
      endcase
    end
  end

  // 오류는 완료와 같은 사이클에 1펄스로 알린다
  assign err = (rd_done && rd_err_r) || (wr_done && wr_err_r);

endmodule

`default_nettype wire
