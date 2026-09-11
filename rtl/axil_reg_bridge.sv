// ============================================================================
// axil_reg_bridge.sv — AXI4-Lite 슬레이브 → reg_top 단순 버스 어댑터
//
// SSOT: AMBA AXI4-Lite (읽기/쓰기 각 채널 VALID/READY 핸드셰이크)
//       reg_top 쪽 버스는 이 레포의 기존 규약이다:
//         addr[AW-1:0] / wr_en / wr_data[31:0] / rd_data[31:0]
//
// ----------------------------------------------------------------------------
// 왜 필요한가 (PLAN W11)
// ----------------------------------------------------------------------------
// 보드(KV260 등)에서는 PS 의 AXI4-Lite 가 유일한 레지스터 통로다. 지금 레포의
// `reg_top` 은 그보다 단순한 버스를 쓴다. **`reg_top` 을 고치지 않고** 이 어댑터를
// 앞에 붙인다 — 시뮬레이션(cocotb)에서 쓰는 경로와 보드 경로가 같은 레지스터
// 구현을 보게 하려는 것이다.
//
// ----------------------------------------------------------------------------
// 읽기 지연 (중요)
// ----------------------------------------------------------------------------
// `reg_top` 의 대부분 레지스터는 **조합**으로 나온다. 그러나 DR1 스크래치 창과
// 트레이스 창은 뒤에 **1사이클 지연 SRAM** 이 있다 (spec/deltarule.md 3.6절).
// 그래서 MMIO 규약은 "같은 주소를 두 번 읽는다" 였다.
//
// 이 브리지는 주소를 걸어 둔 채 **RD_LAT 사이클을 기다렸다가** RVALID 를 올린다.
// 즉 **AXI 쪽에서는 한 번만 읽으면 된다.** 두 번 읽어도 결과는 같으니
// 기존 호스트 코드(`tools/orbit_device.py`)는 그대로 동작한다.
//
// ----------------------------------------------------------------------------
// 지원 범위 (v1)
// ----------------------------------------------------------------------------
//   - 32비트 데이터, WSTRB 는 **무시한다** (레지스터가 전부 32비트 단위다)
//   - 동시 읽기/쓰기 없음. 한 번에 하나씩 (AXI4-Lite 가 요구하지 않는다)
//   - 응답은 항상 OKAY. 디코드 실패도 OKAY + rd_data=0 이다 (reg_top 규약)
// ============================================================================
`timescale 1ns/1ps
`default_nettype none

module axil_reg_bridge #(
  parameter int AW     = 20,   // reg_top 주소 폭
  parameter int RD_LAT = 2     // 주소를 건 뒤 rd_data 를 샘플링하기까지 대기 사이클
)(
  input  wire         clk,
  input  wire         rst_n,

  // ── AXI4-Lite 슬레이브 ──
  input  wire [AW-1:0] s_axil_awaddr,
  input  wire          s_axil_awvalid,
  output logic         s_axil_awready,

  input  wire [31:0]   s_axil_wdata,
  /* verilator lint_off UNUSEDSIGNAL */
  input  wire [3:0]    s_axil_wstrb,    // v1 은 무시한다 (전부 32비트 레지스터)
  /* verilator lint_on UNUSEDSIGNAL */
  input  wire          s_axil_wvalid,
  output logic         s_axil_wready,

  output logic [1:0]   s_axil_bresp,
  output logic         s_axil_bvalid,
  input  wire          s_axil_bready,

  input  wire [AW-1:0] s_axil_araddr,
  input  wire          s_axil_arvalid,
  output logic         s_axil_arready,

  output logic [31:0]  s_axil_rdata,
  output logic [1:0]   s_axil_rresp,
  output logic         s_axil_rvalid,
  input  wire          s_axil_rready,

  // ── reg_top 쪽 ──
  output logic [AW-1:0] reg_addr,
  output logic          reg_wr_en,
  output logic [31:0]   reg_wr_data,
  input  wire  [31:0]   reg_rd_data
);

  localparam logic [1:0] RESP_OKAY = 2'b00;
  localparam int LATW = (RD_LAT <= 1) ? 1 : $clog2(RD_LAT + 1);

  typedef enum logic [2:0] {
    ST_IDLE  = 3'd0,
    ST_WRITE = 3'd1,   // reg_wr_en 1사이클
    ST_BRESP = 3'd2,
    ST_RWAIT = 3'd3,   // 주소를 걸고 RD_LAT 사이클 대기
    ST_RRESP = 3'd4
  } state_t;

  state_t state;

  logic [AW-1:0]   addr_r;
  logic [31:0]     wdata_r;
  logic [LATW-1:0] lat_cnt;
  logic            aw_seen, w_seen;

  // reg_top 버스: 쓰기는 ST_WRITE 에서 1사이클, 읽기는 ST_RWAIT/ST_RRESP 동안 주소 유지
  assign reg_addr    = addr_r;
  assign reg_wr_en   = (state == ST_WRITE);
  assign reg_wr_data = wdata_r;

  assign s_axil_awready = (state == ST_IDLE) && !aw_seen;
  assign s_axil_wready  = (state == ST_IDLE) && !w_seen;
  assign s_axil_arready = (state == ST_IDLE) && !aw_seen && !w_seen;

  assign s_axil_bresp  = RESP_OKAY;
  assign s_axil_rresp  = RESP_OKAY;
  assign s_axil_bvalid = (state == ST_BRESP);
  assign s_axil_rvalid = (state == ST_RRESP);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state        <= ST_IDLE;
      addr_r       <= '0;
      wdata_r      <= '0;
      lat_cnt      <= '0;
      aw_seen      <= 1'b0;
      w_seen       <= 1'b0;
      s_axil_rdata <= 32'd0;
    end else begin
      case (state)
        ST_IDLE: begin
          // 쓰기: AW 와 W 가 **따로** 와도 된다 (AXI4-Lite 는 순서를 강제하지 않는다)
          if (s_axil_awvalid && s_axil_awready) begin
            addr_r  <= s_axil_awaddr;
            aw_seen <= 1'b1;
          end
          if (s_axil_wvalid && s_axil_wready) begin
            wdata_r <= s_axil_wdata;
            w_seen  <= 1'b1;
          end

          // 둘 다 모였으면 쓰기를 실행한다
          if ((aw_seen || (s_axil_awvalid && s_axil_awready)) &&
              (w_seen  || (s_axil_wvalid  && s_axil_wready))) begin
            state   <= ST_WRITE;
            aw_seen <= 1'b0;
            w_seen  <= 1'b0;
          end else if (s_axil_arvalid && s_axil_arready) begin
            addr_r  <= s_axil_araddr;
            lat_cnt <= '0;
            state   <= ST_RWAIT;
          end
        end

        ST_WRITE: state <= ST_BRESP;

        ST_BRESP: if (s_axil_bready) state <= ST_IDLE;

        ST_RWAIT: begin
          // 주소는 addr_r 에 걸려 있다. 스크래치/트레이스 창의 1사이클 지연을
          // 여기서 흡수한다 — **AXI 쪽은 한 번만 읽으면 된다.**
          if (lat_cnt >= LATW'(RD_LAT)) begin
            s_axil_rdata <= reg_rd_data;
            state        <= ST_RRESP;
          end else begin
            lat_cnt <= lat_cnt + LATW'(1);
          end
        end

        ST_RRESP: if (s_axil_rready) state <= ST_IDLE;

        default: state <= ST_IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
