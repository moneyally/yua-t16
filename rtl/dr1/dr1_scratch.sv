// ============================================================================
// dr1_scratch.sv — DR1 입출력 벡터 스크래치 (q, k, v, o)
//
// SSOT: spec/deltarule.md 3.3절(주소 필드), 3.5절(평탄화·바이트 순서)
//
// ----------------------------------------------------------------------------
// 왜 이 모듈이 필요한가
// ----------------------------------------------------------------------------
// W6 까지 dr1_top 은 상태(state_sram)만 있었고 q/k/v 를 받을 곳이 없었다.
// `DELTA_STEP` 이 실제로 돌려면 **호스트가 벡터를 넣고 결과를 꺼내는 메모리**가
// 있어야 한다. DDR/HBM 경로는 이 레포에 없으므로(docs/DESIGN.md 10절),
// v1 은 **온칩 스크래치**로 한다. 이것은 스텁이 아니라 실제로 동작하는 메모리다.
//
// ----------------------------------------------------------------------------
// 포트 계약
// ----------------------------------------------------------------------------
//   host 포트 : 레지스터 버스(reg_top 의 DR1 스크래치 창)에서 온다. 32비트 MMIO
//               워드 하나가 **Q1.15 원소 하나**를 담는다 (하위 16비트).
//               낭비지만 주소 계산이 단순하고, 호스트 쪽 테스트가 읽기 쉽다.
//   dr1 포트  : dr1_top 이 쓴다. 원소 단위 16비트.
//   **둘 다 읽기 지연 정확히 1사이클** (출력 레지스터) → falling edge 샘플링.
//
//   같은 주소에 host 쓰기와 dr1 쓰기가 겹치면 **미정의**다.
//   호스트는 `DR1_STATUS.busy` 가 1 인 동안 쓰면 안 된다 (spec 5.1절).
//
// BRAM 추론 스타일은 state_sram.sv 와 같다: mem 에 리셋을 걸지 않고,
// 포트마다 always_ff 하나씩. scripts/synth_gate.sh --check-mem 으로 확인한다.
// ============================================================================
`timescale 1ns/1ps
`default_nettype none

module dr1_scratch #(
  parameter int DEPTH = 256,     // Q1.15 원소 개수. d=64 × (q,k,v,o) = 256
  parameter int W     = 16
)(
  input  wire                      clk,
  input  wire                      rst_n,

  // ── 호스트 포트 (MMIO) ──
  input  wire                      h_en,
  input  wire                      h_we,
  input  wire [$clog2(DEPTH)-1:0]  h_addr,
  input  wire [W-1:0]              h_wdata,
  output logic [W-1:0]             h_rdata,

  // ── DR1 포트 ──
  input  wire                      d_rd_en,
  input  wire [$clog2(DEPTH)-1:0]  d_rd_addr,
  output logic [W-1:0]             d_rd_data,
  input  wire                      d_wr_en,
  input  wire [$clog2(DEPTH)-1:0]  d_wr_addr,
  input  wire [W-1:0]              d_wr_data
);

  logic [W-1:0] mem [0:DEPTH-1];

  // 호스트 포트 — 쓰기/읽기 같은 always_ff (단순 이중 포트 BRAM 추론)
  always_ff @(posedge clk) begin
    if (h_en && h_we) mem[h_addr] <= h_wdata;
    if (h_en)         h_rdata     <= mem[h_addr];
  end

  // DR1 포트
  always_ff @(posedge clk) begin
    if (d_wr_en) mem[d_wr_addr] <= d_wr_data;
    if (d_rd_en) d_rd_data      <= mem[d_rd_addr];
  end

  // 시뮬레이션 초기값. 합성 경로에는 없다 (CLAUDE.md 규칙 1 — SIM-ONLY).
`ifdef COCOTB_SIM
  initial begin
    h_rdata   = '0;
    d_rd_data = '0;
    for (int i = 0; i < DEPTH; i++) mem[i] = '0;
  end
`endif

  // rst_n 은 BRAM 추론을 깨지 않으려고 mem 에 걸지 않는다. 포트 목록에는 남겨
  // 두어 상위 모듈 결선을 다른 SRAM 과 똑같이 유지한다.
  /* verilator lint_off UNUSEDSIGNAL */
  wire _unused_rst = &{1'b0, rst_n, 1'b0};
  /* verilator lint_on UNUSEDSIGNAL */

endmodule

`default_nettype wire
