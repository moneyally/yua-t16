// ============================================================================
// state_sram.sv — DR1 상태 행렬 S 저장소 (D행 × D·W비트, 1R1W)
//
// 대응 골든:  sim/golden/deltarule.py 의 상태 배열 S 자체.
//             행 r 의 내용 = S[r][0..D-1], 평탄화 규칙은 spec/deltarule.md 3.5절
//             (원소 i = bits[i*W +: W]).  tools/orbit_pack.py pack_vec/unpack_vec.
//
// SSOT: docs/DESIGN.md 5절 "state_sram   d×d Q1.15, (v1: 단순 1R1W + 직렬)"
//
// ----------------------------------------------------------------------------
// 타이밍 계약 — **읽기 지연은 정확히 1사이클**
// ----------------------------------------------------------------------------
//   rd_en 을 올린 사이클의 다음 클럭 엣지에서 rd_data 가 유효해진다.
//   rd_data 는 출력 레지스터다. 따라서 테스트벤치는 **falling edge 에서
//   샘플링**해야 한다 — RisingEdge 직후에 읽으면 논블로킹 대입 전이라
//   이전 값을 본다 (docs/BUGS.md BUG-008a/b 에서 cdc_fifo 가 정확히 이걸로 틀렸다).
//
// ----------------------------------------------------------------------------
// 같은 행 동시 읽기/쓰기 = **이전 값 읽기 (read-before-write)**
// ----------------------------------------------------------------------------
//   같은 always_ff 안에서 둘 다 논블로킹 대입이므로, 읽기는 이번 엣지의 쓰기가
//   반영되기 **전** 값을 가져간다. matvec_unit 과 update_unit 이 같은 행을
//   읽고 쓰는 파이프라인을 만들 때 이 정의에 의존한다.
//
// ----------------------------------------------------------------------------
// Vivado BRAM 추론 스타일
// ----------------------------------------------------------------------------
//   - 메모리 배열 `mem` 에는 **리셋이 없다** (리셋이 있으면 BRAM 추론이 깨진다)
//   - 메모리 접근은 **always_ff 하나**에만 있다
//   - 출력 레지스터를 그 안에 두어 동기 읽기 BRAM 으로 매핑되게 한다
//   - 클리어 FSM 은 **별도 always_ff** (여긴 리셋 있음)
//   yosys 검사: scripts/synth_gate.sh --check-mem state_sram
//              → $mem_v2 셀 1개로 인식되어야 한다. 플립플롭으로 풀리면 실패.
// ============================================================================
`timescale 1ns/1ps
`default_nettype none

module state_sram #(
  parameter int D = 16,               // 행/열 개수 (헤드 차원)
  parameter int W = 16                // 원소 폭 (Q1.15)
)(
  input  wire                 clk,
  input  wire                 rst_n,

  // 읽기 포트 (동기, 지연 1사이클)
  input  wire                 rd_en,
  input  wire [$clog2(D)-1:0] rd_row,
  output logic [D*W-1:0]      rd_data,

  // 쓰기 포트
  input  wire                 wr_en,
  input  wire [$clog2(D)-1:0] wr_row,
  input  wire [D*W-1:0]       wr_data,

  // 전체 클리어 (DELTA_INIT)
  input  wire                 clr_start,
  output logic                clr_busy
);

  localparam int AW = $clog2(D);

  // --------------------------------------------------------------------------
  // 클리어 FSM — D 사이클 동안 전 행에 0 을 쓴다.
  // clr_busy 는 clr_start 다음 사이클부터 D 사이클 동안 1 이다.
  // --------------------------------------------------------------------------
  logic [AW-1:0] clr_row;
  logic          clr_run;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      clr_run <= 1'b0;
      clr_row <= '0;
    end else begin
      if (clr_start && !clr_run) begin
        clr_run <= 1'b1;
        clr_row <= '0;
      end else if (clr_run) begin
        if (clr_row == AW'(D-1)) begin
          clr_run <= 1'b0;
          clr_row <= '0;
        end else begin
          clr_row <= clr_row + AW'(1);
        end
      end
    end
  end

  assign clr_busy = clr_run;

  // --------------------------------------------------------------------------
  // 쓰기 포트 먹싱 — 클리어가 이기면 외부 쓰기는 무시된다.
  // (DELTA_INIT 중에 STEP 이 들어올 수 없으므로 dr1_top 이 이를 보장한다)
  // --------------------------------------------------------------------------
  wire             wr_en_eff   = clr_run ? 1'b1      : wr_en;
  wire [AW-1:0]    wr_row_eff  = clr_run ? clr_row   : wr_row;
  wire [D*W-1:0]   wr_data_eff = clr_run ? '0        : wr_data;

  // --------------------------------------------------------------------------
  // 메모리 — always_ff 하나, 리셋 없음. BRAM 으로 추론되어야 한다.
  // 같은 엣지에서 읽기와 쓰기가 같은 행이면 **이전 값**이 읽힌다.
  // --------------------------------------------------------------------------
  logic [D*W-1:0] mem [0:D-1];

  always_ff @(posedge clk) begin
    if (wr_en_eff)
      mem[wr_row_eff] <= wr_data_eff;
    if (rd_en)
      rd_data <= mem[rd_row];
  end

`ifdef COCOTB_SIM
  // 시뮬레이션 전용: X 로 시작하지 않게. 합성에는 들어가지 않는다 (CLAUDE.md 규칙 1).
  initial begin
    rd_data = '0;
    for (int i = 0; i < D; i++) mem[i] = '0;
  end
`endif

endmodule

`default_nettype wire
