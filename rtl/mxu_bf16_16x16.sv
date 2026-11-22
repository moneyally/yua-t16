// mxu_bf16_16x16.sv — BF16 16×16 Matrix Multiply Unit for ORBIT-G3
// SSOT: ORBIT_G3_RTL_PLAN.md (G3-RTL-003), ORBIT_G3_ARCHITECTURE.md
//
// Computes C[16×16] += A[16×K] × B[K×16] in BF16, accumulated in FP32.
// One K-step per cycle: 256 BF16 MACs/cycle.
//
// Interface:
//   - a_row: 16 BF16 values, 평탄화 (one row of A, column k)
//   - b_col: 16 BF16 values, 평탄화 (one column of B, row k)
//   - acc_out: 16×16 FP32 누산기 (256 값), 행 우선 평탄화
//
// ── 왜 평탄화인가 ────────────────────────────────────────────────
//
// 전에는 unpacked 배열 포트였다 (`input logic [15:0] a_row [0:ROWS-1]`).
// 읽기는 편했지만 **Yosys 가 그 포트 선언을 파싱하지 못한다** — 0.33
// 도, 0.71 도 같은 줄에서 멈춘다. 포트만 평탄화한 래퍼를 씌워도
// 안 된다. Yosys 가 이 파일 자체를 읽어야 하기 때문이다.
//
// 그래서 BF16 경로(타일 → 128×128 → `mxu_core`) 의 면적을 한 번도
// 재지 못했다. 공유 변환기가 아끼는 "타일당 143,394 셀" 조차 벤치
// 두 개를 더하고 뺀 추정값이었다. 포트를 packed 로 바꾸면 그 전부가
// 실측 가능해진다. 인덱싱은 `[i*COLS + j]` 로 한 줄 늘어날 뿐이다.
//
// 누산 방식: 고정소수점 (bf16_pkg 의 ACC_W=48, ACC_F=16)
//   Stage 1: BF16 × BF16 → 고정소수점 증분 (조합, 시프트 정렬)
//   Stage 2: 고정소수점 포화 덧셈 (등록)
//   출력   : 고정소수점 → FP32 변환 (조합, 누산 루프 밖)
//
// 이전에는 PE 마다 완전한 FP32 가산기(정렬 + 25단 정규화 + 반올림)를
// 매 사이클 돌렸다. 합성하면 PE 하나가 4,369 셀, 타일이 1,118,464 셀,
// PAR=8 이면 71,618,048 셀이 되어 합성 자체가 불가능했다.
// 정규화와 반올림을 출력으로 한 번만 밀어내면 누산 루프에는 시프트와
// 덧셈만 남는다. 정수 누산이라 K 에 대해 중간 반올림도 없다.
//
// BF16 format: {sign[15], exponent[14:7], mantissa[6:0]}
//
// 산술(bf16_mul / fp32_add)은 rtl/bf16_pkg.sv 로 옮겼다. 이 모듈 안에만
// 있던 탓에 mxu_bf16_128x128 이 누적 덧셈을 따로 구현했고 거기서 버그가
// 났다. 이제 두 모듈이 같은 함수를 쓴다.
`timescale 1ns/1ps
`default_nettype none

module mxu_bf16_16x16 #(
  parameter int ROWS = 16,
  parameter int COLS = 16,

  // 출력 방식 선택.
  //
  // FP32_OUT=1 (기본): acc_out 에 16×16 FP32 를 조합으로 전부 내보낸다.
  //   PE 마다 고정소수점→FP32 변환기가 하나씩 필요하다 (PE 당 720 셀,
  //   PE 비용의 27%). 기존 인터페이스 그대로다.
  //
  // FP32_OUT=0 + STREAM_RD=1: acc_out 을 0 으로 묶고, 변환기 하나를
  //   타일 전체가 공유한다. rd_idx 로 원소를 골라 rd_fp32 로 읽는다.
  //   읽기는 GEMM 타일당 한 번뿐이므로 순차 읽기로 충분하다.
  //   변환기 255개가 사라지는 대신 읽기 먹스 트리가 들어온다.
  parameter int FP32_OUT  = 1,
  parameter int STREAM_RD = 0,

  // 공유 변환기를 몇 개 둘지 (STREAM_RD=1 일 때만 의미가 있다).
  //
  // 1 이면 타일 전체가 변환기 하나를 공유한다. 면적은 가장 작지만
  // 사이클당 누산기 하나밖에 못 읽어 쓰기가 ACCS 사이클이 된다
  // (측정: 처리량 5.5~16.3배 손실).
  //
  // RD_LANES 개를 두면 **한 사이클에 그만큼** 읽는다. 출력이 행
  // 우선이라 DMA 비트 하나에 들어가는 WPB 개는 같은 행의 연속된
  // 열이고 그건 같은 타일 안에 있으므로, RD_LANES = WPB 면 비트
  // 하나를 한 사이클에 채운다.
  //
  // COLS 를 나누는 2의 거듭제곱이어야 한다. `rd_idx` 의 하위
  // log2(RD_LANES) 비트는 무시된다 (레인 0 이 그 그룹의 첫 열이다).
  parameter int RD_LANES  = 1
)(
  input  logic        clk,
  input  logic        rst_n,

  // Control
  input  logic        en,           // MAC enable (1 = accumulate this cycle)
  input  logic        acc_clr,      // clear accumulators to 0

  // Input: one column-step of A and B
  input  logic [ROWS*16-1:0] a_row,   // 16 BF16, a_row[16*r +: 16]
  input  logic [COLS*16-1:0] b_col,   // 16 BF16, b_col[16*c +: 16]

  // Output: 16×16 FP32 accumulator (FP32_OUT=0 이면 전부 0)
  // 행 우선 평탄화: 원소 (r,c) 는 acc_out[32*(r*COLS + c) +: 32]
  output logic [ROWS*COLS*32-1:0] acc_out,

  // 공유 변환기 순차 읽기 (STREAM_RD=1 일 때만 유효)
  input  logic [$clog2(ROWS*COLS)-1:0] rd_idx,
  output logic [RD_LANES*32-1:0]       rd_fp32,

  // Status
  output logic        busy
);

  assign busy = en;

  // ═══════════════════════════════════════════════════════════
  // 16×16 MAC array — 고정소수점 누산
  // ═══════════════════════════════════════════════════════════

  // 누산기는 모듈 레벨에 둔다. 공유 변환기가 인덱스로 고를 수 있어야 한다.
  logic signed [bf16_pkg::ACC_W-1:0] acc_fxp [0:ROWS-1][0:COLS-1];

  genvar r, c, ln;
  generate
    for (r = 0; r < ROWS; r = r + 1) begin : GEN_ROW
      for (c = 0; c < COLS; c = c + 1) begin : GEN_COL

        // 곱 → 고정소수점 증분 (조합)
        logic signed [bf16_pkg::ACC_W-1:0] incr;
        assign incr = bf16_pkg::bf16_mul_fxp(a_row[16*r +: 16],
                                             b_col[16*c +: 16]);

        // 누산기: 매 사이클 경로는 시프트 + 덧셈뿐이다.
        always_ff @(posedge clk or negedge rst_n) begin
          if (!rst_n)       acc_fxp[r][c] <= '0;
          else if (acc_clr) acc_fxp[r][c] <= '0;
          else if (en)      acc_fxp[r][c] <= bf16_pkg::fxp_add_sat(acc_fxp[r][c], incr);
        end

      end
    end
  endgenerate

  initial begin
    if (STREAM_RD != 0) begin
      if (RD_LANES < 1 || (COLS % RD_LANES) != 0)
        $fatal(1, "mxu_bf16_16x16: RD_LANES(%0d) 가 COLS(%0d) 를 나누지 않는다",
               RD_LANES, COLS);
      if ((RD_LANES & (RD_LANES - 1)) != 0)
        $fatal(1, "mxu_bf16_16x16: RD_LANES(%0d) 가 2의 거듭제곱이 아니다",
               RD_LANES);
    end
  end

  // ─── 출력 경로 ────────────────────────────────────────────────
  generate
    if (FP32_OUT != 0) begin : G_PER_PE_CONV
      // PE 마다 변환기. 기존 인터페이스.
      for (r = 0; r < ROWS; r = r + 1) begin : OR_
        for (c = 0; c < COLS; c = c + 1) begin : OC_
          assign acc_out[32*(r*COLS + c) +: 32] =
                     bf16_pkg::fxp_to_fp32(acc_fxp[r][c]);
        end
      end
    end else begin : G_NO_CONV
      for (r = 0; r < ROWS; r = r + 1) begin : TR_
        for (c = 0; c < COLS; c = c + 1) begin : TC_
          assign acc_out[32*(r*COLS + c) +: 32] = 32'd0;
        end
      end
    end
  endgenerate

  generate
    if (STREAM_RD != 0) begin : G_SHARED_CONV
      // 변환기 RD_LANES 개를 타일이 공유한다. 한 사이클에 같은 행의
      // 연속된 RD_LANES 열을 변환한다.
      localparam int RW = $clog2(ROWS);
      localparam int CW = $clog2(COLS);
      // 열 인덱스를 RD_LANES 경계로 내림한다. 그래야 먹스가 열
      // 그룹(COLS/RD_LANES 개)만 고르면 되고, 정렬이 안 맞은 인덱스가
      // 와도 조용히 틀리는 대신 그 그룹의 처음부터 읽는다.
      wire [RW-1:0] rd_r = rd_idx[RW+CW-1:CW];
      wire [CW-1:0] rd_c = rd_idx[CW-1:0] & ~(CW'(RD_LANES-1));

      for (ln = 0; ln < RD_LANES; ln = ln + 1) begin : G_LANE
        logic signed [bf16_pkg::ACC_W-1:0] lane_sel;
        assign lane_sel = acc_fxp[rd_r][rd_c + CW'(ln)];
        assign rd_fp32[32*ln +: 32] = bf16_pkg::fxp_to_fp32(lane_sel);
      end
    end else begin : G_NO_STREAM
      assign rd_fp32 = '0;
    end
  endgenerate

endmodule

`default_nettype wire
