// mxu_bf16_16x16.sv — BF16 16×16 Matrix Multiply Unit for ORBIT-G3
// SSOT: ORBIT_G3_RTL_PLAN.md (G3-RTL-003), ORBIT_G3_ARCHITECTURE.md
//
// Computes C[16×16] += A[16×K] × B[K×16] in BF16, accumulated in FP32.
// One K-step per cycle: 256 BF16 MACs/cycle.
//
// Interface:
//   - a_row[0:15]: 16 BF16 values (one row of A, column k)
//   - b_col[0:15]: 16 BF16 values (one column of B, row k)
//   - acc_out[0:15][0:15]: 16×16 FP32 accumulator (256 values)
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
  parameter int STREAM_RD = 0
)(
  input  logic        clk,
  input  logic        rst_n,

  // Control
  input  logic        en,           // MAC enable (1 = accumulate this cycle)
  input  logic        acc_clr,      // clear accumulators to 0

  // Input: one column-step of A and B
  input  logic [15:0] a_row [0:ROWS-1],   // 16 BF16 values
  input  logic [15:0] b_col [0:COLS-1],   // 16 BF16 values

  // Output: 16×16 FP32 accumulator (FP32_OUT=0 이면 전부 0)
  output logic [31:0] acc_out [0:ROWS-1][0:COLS-1],

  // 공유 변환기 순차 읽기 (STREAM_RD=1 일 때만 유효)
  input  logic [$clog2(ROWS*COLS)-1:0] rd_idx,
  output logic [31:0]                  rd_fp32,

  // Status
  output logic        busy
);

  assign busy = en;

  // ═══════════════════════════════════════════════════════════
  // 16×16 MAC array — 고정소수점 누산
  // ═══════════════════════════════════════════════════════════

  // 누산기는 모듈 레벨에 둔다. 공유 변환기가 인덱스로 고를 수 있어야 한다.
  logic signed [bf16_pkg::ACC_W-1:0] acc_fxp [0:ROWS-1][0:COLS-1];

  genvar r, c;
  generate
    for (r = 0; r < ROWS; r = r + 1) begin : GEN_ROW
      for (c = 0; c < COLS; c = c + 1) begin : GEN_COL

        // 곱 → 고정소수점 증분 (조합)
        logic signed [bf16_pkg::ACC_W-1:0] incr;
        assign incr = bf16_pkg::bf16_mul_fxp(a_row[r], b_col[c]);

        // 누산기: 매 사이클 경로는 시프트 + 덧셈뿐이다.
        always_ff @(posedge clk or negedge rst_n) begin
          if (!rst_n)       acc_fxp[r][c] <= '0;
          else if (acc_clr) acc_fxp[r][c] <= '0;
          else if (en)      acc_fxp[r][c] <= bf16_pkg::fxp_add_sat(acc_fxp[r][c], incr);
        end

      end
    end
  endgenerate

  // ─── 출력 경로 ────────────────────────────────────────────────
  generate
    if (FP32_OUT != 0) begin : G_PER_PE_CONV
      // PE 마다 변환기. 기존 인터페이스.
      for (r = 0; r < ROWS; r = r + 1) begin : OR_
        for (c = 0; c < COLS; c = c + 1) begin : OC_
          assign acc_out[r][c] = bf16_pkg::fxp_to_fp32(acc_fxp[r][c]);
        end
      end
    end else begin : G_NO_CONV
      for (r = 0; r < ROWS; r = r + 1) begin : TR_
        for (c = 0; c < COLS; c = c + 1) begin : TC_
          assign acc_out[r][c] = 32'd0;
        end
      end
    end
  endgenerate

  generate
    if (STREAM_RD != 0) begin : G_SHARED_CONV
      // 타일 전체가 변환기 하나를 공유한다.
      localparam int RW = $clog2(ROWS);
      localparam int CW = $clog2(COLS);
      logic signed [bf16_pkg::ACC_W-1:0] rd_sel;
      assign rd_sel  = acc_fxp[rd_idx[RW+CW-1:CW]][rd_idx[CW-1:0]];
      assign rd_fp32 = bf16_pkg::fxp_to_fp32(rd_sel);
    end else begin : G_NO_STREAM
      assign rd_fp32 = 32'd0;
    end
  endgenerate

endmodule

`default_nettype wire
