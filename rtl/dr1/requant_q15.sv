// ============================================================================
// requant_q15.sv — Q(x).FRAC 누산기 → Q1.15 재양자화
//
// 대응 골든 함수:  sim/golden/deltarule.py  requantize_q15()
//   비트 단위로 일치해야 한다. tb/tb_requant_q15.py 가 무작위 1만 개로 대조한다.
//
// SSOT: docs/DESIGN.md 3절
//   - round-half-to-even (반올림 방식 고정)
//   - 오버플로는 포화(saturate), 발생을 알린다 (트레이스 링 SAT_EVENT)
//
// 순수 조합 논리다. matvec_unit 과 update_unit 이 둘 다 이 모듈을 쓴다 —
// 반올림 규칙이 두 곳에 있으면 갈라진다.
// ============================================================================
`timescale 1ns/1ps
`default_nettype none

module requant_q15 #(
  parameter int ACC_W = 40,   // 누산기 폭 (DESIGN.md 3절: 40비트)
  parameter int FRAC  = 15    // 내릴 비트 수 (Q2.30 → Q1.15)
)(
  input  wire signed [ACC_W-1:0] acc,
  output wire signed [15:0]      q15,
  output wire                    sat    // 1 이면 포화가 일어났다 = SAT_EVENT
);

  localparam signed [ACC_W-1:0] Q15_MAX_EXT = 40'sd32767;
  localparam signed [ACC_W-1:0] Q15_MIN_EXT = -40'sd32768;

  // --------------------------------------------------------------------------
  // round-half-to-even
  //   q    = acc >>> FRAC              (산술 시프트 = floor)
  //   rem  = acc 의 하위 FRAC 비트     (항상 0 이상)
  //   rem > half            → +1
  //   rem == half && q 홀수 → +1  (짝수로 맞춘다)
  //   그 외                  → 그대로
  // 음수에서도 골든과 같은 답이 나온다: 파이썬 `>>` 도 floor 이고 나머지가
  // 음이 아니기 때문이다. 예: -1.5 → -2, -2.5 → -2.
  // --------------------------------------------------------------------------
  wire signed [ACC_W-1:0] q_floor = acc >>> FRAC;
  wire        [FRAC-1:0]  rem     = acc[FRAC-1:0];
  localparam  [FRAC-1:0]  HALF    = {1'b1, {(FRAC-1){1'b0}}};

  wire round_up = (rem > HALF) || ((rem == HALF) && q_floor[0]);
  wire signed [ACC_W-1:0] rounded = q_floor + (round_up ? 40'sd1 : 40'sd0);

  // --------------------------------------------------------------------------
  // 포화
  // --------------------------------------------------------------------------
  wire over  = (rounded > Q15_MAX_EXT);
  wire under = (rounded < Q15_MIN_EXT);

  assign sat = over | under;
  assign q15 = over  ? 16'sh7FFF
             : under ? 16'sh8000
             :         rounded[15:0];

endmodule

`default_nettype wire
