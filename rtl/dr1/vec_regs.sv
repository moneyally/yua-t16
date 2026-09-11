// ============================================================================
// vec_regs.sv — DR1 입력 벡터 레지스터 (q, k, v — 각 D×W)
//
// 대응 골든:  sim/golden/deltarule.py 의 인자 q, k, v 그 자체.
//             평탄화 규칙은 spec/deltarule.md 3.5절 = tools/orbit_pack.py pack_vec.
//               원소 i = bits[i*W +: W]   (i=0 이 최하위)
//
// SSOT: docs/DESIGN.md 5절 "vec_regs   q, k, v, err (각 d×16bit)"
//       (err 는 dr1_top 쪽에 두었다 — 계산 결과라서 적재 포트가 필요 없다)
//
// 적재는 한 원소씩 한다. 호스트가 스크래치를 채우고 dr1_top 이 여기로 옮기는
// 구조이므로, 폭 넓은 버스를 끌어오는 대신 인덱스로 순차 적재한다.
// ============================================================================
`timescale 1ns/1ps
`default_nettype none

module vec_regs #(
  parameter int D = 16,
  parameter int W = 16
)(
  input  wire                 clk,
  input  wire                 rst_n,

  // 적재 포트
  input  wire                 ld_en,
  input  wire [1:0]           ld_sel,    // SEL_Q / SEL_K / SEL_V
  input  wire [$clog2(D)-1:0] ld_idx,
  input  wire [W-1:0]         ld_data,

  // 평탄화 출력 — 원소 i = bits[i*W +: W]
  output logic [D*W-1:0]      q_flat,
  output logic [D*W-1:0]      k_flat,
  output logic [D*W-1:0]      v_flat
);

  // ld_sel 인코딩. tb/tb_vec_regs.py 와 dr1_top 이 같은 값을 쓴다.
  localparam logic [1:0] SEL_Q = 2'd0;
  localparam logic [1:0] SEL_K = 2'd1;
  localparam logic [1:0] SEL_V = 2'd2;

  // 평탄화 벡터를 그대로 레지스터로 둔다. 부분 선택 대입으로 원소 하나만 쓴다 —
  // `flat[idx*W +: W] <= data` 는 합성 가능하다 (가변 오프셋 부분 선택).
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      q_flat <= '0;
      k_flat <= '0;
      v_flat <= '0;
    end else if (ld_en) begin
      case (ld_sel)
        SEL_Q:   q_flat[ld_idx*W +: W] <= ld_data;
        SEL_K:   k_flat[ld_idx*W +: W] <= ld_data;
        SEL_V:   v_flat[ld_idx*W +: W] <= ld_data;
        default: ;   // 2'd3 은 미사용 — 아무것도 하지 않는다
      endcase
    end
  end

endmodule

`default_nettype wire
