// ============================================================================
// err_unit.sv — err = v − α·p   (원소별, D개 동시)
//
// 대응 골든:  sim/golden/deltarule.py  compute_err(v, p, alpha_uq15)
//                                       -> (err_q15, sat_count)
//             비트 단위로 일치해야 한다. tb/tb_err_unit.py 가 대조한다.
//
// SSOT: docs/DESIGN.md 2절 — **err = v − α·p** 다. `err = v − p` 는 α=1 일 때만 같다.
//       3절 — 곱은 Q2.30, 재양자화 round-half-to-even, 포화는 센다.
//
// ----------------------------------------------------------------------------
// 포화가 **두 번** 날 수 있다. 골든과 같은 순서로 세야 한다:
//   1) ap = requant_q15(α · p[i])    ← 재양자화 포화
//   2) err[i] = sat(v[i] − ap)       ← 뺄셈 포화
// 둘 다 각각 1회로 센다 (골든의 q15_mul + sat_q15 와 1:1).
//
// 순수 조합 논리다. 레지스터는 부르는 쪽(dr1_top)이 둔다 — matvec/update 처럼
// FSM 을 따로 두기에는 일이 1사이클짜리다.
// ============================================================================
`timescale 1ns/1ps
`default_nettype none

module err_unit #(
  parameter int D     = 16,
  parameter int W     = 16,
  parameter int ACC_W = 40
)(
  input  wire [15:0]      alpha_uq15,   // UQ1.15, 0x8000 = 1.0. 클램프는 dr1_top 이 이미 했다
  input  wire [D*W-1:0]   v_flat,       // 평탄화: 원소 i = bits[i*W +: W]
  input  wire [D*W-1:0]   p_flat,

  output logic [D*W-1:0]  err_flat,
  output logic [31:0]     sat_count     // 이 계산에서 발생한 포화 횟수
);

  // α 는 부호 없는 UQ1.15 → 부호 있는 17비트로 확장해서 곱한다 (0x8000 이 양수로 남는다)
  wire signed [16:0] alpha_s = $signed({1'b0, alpha_uq15});

  localparam signed [W:0] Q15_MAX_EXT = 17'sd32767;
  localparam signed [W:0] Q15_MIN_EXT = -17'sd32768;

  logic [31:0] sat_sum;

  // 포화 비트를 평평한 벡터로 모은다. generate 블록 안의 신호를 루프 변수로
  // 인덱싱할 수 없기 때문이다 (`g_elem[j].x` 는 상수 인덱스만 허용).
  // 원소 i 의 재양자화 포화 = sat_bits[2*i], 뺄셈 포화 = sat_bits[2*i+1].
  logic [2*D-1:0] sat_bits;

  genvar i;
  generate
    for (i = 0; i < D; i = i + 1) begin : g_elem
      wire signed [W-1:0] v_i = $signed(v_flat[i*W +: W]);
      wire signed [W-1:0] p_i = $signed(p_flat[i*W +: W]);

      // 1) ap = requant_q15(α · p)  — 반올림 규칙은 여기서도 같은 모듈 하나뿐이다
      wire signed [ACC_W-1:0] ap_acc = ACC_W'(alpha_s * p_i);
      wire signed [W-1:0]     ap;
      wire                    ap_sat;

      requant_q15 #(.ACC_W(ACC_W), .FRAC(15)) u_rq (
        .acc (ap_acc),
        .q15 (ap),
        .sat (ap_sat)
      );

      // 2) err = sat(v − ap)  — 17비트로 빼고 Q1.15 로 포화
      wire signed [W:0] diff = {v_i[W-1], v_i} - {ap[W-1], ap};
      wire over  = (diff > Q15_MAX_EXT);
      wire under = (diff < Q15_MIN_EXT);

      assign err_flat[i*W +: W] = over  ? 16'sh7FFF
                                : under ? 16'sh8000
                                        : diff[W-1:0];

      wire sub_sat = over | under;
      assign sat_bits[2*i]     = ap_sat;
      assign sat_bits[2*i + 1] = sub_sat;
    end
  endgenerate

  always_comb begin
    sat_sum = '0;
    for (int j = 0; j < 2*D; j++) begin
      sat_sum = sat_sum + (sat_bits[j] ? 32'd1 : 32'd0);
    end
  end

  assign sat_count = sat_sum;

endmodule

`default_nettype wire
