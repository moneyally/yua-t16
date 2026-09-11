// ============================================================================
// update_unit.sv — S_row_new = α·S_row + β·err_i·kᵀ   (상태 한 행 갱신)
//
// 대응 골든:  sim/golden/deltarule.py  update_row(S_row, alpha, beta, err_i, k)
//                                       -> (row_q15, sat_count)
//             비트 단위로 일치해야 한다. tb/tb_update_unit.py 가 300세트로 대조.
//
// SSOT: docs/DESIGN.md 3절 — "α·S 와 β·err·kᵀ 는 **Q2.30 누산기에서 합산한 뒤
//       1회만** 재양자화한다". 항마다 재양자화하면 오차가 1 LSB 를 넘는다
//       (실측 1.06 → 0.63 LSB).
//
// ----------------------------------------------------------------------------
// 구조 — 행 하나를 D개 PE 가 동시에 처리한다 (CLAUDE.md 0절: mac_array 를
// 새로 만들지 말고 진화시킨다. 여기 쓰는 셀은 기존 `rtl/mac_pe.sv` 다)
// ----------------------------------------------------------------------------
//   열 j 마다 PE 하나:
//     MAC1:  acc_j  = α      · S_row[j]     (a=17비트 부호 있음, b=Q1.15)
//     MAC2:  acc_j += berr   · k[j]
//     CAP :  row[j] = requant_q15(acc_j)    ← 재양자화는 여기 한 번뿐
//
//   berr = requant_q15(β · err_i)  — 골든의 q15_mul(β, err) 과 같은 조각.
//   requant_q15 를 여기서도 쓰기 때문에 반올림 규칙은 여전히 한 곳에만 있다.
//
//   누산기 포화: |α·S| ≤ 2^30, |berr·k| ≤ 2^30 이므로 합은 2^31 미만이다.
//   40비트 누산기(±2^39)는 **구조적으로 넘칠 수 없다.** 골든의 _sat_acc 도
//   같은 이유로 절대 발동하지 않는다 — 그래서 sat_count 는 berr 재양자화 1회와
//   행 재양자화 D회에서만 나온다.
//
//   사이클: start 부터 done 까지 **5** (D 와 무관). 계약 상한 D+6 이내.
//   D 와 무관한 이유는 열 방향을 펼쳤기 때문이다. 행 방향(D행)은 dr1_top 이
//   돌린다 — W7 에서 "행 스트리밍" 이 된다.
//
// ----------------------------------------------------------------------------
// α, β 는 **이미 [0, 0x8000] 로 정리되어 들어온다.**
// 0x8000 초과 클램프와 CLAMP_EVENT 는 디스크립터를 해석하는 dr1_top 의 일이다
// (spec/deltarule.md 1절). 같은 규칙을 두 곳에 두지 않는다 — BUG-006 교훈.
// ----------------------------------------------------------------------------
`timescale 1ns/1ps
`default_nettype none

module update_unit #(
  parameter int D     = 16,
  parameter int W     = 16,
  parameter int ACC_W = 40
)(
  input  wire                 clk,
  input  wire                 rst_n,

  input  wire                 start,
  input  wire [15:0]          alpha_uq15,   // UQ1.15, 0x8000 = 1.0
  input  wire [15:0]          beta_uq15,    // UQ1.15
  input  wire signed [W-1:0]  err_i,        // Q1.15, 행 i 의 err
  input  wire [D*W-1:0]       s_row_flat,   // 평탄화: 원소 j = bits[j*W +: W]
  input  wire [D*W-1:0]       k_flat,

  output logic [D*W-1:0]      row_flat,     // 갱신된 행
  output logic [31:0]         sat_count,    // 이번 실행의 포화 횟수
  output logic [31:0]         cycles,       // 이번 실행의 실측 사이클 수
  output logic                busy,
  output logic                done          // 정확히 1사이클 펄스
);

  typedef enum logic [2:0] {
    ST_IDLE = 3'd0,
    ST_MAC1 = 3'd1,   // acc  = α · S_row
    ST_MAC2 = 3'd2,   // acc += berr · k
    ST_CAP  = 3'd3,   // requant + 기록
    ST_DONE = 3'd4
  } state_t;

  state_t state, state_n;

  // --------------------------------------------------------------------------
  // 입력 래치 — start 사이클에 잡고, 이후 PE 는 래치된 값만 본다
  // --------------------------------------------------------------------------
  logic [15:0]        alpha_r, beta_r;
  logic signed [W-1:0] err_r;
  logic [D*W-1:0]     s_row_r, k_r;

  // --------------------------------------------------------------------------
  // berr = requant_q15(β · err)   — 골든 q15_mul(β, err_i)
  // β 는 부호 없는 UQ1.15 이므로 부호 있는 17비트로 확장해서 곱한다.
  // --------------------------------------------------------------------------
  wire signed [16:0]       beta_s   = $signed({1'b0, beta_r});
  wire signed [ACC_W-1:0]  berr_acc = ACC_W'(beta_s * err_r);

  logic signed [W-1:0] berr;
  logic                berr_sat;

  requant_q15 #(.ACC_W(ACC_W), .FRAC(15)) u_rq_berr (
    .acc (berr_acc),
    .q15 (berr),
    .sat (berr_sat)
  );

  // --------------------------------------------------------------------------
  // PE 배열 — 열 j 하나에 PE 하나. a 는 방송, b 는 열별.
  // --------------------------------------------------------------------------
  wire signed [16:0] alpha_s  = $signed({1'b0, alpha_r});
  wire signed [16:0] berr_ext = $signed({berr[W-1], berr});   // 부호 확장 16 -> 17

  wire signed [16:0] pe_a = (state == ST_MAC1) ? alpha_s : berr_ext;
  wire               pe_en      = (state == ST_MAC1) || (state == ST_MAC2);
  wire               pe_acc_clr = (state == ST_IDLE);

  wire signed [ACC_W-1:0] pe_acc [0:D-1];
  logic signed [W-1:0]    row_elem [0:D-1];
  logic                   row_sat  [0:D-1];

  genvar j;
  generate
    for (j = 0; j < D; j = j + 1) begin : g_pe
      wire signed [W-1:0] s_j = $signed(s_row_r[j*W +: W]);
      wire signed [W-1:0] k_j = $signed(k_r    [j*W +: W]);
      wire signed [W-1:0] pe_b = (state == ST_MAC1) ? s_j : k_j;

      mac_pe #(.A_W(17), .B_W(W), .ACC_W(ACC_W)) u_pe (
        .clk     (clk),
        .rst_n   (rst_n),
        .en      (pe_en),
        .acc_clr (pe_acc_clr),
        .a       (pe_a),
        .b       (pe_b),
        .acc     (pe_acc[j])
      );

      // 재양자화는 열마다 1회 — 여기 말고 다른 곳에서 내리지 않는다
      requant_q15 #(.ACC_W(ACC_W), .FRAC(15)) u_rq (
        .acc (pe_acc[j]),
        .q15 (row_elem[j]),
        .sat (row_sat[j])
      );
    end
  endgenerate

  // 이번 행에서 포화한 열의 개수
  logic [31:0] row_sat_n;
  always_comb begin
    row_sat_n = '0;
    for (int i = 0; i < D; i++) begin
      row_sat_n = row_sat_n + (row_sat[i] ? 32'd1 : 32'd0);
    end
  end

  // --------------------------------------------------------------------------
  // FSM
  // --------------------------------------------------------------------------
  always_comb begin
    state_n = state;
    case (state)
      ST_IDLE: if (start) state_n = ST_MAC1;
      ST_MAC1: state_n = ST_MAC2;
      ST_MAC2: state_n = ST_CAP;
      ST_CAP:  state_n = ST_DONE;
      ST_DONE: state_n = ST_IDLE;
      default: state_n = ST_IDLE;
    endcase
  end

  assign busy = (state != ST_IDLE);
  assign done = (state == ST_DONE);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state     <= ST_IDLE;
      alpha_r   <= '0;
      beta_r    <= '0;
      err_r     <= '0;
      s_row_r   <= '0;
      k_r       <= '0;
      row_flat  <= '0;
      sat_count <= '0;
      cycles    <= '0;
    end else begin
      state <= state_n;

      // 사이클 카운터 — matvec_unit 과 같은 규약 (start 수락 사이클이 1)
      if (state == ST_IDLE && start) cycles <= 32'd1;
      else if (state != ST_IDLE)     cycles <= cycles + 32'd1;

      case (state)
        ST_IDLE: begin
          if (start) begin
            alpha_r <= alpha_uq15;
            beta_r  <= beta_uq15;
            err_r   <= err_i;
            s_row_r <= s_row_flat;
            k_r     <= k_flat;
          end
        end

        ST_CAP: begin
          for (int i = 0; i < D; i++) begin
            row_flat[i*W +: W] <= row_elem[i];
          end
          // 골든과 같은 셈: berr 재양자화 1회 + 행 재양자화 D회
          sat_count <= row_sat_n + (berr_sat ? 32'd1 : 32'd0);
        end

        default: ;
      endcase
    end
  end

endmodule

`default_nettype wire
