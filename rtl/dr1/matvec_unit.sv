// ============================================================================
// matvec_unit.sv — y = S · x   (행렬-벡터 곱)
//
// 대응 골든:  sim/golden/deltarule.py  matvec(S, x) -> (y_q15, sat_count)
//             비트 단위로 일치해야 한다. tb/tb_matvec_unit.py 가 무작위 500쌍으로 대조.
//
// SSOT: docs/DESIGN.md 5절 "matvec_unit   p = S·k, o = S·q (행 단위 순회, d 사이클)"
//       docs/DESIGN.md 3절 곱 Q2.30 / 누산 40비트 / requantize 1회
//
// ----------------------------------------------------------------------------
// 동작
// ----------------------------------------------------------------------------
//   행 r 마다:  acc = Σ_j S[r][j] · x[j]     (D개 곱을 한 사이클에, 40비트 누산)
//               y[r] = requant_q15(acc)
//
//   state_sram 의 읽기 지연이 1사이클이므로 2단 파이프라인이다:
//     사이클 c   : rd_row = r 요청
//     사이클 c+1 : rd_data = S[r] 도착 → 곱·누산·requant → y[r] 기록
//   즉 요청과 소비가 한 사이클 겹쳐서, D행을 D+1 사이클에 처리한다.
//
//   사이클 수: start 부터 done 까지 D+3 (D=16 이면 19). 계약 상한 D+4 이내.
//   cycles 출력은 마지막 실행의 실측값이다 (DR1_CYCLES 레지스터로 나간다).
//
// ----------------------------------------------------------------------------
// done 은 y_flat 이 유효해진 **다음** 사이클에 정확히 1펄스다.
// docs/DESIGN.md 5.1 완료 신호 계약: 폭은 정확히 1사이클, 레벨이 아니다.
// ----------------------------------------------------------------------------
`timescale 1ns/1ps
`default_nettype none

module matvec_unit #(
  parameter int D     = 16,
  parameter int W     = 16,
  parameter int ACC_W = 40
)(
  input  wire                 clk,
  input  wire                 rst_n,

  input  wire                 start,
  input  wire [D*W-1:0]       x_flat,     // 평탄화: 원소 i = bits[i*W +: W]

  // state_sram 읽기 포트를 **직접 구동**한다
  output logic                sram_rd_en,
  output logic [$clog2(D)-1:0] sram_rd_row,
  input  wire [D*W-1:0]       sram_rd_data,

  output logic [D*W-1:0]      y_flat,
  output logic [31:0]         sat_count,  // 이번 실행에서 발생한 포화 횟수
  output logic [31:0]         cycles,     // 이번 실행의 실측 사이클 수
  output logic                busy,
  output logic                done        // 정확히 1사이클 펄스
);

  localparam int AW = $clog2(D);
  localparam int PW = AW + 1;                       // 포인터 폭 (D 까지 세야 한다)
  localparam logic [PW-1:0] D_CNT = PW'(D);

  typedef enum logic [1:0] {
    ST_IDLE  = 2'd0,
    ST_ISSUE = 2'd1,   // 행 요청을 흘리면서 도착한 행을 소비
    ST_DRAIN = 2'd2,   // 마지막 행이 도착하기를 기다림
    ST_DONE  = 2'd3
  } state_t;

  state_t state, state_n;

  logic [PW-1:0] issue_ptr;    // 다음에 요청할 행 (D 까지 세므로 PW 비트)
  logic [PW-1:0] consume_ptr;  // 다음에 소비할 행
  logic        rd_pending;   // 직전 사이클에 요청했다 = 이번 사이클에 데이터가 있다

  // --------------------------------------------------------------------------
  // 곱·누산 — 도착한 행(sram_rd_data)과 x 의 내적
  // D개 곱을 한 사이클에 한다. D=16, W=16 이면 16개 16×16 곱 + 덧셈 트리다.
  // --------------------------------------------------------------------------
  logic signed [ACC_W-1:0] dot;

  always_comb begin
    dot = '0;
    for (int j = 0; j < D; j++) begin
      dot = dot + ACC_W'($signed(sram_rd_data[j*W +: W]) * $signed(x_flat[j*W +: W]));
    end
  end

  // requant 는 별도 모듈 — update_unit 도 같은 것을 쓴다 (반올림 규칙 단일 출처)
  logic signed [W-1:0] y_elem;
  logic                y_sat;

  requant_q15 #(.ACC_W(ACC_W), .FRAC(15)) u_rq (
    .acc (dot),
    .q15 (y_elem),
    .sat (y_sat)
  );

  // --------------------------------------------------------------------------
  // FSM
  // --------------------------------------------------------------------------
  always_comb begin
    state_n = state;
    case (state)
      ST_IDLE:  if (start) state_n = ST_ISSUE;
      ST_ISSUE: if (issue_ptr >= D_CNT) state_n = ST_DRAIN;   // 요청을 다 흘렸다
      ST_DRAIN: if (consume_ptr >= D_CNT) state_n = ST_DONE;  // 마지막 행까지 소비했다
      ST_DONE:  state_n = ST_IDLE;
      default:  state_n = ST_IDLE;
    endcase
  end

  assign busy       = (state != ST_IDLE);
  assign done       = (state == ST_DONE);
  assign sram_rd_en = (state == ST_ISSUE) && (issue_ptr < D_CNT);
  assign sram_rd_row = issue_ptr[AW-1:0];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= ST_IDLE;
      issue_ptr   <= '0;
      consume_ptr <= '0;
      rd_pending  <= 1'b0;
      y_flat      <= '0;
      sat_count   <= '0;
      cycles      <= '0;
    end else begin
      state <= state_n;

      // 사이클 카운터: start 에서 0 으로, busy 동안 증가
      if (state == ST_IDLE && start)      cycles <= 32'd1;
      else if (state != ST_IDLE)          cycles <= cycles + 32'd1;

      case (state)
        ST_IDLE: begin
          if (start) begin
            issue_ptr   <= '0;
            consume_ptr <= '0;
            rd_pending  <= 1'b0;
            sat_count   <= '0;
          end
        end

        ST_ISSUE, ST_DRAIN: begin
          // 요청을 흘린다
          if (sram_rd_en) issue_ptr <= issue_ptr + PW'(1);
          rd_pending <= sram_rd_en;

          // 직전 사이클에 요청한 행이 이번 사이클에 도착해 있다
          if (rd_pending) begin
            y_flat[consume_ptr[AW-1:0]*W +: W] <= y_elem;
            if (y_sat) sat_count <= sat_count + 32'd1;
            consume_ptr <= consume_ptr + PW'(1);
          end
        end

        ST_DONE: ;
        default: ;
      endcase
    end
  end

endmodule

`default_nettype wire
