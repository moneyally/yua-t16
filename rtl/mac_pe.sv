`timescale 1ns/1ps
`default_nettype none

// mac_pe.sv — INT8 곱 + 32비트 누산 PE.
//
// ── 임계 경로 ────────────────────────────────────────────────────
//
// 이 설계의 Fmax 를 정하는 것은 **누산기의 캐리 체인**이다. 실측으로
// 확인했다 (ECP5-85K, nextpnr 타이밍 DB):
//
//   경로: acc FF.Q → CCU2C 캐리 체인(32비트) → acc FF.D
//
// 32비트 덧셈이 피드백 루프 안에 있으므로 체인 길이가 그대로 주기에
// 들어간다. `ACC_SPLIT=1` 이 그것을 반으로 줄인다.
module mac_pe #(
  // 누산기 캐리 체인 분할.
  //
  //   0: 32비트 덧셈을 한 번에 한다. 캐리 체인 32비트가 루프 안에 있다.
  //   1: 16비트 두 덩이로 쪼갠다. 루프 안의 체인이 16비트가 된다.
  //
  // 분할의 핵심은 **최종 합산을 읽기 경로로 옮기는 것**이다. 상위
  // 자리올림을 한 사이클 늦게 더하되, 출력에서 밀린 몫을 조합으로
  // 합쳐 준다. 그래서 지연(latency) 이 바뀌지 않고 FSM 을 손대지
  // 않아도 된다 — `acc` 는 예전과 똑같이 매 사이클 올바른 누적값이다.
  //
  // 오버플로 동작도 같다 (둘 다 2^32 모듈로 감싼다).
  parameter int ACC_SPLIT = 0,

  // 곱셈기 출력을 레지스터에 받는다 (파이프라인 1단 추가).
  //
  // **이것이 Fmax 의 지배 항목이다.** 실측 임계 경로 (ECP5-85K):
  //
  //   피연산자 FF → 배선 1.36 → MULT18X18D 조합 3.93ns
  //                → 배선 1.39 → 32비트 CCU2C 체인 → acc FF
  //   합계 6.53ns 논리 + 3.87ns 배선 → 101 MHz
  //
  // 곱셈과 덧셈이 한 사이클에 직렬로 들어 있다. 그래서 누산기만
  // 쪼개면(ACC_SPLIT) 2.6% 밖에 오르지 않는다 (101.09 → 103.69 MHz).
  // 둘 사이에 레지스터를 넣으면 경로가 반으로 갈린다.
  //
  // 대가: 누산 결과가 한 사이클 늦게 확정된다. `en` 펄스 수는 그대로
  // Kt 개이지만, 마지막 곱이 누산기에 들어가기까지 한 사이클이 더
  // 걸린다. `gemm_core` 의 drain(C_POST) 을 그만큼 늘려야 한다.
  parameter int MUL_REG = 0
)(
  input  logic              clk,
  input  logic              rst_n,
  input  logic              en,
  input  logic              acc_clr,
  input  logic signed [7:0]  a,
  input  logic signed [7:0]  b,
  output logic signed [31:0] acc
);

  // ----------------------------
  // multiply (explicit width)
  // ----------------------------
  logic signed [15:0] prod_c;
  always_comb begin
    prod_c = $signed(a) * $signed(b);
  end

  // 파이프라인 단계. MUL_REG=1 이면 곱과 제어를 한 사이클 늦춘다 —
  // 제어도 같이 늦춰야 누산기가 보는 것이 일관된다 (en 만 늦추고
  // acc_clr 을 그대로 두면, 날아오던 곱이 클리어에 지워진다).
  logic signed [15:0] prod;
  logic               en_q, clr_q;

  generate
  if (MUL_REG == 0) begin : G_MUL_COMB
    assign prod  = prod_c;
    assign en_q  = en;
    assign clr_q = acc_clr;
  end else begin : G_MUL_REG
    logic signed [15:0] prod_r;
    logic               en_r, clr_r;
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        prod_r <= 16'sd0;
        en_r   <= 1'b0;
        clr_r  <= 1'b0;
      end else begin
        prod_r <= prod_c;
        en_r   <= en;
        clr_r  <= acc_clr;
      end
    end
    assign prod  = prod_r;
    assign en_q  = en_r;
    assign clr_q = clr_r;
  end
  endgenerate

  generate
  if (ACC_SPLIT == 0) begin : G_ACC32

    logic signed [31:0] acc_r;

`ifdef COCOTB_SIM
    initial acc_r = 32'sd0;
`endif

    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n)      acc_r <= 32'sd0;
      else if (clr_q)  acc_r <= 32'sd0;
      else if (en_q)   acc_r <= acc_r + {{16{prod[15]}}, prod};
    end

    assign acc = acc_r;

  end else begin : G_ACC_SPLIT

    // 하위 16비트는 부호 없이 더하고 자리올림을 따로 뽑는다.
    logic [15:0] acc_lo;
    logic [15:0] acc_hi;
    logic        carry_r;   // 직전 사이클의 하위 자리올림
    logic [15:0] hi_r;      // 직전 사이클의 상위 증분 (부호 확장분)

    wire [16:0] lo_sum = {1'b0, acc_lo} + {1'b0, prod[15:0]};

`ifdef COCOTB_SIM
    initial begin
      acc_lo  = 16'd0;
      acc_hi  = 16'd0;
      carry_r = 1'b0;
      hi_r    = 16'd0;
    end
`endif

    // 비동기 리셋과 동기 클리어를 **같은 조건에 합치면 안 된다.**
    // `if (!rst_n || acc_clr)` 로 썼더니 Yosys 가 비동기 리셋을 알아보지
    // 못하고 "Multiple edge sensitive events found for this signal" 로
    // 멈췄다. 시뮬레이션 동작은 같지만 합성이 안 된다.
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        acc_lo  <= 16'd0;
        acc_hi  <= 16'd0;
        carry_r <= 1'b0;
        hi_r    <= 16'd0;
      end else if (clr_q) begin
        acc_lo  <= 16'd0;
        acc_hi  <= 16'd0;
        carry_r <= 1'b0;
        hi_r    <= 16'd0;
      end else if (en_q) begin
        acc_lo  <= lo_sum[15:0];
        carry_r <= lo_sum[16];
        hi_r    <= {16{prod[15]}};
        // 한 사이클 밀린 몫만 누적한다. 루프 안의 체인은 16비트다.
        acc_hi  <= acc_hi + hi_r + {15'd0, carry_r};
      end
    end

    // 아직 누적되지 않은 마지막 몫을 **읽기 경로에서** 합친다.
    // 이 덧셈은 피드백 루프 밖이므로 Fmax 에 들어가지 않는다.
    assign acc = $signed({acc_hi + hi_r + {15'd0, carry_r}, acc_lo});

  end
  endgenerate

endmodule

`default_nettype wire
