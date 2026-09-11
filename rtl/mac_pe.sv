// mac_pe.sv — 단일 곱셈-누산 셀
//
// 폭이 파라미터다. **기본값은 예전과 완전히 같다** (8×8 → 32비트):
// `mac_array` 는 파라미터를 주지 않으므로 한 비트도 달라지지 않는다.
// 셀 수로 확인한다 — docs/LOG.md 세션 10.
//
// 왜 파라미터화하는가: ORBIT-DR1 의 `update_unit` 이 Q1.15 외적 누산에
// 같은 셀을 쓴다 (CLAUDE.md 0절 "mac_array 를 새로 만들지 말고 진화시킨다").
// DR1 은 α(UQ1.15, 0x8000 까지)를 **부호 있는 17비트**로 넣기 때문에
// a 와 b 의 폭이 서로 다르다. 그래서 A_W/B_W 를 따로 둔다.
//
//   DR1 용:  mac_pe #(.A_W(17), .B_W(16), .ACC_W(40), .DUAL(1))
//   G2 용:   mac_pe                     (8, 8, 32, DUAL=0 — 기존)
//
// DUAL=1 은 **한 사이클에 곱 두 개**를 누산한다: acc += a*b + a2*b2.
// 왜 필요한가: 델타룰의 행 갱신은 `α·S[j] + β·err·k[j]` 로 **항이 정확히 2개**다.
// K 방향 누산이 없으므로 누산기를 두 사이클 돌릴 이유가 없다. DUAL=0 일 때는
// 두 번째 곱셈기가 아예 생성되지 않는다 — mac_array 의 셀 수는 그대로다.
`timescale 1ns/1ps
`default_nettype none

module mac_pe #(
  parameter int A_W   = 8,
  parameter int B_W   = 8,
  parameter int ACC_W = 32,
  parameter int DUAL  = 0    // 1 이면 a2*b2 도 같은 사이클에 더한다
)(
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    en,
  input  logic                    acc_clr,
  input  logic signed [A_W-1:0]   a,
  input  logic signed [B_W-1:0]   b,
  // 두 번째 곱 — DUAL=0 이면 **쓰이지 않는다**. 상위 모듈은 0 으로 묶으면 된다.
  input  logic signed [A_W-1:0]   a2,
  input  logic signed [B_W-1:0]   b2,
  output logic signed [ACC_W-1:0] acc
);

  localparam int P_W = A_W + B_W;   // 곱의 폭 (부호 있는 곱은 A_W+B_W 로 충분하다)

  // ----------------------------
  // multiply (explicit width)
  // ----------------------------
  logic signed [P_W-1:0] prod;
  always_comb begin
    prod = $signed(a) * $signed(b);
  end

  // 두 번째 곱. DUAL=0 이면 **곱셈기도 덧셈기도 만들지 않는다.**
  //
  // ⚠ 여기를 `addend` 중간 wire 하나로 합쳐 썼다가 **mac_array 가 196,352 →
  //   287,488 셀로 늘었다.** 논리적으로는 같은 식인데 yosys 가 다르게 접었다.
  //   셀 수 검사(scripts/synth_gate.sh)가 잡아냈다 — 그래서 DUAL=0 경로를
  //   **예전 코드 그대로** 두고, 누산 문장 자체를 generate 로 가른다.
  //   "같은 식이니 괜찮겠지" 가 통하지 않는 자리다 (docs/BUGS.md BUG-011).
  logic signed [P_W-1:0] prod2;
  generate
    if (DUAL != 0) begin : g_dual
      always_comb begin
        prod2 = $signed(a2) * $signed(b2);
      end
    end else begin : g_single
      always_comb begin
        prod2 = '0;
      end
      /* verilator lint_off UNUSEDSIGNAL */
      wire _unused_dual = &{1'b0, a2, b2, prod2, 1'b0};
      /* verilator lint_on UNUSEDSIGNAL */
    end
  endgenerate

`ifdef COCOTB_SIM
  initial acc = '0;
`endif

  // ----------------------------
  // accumulator
  // ----------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      acc <= '0;
    end
    else if (acc_clr) begin
      acc <= '0;
    end
 else if (en) begin
   acc <= acc + {{(ACC_W-P_W){prod[P_W-1]}}, prod}
              + (DUAL != 0 ? {{(ACC_W-P_W){prod2[P_W-1]}}, prod2} : '0);
 end
  end
endmodule

`default_nettype wire
