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
//   DR1 용:  mac_pe #(.A_W(17), .B_W(16), .ACC_W(40))
//   G2 용:   mac_pe                     (8, 8, 32 — 기존)
`timescale 1ns/1ps
`default_nettype none

module mac_pe #(
  parameter int A_W   = 8,
  parameter int B_W   = 8,
  parameter int ACC_W = 32
)(
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    en,
  input  logic                    acc_clr,
  input  logic signed [A_W-1:0]   a,
  input  logic signed [B_W-1:0]   b,
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
   acc <= acc + {{(ACC_W-P_W){prod[P_W-1]}}, prod};
 end
  end
endmodule

`default_nettype wire
