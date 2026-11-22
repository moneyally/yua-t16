`timescale 1ns/1ps
`default_nettype none

// mac_array.sv — N×N INT8 MAC 배열 (외적 누산)
//
// 사이클마다 a_row (N개) 와 b_col (N개) 의 외적을 받아 N² 누산기에 더한다.
// Kt 사이클을 돌면 A(N×Kt) · B(Kt×N) 의 N×N 결과가 나온다.
//
// N 을 키우면 MAC 은 N² 로, 피연산자 바이트는 N 으로 늘어난다. 즉 산술
// 강도가 N 에 비례한다 — 대역폭 대비 연산을 늘리는 정석 경로다.
// spec/memory-bandwidth-analysis.md 참조.
module mac_array #(
  parameter int N = 16,
  // 누산기 캐리 체인 분할. 자세한 설명은 mac_pe.sv 에 있다.
  // 이 설계의 Fmax 를 정하는 경로이므로 여기서 바꿔 A/B 할 수 있게 둔다.
  parameter int ACC_SPLIT = 0,
  // 곱셈기 출력 레지스터. Fmax 의 지배 항목이다 (mac_pe.sv 참고).
  parameter int MUL_REG = 0
)(
  input  logic clk,
  input  logic rst_n,
  input  logic en,
  input  logic acc_clr,
  // 평탄화된 피연산자. a_row[8*i +: 8] 가 i 번째 값이다.
  //
  // 전에는 unpacked 배열 포트였다. Yosys 가 그 선언을 파싱하지 못해
  // (`mxu_bf16_16x16` 과 같은 한계) **`gemm_core` 를 합성할 수 없었다.**
  // 그래서 이 설계의 클럭 주파수가 한 번도 측정되지 않았고, 모든
  // 처리량 수치(MAC/사이클) 를 TOPS 로 바꿀 수 없었다.
  input  logic [N*8-1:0] a_row,
  input  logic [N*8-1:0] b_col,
  // Packed flat output: acc_out_flat[IDX*32 +: 32] = result for
  // row=IDX/N, col=IDX%N.
  // Exported as a packed vector to work around the Icarus 12
  // unpacked-array-output limitation.
  output logic [32*N*N-1:0] acc_out_flat
);

  genvar i, j;
  generate
    for (i = 0; i < N; i++) begin : ROW
      for (j = 0; j < N; j++) begin : COL
        localparam int IDX = (i * N) + j;
        localparam int LSB = IDX * 32;

        mac_pe #(.ACC_SPLIT(ACC_SPLIT), .MUL_REG(MUL_REG)) u_pe (
          .clk     (clk),
          .rst_n   (rst_n),
          .en      (en),
          .acc_clr (acc_clr),
          .a       ($signed(a_row[8*i +: 8])),
          .b       ($signed(b_col[8*j +: 8])),
          .acc     (acc_out_flat[LSB +: 32])
        );
      end
    end
  endgenerate

endmodule

`default_nettype wire
