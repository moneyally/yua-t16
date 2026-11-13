// rd_bench.sv — 공유 변환기 읽기 경로 면적 측정 (합성 전용)
//
// Yosys 0.33 이 mxu_bf16_16x16 의 unpacked 배열 포트를 파싱하지 못해
// 타일 전체는 합성할 수 없다. 대신 "공유 읽기 모드에서 새로 들어오는
// 비용" 만 따로 잰다: 256:1 먹스 트리 + 변환기 1개.
// PE 측 절감(변환기 255개 제거)은 pe_bench 로 이미 측정했다.
`timescale 1ns/1ps
`default_nettype none

module rd_bench #(
  parameter int N = 16                       // N×N 누산기
)(
  input  logic [N*N*48-1:0] acc_flat_in,     // 평탄화된 고정소수점 누산기
  input  logic [$clog2(N*N)-1:0] rd_idx,
  output logic [31:0]       rd_fp32
);
  logic signed [47:0] sel;
  assign sel     = $signed(acc_flat_in[rd_idx*48 +: 48]);
  assign rd_fp32 = bf16_pkg::fxp_to_fp32(sel);
endmodule

// 변환기 단독 (먹스 제외)
module conv_bench (
  input  logic [47:0] acc_in,
  output logic [31:0] fp32_out
);
  assign fp32_out = bf16_pkg::fxp_to_fp32($signed(acc_in));
endmodule

`default_nettype wire
