// fmax_harness.sv — 실제 배치배선으로 임계 경로 주파수를 재기 위한 래퍼.
//                   (합성/P&R 전용, 기능 검증용이 아니다)
//
// ── 왜 필요한가 ──────────────────────────────────────────────────
//
// DUT 의 포트를 다 꺼내면 I/O 가 수백~수천 개가 된다 (`mac_array` 는
// acc_out_flat 이 32·N² 비트다). ECP5-85K 의 사용자 I/O 는 365개라
// 배치가 안 된다. 그런데 Fmax 를 재려면 실제 배치배선을 통과해야 한다.
//
// 그래서 표준 기법을 쓴다.
//
//   입력: LFSR 로 만든다. 상수가 아니므로 합성이 내부 논리를 상수
//         접기로 지워버리지 않는다.
//   출력: XOR 로 접어 1비트로 만든다. 모든 출력이 살아 있어야 하므로
//         내부 논리가 제거되지 않는다.
//
// ── 접기가 임계 경로가 되면 안 된다 ─────────────────────────────
//
// 처음에 N=16 (256 PE) 을 한 사이클에 접었다가 측정을 망쳤다.
// 256-way XOR 은 LUT4 로 4단이고 일반 배선을 타므로
// 4 × (논리 0.45 + 배선 ~0.5) ≈ 3.8ns 다. 반면 재려는 32비트 캐리
// 체인은 **전용 캐리 배선**을 써서 ~1.1ns 다. 즉 하네스가 임계 경로가
// 되어 DUT 가 아니라 내 XOR 트리의 주파수를 재게 된다.
//
// 그래서 A/B 는 N 을 작게 둔 구성(N=4, N=8) 으로 한다. 누산기 경로는
// PE 안에 있어 N 과 무관하고, N 이 커질 때 더해지는 것은 배선 혼잡뿐
// 이므로 N=4 와 N=8 을 비교하면 그 혼잡분까지 보인다.
//
// ── 이 수치가 말하는 것과 말하지 않는 것 ─────────────────────────
//
// 말하는 것: 이 RTL 을 ECP5-85K 에 올렸을 때의 임계 경로 주파수와
//            자원 사용량. 둘 다 실제 타이밍 데이터베이스 기준이다.
// 말하지 않는 것: LFSR 이 제어 입력을 무작위로 흔들므로 **기능은
//            의미가 없다.** 기능 검증은 cocotb 회귀가 한다.
//            디바이스가 다르면 (Versal DSP58 은 누산기가 내장이고
//            600MHz 급이다) 숫자가 달라진다.
`timescale 1ns/1ps
`default_nettype none

module fmax_mac_array #(
  parameter int N = 4,
  parameter int ACC_SPLIT = 0
)(
  input  logic clk,
  input  logic rst_n,
  input  logic seed,
  output logic probe
);
  // 64비트 LFSR (x^64+x^63+x^61+x^60+1). seed 로 상수 접기를 막는다.
  logic [63:0] lfsr;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) lfsr <= 64'h0123_4567_89ab_cdef;
    else        lfsr <= {lfsr[62:0],
                         lfsr[63] ^ lfsr[62] ^ lfsr[60] ^ lfsr[59] ^ seed};
  end

  // 피연산자는 위상이 다른 비트열로 채운다.
  logic [N*8-1:0] a_r, b_r;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      a_r <= '0;
      b_r <= '0;
    end else begin
      a_r <= {a_r[N*8-2:0], lfsr[63]};
      b_r <= {b_r[N*8-2:0], lfsr[31]};
    end
  end

  logic [32*N*N-1:0] acc;

  mac_array #(.N(N), .ACC_SPLIT(ACC_SPLIT)) u_dut (
    .clk(clk), .rst_n(rst_n),
    .en(lfsr[0]), .acc_clr(lfsr[1]),
    .a_row(a_r), .b_col(b_r),
    .acc_out_flat(acc)
  );

  // 접기: 조합 XOR 한 단계 뒤 레지스터, 다시 1비트.
  logic [31:0] fold_c;
  always_comb begin
    fold_c = '0;
    for (int w = 0; w < N*N; w++)
      fold_c = fold_c ^ acc[32*w +: 32];
  end

  logic [31:0] fold_r;
  logic        probe_r;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      fold_r  <= '0;
      probe_r <= 1'b0;
    end else begin
      fold_r  <= fold_c;
      probe_r <= ^fold_r;
    end
  end
  assign probe = probe_r;
endmodule

// ── PE 하나만 재는 하네스 (누산기 A/B 의 본체) ──────────────────
//
// 배열을 재려면 N² 개 누산기를 접어야 하는데, 그 접기가 임계 경로가
// 되기 쉽다. 실제로 그렇게 당했다 — `for` 루프로 쓴 XOR 축약이 64단
// **직렬 체인**으로 합성되어 N=8 에서 30 MHz 가 나왔고, 임계 경로
// 보고서가 `fold_c_PFUMX...L6MUX21` 체인을 가리켰다. DUT 가 아니라
// 하네스를 재고 있었다.
//
// 바꾼 것이 누산기이므로 PE 하나만 재면 된다. 누산기 32비트를 1비트로
// 접는 것은 LUT4 두 단이라 무시할 수 있고, `acc` 를 모듈 안에 두므로
// mac_pe 를 단독 top 으로 쟀을 때처럼 피드백이 I/O 를 경유하지도
// 않는다 (그때 배선 1.10ns 가 끼어 252.78 MHz 가 왜곡됐다).
module fmax_pe #(
  parameter int ACC_SPLIT = 0,
  parameter int MUL_REG   = 0
)(
  input  logic clk,
  input  logic rst_n,
  input  logic seed,
  output logic probe
);
  logic [63:0] lfsr;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) lfsr <= 64'h0123_4567_89ab_cdef;
    else        lfsr <= {lfsr[62:0],
                         lfsr[63] ^ lfsr[62] ^ lfsr[60] ^ lfsr[59] ^ seed};
  end

  logic signed [31:0] acc;

  mac_pe #(.ACC_SPLIT(ACC_SPLIT), .MUL_REG(MUL_REG)) u_dut (
    .clk(clk), .rst_n(rst_n),
    .en(lfsr[0]), .acc_clr(lfsr[1]),
    .a($signed(lfsr[15:8])), .b($signed(lfsr[47:40])),
    .acc(acc)
  );

  logic probe_r;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) probe_r <= 1'b0;
    else        probe_r <= ^acc;
  end
  assign probe = probe_r;
endmodule

module fmax_pe_s0 (input logic clk, rst_n, seed, output logic probe);
  fmax_pe #(.ACC_SPLIT(0)) u (
    .clk(clk), .rst_n(rst_n), .seed(seed), .probe(probe));
endmodule

module fmax_pe_s1 (input logic clk, rst_n, seed, output logic probe);
  fmax_pe #(.ACC_SPLIT(1)) u (
    .clk(clk), .rst_n(rst_n), .seed(seed), .probe(probe));
endmodule

// 곱셈기 레지스터. 임계 경로의 지배 항목을 쪼갠다.
module fmax_pe_m1 (input logic clk, rst_n, seed, output logic probe);
  fmax_pe #(.ACC_SPLIT(0), .MUL_REG(1)) u (
    .clk(clk), .rst_n(rst_n), .seed(seed), .probe(probe));
endmodule

// 둘 다.
module fmax_pe_m1s1 (input logic clk, rst_n, seed, output logic probe);
  fmax_pe #(.ACC_SPLIT(1), .MUL_REG(1)) u (
    .clk(clk), .rst_n(rst_n), .seed(seed), .probe(probe));
endmodule

// ── A/B top ─────────────────────────────────────────────────────
// `chparam` 은 ECP5 cells_sim 을 끌어와 다른 문제를 일으키므로 이름을
// 따로 둔다. N=4 는 DSP 16개, N=8 은 64개로 둘 다 ECP5-85K 에 들어간다.
module fmax_arr4_s0 (input logic clk, rst_n, seed, output logic probe);
  fmax_mac_array #(.N(4), .ACC_SPLIT(0)) u (
    .clk(clk), .rst_n(rst_n), .seed(seed), .probe(probe));
endmodule

module fmax_arr4_s1 (input logic clk, rst_n, seed, output logic probe);
  fmax_mac_array #(.N(4), .ACC_SPLIT(1)) u (
    .clk(clk), .rst_n(rst_n), .seed(seed), .probe(probe));
endmodule

module fmax_arr8_s0 (input logic clk, rst_n, seed, output logic probe);
  fmax_mac_array #(.N(8), .ACC_SPLIT(0)) u (
    .clk(clk), .rst_n(rst_n), .seed(seed), .probe(probe));
endmodule

module fmax_arr8_s1 (input logic clk, rst_n, seed, output logic probe);
  fmax_mac_array #(.N(8), .ACC_SPLIT(1)) u (
    .clk(clk), .rst_n(rst_n), .seed(seed), .probe(probe));
endmodule

`default_nettype wire
