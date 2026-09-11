// ============================================================================
// dr1_tb_wrap.sv — dr1_top + dr1_scratch 결선 (배선만)
//
// `matvec_tb_wrap.sv` 와 같은 역할이다: 두 모듈을 묶고 **호스트 포트를 밖으로**
// 내보내서 테스트벤치가 q/k/v 를 넣고 o 를 꺼낼 수 있게 한다.
// 로직은 한 줄도 없다 — 여기서 뭔가를 계산하면 그것은 검증되지 않은 코드다.
//
// g2_ctrl_top 도 같은 방식으로 둘을 묶는다. 이 파일이 그 결선의 참조다.
// ============================================================================
`timescale 1ns/1ps
`default_nettype none

module dr1_tb_wrap #(
  parameter int D       = 16,
  parameter int W       = 16,
  parameter int SCRATCH = 1024
)(
  input  wire         clk,
  input  wire         rst_n,

  // 디스크립터 명령
  input  wire         cmd_valid,
  output wire         cmd_ready,
  input  wire [7:0]   cmd_opcode,
  input  wire [7:0]   cmd_slot,
  input  wire [63:0]  cmd_q_addr,
  input  wire [63:0]  cmd_k_addr,
  input  wire [63:0]  cmd_dst_addr,
  input  wire [63:0]  cmd_v_addr,
  input  wire [15:0]  cmd_alpha,
  input  wire [15:0]  cmd_beta,

  // 호스트 스크래치 포트 (테스트벤치 = 호스트)
  input  wire                       h_en,
  input  wire                       h_we,
  input  wire [$clog2(SCRATCH)-1:0] h_addr,
  input  wire [W-1:0]               h_wdata,
  output wire [W-1:0]               h_rdata,

  // 덤프 스트림
  output wire                 dump_valid,
  output wire [$clog2(D)-1:0] dump_row,
  output wire [D*W-1:0]       dump_data,

  // 완료·상태
  output wire        busy,
  output wire        done_ok,
  output wire        done_err,
  output wire        done_pulse,
  output wire [7:0]  fault_code,
  output wire        sat_event,
  output wire        clamp_event,
  output wire [31:0] dr1_status,
  output wire [31:0] dr1_sat_count,
  output wire [31:0] dr1_clamp_count,
  output wire [31:0] dr1_cycles,
  input  wire        sat_count_clr,
  input  wire        clamp_count_clr
);

  localparam int SAW = $clog2(SCRATCH);

  wire            scr_rd_en, scr_wr_en;
  wire [SAW-1:0]  scr_rd_addr, scr_wr_addr;
  wire [W-1:0]    scr_rd_data, scr_wr_data;

  dr1_top #(.D(D), .W(W), .NUM_SLOTS(1), .SCRATCH(SCRATCH)) u_dr1 (
    .clk(clk), .rst_n(rst_n),
    .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
    .cmd_opcode(cmd_opcode), .cmd_slot(cmd_slot),
    .cmd_q_addr(cmd_q_addr), .cmd_k_addr(cmd_k_addr),
    .cmd_dst_addr(cmd_dst_addr), .cmd_v_addr(cmd_v_addr),
    .cmd_alpha(cmd_alpha), .cmd_beta(cmd_beta),
    .scr_rd_en(scr_rd_en), .scr_rd_addr(scr_rd_addr), .scr_rd_data(scr_rd_data),
    .scr_wr_en(scr_wr_en), .scr_wr_addr(scr_wr_addr), .scr_wr_data(scr_wr_data),
    .dump_valid(dump_valid), .dump_row(dump_row), .dump_data(dump_data),
    .busy(busy), .done_ok(done_ok), .done_err(done_err), .done_pulse(done_pulse),
    .fault_code(fault_code),
    .sat_event(sat_event), .clamp_event(clamp_event),
    .dr1_status(dr1_status), .dr1_sat_count(dr1_sat_count),
    .dr1_clamp_count(dr1_clamp_count), .dr1_cycles(dr1_cycles),
    .sat_count_clr(sat_count_clr), .clamp_count_clr(clamp_count_clr)
  );

  dr1_scratch #(.DEPTH(SCRATCH), .W(W)) u_scratch (
    .clk(clk), .rst_n(rst_n),
    .h_en(h_en), .h_we(h_we), .h_addr(h_addr), .h_wdata(h_wdata), .h_rdata(h_rdata),
    .d_rd_en(scr_rd_en), .d_rd_addr(scr_rd_addr), .d_rd_data(scr_rd_data),
    .d_wr_en(scr_wr_en), .d_wr_addr(scr_wr_addr), .d_wr_data(scr_wr_data)
  );

endmodule

`default_nettype wire
