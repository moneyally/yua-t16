`timescale 1ns/1ps
`default_nettype none

module wgt_sram #(
  parameter int MAX_KT = 256
)(
  input  logic clk,
  input  logic rst_n,

  input  logic we,
  input  logic re,

  input  logic [$clog2(MAX_KT)-1:0] waddr,
  input  logic [127:0]              wdata,

  input  logic [$clog2(MAX_KT)-1:0] raddr,
  output logic [127:0]              rdata
);

  localparam int AW = $clog2(MAX_KT);

  logic [127:0] mem [0:MAX_KT-1];
  integer i;

  // X 주소 가드는 **시뮬레이션 전용**이다.
  // `===` 는 합성 가능한 연산자가 아니다. 가드를 그대로 두면 yosys 가
  // has_x_addr 를 상수 1 로 접어버려서 `we && !has_x_addr(...)` 가 항상 거짓이 되고,
  // **wgt_sram 전체가 합성에서 사라진다** (측정: 0 cells vs 가드 제거 시 99,294 cells).
  // docs/BUGS.md BUG-006 참조. COCOTB_SIM 밖에서는 상수 0 이어야 한다.
  function automatic logic has_x_addr(input logic [AW-1:0] a);
`ifdef COCOTB_SIM
    has_x_addr = (^a === 1'bx);
`else
    has_x_addr = 1'b0;
`endif
  endfunction

`ifdef COCOTB_SIM
  initial begin
    rdata = '0;
    for (i = 0; i < MAX_KT; i = i + 1)
      mem[i] = '0;
  end
`endif

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rdata <= '0;
`ifdef COCOTB_SIM
      for (i = 0; i < MAX_KT; i = i + 1)
        mem[i] <= '0;
`endif
    end else begin
      if (we && !has_x_addr(waddr))
        mem[waddr] <= wdata;

      if (re && !has_x_addr(raddr))
        rdata <= mem[raddr];
      else
        rdata <= rdata; // ✅ HOLD (reset에서 0이므로 X-safe)
    end
  end

endmodule

`default_nettype wire
