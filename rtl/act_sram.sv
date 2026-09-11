`timescale 1ns/1ps
`default_nettype none

module act_sram #(
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

  // --------------------------------------------
  // X-safe address check (Icarus friendly)
  // --------------------------------------------
  // X 주소 검사는 **시뮬레이션 전용**이다. `!==` 는 합성 가능한 연산자가 아니다.
  // wgt_sram 에서 같은 종류의 가드가 메모리를 통째로 삭제했다 (docs/BUGS.md BUG-006).
  // 여기서는 yosys 가 무해한 쪽(항상 valid)으로 접어서 피해가 없었지만
  // (셀 수 99,294 로 동일) 도구 판단에 기대지 않고 명시한다. docs/BUGS.md BUG-007.
  function automatic logic addr_is_valid(input logic [AW-1:0] a);
    integer k;
    begin
      addr_is_valid = 1'b1;
`ifdef COCOTB_SIM
      for (k = 0; k < AW; k = k + 1) begin
        if (a[k] !== 1'b0 && a[k] !== 1'b1)
          addr_is_valid = 1'b0;
      end
`endif
    end
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
      // ---------------- write ----------------
      if (we && addr_is_valid(waddr))
        mem[waddr] <= wdata;

      // ---------------- read (X-safe) ----------------
      if (re && addr_is_valid(raddr))
        rdata <= mem[raddr];
      else
        rdata <= rdata; // ✅ HOLD (reset에서 0이므로 X-safe)
    end
  end

endmodule

`default_nettype wire
