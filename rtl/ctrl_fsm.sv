`timescale 1ns/1ps
`default_nettype none

module ctrl_fsm #(
  parameter int DESC_SIZE = 64
)(
  input  logic        clk,
  input  logic        rst_n,

  input  logic        desc_valid,
  input  logic [7:0]  desc_bytes [0:DESC_SIZE-1],
  output logic        desc_ready,

  output logic        cmd_valid,
  input  logic        cmd_ready,
  output logic [63:0] act_addr,
  output logic [63:0] wgt_addr,
  output logic [63:0] out_addr,
  output logic [31:0] Kt,

  input  logic        core_done,
  output logic        busy,
  output logic        done_pulse
);

  typedef enum logic [2:0] {
    ST_IDLE,
    ST_LATCH,
    ST_DECODE,
    ST_DISPATCH,
    ST_WAIT,
    ST_DONE
  } state_t;

  state_t state, state_n;

  logic [7:0] latched [0:DESC_SIZE-1];

  logic [7:0]  desc_type_r;
  logic [63:0] act_addr_r, wgt_addr_r, out_addr_r;
  logic [31:0] Kt_r;

  logic core_done_seen;

  function automatic logic [63:0] u64_le(input int base);
    u64_le = {
      latched[base+7], latched[base+6], latched[base+5], latched[base+4],
      latched[base+3], latched[base+2], latched[base+1], latched[base+0]
    };
  endfunction

  function automatic logic [31:0] u32_le(input int base);
    u32_le = { latched[base+3], latched[base+2], latched[base+1], latched[base+0] };
  endfunction

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) state <= ST_IDLE;
    else        state <= state_n;
  end

  integer i;
  always_ff @(posedge clk) begin
    if (state == ST_LATCH) begin
      for (i = 0; i < DESC_SIZE; i = i + 1)
        latched[i] <= desc_bytes[i];
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      desc_type_r <= 8'd0;
      act_addr_r  <= 64'd0;
      wgt_addr_r  <= 64'd0;
      out_addr_r  <= 64'd0;
      Kt_r        <= 32'd0;
    end else if (state == ST_DECODE) begin
      desc_type_r <= latched[0];
      act_addr_r  <= u64_le(16);
      wgt_addr_r  <= u64_le(24);
      out_addr_r  <= u64_le(32);
      Kt_r        <= u32_le(40);
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      core_done_seen <= 1'b0;
    end else begin
      // FIX v0.2: ST_DISPATCH 진입 시에도 core_done_seen 클리어
      // Kt=1 엣지케이스에서 gemm_core가 빠르게 완료되면
      // ST_DISPATCH 상태에서 core_done=1이 이미 세팅될 수 있음
      if (state == ST_IDLE || state == ST_DISPATCH) begin
        core_done_seen <= 1'b0;
      end else if (core_done === 1'b1) begin
        core_done_seen <= 1'b1;
      end
    end
  end

  always_comb begin
    desc_ready = 1'b0;
    cmd_valid  = 1'b0;
    busy       = 1'b0;
    done_pulse = 1'b0;

    act_addr = act_addr_r;
    wgt_addr = wgt_addr_r;
    out_addr = out_addr_r;
    Kt       = Kt_r;

    case (state)
      ST_IDLE: begin
        desc_ready = 1'b1;
      end
      ST_LATCH:   busy = 1'b1;
      ST_DECODE:  busy = 1'b1;
      ST_DISPATCH: begin
        busy      = 1'b1;
        cmd_valid = 1'b1;
      end
      ST_WAIT: begin
        busy = 1'b1;
      end
      ST_DONE: begin
        done_pulse = 1'b1;
      end
      default: ;
    endcase
  end

  always_comb begin
    state_n = state;
    case (state)
      ST_IDLE: begin
        if (desc_valid) state_n = ST_LATCH;
      end
      ST_LATCH:  state_n = ST_DECODE;
      ST_DECODE: begin
        if (latched[0] == 8'h02) state_n = ST_DISPATCH;
        else                     state_n = ST_DONE;
      end
      ST_DISPATCH: begin
        if (cmd_ready) state_n = ST_WAIT;
      end
      ST_WAIT: begin
        if (core_done_seen) state_n = ST_DONE;
      end
      ST_DONE: state_n = ST_IDLE;
      default: state_n = ST_IDLE;
    endcase
  end

endmodule

`default_nettype wire
