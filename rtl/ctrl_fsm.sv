// ctrl_fsm.sv — 디스크립터 수신 → 디코드 → 코어 발행
//
// 발행 경로와 완료 경로를 분리했다.
//
//   발행: ST_IDLE → ST_LATCH → ST_DECODE → ST_DISPATCH → ST_IDLE
//   완료: core_done 을 그대로 흘려보낸다 (GEMM 디스크립터)
//         GEMM 이 아닌 디스크립터는 ST_DONE 에서 자체 생성
//
// 이전에는 ST_DISPATCH 다음이 ST_WAIT 였고 거기서 core_done 을 기다린
// 뒤에야 ST_IDLE 로 돌아가 desc_ready 를 올렸다. gemm_core 가 쓰기 FSM
// 분리로 다음 명령을 일찍 받을 수 있게 됐는데도 시스템 처리량이 그대로
// 였던 이유가 이것이다 (gemm_top 경유 274.8 사이클/op vs 코어 직접
// 210.1). 이제 코어가 명령을 받는 즉시 다음 디스크립터를 받는다.
//
// gemm_top 의 커맨드 브리지가 cmd_valid 를 코어 수용 시점까지 유지하고
// cmd_ready_ctrl 을 그때 올리므로, ST_DISPATCH 를 빠져나오는 순간
// 래치/디코드 레지스터를 덮어써도 안전하다.
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
    ST_DONE
  } state_t;

  state_t state, state_n;

  logic [7:0] latched [0:DESC_SIZE-1];

  logic [7:0]  desc_type_r;
  logic [63:0] act_addr_r, wgt_addr_r, out_addr_r;
  logic [31:0] Kt_r;

  // 코어에 떠 있는 GEMM 명령 수. busy 판정에 쓴다.
  logic [1:0] outstanding;

  // 내보내야 할 done 펄스 큐.
  // core_done 과 ST_DONE(비 GEMM) 이 같은 사이클에 겹칠 수 있으므로
  // 하나로 합치지 않고 큐에 쌓아 한 사이클짜리 펄스로 하나씩 낸다.
  logic [2:0] done_q;
  logic       done_r;

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

  wire issue      = (state == ST_DISPATCH) && cmd_ready;
  wire local_done = (state == ST_DONE);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      outstanding <= 2'd0;
    end else begin
      case ({issue, core_done})
        2'b10:   outstanding <= outstanding + 2'd1;
        2'b01:   if (outstanding != 2'd0) outstanding <= outstanding - 2'd1;
        default: ;   // 둘 다 또는 아무것도 → 변화 없음
      endcase
    end
  end

  // 완료 펄스 큐. 한 사이클 펄스 뒤 반드시 한 사이클 비운다.
  wire [2:0] done_inc = (core_done ? 3'd1 : 3'd0) + (local_done ? 3'd1 : 3'd0);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      done_q <= 3'd0;
      done_r <= 1'b0;
    end else if (!done_r && (done_q + done_inc) != 3'd0) begin
      done_r <= 1'b1;
      done_q <= done_q + done_inc - 3'd1;
    end else begin
      done_r <= 1'b0;
      done_q <= done_q + done_inc;
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
      ST_DONE: ;   // 펄스는 아래 큐가 낸다
      default: ;
    endcase

    // 코어에 아직 떠 있는 명령이 있으면 IDLE 이어도 바쁘다.
    if (outstanding != 2'd0 || done_q != 3'd0 || done_r) busy = 1'b1;

    done_pulse = done_r;
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
        // 코어가 명령을 받는 즉시 다음 디스크립터를 받으러 돌아간다.
        // 완료는 core_done 이 따로 알려준다.
        if (cmd_ready) state_n = ST_IDLE;
      end
      ST_DONE: state_n = ST_IDLE;
      default: state_n = ST_IDLE;
    endcase
  end

endmodule

`default_nettype wire
