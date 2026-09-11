// desc_fsm_v2.sv — Descriptor FSM v2 for ORBIT-G2
// SSOT: ORBIT_G2_RTL_SKELETONS.md section 2.1, ctrl_fsm.sv design philosophy
//
// Extension of ctrl_fsm with:
//   - CRC-8 check on descriptor payload
//   - Timeout counter (configurable cycles)
//   - Illegal opcode detection
//   - Fault code output (fault_valid + fault_code)
//   - Queue class input (which queue the descriptor came from)
//   - Integration-ready with desc_queue (pop_valid/pop_ready/pop_data)
//
// Handshake contract:
//   desc_valid/desc_ready: descriptor input (from desc_queue pop interface)
//   cmd_valid/cmd_ready:   command output (to compute engines)
//   core_done:             completion feedback (from compute engines)
//
// Opcode map (byte 0 of descriptor):
//   0x01 = NOP
//   0x02 = GEMM
//   0x03 = KVC_OP
//   0x04 = VPU_OP
//   others = ILLEGAL
`timescale 1ns/1ps
`default_nettype none

module desc_fsm_v2 #(
  parameter int DESC_SIZE       = 64,   // bytes per descriptor
  parameter int TIMEOUT_DEFAULT = 32'd100_000  // default timeout cycles
)(
  input  logic        clk,
  input  logic        rst_n,

  // Descriptor input (from desc_queue pop)
  input  logic        desc_valid,
  input  logic [7:0]  desc_bytes [0:DESC_SIZE-1],
  output logic        desc_ready,

  // Queue class (which queue this descriptor came from)
  input  logic [1:0]  queue_class,   // 0=compute, 1=utility, 2=telemetry, 3=hipri

  // Command output (to compute engines)
  output logic        cmd_valid,
  input  logic        cmd_ready,
  output logic [7:0]  cmd_opcode,
  output logic [63:0] act_addr,
  output logic [63:0] wgt_addr,
  output logic [63:0] out_addr,
  output logic [31:0] Kt,

  // Completion feedback
  input  logic        core_done,
  // 엔진 쪽 실패 (docs/DESIGN.md 5.1, spec/deltarule.md 4절).
  // core_done 과 달리 **실패**를 보고한다. 예전에는 이 입력이 없어서 엔진이
  // 실패해도 desc_fsm 은 ST_DONE 으로 가 done_ok 를 냈다 — BUG-001 과 같은 종류다.
  // 안 쓰는 상위 모듈은 1'b0 으로 묶으면 동작이 예전과 완전히 같다.
  input  logic        core_err,
  input  logic [7:0]  core_fault_code,

  // Timeout configuration (from register interface)
  input  logic [31:0] timeout_cycles,

  // Fault output
  output logic        fault_valid,
  output logic [7:0]  fault_code,
  // Fault codes:
  //   0x00 = no fault
  //   0x01 = illegal opcode
  //   0x02 = CRC mismatch
  //   0x03 = timeout
  //   0x04 = reserved

  // Status — 완료 신호 계약: docs/DESIGN.md 5.1절
  //   done_ok    : 성공 완료. 정확히 1사이클. **완료 IRQ 는 이것에만 걸린다.**
  //   done_err   : 실패 종료. 정확히 1사이클.
  //   done_pulse : 리타이어 (= done_ok | done_err). 자원 회수(OOM 감소)는 이것에 걸린다.
  // 한 트랜잭션은 done_ok 와 done_err 중 정확히 하나만 낸다.
  // 예전에는 done_pulse 하나뿐이어서 fault 난 디스크립터도 완료 IRQ 를 올렸다
  // (docs/BUGS.md BUG-001).
  output logic        busy,
  output logic        done_pulse,
  output logic        done_ok,
  output logic        done_err
);

  // ---------------------------------------------------------------
  // State machine (extends ctrl_fsm pattern)
  // ---------------------------------------------------------------
  typedef enum logic [3:0] {
    ST_IDLE,
    ST_LATCH,
    ST_CRC_CHECK,
    ST_DECODE,
    ST_DISPATCH,
    ST_WAIT,
    ST_DONE,
    ST_FAULT
  } state_t;

  state_t state, state_n;

  // ---------------------------------------------------------------
  // Latched descriptor
  // ---------------------------------------------------------------
  logic [7:0] latched [0:DESC_SIZE-1];

  // Decode fields
  logic [7:0]  desc_opcode_r;
  // CRC is checked in ST_CRC_CHECK via computed_crc vs latched[DESC_SIZE-1] directly
  logic [63:0] act_addr_r, wgt_addr_r, out_addr_r;
  logic [31:0] Kt_r;
  // TODO: queue_class_r is latched but not yet consumed in skeleton.
  // Will be used for: priority arbitration, trace event tagging, fault routing.
  // Consumer logic to be added when desc_queue <-> desc_fsm_v2 integration happens.
  /* verilator lint_off UNUSEDSIGNAL */
  logic [1:0]  queue_class_r;
  /* verilator lint_on UNUSEDSIGNAL */

  // ---------------------------------------------------------------
  // Timeout counter
  // ---------------------------------------------------------------
  logic [31:0] timeout_cnt;
  logic [31:0] timeout_limit;

  // ---------------------------------------------------------------
  // CRC-8 computation (simple XOR-based, polynomial 0x07)
  // ---------------------------------------------------------------
  function automatic logic [7:0] crc8_byte(input logic [7:0] crc, input logic [7:0] data);
    logic [7:0] c;
    integer i;
    begin
      c = crc ^ data;
      for (i = 0; i < 8; i = i + 1) begin
        if (c[7])
          c = {c[6:0], 1'b0} ^ 8'h07;
        else
          c = {c[6:0], 1'b0};
      end
      crc8_byte = c;
    end
  endfunction

  logic [7:0] computed_crc;
  logic        crc_ok;

  // CRC is computed over bytes [0:DESC_SIZE-2], compared to byte [DESC_SIZE-1]
  always_comb begin
    automatic logic [7:0] acc = 8'h00;
    integer j;
    for (j = 0; j < DESC_SIZE - 1; j = j + 1)
      acc = crc8_byte(acc, latched[j]);
    computed_crc = acc;
    crc_ok = (computed_crc == latched[DESC_SIZE-1]);
  end

  // ---------------------------------------------------------------
  // Little-endian decode helpers (same as ctrl_fsm)
  // ---------------------------------------------------------------
  function automatic logic [63:0] u64_le(input int base);
    u64_le = {
      latched[base+7], latched[base+6], latched[base+5], latched[base+4],
      latched[base+3], latched[base+2], latched[base+1], latched[base+0]
    };
  endfunction

  function automatic logic [31:0] u32_le(input int base);
    u32_le = { latched[base+3], latched[base+2], latched[base+1], latched[base+0] };
  endfunction

  // ---------------------------------------------------------------
  // Opcode validation
  // ---------------------------------------------------------------
  // 0x50 DELTA_INIT / 0x51 DELTA_STEP / 0x52 DELTA_DUMP 는 dr1_top 이 처리한다
  // (spec/deltarule.md 2절). W6 까지는 0x51 을 일부러 뺐었다 — 계산 경로가 없어서
  // 조용히 아무것도 안 하게 되기 때문이다. **W7 에서 구현되어 이제 받는다.**
  function automatic logic opcode_valid(input logic [7:0] op);
    opcode_valid = (op == 8'h01) || (op == 8'h02) || (op == 8'h03) || (op == 8'h04)
                || (op == 8'h50) || (op == 8'h51) || (op == 8'h52);
  endfunction

  // ---------------------------------------------------------------
  // ST_DONE 진입 경로 구분 (docs/DESIGN.md 5.1절, docs/BUGS.md BUG-001)
  // ST_FAULT 를 거쳐 들어왔는지 기억한다. ST_DONE 자체는 두 경로를 구분할 수 없다.
  // ---------------------------------------------------------------
  logic came_from_fault;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      came_from_fault <= 1'b0;
    end else if (state == ST_IDLE) begin
      came_from_fault <= 1'b0;          // 새 트랜잭션 시작 시 클리어
    end else if (state == ST_FAULT) begin
      came_from_fault <= 1'b1;
    end
  end

  // ---------------------------------------------------------------
  // core_done capture (same pattern as ctrl_fsm)
  // ---------------------------------------------------------------
  logic core_done_seen;
  logic core_err_seen;
  logic [7:0] core_fault_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      core_done_seen <= 1'b0;
      core_err_seen  <= 1'b0;
      core_fault_r   <= 8'h00;
    end else begin
      if (state == ST_IDLE || state == ST_DISPATCH) begin
        core_done_seen <= 1'b0;
        core_err_seen  <= 1'b0;
      end else begin
        if (core_done == 1'b1) core_done_seen <= 1'b1;
        if (core_err  == 1'b1) begin
          core_err_seen <= 1'b1;
          core_fault_r  <= core_fault_code;
        end
      end
    end
  end

  // ---------------------------------------------------------------
  // Latch descriptor
  // ---------------------------------------------------------------
  integer i;
  always_ff @(posedge clk) begin
    if (state == ST_LATCH) begin
      for (i = 0; i < DESC_SIZE; i = i + 1)
        latched[i] <= desc_bytes[i];
    end
  end

  // ---------------------------------------------------------------
  // Decode register latch
  // ---------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      desc_opcode_r <= 8'd0;
      act_addr_r    <= 64'd0;
      wgt_addr_r    <= 64'd0;
      out_addr_r    <= 64'd0;
      Kt_r          <= 32'd0;
      queue_class_r <= 2'd0;
    end else if (state == ST_DECODE) begin
      desc_opcode_r <= latched[0];
      act_addr_r    <= u64_le(16);
      wgt_addr_r    <= u64_le(24);
      out_addr_r    <= u64_le(32);
      Kt_r          <= u32_le(40);
      queue_class_r <= queue_class;
    end
  end

  // ---------------------------------------------------------------
  // Timeout counter
  // ---------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      timeout_cnt   <= 32'd0;
      timeout_limit <= TIMEOUT_DEFAULT;
    end else begin
      // Latch timeout config at dispatch
      if (state == ST_DISPATCH && cmd_ready)
        timeout_limit <= (timeout_cycles != 32'd0) ? timeout_cycles : TIMEOUT_DEFAULT;

      // Count in WAIT state
      if (state == ST_WAIT)
        timeout_cnt <= timeout_cnt + 1'b1;
      else
        timeout_cnt <= 32'd0;
    end
  end

  wire timeout_expired = (state == ST_WAIT) && (timeout_cnt >= timeout_limit);

  // ---------------------------------------------------------------
  // Fault register
  // ---------------------------------------------------------------
  logic [7:0] fault_code_r;
  logic       fault_valid_r;

  // NOTE: fault_code_r 는 아래 "Fault code latch" always_ff 하나에서만 구동한다.
  // 예전에는 이 블록도 리셋 시 fault_code_r <= 0 을 했는데, 그러면 같은 변수를
  // 두 개의 always_ff 가 구동하는 다중 드라이버가 된다. yosys 가
  // "multiple conflicting drivers for desc_fsm_v2.\fault_code_r" 로 잡아냈고
  // (scripts/synth_gate.sh STAGE 1), Vivado·verilator 도 MULTIDRIVEN 으로 거부한다.
  // 시뮬레이션에서는 우연히 동작해서 기존 테스트가 전부 통과했다.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      fault_valid_r <= 1'b0;
    end else begin
      if (state == ST_FAULT) begin
        fault_valid_r <= 1'b1;
      end else if (state == ST_IDLE) begin
        fault_valid_r <= 1'b0;
      end
    end
  end

  // ---------------------------------------------------------------
  // FSM sequential
  // ---------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      state <= ST_IDLE;
    else
      state <= state_n;
  end

  // ---------------------------------------------------------------
  // FSM combinational
  // ---------------------------------------------------------------
  always_comb begin
    state_n    = state;
    desc_ready = 1'b0;
    cmd_valid  = 1'b0;
    busy       = 1'b0;
    done_pulse = 1'b0;

    cmd_opcode = desc_opcode_r;
    act_addr   = act_addr_r;
    wgt_addr   = wgt_addr_r;
    out_addr   = out_addr_r;
    Kt         = Kt_r;

    fault_valid = fault_valid_r;
    fault_code  = fault_code_r;

    case (state)
      ST_IDLE: begin
        desc_ready = 1'b1;
        if (desc_valid) state_n = ST_LATCH;
      end

      ST_LATCH: begin
        busy    = 1'b1;
        state_n = ST_CRC_CHECK;
      end

      ST_CRC_CHECK: begin
        busy = 1'b1;
        if (!crc_ok) begin
          state_n = ST_FAULT;
        end else begin
          state_n = ST_DECODE;
        end
      end

      ST_DECODE: begin
        busy = 1'b1;
        if (!opcode_valid(latched[0])) begin
          state_n = ST_FAULT;
        end else if (latched[0] == 8'h01) begin
          // NOP — skip dispatch
          state_n = ST_DONE;
        end else begin
          state_n = ST_DISPATCH;
        end
      end

      ST_DISPATCH: begin
        busy      = 1'b1;
        cmd_valid = 1'b1;
        if (cmd_ready) state_n = ST_WAIT;
      end

      ST_WAIT: begin
        busy = 1'b1;
        // 순서가 중요하다: 엔진 실패가 완료보다 먼저 검사돼야 한다.
        // 엔진이 done_err 를 내면서 done_pulse 도 같이 내는 경우(리타이어 규약)
        // core_done 과 core_err 이 같은 사이클에 올 수 있다 — 그때 실패가 이긴다.
        if (timeout_expired || core_err_seen)
          state_n = ST_FAULT;
        else if (core_done_seen)
          state_n = ST_DONE;
      end

      ST_DONE: begin
        done_pulse = 1'b1;
        state_n    = ST_IDLE;
      end

      ST_FAULT: begin
        busy    = 1'b1;
        state_n = ST_DONE;  // transition to DONE to re-enter IDLE
      end

      default: state_n = ST_IDLE;
    endcase
  end

  // ---------------------------------------------------------------
  // Fault code latch (set on entry to ST_FAULT)
  // ---------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      fault_code_r <= 8'h00;
    end else begin
      // Detect the transition INTO ST_FAULT
      if (state != ST_FAULT && state_n == ST_FAULT) begin
        if (state == ST_CRC_CHECK)
          fault_code_r <= 8'h02;  // CRC mismatch
        else if (state == ST_DECODE)
          fault_code_r <= 8'h01;  // illegal opcode
        else if (state == ST_WAIT)
          // 엔진이 낸 코드를 그대로 올린다 (DR1 은 0x05/0x06/0x07).
          // 엔진 실패가 아니면 타임아웃이다.
          fault_code_r <= core_err_seen ? core_fault_r : 8'h03;
        else
          fault_code_r <= 8'h04;  // reserved
      end
    end
  end

  // ---------------------------------------------------------------
  // 완료 신호 분리 (docs/DESIGN.md 5.1절)
  // done_pulse 는 ST_DONE 에서 1사이클 — 성공/실패 모두 여기를 지난다 (리타이어).
  // 진입 경로로 갈라서 done_ok / done_err 를 만든다.
  // ---------------------------------------------------------------
  assign done_ok  = done_pulse && !came_from_fault;
  assign done_err = done_pulse &&  came_from_fault;

endmodule

`default_nettype wire
