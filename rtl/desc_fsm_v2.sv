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

  // Status
  output logic        busy,
  output logic        done_pulse
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
  function automatic logic opcode_valid(input logic [7:0] op);
    opcode_valid = (op == 8'h01) || (op == 8'h02) || (op == 8'h03) || (op == 8'h04);
  endfunction

  // ---------------------------------------------------------------
  // core_done capture (same pattern as ctrl_fsm)
  // ---------------------------------------------------------------
  logic core_done_seen;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      core_done_seen <= 1'b0;
    end else begin
      if (state == ST_IDLE || state == ST_DISPATCH)
        core_done_seen <= 1'b0;
      else if (core_done == 1'b1)
        core_done_seen <= 1'b1;
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

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      fault_code_r  <= 8'h00;
      fault_valid_r <= 1'b0;
    end else begin
      if (state == ST_FAULT) begin
        fault_valid_r <= 1'b1;
        // fault_code_r is set in state_n logic
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
        if (timeout_expired)
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
          fault_code_r <= 8'h03;  // timeout
        else
          fault_code_r <= 8'h04;  // reserved
      end
    end
  end

endmodule

`default_nettype wire
