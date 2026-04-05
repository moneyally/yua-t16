// g3_desc_fsm.sv — ORBIT-G3 Descriptor FSM
// SSOT: ORBIT_G3_RTL_ISSUES.md (G3-RTL-005), ORBIT_G3_REG_SPEC.md
//
// Extends G2 desc_fsm_v2 philosophy to G3 targets:
//   - MXU BF16 forward (implemented this turn)
//   - Backward, Optimizer, Collective (decode only, stub dispatch)
//
// State machine: IDLE -> LATCH -> CRC_CHECK -> DECODE -> DISPATCH -> WAIT -> DONE/FAULT
// Same as G2 desc_fsm_v2 but with multi-target dispatch.
//
// Opcodes (G2 compatible + G3 extensions):
//   0x01 = NOP
//   0x02 = GEMM INT8 (G2 legacy)
//   0x10 = MXU_FWD (G3 BF16 MXU forward)  // localparam, TODO: confirm in spec
//   0x20 = BACKWARD (G3, stub this turn)
//   0x30 = OPTIMIZER (G3, stub this turn)
//   0x40 = COLLECTIVE (G3, stub this turn)
`timescale 1ns/1ps
`default_nettype none

module g3_desc_fsm #(
  parameter int DESC_SIZE       = 64,
  parameter int TIMEOUT_DEFAULT = 32'd100_000
)(
  input  logic        clk,
  input  logic        rst_n,

  // Descriptor input
  input  logic        desc_valid,
  input  logic [7:0]  desc_bytes [0:DESC_SIZE-1],
  output logic        desc_ready,

  // Queue class
  input  logic [1:0]  queue_class,

  // MXU dispatch (active this turn)
  output logic        mxu_cmd_valid,
  input  logic        mxu_cmd_ready,
  output logic [31:0] mxu_cfg0,
  output logic [31:0] mxu_cfg1,
  output logic [63:0] mxu_act_addr,
  output logic [63:0] mxu_wgt_addr,
  output logic [63:0] mxu_out_addr,

  // G2 legacy GEMM dispatch (pass-through)
  output logic        gemm_cmd_valid,
  input  logic        gemm_cmd_ready,
  output logic [63:0] gemm_act_addr,
  output logic [63:0] gemm_wgt_addr,
  output logic [63:0] gemm_out_addr,
  output logic [31:0] gemm_Kt,

  // Backward dispatch (stub)
  output logic        bkwd_cmd_valid,
  // Optimizer dispatch (stub)
  output logic        opt_cmd_valid,
  // Collective dispatch (stub)
  output logic        coll_cmd_valid,

  // Completion
  input  logic        core_done,      // from active target
  input  logic [31:0] timeout_cycles,

  // Fault
  output logic        fault_valid,
  output logic [7:0]  fault_code,
  // 0x00 = none, 0x01 = illegal opcode, 0x02 = CRC fail,
  // 0x03 = timeout, 0x04 = unsupported target

  // Status
  output logic        busy,
  output logic        done_pulse,
  output logic [7:0]  current_opcode,
  output logic [1:0]  current_qclass
);

  // ═══════════════════════════════════════════════════════════
  // Opcodes
  // ═══════════════════════════════════════════════════════════
  localparam logic [7:0] OP_NOP       = 8'h01;
  localparam logic [7:0] OP_GEMM_INT8 = 8'h02;  // G2 legacy
  localparam logic [7:0] OP_KVC       = 8'h03;   // G2 (unsupported)
  localparam logic [7:0] OP_VPU       = 8'h04;   // G2 (unsupported)
  // G3 extensions — TODO: confirm final values in ORBIT_G3_REG_SPEC
  localparam logic [7:0] OP_MXU_FWD   = 8'h10;
  localparam logic [7:0] OP_BACKWARD  = 8'h20;
  localparam logic [7:0] OP_OPTIMIZER = 8'h30;
  localparam logic [7:0] OP_COLLECTIVE = 8'h40;

  // ═══════════════════════════════════════════════════════════
  // State machine
  // ═══════════════════════════════════════════════════════════
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

  // ═══════════════════════════════════════════════════════════
  // Latched descriptor
  // ═══════════════════════════════════════════════════════════
  logic [7:0] latched [0:DESC_SIZE-1];
  logic [7:0] opcode_r;
  logic [1:0] qclass_r;

  // Decoded fields (LE helpers, same as G2)
  function automatic logic [63:0] u64_le(input int base);
    u64_le = {latched[base+7], latched[base+6], latched[base+5], latched[base+4],
              latched[base+3], latched[base+2], latched[base+1], latched[base+0]};
  endfunction

  function automatic logic [31:0] u32_le(input int base);
    u32_le = {latched[base+3], latched[base+2], latched[base+1], latched[base+0]};
  endfunction

  // Decoded address fields
  logic [63:0] act_addr_r, wgt_addr_r, out_addr_r;
  logic [31:0] cfg0_r, cfg1_r, kt_r;

  // ═══════════════════════════════════════════════════════════
  // CRC-8 (same as G2)
  // ═══════════════════════════════════════════════════════════
  function automatic logic [7:0] crc8_byte(input logic [7:0] crc, input logic [7:0] data);
    logic [7:0] c;
    integer i;
    c = crc ^ data;
    for (i = 0; i < 8; i = i + 1) begin
      if (c[7]) c = {c[6:0], 1'b0} ^ 8'h07;
      else      c = {c[6:0], 1'b0};
    end
    crc8_byte = c;
  endfunction

  logic [7:0] computed_crc;
  logic       crc_ok;

  always_comb begin
    logic [7:0] acc;
    integer j;
    acc = 8'h00;
    for (j = 0; j < DESC_SIZE - 1; j = j + 1)
      acc = crc8_byte(acc, latched[j]);
    computed_crc = acc;
    crc_ok = (computed_crc == latched[DESC_SIZE-1]);
  end

  // ═══════════════════════════════════════════════════════════
  // Opcode validation
  // ═══════════════════════════════════════════════════════════
  function automatic logic opcode_known(input logic [7:0] op);
    opcode_known = (op == OP_NOP) || (op == OP_GEMM_INT8) ||
                   (op == OP_MXU_FWD) || (op == OP_BACKWARD) ||
                   (op == OP_OPTIMIZER) || (op == OP_COLLECTIVE);
  endfunction

  function automatic logic opcode_dispatchable(input logic [7:0] op);
    // Only NOP, GEMM_INT8, MXU_FWD have real targets this turn
    opcode_dispatchable = (op == OP_NOP) || (op == OP_GEMM_INT8) || (op == OP_MXU_FWD);
  endfunction

  // ═══════════════════════════════════════════════════════════
  // Dispatch target
  // ═══════════════════════════════════════════════════════════
  typedef enum logic [2:0] {
    TGT_NONE,
    TGT_NOP,
    TGT_GEMM,
    TGT_MXU,
    TGT_BKWD,
    TGT_OPT,
    TGT_COLL
  } target_t;

  target_t dispatch_target;

  always_comb begin
    case (opcode_r)
      OP_NOP:        dispatch_target = TGT_NOP;
      OP_GEMM_INT8:  dispatch_target = TGT_GEMM;
      OP_MXU_FWD:    dispatch_target = TGT_MXU;
      OP_BACKWARD:   dispatch_target = TGT_BKWD;
      OP_OPTIMIZER:  dispatch_target = TGT_OPT;
      OP_COLLECTIVE: dispatch_target = TGT_COLL;
      default:       dispatch_target = TGT_NONE;
    endcase
  end

  // ═══════════════════════════════════════════════════════════
  // Timeout counter
  // ═══════════════════════════════════════════════════════════
  logic [31:0] timeout_cnt;
  logic [31:0] timeout_limit;
  wire timeout_expired = (state == ST_WAIT) && (timeout_cnt >= timeout_limit);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      timeout_cnt   <= 32'd0;
      timeout_limit <= TIMEOUT_DEFAULT;
    end else begin
      if (state == ST_DISPATCH)
        timeout_limit <= (timeout_cycles != 0) ? timeout_cycles : TIMEOUT_DEFAULT;
      if (state == ST_WAIT)
        timeout_cnt <= timeout_cnt + 1'b1;
      else
        timeout_cnt <= 32'd0;
    end
  end

  // ═══════════════════════════════════════════════════════════
  // core_done capture
  // ═══════════════════════════════════════════════════════════
  logic core_done_seen;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      core_done_seen <= 1'b0;
    else if (state == ST_IDLE || state == ST_DISPATCH)
      core_done_seen <= 1'b0;
    else if (core_done == 1'b1)
      core_done_seen <= 1'b1;
  end

  // ═══════════════════════════════════════════════════════════
  // Latch descriptor
  // ═══════════════════════════════════════════════════════════
  integer i;
  always_ff @(posedge clk) begin
    if (state == ST_LATCH)
      for (i = 0; i < DESC_SIZE; i = i + 1)
        latched[i] <= desc_bytes[i];
  end

  // Decode register latch
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      opcode_r   <= 8'd0;
      act_addr_r <= 64'd0;
      wgt_addr_r <= 64'd0;
      out_addr_r <= 64'd0;
      cfg0_r     <= 32'd0;
      cfg1_r     <= 32'd0;
      kt_r       <= 32'd0;
      qclass_r   <= 2'd0;
    end else if (state == ST_DECODE) begin
      opcode_r   <= latched[0];
      act_addr_r <= u64_le(16);
      wgt_addr_r <= u64_le(24);
      out_addr_r <= u64_le(32);
      kt_r       <= u32_le(40);
      cfg0_r     <= u32_le(44);
      cfg1_r     <= u32_le(48);
      qclass_r   <= queue_class;
    end
  end

  // ═══════════════════════════════════════════════════════════
  // Fault code latch
  // ═══════════════════════════════════════════════════════════
  logic [7:0] fault_code_r;
  logic       fault_valid_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      fault_code_r  <= 8'h00;
      fault_valid_r <= 1'b0;
    end else begin
      if (state == ST_FAULT)
        fault_valid_r <= 1'b1;
      else if (state == ST_IDLE)
        fault_valid_r <= 1'b0;
    end
  end

  // Set fault code on transition INTO ST_FAULT
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      fault_code_r <= 8'h00;
    else if (state != ST_FAULT && state_n == ST_FAULT) begin
      if (state == ST_CRC_CHECK)     fault_code_r <= 8'h02; // CRC fail
      else if (state == ST_DECODE)   fault_code_r <= 8'h01; // illegal opcode
      else if (state == ST_WAIT)     fault_code_r <= 8'h03; // timeout
      else                           fault_code_r <= 8'h04; // unsupported
    end
  end

  // ═══════════════════════════════════════════════════════════
  // FSM sequential
  // ═══════════════════════════════════════════════════════════
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) state <= ST_IDLE;
    else        state <= state_n;
  end

  // ═══════════════════════════════════════════════════════════
  // FSM combinational + outputs
  // ═══════════════════════════════════════════════════════════
  logic target_ready;

  always_comb begin
    // Target-specific ready mux
    case (dispatch_target)
      TGT_GEMM: target_ready = gemm_cmd_ready;
      TGT_MXU:  target_ready = mxu_cmd_ready;
      default:  target_ready = 1'b1; // NOP, stubs
    endcase
  end

  always_comb begin
    state_n        = state;
    desc_ready     = 1'b0;
    busy           = 1'b0;
    done_pulse     = 1'b0;
    mxu_cmd_valid  = 1'b0;
    gemm_cmd_valid = 1'b0;
    bkwd_cmd_valid = 1'b0;
    opt_cmd_valid  = 1'b0;
    coll_cmd_valid = 1'b0;
    fault_valid    = fault_valid_r;
    fault_code     = fault_code_r;
    current_opcode = opcode_r;
    current_qclass = qclass_r;

    // Output decoded fields
    mxu_cfg0     = cfg0_r;
    mxu_cfg1     = cfg1_r;
    mxu_act_addr = act_addr_r;
    mxu_wgt_addr = wgt_addr_r;
    mxu_out_addr = out_addr_r;
    gemm_act_addr = act_addr_r;
    gemm_wgt_addr = wgt_addr_r;
    gemm_out_addr = out_addr_r;
    gemm_Kt       = kt_r;

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
        if (!crc_ok)
          state_n = ST_FAULT;
        else
          state_n = ST_DECODE;
      end

      ST_DECODE: begin
        busy = 1'b1;
        if (!opcode_known(latched[0]))
          state_n = ST_FAULT;  // illegal opcode
        else if (!opcode_dispatchable(latched[0]))
          state_n = ST_FAULT;  // known but unsupported (backward/opt/coll)
        else if (latched[0] == OP_NOP)
          state_n = ST_DONE;   // NOP: skip dispatch
        else
          state_n = ST_DISPATCH;
      end

      ST_DISPATCH: begin
        busy = 1'b1;
        // Assert target cmd_valid
        case (dispatch_target)
          TGT_GEMM: gemm_cmd_valid = 1'b1;
          TGT_MXU:  mxu_cmd_valid  = 1'b1;
          default: ;
        endcase
        if (target_ready) state_n = ST_WAIT;
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
        state_n = ST_DONE;
      end

      default: state_n = ST_IDLE;
    endcase
  end

endmodule

`default_nettype wire
