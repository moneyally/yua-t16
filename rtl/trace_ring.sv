// trace_ring.sv — Trace Ring Buffer for ORBIT-G2
// SSOT: ORBIT_G2_REG_SPEC.md section 11, ORBIT_G2_RTL_SKELETONS.md section 2.4
//
// Last-N event ring buffer (default 1024 entries).
// Stores descriptor, DMA, and fault events for post-mortem debug.
//
// Features:
//   - Circular overwrite (newest overwrites oldest)
//   - Freeze mode: stops advancing, counts dropped events
//   - Fatal-only filter: only records events with fatal flag set
//   - Wrap IRQ: pulse when tail passes head
//
// Register interface:
//   TRACE_HEAD     (0x803A_0000) RO — ring head
//   TRACE_TAIL     (0x803A_0004) RO — ring tail
//   TRACE_BASE_LO  (0x803A_0008) RW — host dump addr (software use)
//   TRACE_BASE_HI  (0x803A_000C) RW — host dump addr (software use)
//   TRACE_CTRL     (0x803A_0010) RW — enable/freeze/fatal_only
//   TRACE_DROP_CNT (0x803A_0014) RO — dropped events while frozen
`timescale 1ns/1ps
`default_nettype none

module trace_ring #(
  parameter int DEPTH     = 1024,      // entries (power of 2)
  parameter int ENTRY_W   = 64,        // bits per trace entry
  parameter int TYPE_W    = 4          // trace type field width
)(
  input  logic              clk,
  input  logic              rst_n,

  // Trace event input
  input  logic              trace_valid,
  input  logic [TYPE_W-1:0] trace_type,    // event type
  input  logic              trace_fatal,   // fatal flag (for fatal_only filter)
  input  logic [ENTRY_W-1:0] trace_payload,

  // Control (from register interface — TRACE_CTRL)
  input  logic              ctrl_enable,    // [0] trace enable
  input  logic              ctrl_freeze,    // [1] freeze
  input  logic              ctrl_fatal_only,// [2] fatal-only mode

  // Status (to register interface)
  output logic [15:0]       ring_head,
  output logic [15:0]       ring_tail,
  output logic [31:0]       drop_count,

  // Read port (for host debug dump)
  input  logic [$clog2(DEPTH)-1:0] rd_addr,
  output logic [ENTRY_W-1:0] rd_data,
  output logic [TYPE_W-1:0]  rd_type,
  output logic               rd_fatal,

  // IRQ output
  output logic              wrap_irq_pulse  // pulse on tail wraparound
);

  localparam int ADDR_W = $clog2(DEPTH);

  // ---------------------------------------------------------------
  // Ring memory
  // ---------------------------------------------------------------
  // Each entry: {type, fatal, payload}
  // BRAM inference checklist (verify in synth report):
  //   [x] mem write is in its own always_ff @(posedge clk) without reset
  //   [x] mem read is combinational assign (rd_data)
  //   [x] no reset loop over mem array
  //   [ ] TODO: confirm BRAM inference in Vivado/Quartus synth report
  localparam int STORE_W = TYPE_W + 1 + ENTRY_W;
  logic [STORE_W-1:0] mem [0:DEPTH-1];

  // Read port — combinational read from ring memory.
  // Entry format in mem: {type[TYPE_W-1:0], fatal, payload[ENTRY_W-1:0]}
  assign rd_data  = mem[rd_addr][ENTRY_W-1:0];
  assign rd_fatal = mem[rd_addr][ENTRY_W];
  assign rd_type  = mem[rd_addr][STORE_W-1:ENTRY_W+1];

  // ---------------------------------------------------------------
  // Pointers
  // ---------------------------------------------------------------
  logic [ADDR_W-1:0] head_r;
  logic [ADDR_W-1:0] tail_r;
  logic [31:0]       drop_cnt_r;
  logic               wrapped_r;  // tail has caught up to head at least once

  // ---------------------------------------------------------------
  // Write logic
  // ---------------------------------------------------------------
  wire event_accepted = trace_valid & ctrl_enable & ~ctrl_freeze &
                        (~ctrl_fatal_only | trace_fatal);

  wire event_dropped  = trace_valid & ctrl_enable & ctrl_freeze;

  // Next tail
  wire [ADDR_W-1:0] tail_next = tail_r + 1'b1;
  wire               will_wrap = event_accepted & (tail_next == head_r);

  // Memory write — separate from reset block for BRAM inference
  always_ff @(posedge clk) begin
    if (event_accepted)
      mem[tail_r] <= {trace_type, trace_fatal, trace_payload};
  end

  // Pointer and state updates
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      tail_r     <= '0;
      head_r     <= '0;
      drop_cnt_r <= 32'd0;
      wrapped_r  <= 1'b0;
    end else begin
      if (event_accepted) begin
        tail_r <= tail_next;

        // If overwriting head, advance head too
        if (wrapped_r && (tail_next == head_r)) begin
          head_r <= head_r + 1'b1;
        end

        // Track wrap
        if (tail_next == '0)
          wrapped_r <= 1'b1;
      end

      // Count drops while frozen
      if (event_dropped)
        drop_cnt_r <= drop_cnt_r + 1'b1;
    end
  end

  // ---------------------------------------------------------------
  // Wrap IRQ pulse
  // ---------------------------------------------------------------
  logic wrap_irq_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      wrap_irq_r <= 1'b0;
    else
      wrap_irq_r <= will_wrap;
  end

  assign wrap_irq_pulse = wrap_irq_r;

  // ---------------------------------------------------------------
  // Status outputs
  // ---------------------------------------------------------------
  assign ring_head  = {{(16-ADDR_W){1'b0}}, head_r};
  assign ring_tail  = {{(16-ADDR_W){1'b0}}, tail_r};
  assign drop_count = drop_cnt_r;

endmodule

`default_nettype wire
