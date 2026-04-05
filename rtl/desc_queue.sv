// desc_queue.sv — Multi-Queue Descriptor Ring for ORBIT-G2
// SSOT: ORBIT_G2_REG_SPEC.md section 4, ORBIT_G2_RTL_SKELETONS.md section 2.2
//
// 4 independent descriptor queues:
//   Q0 = compute, Q1 = utility, Q2 = telemetry, Q3 = hipri
//
// Each queue is a circular buffer with head/tail pointers.
// Backpressure: push fails when queue is full (push_ready deasserted).
// Overflow: sticky flag set when push attempted on full queue (W1C via reg interface).
//
// Register interface signals map to REG_SPEC:
//   - q_head[i], q_tail[i]: read via Qx_STATUS (0x8030_3000 + 4*i)
//   - q_depth[i]: derived from head/tail
//   - overflow_flags: Q_OVERFLOW (0x8030_3010) W1C
`timescale 1ns/1ps
`default_nettype none

module desc_queue #(
  parameter int NUM_QUEUES  = 4,
  parameter int QUEUE_DEPTH = 64,     // entries per queue (power of 2)
  parameter int DESC_W      = 128     // descriptor width in bits
)(
  input  logic        clk,
  input  logic        rst_n,

  // Push interface (per queue)
  input  logic [NUM_QUEUES-1:0]           push_valid,
  output logic [NUM_QUEUES-1:0]           push_ready,
  input  logic [DESC_W-1:0]              push_data   [0:NUM_QUEUES-1],

  // Pop interface (per queue)
  input  logic [NUM_QUEUES-1:0]           pop_valid,
  output logic [NUM_QUEUES-1:0]           pop_ready,
  output logic [DESC_W-1:0]              pop_data    [0:NUM_QUEUES-1],

  // Status (directly maps to REG_SPEC Qx_STATUS)
  // [15:0] = HEAD, [31:16] = TAIL per queue
  output logic [15:0]                    q_head      [0:NUM_QUEUES-1],
  output logic [15:0]                    q_tail      [0:NUM_QUEUES-1],
  output logic [15:0]                    q_depth     [0:NUM_QUEUES-1],

  // Overflow flags (sticky, W1C)
  output logic [NUM_QUEUES-1:0]          overflow_flags,
  input  logic [NUM_QUEUES-1:0]          overflow_clr,  // W1C clear from register interface

  // Error
  output logic                           any_overflow
);

  localparam int ADDR_W = $clog2(QUEUE_DEPTH);
  localparam int PTR_W  = ADDR_W + 1;  // extra bit for full/empty distinction

  // ---------------------------------------------------------------
  // Per-queue storage and pointers
  // ---------------------------------------------------------------
  logic [DESC_W-1:0] mem [0:NUM_QUEUES-1][0:QUEUE_DEPTH-1];

  logic [PTR_W-1:0] wr_ptr [0:NUM_QUEUES-1];
  logic [PTR_W-1:0] rd_ptr [0:NUM_QUEUES-1];

  // ---------------------------------------------------------------
  // Per-queue logic (generate)
  // ---------------------------------------------------------------
  genvar qi;
  generate
    for (qi = 0; qi < NUM_QUEUES; qi = qi + 1) begin : GEN_QUEUE

      // Derived signals
      wire [ADDR_W-1:0] wr_addr = wr_ptr[qi][ADDR_W-1:0];
      wire [ADDR_W-1:0] rd_addr = rd_ptr[qi][ADDR_W-1:0];
      wire               q_full  = (wr_ptr[qi][PTR_W-1] != rd_ptr[qi][PTR_W-1]) &&
                                    (wr_ptr[qi][ADDR_W-1:0] == rd_ptr[qi][ADDR_W-1:0]);
      wire               q_empty = (wr_ptr[qi] == rd_ptr[qi]);

      // Push/pop handshake
      assign push_ready[qi] = ~q_full;
      assign pop_ready[qi]  = ~q_empty;

      // Status outputs
      assign q_head[qi]  = {{(16-PTR_W){1'b0}}, rd_ptr[qi]};
      assign q_tail[qi]  = {{(16-PTR_W){1'b0}}, wr_ptr[qi]};
      assign q_depth[qi] = 16'(wr_ptr[qi] - rd_ptr[qi]);

      // Write path
      always_ff @(posedge clk) begin
        if (push_valid[qi] & push_ready[qi])
          mem[qi][wr_addr] <= push_data[qi];
      end

      // Write pointer
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
          wr_ptr[qi] <= '0;
        else if (push_valid[qi] & push_ready[qi])
          wr_ptr[qi] <= wr_ptr[qi] + 1'b1;
      end

      // Read pointer
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
          rd_ptr[qi] <= '0;
        else if (pop_valid[qi] & pop_ready[qi])
          rd_ptr[qi] <= rd_ptr[qi] + 1'b1;
      end

      // Read data (combinational for low latency — consumer sees data same cycle as pop_ready)
      assign pop_data[qi] = mem[qi][rd_addr];

      // Overflow detection (sticky)
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
          overflow_flags[qi] <= 1'b0;
        else if (overflow_clr[qi])
          overflow_flags[qi] <= 1'b0;    // W1C
        else if (push_valid[qi] & q_full)
          overflow_flags[qi] <= 1'b1;    // sticky set
      end

    end
  endgenerate

  // Any overflow status
  assign any_overflow = |overflow_flags;

endmodule

`default_nettype wire
