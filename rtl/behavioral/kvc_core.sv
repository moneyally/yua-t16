// kvc_core.sv — Behavioral KV-Cache Controller for Icarus Verilog 12
// Stores K and V tensors for transformer attention layers.
// Blocking assignments throughout; flat packed arrays; no dynamic bit-selects
// inside always blocks (unpacked via generate/assign outside always).
`timescale 1ns/1ps
`default_nettype none

module kvc_core #(
  parameter int NUM_LAYERS = 4,    // reduced for simulation
  parameter int NUM_HEADS  = 4,    // reduced for simulation
  parameter int HEAD_DIM   = 16,   // reduced for simulation (real: 128)
  parameter int MAX_SEQ    = 64,   // max sequence length per slot
  parameter int MAX_SEQS   = 4     // max concurrent sequences
)(
  input  logic        clk,
  input  logic        rst_n,

  // Operation control
  input  logic        start,
  output logic        busy,
  output logic        done,

  // Operation type
  // 0 = KVC_WRITE (store K and V for current token)
  // 1 = KVC_READ  (read K and V for all tokens up to seq_len)
  input  logic        op_type,

  // Descriptor fields
  input  logic [7:0]  seq_id,      // sequence identifier
  input  logic [7:0]  layer_id,    // transformer layer
  input  logic [15:0] seq_pos,     // current token position (for WRITE)
  input  logic [15:0] seq_len,     // tokens to read (for READ)

  // K data interface (flat packed FP16)
  // Size: NUM_HEADS * HEAD_DIM elements = NUM_HEADS*HEAD_DIM*16 bits
  input  logic [NUM_HEADS*HEAD_DIM*16-1:0] k_in_flat,   // K to write
  input  logic [NUM_HEADS*HEAD_DIM*16-1:0] v_in_flat,   // V to write
  output logic [NUM_HEADS*HEAD_DIM*MAX_SEQ*16-1:0] k_out_flat, // K read out
  output logic [NUM_HEADS*HEAD_DIM*MAX_SEQ*16-1:0] v_out_flat  // V read out
);

  // ── Unpack k_in / v_in via continuous assigns (outside always) ────────────
  // k_in_r[head_id][elem] — one token's K data per head
  wire [15:0] k_in_r [0:NUM_HEADS-1][0:HEAD_DIM-1];
  wire [15:0] v_in_r [0:NUM_HEADS-1][0:HEAD_DIM-1];

  // k_out_r[tok][head_id][elem] — all tokens' K data
  reg  [15:0] k_out_r [0:MAX_SEQ-1][0:NUM_HEADS-1][0:HEAD_DIM-1];
  reg  [15:0] v_out_r [0:MAX_SEQ-1][0:NUM_HEADS-1][0:HEAD_DIM-1];

  // Packing convention for k_in_flat:
  //   element [head_id][elem] at bit offset: (head_id * HEAD_DIM + elem) * 16
  genvar gh, ge, gt;
  generate
    for (gh = 0; gh < NUM_HEADS; gh = gh+1) begin : UNPACK_IN_HEAD
      for (ge = 0; ge < HEAD_DIM; ge = ge+1) begin : UNPACK_IN_ELEM
        assign k_in_r[gh][ge] = k_in_flat[(gh*HEAD_DIM + ge)*16 +: 16];
        assign v_in_r[gh][ge] = v_in_flat[(gh*HEAD_DIM + ge)*16 +: 16];
      end
    end

    // Packing convention for k_out_flat:
    //   element [head_id][elem] at token t at bit offset:
    //   ((t * NUM_HEADS + head_id) * HEAD_DIM + elem) * 16
    for (gt = 0; gt < MAX_SEQ; gt = gt+1) begin : PACK_OUT_TOK
      for (gh = 0; gh < NUM_HEADS; gh = gh+1) begin : PACK_OUT_HEAD
        for (ge = 0; ge < HEAD_DIM; ge = ge+1) begin : PACK_OUT_ELEM
          assign k_out_flat[((gt*NUM_HEADS + gh)*HEAD_DIM + ge)*16 +: 16] = k_out_r[gt][gh][ge];
          assign v_out_flat[((gt*NUM_HEADS + gh)*HEAD_DIM + ge)*16 +: 16] = v_out_r[gt][gh][ge];
        end
      end
    end
  endgenerate

  // ── Internal KV storage ───────────────────────────────────────────────────
  // kv_store[seq_id][layer_id][seq_pos][head_id][kv][elem]
  // kv=0 → K, kv=1 → V
  // Dimensions: MAX_SEQS * NUM_LAYERS * MAX_SEQ * NUM_HEADS * 2 * HEAD_DIM FP16 values
  reg [15:0] kv_store [0:MAX_SEQS-1][0:NUM_LAYERS-1][0:MAX_SEQ-1][0:NUM_HEADS-1][0:1][0:HEAD_DIM-1];

  // ── State machine ─────────────────────────────────────────────────────────
  localparam ST_IDLE  = 2'd0;
  localparam ST_EXEC  = 2'd1;
  localparam ST_DONE  = 2'd2;

  localparam OP_WRITE = 1'b0;
  localparam OP_READ  = 1'b1;

  reg [1:0]  state;
  reg        op_r;
  reg [7:0]  seq_id_r;
  reg [7:0]  layer_id_r;
  reg [15:0] seq_pos_r;
  reg [15:0] seq_len_r;

  // Loop counters for multi-cycle operations
  reg [15:0] tok_idx;   // current token being processed
  reg [3:0]  head_idx;  // current head being processed
  // For READ: head_idx goes 0..NUM_HEADS-1 per token

  integer init_s, init_l, init_t, init_h, init_e;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state      = ST_IDLE;
      busy       = 1'b0;
      done       = 1'b0;
      tok_idx    = 16'd0;
      head_idx   = 4'd0;
      op_r       = 1'b0;
      seq_id_r   = 8'd0;
      layer_id_r = 8'd0;
      seq_pos_r  = 16'd0;
      seq_len_r  = 16'd0;

      // Clear output registers
      for (init_t = 0; init_t < MAX_SEQ; init_t = init_t+1)
        for (init_h = 0; init_h < NUM_HEADS; init_h = init_h+1)
          for (init_e = 0; init_e < HEAD_DIM; init_e = init_e+1) begin
            k_out_r[init_t][init_h][init_e] = 16'h0;
            v_out_r[init_t][init_h][init_e] = 16'h0;
          end

      // Clear KV store
      for (init_s = 0; init_s < MAX_SEQS; init_s = init_s+1)
        for (init_l = 0; init_l < NUM_LAYERS; init_l = init_l+1)
          for (init_t = 0; init_t < MAX_SEQ; init_t = init_t+1)
            for (init_h = 0; init_h < NUM_HEADS; init_h = init_h+1)
              for (init_e = 0; init_e < HEAD_DIM; init_e = init_e+1) begin
                kv_store[init_s][init_l][init_t][init_h][0][init_e] = 16'h0;
                kv_store[init_s][init_l][init_t][init_h][1][init_e] = 16'h0;
              end
    end else begin
      done = 1'b0;  // default

      case (state)

        ST_IDLE: begin
          if (start) begin
            op_r       = op_type;
            seq_id_r   = seq_id;
            layer_id_r = layer_id;
            seq_pos_r  = seq_pos;
            seq_len_r  = seq_len;
            tok_idx    = 16'd0;
            head_idx   = 4'd0;
            busy       = 1'b1;
            state      = ST_EXEC;
          end
        end

        ST_EXEC: begin
          if (op_r == OP_WRITE) begin
            // KVC_WRITE: Store all heads for current token
            // Process one head per clock cycle
            if (head_idx < NUM_HEADS[3:0]) begin
              begin : WRITE_HEAD
                integer he, si, li, ti;
                si = seq_id_r;
                li = layer_id_r;
                ti = seq_pos_r;
                he = head_idx;
                // Store K and V for this head
                for (int ei = 0; ei < HEAD_DIM; ei = ei+1) begin
                  kv_store[si][li][ti][he][0][ei] = k_in_r[he][ei];
                  kv_store[si][li][ti][he][1][ei] = v_in_r[he][ei];
                end
              end
              head_idx = head_idx + 4'd1;
            end else begin
              // All heads written
              state    = ST_DONE;
              busy     = 1'b0;
              done     = 1'b1;
            end
          end else begin
            // OP_READ: Read K and V for tok_idx, head_idx
            if (tok_idx < seq_len_r) begin
              if (head_idx < NUM_HEADS[3:0]) begin
                begin : READ_HEAD
                  integer he, si, li, ti;
                  si = seq_id_r;
                  li = layer_id_r;
                  ti = tok_idx;
                  he = head_idx;
                  for (int ei = 0; ei < HEAD_DIM; ei = ei+1) begin
                    k_out_r[ti][he][ei] = kv_store[si][li][ti][he][0][ei];
                    v_out_r[ti][he][ei] = kv_store[si][li][ti][he][1][ei];
                  end
                end
                head_idx = head_idx + 4'd1;
              end else begin
                // Move to next token
                head_idx = 4'd0;
                tok_idx  = tok_idx + 16'd1;
              end
            end else begin
              // All tokens read
              state    = ST_DONE;
              busy     = 1'b0;
              done     = 1'b1;
            end
          end
        end

        ST_DONE: begin
          state = ST_IDLE;
        end

        default: begin
          state = ST_IDLE;
        end

      endcase
    end
  end

endmodule
`default_nettype wire
