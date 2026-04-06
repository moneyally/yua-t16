// scale_fabric_ctrl.sv — ORBIT-G3 Scale-up Fabric Controller
// SSOT: ORBIT_G3_RTL_ISSUES.md (G3-RTL-021)
//
// 2-peer fabric controller with framed valid/ready streaming.
// Half-duplex: send OR recv, not both simultaneously.
// Payload width: 32-bit (one FP32 element per beat).
// Frame = N beats terminated by `last`. N is variable.
//
// Paths:
//   SEND: local_tx → fabric_tx (pass-through with frame tracking)
//   RECV: fabric_rx → peer_rx (pass-through with frame tracking)
//
// Errors:
//   0x01: start while busy
//   0x02: frame abort (reset mid-frame)
`timescale 1ns/1ps
`default_nettype none

module scale_fabric_ctrl (
  input  logic        clk,
  input  logic        rst_n,

  // Control
  input  logic        send_start,    // initiate SEND mode
  input  logic        recv_start,    // initiate RECV mode

  // Status
  output logic        busy,
  output logic        done_pulse,
  output logic [7:0]  err_code,

  // ── Local engine side (upstream) ────────────────────────
  // SEND: local engine pushes payload to fabric
  input  logic        local_tx_valid,
  output logic        local_tx_ready,
  input  logic [31:0] local_tx_data,
  input  logic        local_tx_last,

  // RECV: local engine consumes payload from peer
  output logic        peer_rx_valid,
  input  logic        peer_rx_ready,
  output logic [31:0] peer_rx_data,
  output logic        peer_rx_last,

  // ── Fabric link side (downstream / PHY stub) ───────────
  // SEND: to link
  output logic        fabric_tx_valid,
  input  logic        fabric_tx_ready,
  output logic [31:0] fabric_tx_data,
  output logic        fabric_tx_last,

  // RECV: from link
  input  logic        fabric_rx_valid,
  output logic        fabric_rx_ready,
  input  logic [31:0] fabric_rx_data,
  input  logic        fabric_rx_last
);

  // ═══════════════════════════════════════════════════════════
  // State machine (half-duplex)
  // ═══════════════════════════════════════════════════════════
  typedef enum logic [2:0] {
    ST_IDLE,
    ST_SEND,
    ST_RECV,
    ST_DONE,
    ST_ERROR
  } state_t;

  state_t st;

  assign busy = (st == ST_SEND || st == ST_RECV);

  // ═══════════════════════════════════════════════════════════
  // SEND path: local_tx → fabric_tx (pass-through)
  // ═══════════════════════════════════════════════════════════
  wire send_active = (st == ST_SEND);
  wire send_beat   = send_active & local_tx_valid & fabric_tx_ready;
  wire send_last   = send_beat & local_tx_last;

  assign fabric_tx_valid = send_active ? local_tx_valid : 1'b0;
  assign fabric_tx_data  = local_tx_data;
  assign fabric_tx_last  = send_active ? local_tx_last : 1'b0;
  assign local_tx_ready  = send_active ? fabric_tx_ready : 1'b0;

  // ═══════════════════════════════════════════════════════════
  // RECV path: fabric_rx → peer_rx (pass-through)
  // ═══════════════════════════════════════════════════════════
  wire recv_active = (st == ST_RECV);
  wire recv_beat   = recv_active & fabric_rx_valid & peer_rx_ready;
  wire recv_last   = recv_beat & fabric_rx_last;

  assign peer_rx_valid   = recv_active ? fabric_rx_valid : 1'b0;
  assign peer_rx_data    = fabric_rx_data;
  assign peer_rx_last    = recv_active ? fabric_rx_last : 1'b0;
  assign fabric_rx_ready = recv_active ? peer_rx_ready : 1'b0;

  // ═══════════════════════════════════════════════════════════
  // Beat counter (for observability / debug)
  // ═══════════════════════════════════════════════════════════
  logic [15:0] beat_cnt;

  // ═══════════════════════════════════════════════════════════
  // FSM
  // ═══════════════════════════════════════════════════════════
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st         <= ST_IDLE;
      done_pulse <= 1'b0;
      err_code   <= 8'd0;
      beat_cnt   <= 16'd0;
    end else begin
      done_pulse <= 1'b0;

      case (st)
        ST_IDLE: begin
          err_code <= 8'd0;
          beat_cnt <= 16'd0;

          if (send_start && recv_start) begin
            // Both requested simultaneously → error
            err_code <= 8'h01;
            st       <= ST_ERROR;
          end else if (send_start) begin
            st <= ST_SEND;
          end else if (recv_start) begin
            st <= ST_RECV;
          end
        end

        ST_SEND: begin
          if (send_beat) begin
            beat_cnt <= beat_cnt + 1;
            if (local_tx_last) begin
              st <= ST_DONE;
            end
          end
          // Detect illegal start during active transfer
          if (send_start || recv_start) begin
            err_code <= 8'h01;
            st       <= ST_ERROR;
          end
        end

        ST_RECV: begin
          if (recv_beat) begin
            beat_cnt <= beat_cnt + 1;
            if (fabric_rx_last) begin
              st <= ST_DONE;
            end
          end
          if (send_start || recv_start) begin
            err_code <= 8'h01;
            st       <= ST_ERROR;
          end
        end

        ST_DONE: begin
          done_pulse <= 1'b1;
          st         <= ST_IDLE;
        end

        ST_ERROR: begin
          done_pulse <= 1'b1;
          st         <= ST_IDLE;
        end

        default: st <= ST_IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
