// dma_bridge.sv — DMA Bridge for ORBIT-G2 Proto-B
// SSOT: ORBIT_G2_REG_SPEC.md section 5, ORBIT_G2_PCIE_BAR_SPEC.md section 4
//
// Converts BAR4 DMA register writes into internal DMA requests.
// State machine matches orbit_dma.py / orbit_protob_mock.py semantics exactly.
//
// Register interface (BAR4 side):
//   DMA_SUBMIT_LO  (0x00) WO — iova low
//   DMA_SUBMIT_HI  (0x04) WO — iova high
//   DMA_LEN        (0x08) WO — byte count
//   DMA_CTRL       (0x0C) RW — START/DIR/QUEUE/QOS/IRQ_EN
//   DMA_STATUS     (0x10) RO — BUSY/DONE/ERR/TIMEOUT/INFLIGHT
//   DMA_ERR_CODE   (0x14) RO — last error code
//   DMA_THROTTLE   (0x18) RW — bandwidth clamp
//   DMA_TIMEOUT    (0x1C) RW — timeout cycles
//
// Internal interface (to memory subsystem):
//   dma_req_valid/ready/addr/len/dir → issue burst request
//   dma_done/dma_err/dma_timeout     ← completion feedback
`timescale 1ns/1ps
`default_nettype none

module dma_bridge (
  input  logic        clk,
  input  logic        rst_n,

  // ── BAR4 register interface (from pcie_ep_versal) ──────────
  input  logic        reg_valid,
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [15:0] reg_addr,    // BAR4 offset (only [4:0] decoded for 8 registers)
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic        reg_wr,
  input  logic [31:0] reg_wdata,
  output logic [31:0] reg_rdata,
  output logic        reg_rvalid,

  // ── Internal DMA request interface (to memory subsystem) ───
  output logic        dma_req_valid,
  input  logic        dma_req_ready,
  output logic [63:0] dma_req_addr,
  output logic [31:0] dma_req_len,
  output logic        dma_req_dir,    // 0=H2D, 1=D2H
  output logic [1:0]  dma_req_queue,
  output logic [3:0]  dma_req_qos,

  // ── Completion feedback ────────────────────────────────────
  input  logic        dma_done,       // transfer complete
  input  logic        dma_err,        // error on transfer
  input  logic [7:0]  dma_err_code_in,
  input  logic        dma_timeout_in, // timeout detected

  // ── IRQ output ─────────────────────────────────────────────
  output logic        irq_dma_done,   // pulse on completion
  output logic        irq_dma_error   // pulse on error
);

  // ═══════════════════════════════════════════════════════════
  // Shadow registers (capture from BAR4 writes)
  // ═══════════════════════════════════════════════════════════
  logic [31:0] submit_lo_r, submit_hi_r;
  logic [31:0] len_r;
  logic [31:0] ctrl_r;
  logic [31:0] throttle_r;
  logic [31:0] timeout_r;

  // Status (read-only, computed)
  logic        st_busy;
  logic        st_done;
  logic        st_err;
  logic        st_timeout;
  logic [7:0]  st_inflight;
  logic [7:0]  err_code_r;

  // ═══════════════════════════════════════════════════════════
  // Register write decode
  // ═══════════════════════════════════════════════════════════
  logic start_pulse;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      submit_lo_r <= 32'd0;
      submit_hi_r <= 32'd0;
      len_r       <= 32'd0;
      ctrl_r      <= 32'd0;
      throttle_r  <= 32'd0;
      timeout_r   <= 32'd0;
    end else if (reg_valid && reg_wr) begin
      case (reg_addr[4:0])
        5'h00: submit_lo_r <= reg_wdata;
        5'h04: submit_hi_r <= reg_wdata;
        5'h08: len_r       <= reg_wdata;
        5'h0C: ctrl_r      <= reg_wdata;
        5'h18: throttle_r  <= reg_wdata;
        5'h1C: timeout_r   <= reg_wdata;
        default: ;
      endcase
    end
  end

  // START is bit[0] of CTRL — pulse on write with START=1
  assign start_pulse = reg_valid && reg_wr && (reg_addr[4:0] == 5'h0C) && reg_wdata[0];

  // ═══════════════════════════════════════════════════════════
  // Register read mux
  // ═══════════════════════════════════════════════════════════
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      reg_rdata  <= 32'd0;
      reg_rvalid <= 1'b0;
    end else begin
      reg_rvalid <= reg_valid && !reg_wr;
      if (reg_valid && !reg_wr) begin
        case (reg_addr[4:0])
          5'h0C: reg_rdata <= ctrl_r;
          5'h10: reg_rdata <= {16'd0, st_inflight, 4'd0, st_timeout, st_err, st_done, st_busy};
          5'h14: reg_rdata <= {24'd0, err_code_r};
          5'h18: reg_rdata <= throttle_r;
          5'h1C: reg_rdata <= timeout_r;
          default: reg_rdata <= 32'd0;
        endcase
      end
    end
  end

  // ═══════════════════════════════════════════════════════════
  // DMA State Machine
  // ═══════════════════════════════════════════════════════════
  typedef enum logic [2:0] {
    ST_IDLE,
    ST_SUBMIT,
    ST_WAIT,
    ST_DONE,
    ST_ERROR
  } dma_state_t;

  dma_state_t state;

  // Timeout counter
  logic [31:0] timeout_cnt;
  wire          timeout_hit = (timeout_r != 0) && (timeout_cnt >= timeout_r);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= ST_IDLE;
      st_busy     <= 1'b0;
      st_done     <= 1'b0;
      st_err      <= 1'b0;
      st_timeout  <= 1'b0;
      st_inflight <= 8'd0;
      err_code_r  <= 8'd0;
      timeout_cnt <= 32'd0;
    end else begin
      case (state)
        ST_IDLE: begin
          if (start_pulse) begin
            // Clear previous status on new submit
            st_done    <= 1'b0;
            st_err     <= 1'b0;
            st_timeout <= 1'b0;
            err_code_r <= 8'd0;
            st_busy    <= 1'b1;
            st_inflight <= 8'd1;
            timeout_cnt <= 32'd0;
            state      <= ST_SUBMIT;
          end
        end

        ST_SUBMIT: begin
          if (dma_req_ready) begin
            state <= ST_WAIT;
          end
        end

        ST_WAIT: begin
          timeout_cnt <= timeout_cnt + 1'b1;

          if (dma_done) begin
            st_done <= 1'b1;
            st_busy <= 1'b0;
            st_inflight <= 8'd0;
            state   <= ST_DONE;
          end else if (dma_err) begin
            st_err     <= 1'b1;
            err_code_r <= dma_err_code_in;
            st_busy    <= 1'b0;
            st_inflight <= 8'd0;
            state      <= ST_ERROR;
          end else if (dma_timeout_in || timeout_hit) begin
            st_timeout <= 1'b1;
            st_busy    <= 1'b0;
            st_inflight <= 8'd0;
            state      <= ST_ERROR;
          end
        end

        ST_DONE: begin
          state <= ST_IDLE;
        end

        ST_ERROR: begin
          state <= ST_IDLE;
        end

        default: state <= ST_IDLE;
      endcase
    end
  end

  // ═══════════════════════════════════════════════════════════
  // Internal DMA request output
  // ═══════════════════════════════════════════════════════════
  assign dma_req_valid = (state == ST_SUBMIT);
  assign dma_req_addr  = {submit_hi_r, submit_lo_r};
  assign dma_req_len   = len_r;
  assign dma_req_dir   = ctrl_r[1];     // DIR bit
  assign dma_req_queue = ctrl_r[3:2];   // QUEUE bits
  assign dma_req_qos   = ctrl_r[7:4];   // QOS bits

  // ═══════════════════════════════════════════════════════════
  // IRQ pulses
  // ═══════════════════════════════════════════════════════════
  wire irq_en = ctrl_r[8];

  logic done_d, err_d;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin done_d <= 1'b0; err_d <= 1'b0; end
    else begin done_d <= st_done; err_d <= st_err | st_timeout; end
  end

  assign irq_dma_done  = irq_en & st_done & ~done_d;
  assign irq_dma_error = irq_en & (st_err | st_timeout) & ~err_d;

endmodule

`default_nettype wire
