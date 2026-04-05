// cdc_fifo.sv — Asynchronous FIFO with Gray-code pointers for ORBIT-G2
// SSOT: ORBIT_G2_RTL_SKELETONS.md section 2.7
//
// Standard async FIFO:
//   - Independent wr_clk / rd_clk
//   - Gray-coded write/read pointers for safe CDC
//   - Registered full/empty flags (conservative, no false negatives)
//   - Parameterized depth (must be power of 2) and data width
`timescale 1ns/1ps
`default_nettype none

module cdc_fifo #(
  parameter int DATA_W = 32,
  parameter int DEPTH  = 16    // must be power of 2, minimum 4
)(
  // Static constraint: DEPTH must be >= 4.
  // DEPTH=2 causes negative bit-select in gray-code full comparison
  // (rd_gray_sync2[PTR_W-3:0] where PTR_W=2 → index -1).
  initial begin
    if (DEPTH < 4) begin
      $error("cdc_fifo: DEPTH must be >= 4 (got %0d). DEPTH=2 causes gray full comparison bug.", DEPTH);
      $finish;
    end
    if ((DEPTH & (DEPTH - 1)) != 0) begin
      $error("cdc_fifo: DEPTH must be power of 2 (got %0d).", DEPTH);
      $finish;
    end
  end
  // Write domain
  input  logic              wr_clk,
  input  logic              wr_rst_n,
  input  logic              wr_valid,
  output logic              wr_ready,
  input  logic [DATA_W-1:0] wr_data,
  output logic              full,

  // Read domain
  input  logic              rd_clk,
  input  logic              rd_rst_n,
  input  logic              rd_valid,   // read request
  output logic              rd_ready,   // data available
  output logic [DATA_W-1:0] rd_data,
  output logic              empty
);

  localparam int ADDR_W = $clog2(DEPTH);
  localparam int PTR_W  = ADDR_W + 1;  // extra MSB for wrap detection

  // ---------------------------------------------------------------
  // Memory
  // ---------------------------------------------------------------
  logic [DATA_W-1:0] mem [0:DEPTH-1];

  // ---------------------------------------------------------------
  // Binary <-> Gray conversion
  // ---------------------------------------------------------------
  function automatic logic [PTR_W-1:0] bin2gray(input logic [PTR_W-1:0] b);
    bin2gray = b ^ (b >> 1);
  endfunction

  function automatic logic [PTR_W-1:0] gray2bin(input logic [PTR_W-1:0] g);
    logic [PTR_W-1:0] b;
    integer i;
    begin
      b[PTR_W-1] = g[PTR_W-1];
      for (i = PTR_W-2; i >= 0; i = i - 1)
        b[i] = b[i+1] ^ g[i];
      gray2bin = b;
    end
  endfunction

  // ---------------------------------------------------------------
  // Write domain
  // ---------------------------------------------------------------
  logic [PTR_W-1:0] wr_bin, wr_bin_next;
  logic [PTR_W-1:0] wr_gray, wr_gray_next;
  logic [PTR_W-1:0] rd_gray_sync1, rd_gray_sync2;  // rd gray -> wr domain

  assign wr_bin_next  = wr_bin + PTR_W'(wr_valid & wr_ready);
  assign wr_gray_next = bin2gray(wr_bin_next);

  always_ff @(posedge wr_clk or negedge wr_rst_n) begin
    if (!wr_rst_n) begin
      wr_bin  <= '0;
      wr_gray <= '0;
    end else begin
      wr_bin  <= wr_bin_next;
      wr_gray <= wr_gray_next;
    end
  end

  // Write to memory
  always_ff @(posedge wr_clk) begin
    if (wr_valid & wr_ready)
      mem[wr_bin[ADDR_W-1:0]] <= wr_data;
  end

  // Synchronize rd_gray into wr domain (2-FF)
  always_ff @(posedge wr_clk or negedge wr_rst_n) begin
    if (!wr_rst_n) begin
      rd_gray_sync1 <= '0;
      rd_gray_sync2 <= '0;
    end else begin
      rd_gray_sync1 <= rd_gray;   // from rd domain
      rd_gray_sync2 <= rd_gray_sync1;
    end
  end

  // Full: gray pointers match with top 2 MSBs inverted, rest identical.
  // Proof (Cummings SNUG-2002):
  //   Binary full = {wr[MSB] != rd[MSB], wr[ADDR-1:0] == rd[ADDR-1:0]}
  //   In gray code, this maps to the top 2 gray bits being inverted
  //   because bin2gray flips MSB-1 when MSB differs.
  //   Verified: DEPTH=4 (PTR_W=3), all 4 full pairs match this pattern.
  logic full_val;
  assign full_val = (wr_gray_next == {~rd_gray_sync2[PTR_W-1:PTR_W-2],
                                       rd_gray_sync2[PTR_W-3:0]});

  always_ff @(posedge wr_clk or negedge wr_rst_n) begin
    if (!wr_rst_n)
      full <= 1'b0;
    else
      full <= full_val;
  end

  assign wr_ready = ~full;

  // ---------------------------------------------------------------
  // Read domain
  // ---------------------------------------------------------------
  logic [PTR_W-1:0] rd_bin, rd_bin_next;
  logic [PTR_W-1:0] rd_gray, rd_gray_next;
  logic [PTR_W-1:0] wr_gray_sync1, wr_gray_sync2;  // wr gray -> rd domain

  wire rd_handshake = rd_valid & rd_ready;

  assign rd_bin_next  = rd_bin + PTR_W'(rd_handshake);
  assign rd_gray_next = bin2gray(rd_bin_next);

  always_ff @(posedge rd_clk or negedge rd_rst_n) begin
    if (!rd_rst_n) begin
      rd_bin  <= '0;
      rd_gray <= '0;
    end else begin
      rd_bin  <= rd_bin_next;
      rd_gray <= rd_gray_next;
    end
  end

  // Read data (registered output for cleaner timing)
  always_ff @(posedge rd_clk or negedge rd_rst_n) begin
    if (!rd_rst_n)
      rd_data <= '0;
    else if (rd_handshake)
      rd_data <= mem[rd_bin[ADDR_W-1:0]];
  end

  // Synchronize wr_gray into rd domain (2-FF)
  always_ff @(posedge rd_clk or negedge rd_rst_n) begin
    if (!rd_rst_n) begin
      wr_gray_sync1 <= '0;
      wr_gray_sync2 <= '0;
    end else begin
      wr_gray_sync1 <= wr_gray;   // from wr domain
      wr_gray_sync2 <= wr_gray_sync1;
    end
  end

  // Empty: gray pointers match exactly
  logic empty_val;
  assign empty_val = (rd_gray_next == wr_gray_sync2);

  always_ff @(posedge rd_clk or negedge rd_rst_n) begin
    if (!rd_rst_n)
      empty <= 1'b1;   // empty after reset
    else
      empty <= empty_val;
  end

  assign rd_ready = ~empty;

endmodule

`default_nettype wire
