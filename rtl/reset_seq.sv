// reset_seq.sv — Reset Sequencer for ORBIT-G2
// SSOT: ORBIT_G2_REG_SPEC.md section 3, ORBIT_G2_DETAIL_BLOCKDIAG.md section 5.2
//
// Reset release order: always_on -> io -> mem -> core
// Each domain release is delayed by RELEASE_DELAY cycles (glitch-free FF chain).
// Reset cause is latched on any reset event and held until software clears via sw_cause_clr.
//
// Supports:
//   - Power-on reset (por_n, active-low)
//   - Software reset (sw_reset pulse, from SW_RESET register bit[0])
//   - Watchdog reset (wdog_reset pulse)
//   - PCIe FLR reset (pcie_flr pulse)
`timescale 1ns/1ps
`default_nettype none

module reset_seq #(
  parameter int RELEASE_DELAY = 4   // cycles between domain releases (min 2)
)(
  input  logic        clk,          // always-on clock domain
  input  logic        por_n,        // power-on reset (active-low, async)

  // Reset trigger inputs (active-high pulses, synchronous to clk)
  input  logic        sw_reset,     // REG_SPEC SW_RESET[0] TRIGGER
  input  logic        wdog_reset,   // watchdog timeout
  input  logic        pcie_flr,     // PCIe FLR

  // Domain reset outputs (active-low, glitch-free)
  output logic        rst_io_n,     // IO domain reset
  output logic        rst_mem_n,    // memory domain reset
  output logic        rst_core_n,   // core/compute domain reset

  // Reset cause (matches REG_SPEC BOOT_CAUSE bit layout)
  // [0]=POR, [1]=WDOG, [2]=SW, [3]=PCIE_FLR
  output logic [3:0]  boot_cause,

  // Software clear for boot_cause (write from register interface)
  input  logic        sw_cause_clr,

  // Status
  output logic        reset_active  // 1 while any domain is held in reset
);

  // ---------------------------------------------------------------
  // Reset cause latch
  // ---------------------------------------------------------------
  // Latched on any reset event, held until sw_cause_clr.
  // por_n is async — synchronize it first.
  logic por_sync_n;
  logic por_meta;

  // 2-FF synchronizer for por_n (async -> sync)
  always_ff @(posedge clk or negedge por_n) begin
    if (!por_n) begin
      por_meta   <= 1'b0;
      por_sync_n <= 1'b0;
    end else begin
      por_meta   <= 1'b1;
      por_sync_n <= por_meta;
    end
  end

  // Combined reset detect (active-high, 1 cycle)
  logic any_reset;
  // Detect por_sync_n still low (POR active after synchronizer)
  logic por_sync_n_d;
  always_ff @(posedge clk or negedge por_n) begin
    if (!por_n)
      por_sync_n_d <= 1'b0;
    else
      por_sync_n_d <= por_sync_n;
  end

  assign any_reset  = (~por_sync_n) | sw_reset | wdog_reset | pcie_flr;

  // Boot cause register
  always_ff @(posedge clk or negedge por_n) begin
    if (!por_n) begin
      boot_cause <= 4'b0001;  // POR is the initial cause
    end else begin
      if (sw_cause_clr) begin
        boot_cause <= 4'b0000;
      end else begin
        // Latch new causes (sticky OR)
        if (~por_sync_n & ~por_sync_n_d)  boot_cause[0] <= 1'b1;  // POR
        if (wdog_reset)                    boot_cause[1] <= 1'b1;  // WDOG
        if (sw_reset)                      boot_cause[2] <= 1'b1;  // SW
        if (pcie_flr)                      boot_cause[3] <= 1'b1;  // PCIE_FLR
      end
    end
  end

  // ---------------------------------------------------------------
  // Sequenced release counter
  // ---------------------------------------------------------------
  // On any_reset: all domains asserted, counter starts from 0.
  // Counter increments each cycle. Domains release at:
  //   io:   count == RELEASE_DELAY
  //   mem:  count == 2 * RELEASE_DELAY
  //   core: count == 3 * RELEASE_DELAY
  //
  // Counter width: needs to count up to 3 * RELEASE_DELAY
  localparam int MAX_COUNT     = 3 * RELEASE_DELAY + 1;
  localparam int CNT_W         = $clog2(MAX_COUNT + 1);
  localparam [CNT_W-1:0] THR_IO   = CNT_W'(RELEASE_DELAY);
  localparam [CNT_W-1:0] THR_MEM  = CNT_W'(2 * RELEASE_DELAY);
  localparam [CNT_W-1:0] THR_CORE = CNT_W'(3 * RELEASE_DELAY);

  logic [CNT_W-1:0] release_cnt;
  logic              sequencing;    // 1 while release sequence is in progress

  always_ff @(posedge clk or negedge por_n) begin
    if (!por_n) begin
      release_cnt <= '0;
      sequencing  <= 1'b1;
    end else begin
      if (any_reset) begin
        release_cnt <= '0;
        sequencing  <= 1'b1;
      end else if (sequencing) begin
        if (release_cnt < CNT_W'(MAX_COUNT))
          release_cnt <= release_cnt + 1'b1;
        else
          sequencing <= 1'b0;
      end
    end
  end

  // ---------------------------------------------------------------
  // Domain reset output FFs (glitch-free)
  // ---------------------------------------------------------------
  // Each output is registered — no combinational glitches on reset release.
  always_ff @(posedge clk or negedge por_n) begin
    if (!por_n) begin
      rst_io_n   <= 1'b0;
      rst_mem_n  <= 1'b0;
      rst_core_n <= 1'b0;
    end else begin
      if (any_reset) begin
        rst_io_n   <= 1'b0;
        rst_mem_n  <= 1'b0;
        rst_core_n <= 1'b0;
      end else begin
        if (release_cnt >= THR_IO)
          rst_io_n <= 1'b1;
        if (release_cnt >= THR_MEM)
          rst_mem_n <= 1'b1;
        if (release_cnt >= THR_CORE)
          rst_core_n <= 1'b1;
      end
    end
  end

  // ---------------------------------------------------------------
  // Status
  // ---------------------------------------------------------------
  assign reset_active = ~rst_io_n | ~rst_mem_n | ~rst_core_n;

endmodule

`default_nettype wire
