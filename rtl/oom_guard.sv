// oom_guard.sv — OOM Guard / Memory Pressure Controller for ORBIT-G2
// SSOT: ORBIT_G2_REG_SPEC.md section 6, ORBIT_G2_RTL_SKELETONS.md section 2.3
//
// Tracks effective memory usage:
//   effective = allocated + dma_reserved + prefetch_reserved
//
// State machine:
//   NORMAL   (0) -> PRESSURE (1) -> CRITICAL (2) -> EMERG (3)
//   with hysteresis on downward transitions
//
// Outputs:
//   admission_stop:  block new descriptor admission (EMERG)
//   prefetch_clamp:  disable prefetch (CRITICAL+)
//
// Register interface signals match REG_SPEC:
//   OOM_USAGE_LO/HI, OOM_RESV_LO/HI, OOM_THRESH0/1/2, OOM_STATE, OOM_ACTION
`timescale 1ns/1ps
`default_nettype none

module oom_guard #(
  parameter int USAGE_W = 40   // width for byte counters (up to 1TB)
)(
  input  logic              clk,
  input  logic              rst_n,

  // Allocation tracking (delta interface)
  input  logic              alloc_inc,           // allocated += alloc_bytes
  input  logic              alloc_dec,           // allocated -= alloc_bytes
  input  logic [31:0]       alloc_bytes,         // size for inc/dec

  // DMA reservation tracking
  input  logic              dma_reserve_inc,     // dma_reserved += rsv_bytes
  input  logic              dma_reserve_dec,     // dma_reserved -= rsv_bytes (on completion)
  input  logic [31:0]       dma_rsv_bytes,

  // Prefetch reservation tracking
  input  logic              prefetch_reserve_inc,
  input  logic              prefetch_reserve_dec,
  input  logic [31:0]       prefetch_rsv_bytes,

  // Threshold configuration (from register interface)
  // REG_SPEC: OOM_THRESH0 = pressure, OOM_THRESH1 = critical, OOM_THRESH2 = emergency
  input  logic [USAGE_W-1:0] thresh_pressure,
  input  logic [USAGE_W-1:0] thresh_critical,
  input  logic [USAGE_W-1:0] thresh_emergency,

  // Status outputs (map to REG_SPEC OOM_STATE)
  output logic [1:0]        pressure_state,      // 0=NORMAL,1=PRESSURE,2=CRITICAL,3=EMERG
  output logic              admission_stop,      // OOM_STATE[8]
  output logic              prefetch_clamp,      // OOM_STATE[9]

  // Usage counters (for register readback)
  output logic [USAGE_W-1:0] allocated_bytes,
  output logic [USAGE_W-1:0] reserved_bytes,     // dma_reserved + prefetch_reserved
  output logic [USAGE_W-1:0] effective_usage,

  // Error
  output logic              underflow_error      // counter went negative (bug indicator)
);

  // ---------------------------------------------------------------
  // Usage counters
  // ---------------------------------------------------------------
  logic [USAGE_W-1:0] alloc_r;
  logic [USAGE_W-1:0] dma_rsv_r;
  logic [USAGE_W-1:0] prefetch_rsv_r;

  // Allocation counter
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      alloc_r <= '0;
    end else begin
      case ({alloc_inc, alloc_dec})
        2'b10:   alloc_r <= alloc_r + USAGE_W'(alloc_bytes);
        2'b01:   alloc_r <= (alloc_r >= USAGE_W'(alloc_bytes)) ? (alloc_r - USAGE_W'(alloc_bytes)) : '0;
        2'b11:   alloc_r <= alloc_r;  // inc and dec same cycle: nop
        default: alloc_r <= alloc_r;
      endcase
    end
  end

  // DMA reservation counter
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      dma_rsv_r <= '0;
    end else begin
      case ({dma_reserve_inc, dma_reserve_dec})
        2'b10:   dma_rsv_r <= dma_rsv_r + USAGE_W'(dma_rsv_bytes);
        2'b01:   dma_rsv_r <= (dma_rsv_r >= USAGE_W'(dma_rsv_bytes)) ? (dma_rsv_r - USAGE_W'(dma_rsv_bytes)) : '0;
        2'b11:   dma_rsv_r <= dma_rsv_r;
        default: dma_rsv_r <= dma_rsv_r;
      endcase
    end
  end

  // Prefetch reservation counter
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      prefetch_rsv_r <= '0;
    end else begin
      case ({prefetch_reserve_inc, prefetch_reserve_dec})
        2'b10:   prefetch_rsv_r <= prefetch_rsv_r + USAGE_W'(prefetch_rsv_bytes);
        2'b01:   prefetch_rsv_r <= (prefetch_rsv_r >= USAGE_W'(prefetch_rsv_bytes)) ?
                                    (prefetch_rsv_r - USAGE_W'(prefetch_rsv_bytes)) : '0;
        2'b11:   prefetch_rsv_r <= prefetch_rsv_r;
        default: prefetch_rsv_r <= prefetch_rsv_r;
      endcase
    end
  end

  // Effective usage (combinational)
  assign effective_usage = alloc_r + dma_rsv_r + prefetch_rsv_r;
  assign allocated_bytes = alloc_r;
  assign reserved_bytes  = dma_rsv_r + prefetch_rsv_r;

  // Underflow detection
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      underflow_error <= 1'b0;
    else begin
      if (alloc_dec && (alloc_r < USAGE_W'(alloc_bytes)))
        underflow_error <= 1'b1;
      if (dma_reserve_dec && (dma_rsv_r < USAGE_W'(dma_rsv_bytes)))
        underflow_error <= 1'b1;
      if (prefetch_reserve_dec && (prefetch_rsv_r < USAGE_W'(prefetch_rsv_bytes)))
        underflow_error <= 1'b1;
    end
  end

  // ---------------------------------------------------------------
  // Pressure state machine
  // ---------------------------------------------------------------
  typedef enum logic [1:0] {
    ST_NORMAL   = 2'd0,
    ST_PRESSURE = 2'd1,
    ST_CRITICAL = 2'd2,
    ST_EMERG    = 2'd3
  } oom_state_t;

  oom_state_t oom_st;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      oom_st <= ST_NORMAL;
    end else begin
      case (oom_st)
        ST_NORMAL: begin
          if (effective_usage >= thresh_emergency)
            oom_st <= ST_EMERG;
          else if (effective_usage >= thresh_critical)
            oom_st <= ST_CRITICAL;
          else if (effective_usage >= thresh_pressure)
            oom_st <= ST_PRESSURE;
        end

        ST_PRESSURE: begin
          if (effective_usage >= thresh_emergency)
            oom_st <= ST_EMERG;
          else if (effective_usage >= thresh_critical)
            oom_st <= ST_CRITICAL;
          else if (effective_usage < thresh_pressure)
            oom_st <= ST_NORMAL;
        end

        ST_CRITICAL: begin
          if (effective_usage >= thresh_emergency)
            oom_st <= ST_EMERG;
          else if (effective_usage < thresh_pressure)
            oom_st <= ST_NORMAL;
          else if (effective_usage < thresh_critical)
            oom_st <= ST_PRESSURE;
        end

        ST_EMERG: begin
          // EMERG exit: step-down only (no direct jump to NORMAL).
          // < critical  -> PRESSURE (skip CRITICAL, but not NORMAL)
          // < emergency -> CRITICAL
          // This ensures EMERG->NORMAL requires passing through intermediate states.
          if (effective_usage < thresh_critical)
            oom_st <= ST_PRESSURE;
          else if (effective_usage < thresh_emergency)
            oom_st <= ST_CRITICAL;
        end

        default: oom_st <= ST_NORMAL;
      endcase
    end
  end

  // ---------------------------------------------------------------
  // Outputs
  // ---------------------------------------------------------------
  assign pressure_state  = oom_st;
  assign admission_stop  = (oom_st == ST_EMERG);
  assign prefetch_clamp  = (oom_st >= ST_CRITICAL);

endmodule

`default_nettype wire
