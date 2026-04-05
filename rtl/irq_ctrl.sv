// irq_ctrl.sv — Interrupt Controller for ORBIT-G2
// SSOT: ORBIT_G2_REG_SPEC.md section 10, ORBIT_G2_DETAIL_BLOCKDIAG.md section 8
//
// Features:
//   - 32-bit IRQ_PENDING: set by hardware sources, W1C by software
//   - 32-bit IRQ_MASK: 1 = masked (blocked), 0 = enabled
//   - 32-bit IRQ_FORCE: test inject (write 1 -> sets pending)
//   - IRQ_CAUSE_LAST: latches last fatal IRQ source (RO, cleared only on reset)
//   - MSI-X ready output: asserted when unmasked pending IRQ exists
//
// IRQ vector map (from REG_SPEC):
//   [0]  IRQ_DESC_DONE       info
//   [1]  IRQ_DMA_DONE        info
//   [2]  IRQ_DMA_ERROR       fatal
//   [3]  IRQ_OOM_PRESSURE    warn
//   [4]  IRQ_OOM_EMERGENCY   fatal
//   [5]  IRQ_TC0_FAULT       fatal
//   [6]  IRQ_TC1_FAULT       fatal
//   [7]  IRQ_HBM_ECC_CORR   warn
//   [8]  IRQ_HBM_ECC_UNCORR fatal
//   [9]  IRQ_ICI_MAILBOX     info
//   [10] IRQ_WATCHDOG        fatal
//   [11] IRQ_TRACE_WRAP      info
//   [31:12] reserved
`timescale 1ns/1ps
`default_nettype none

module irq_ctrl (
  // Note: 12 sources defined in REG_SPEC (bits [11:0]), bus is 32-bit for future expansion
  input  logic        clk,
  input  logic        rst_n,

  // Hardware IRQ sources (active-high pulses, 1 cycle)
  input  logic [31:0] irq_sources,

  // Register interface: W1C write to IRQ_PENDING
  input  logic        pending_w1c_en,     // host writes to IRQ_PENDING
  input  logic [31:0] pending_w1c_data,   // bits to clear (W1C)

  // Register interface: IRQ_MASK (RW)
  input  logic        mask_wr_en,
  input  logic [31:0] mask_wr_data,

  // Register interface: IRQ_FORCE (write 1 = inject)
  input  logic        force_wr_en,
  input  logic [31:0] force_wr_data,

  // Status outputs (to register readback)
  output logic [31:0] irq_pending,
  output logic [31:0] irq_mask,
  output logic [31:0] irq_cause_last,     // last fatal cause

  // Fatal classification bitmap (which bits are fatal)
  // Default: bits 2,4,5,6,8,10 are fatal (from REG_SPEC)
  // Can be overridden via parameter if needed

  // IRQ output to host / MSI-X interface
  output logic        irq_out,            // level: any unmasked pending
  output logic        msix_req,           // pulse: new unmasked IRQ detected
  output logic [4:0]  msix_vector         // lowest pending unmasked vector number
);

  // ---------------------------------------------------------------
  // Fatal bitmap (hardcoded from SSOT)
  // ---------------------------------------------------------------
  localparam logic [31:0] FATAL_BITMAP = 32'b0000_0000_0000_0000_0000_0101_0111_0100;
  // bits: 2(DMA_ERROR), 4(OOM_EMERG), 5(TC0_FAULT), 6(TC1_FAULT),
  //       8(HBM_ECC_UNCORR), 10(WATCHDOG)

  // ---------------------------------------------------------------
  // IRQ_PENDING register (W1C)
  // ---------------------------------------------------------------
  logic [31:0] pending_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      pending_r <= 32'd0;
    end else begin
      logic [31:0] set_bits;
      logic [31:0] clr_bits;

      // Sources that want to set pending
      set_bits = irq_sources;

      // Force inject
      if (force_wr_en)
        set_bits = set_bits | force_wr_data;

      // W1C clear
      clr_bits = 32'd0;
      if (pending_w1c_en)
        clr_bits = pending_w1c_data;

      // ── W1C Priority Policy: SET WINS ──
      // If a HW source fires the same cycle SW writes W1C on the same bit,
      // the pending bit stays SET. This prevents interrupt loss.
      //   Formula: (pending & ~clr) | set
      //   - clr is applied first (removes old pending)
      //   - set is applied second (re-asserts if HW fired this cycle)
      // Contrast: (pending | set) & ~clr would be CLEAR WINS (rejected).
      // Validated in: tb_irq_ctrl_w1c::test_set_wins_over_clear
      pending_r <= (pending_r & ~clr_bits) | set_bits;
    end
  end

  assign irq_pending = pending_r;

  // ---------------------------------------------------------------
  // IRQ_MASK register (RW)
  // ---------------------------------------------------------------
  logic [31:0] mask_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      mask_r <= 32'hFFFF_FFFF;  // all masked by default after reset
    else if (mask_wr_en)
      mask_r <= mask_wr_data;
  end

  assign irq_mask = mask_r;

  // ---------------------------------------------------------------
  // IRQ output
  // ---------------------------------------------------------------
  wire [31:0] active_irqs = pending_r & ~mask_r;

  assign irq_out = |active_irqs;

  // ---------------------------------------------------------------
  // MSI-X request (edge detect on new active IRQ)
  // ---------------------------------------------------------------
  logic [31:0] active_irqs_d;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
      active_irqs_d <= 32'd0;
    else
      active_irqs_d <= active_irqs;
  end

  wire [31:0] new_active = active_irqs & ~active_irqs_d;
  assign msix_req = |new_active;

  // Priority encoder: lowest set bit of new_active
  always_comb begin
    msix_vector = 5'd0;
    for (int i = 31; i >= 0; i = i - 1) begin
      if (new_active[i])
        msix_vector = i[4:0];
    end
  end

  // ---------------------------------------------------------------
  // IRQ_CAUSE_LAST (last fatal cause, latched until reset)
  // ---------------------------------------------------------------
  logic [31:0] cause_last_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cause_last_r <= 32'd0;
    end else begin
      // Latch any new fatal source
      logic [31:0] fatal_events;
      fatal_events = irq_sources & FATAL_BITMAP;
      if (|fatal_events)
        cause_last_r <= fatal_events;
    end
  end

  assign irq_cause_last = cause_last_r;

endmodule

`default_nettype wire
