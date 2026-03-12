// gemm_wb_wrapper.v — Wishbone wrapper for gemm_int4_sky130
// Connects the GEMM INT4 tile to Caravel's Wishbone bus.
//
// Memory map (base = 0x3000_0000):
//   0x000        CTRL    [0]=start, [1]=rst_sync
//   0x004        STATUS  [0]=busy,  [1]=done
//   0x008~0x207  A_BUF   16×16 × 8-bit activations  (2048 bits = 256 bytes)
//   0x208~0x307  B_BUF   16×16 × 4-bit weights      (1024 bits = 128 bytes)
//   0x308~0x387  SC_BUF  16 × 16-bit Q8.8 scale     (256 bits = 32 bytes)
//   0x388~0x787  C_BUF   16×16 × 32-bit output      (8192 bits = 1024 bytes)
//
// Total addressable space: 0x788 = 1928 bytes
// All registers are 32-bit aligned.
//
// Wishbone interface: standard single-cycle (no pipelining)
//   wbs_stb_i + wbs_cyc_i + wbs_we_i -> single-cycle ack
//
`timescale 1ns/1ps

module gemm_wb_wrapper (
    // Wishbone slave
    input  wire        wb_clk_i,
    input  wire        wb_rst_i,
    input  wire        wbs_stb_i,
    input  wire        wbs_cyc_i,
    input  wire        wbs_we_i,
    input  wire [3:0]  wbs_sel_i,
    input  wire [31:0] wbs_dat_i,
    input  wire [31:0] wbs_adr_i,
    output reg         wbs_ack_o,
    output reg  [31:0] wbs_dat_o,

    // IO pads (unused here, wired to 0)
    input  wire [37:0] io_in,
    output wire [37:0] io_out,
    output wire [37:0] io_oeb,

    // IRQ
    output wire [2:0] irq
);

// ─── Tie-offs ────────────────────────────────────────────────────────────────
assign io_out = 38'd0;
assign io_oeb = {38{1'b1}};  // all inputs (high = input)
assign irq    = 3'b000;

// ─── Internal buffers ────────────────────────────────────────────────────────
// A: 2048 bits = 64 words × 32-bit
reg [31:0] a_mem  [0:63];
// B: 1024 bits = 32 words × 32-bit
reg [31:0] b_mem  [0:31];
// Scale: 256 bits = 8 words × 32-bit
reg [31:0] sc_mem [0:7];
// C: 8192 bits = 256 words × 32-bit (read-only from CPU perspective)
wire [31:0] c_mem [0:255];

// Pack buffers to flat signals
wire [2047:0] a_flat;
wire [1023:0] b_flat;
wire [255:0]  sc_flat;
wire [8191:0] c_flat;

genvar fi;
generate
    for (fi = 0; fi < 64; fi = fi+1) begin : PA
        assign a_flat[fi*32 +: 32] = a_mem[fi];
    end
    for (fi = 0; fi < 32; fi = fi+1) begin : PB
        assign b_flat[fi*32 +: 32] = b_mem[fi];
    end
    for (fi = 0; fi < 8; fi = fi+1) begin : PSC
        assign sc_flat[fi*32 +: 32] = sc_mem[fi];
    end
    for (fi = 0; fi < 256; fi = fi+1) begin : PC
        assign c_mem[fi] = c_flat[fi*32 +: 32];
    end
endgenerate

// ─── CTRL/STATUS registers ───────────────────────────────────────────────────
reg  start_r;
wire busy_w, done_w;

// ─── GEMM instance ───────────────────────────────────────────────────────────
gemm_int4_sky130 u_gemm (
    .clk          (wb_clk_i),
    .rst_n        (~wb_rst_i),
    .start        (start_r),
    .busy         (busy_w),
    .done         (done_w),
    .a_flat       (a_flat),
    .b_flat       (b_flat),
    .scale_q88_flat(sc_flat),
    .c_flat       (c_flat)
);

// ─── Wishbone address decode ─────────────────────────────────────────────────
// Offsets (byte address from base):
//   0x000       CTRL
//   0x004       STATUS
//   0x008~0x207 A_BUF  (0x008 + i*4, i=0..63)
//   0x208~0x307 B_BUF  (0x208 + i*4, i=0..31)
//   0x308~0x387 SC_BUF (0x308 + i*4, i=0..7)
//   0x388~0x787 C_BUF  (0x388 + i*4, i=0..255)

wire [11:0] offset = wbs_adr_i[11:0];  // bottom 12 bits = offset within module

wire sel = wbs_stb_i & wbs_cyc_i;

// ─── Read mux ────────────────────────────────────────────────────────────────
always @(*) begin
    wbs_dat_o = 32'd0;
    if (sel) begin
        if (offset == 12'h000)
            wbs_dat_o = {30'd0, start_r, 1'b0};
        else if (offset == 12'h004)
            wbs_dat_o = {30'd0, done_w, busy_w};
        else if (offset >= 12'h008 && offset < 12'h208)
            wbs_dat_o = a_mem[(offset - 12'h008) >> 2];
        else if (offset >= 12'h208 && offset < 12'h308)
            wbs_dat_o = b_mem[(offset - 12'h208) >> 2];
        else if (offset >= 12'h308 && offset < 12'h388)
            wbs_dat_o = sc_mem[(offset - 12'h308) >> 2];
        else if (offset >= 12'h388 && offset < 12'h788)
            wbs_dat_o = c_mem[(offset - 12'h388) >> 2];
    end
end

// ─── Write logic + ACK ───────────────────────────────────────────────────────
integer wi;
always @(posedge wb_clk_i or posedge wb_rst_i) begin
    if (wb_rst_i) begin
        wbs_ack_o <= 1'b0;
        start_r   <= 1'b0;
        for (wi = 0; wi < 64; wi = wi+1) a_mem[wi]  <= 32'd0;
        for (wi = 0; wi < 32; wi = wi+1) b_mem[wi]  <= 32'd0;
        for (wi = 0; wi <  8; wi = wi+1) sc_mem[wi] <= 32'd0;
    end else begin
        wbs_ack_o <= sel;  // single-cycle ack
        start_r   <= 1'b0; // auto-clear

        if (sel & wbs_we_i) begin
            if (offset == 12'h000)
                start_r <= wbs_dat_i[0];
            else if (offset >= 12'h008 && offset < 12'h208)
                a_mem[(offset - 12'h008) >> 2] <= wbs_dat_i;
            else if (offset >= 12'h208 && offset < 12'h308)
                b_mem[(offset - 12'h208) >> 2] <= wbs_dat_i;
            else if (offset >= 12'h308 && offset < 12'h388)
                sc_mem[(offset - 12'h308) >> 2] <= wbs_dat_i;
            // C_BUF is read-only
        end
    end
end

endmodule
