// gemm_int4_sky130.v — SKY130B-compatible INT4 GEMM tile
// Pure Verilog 2005, no SystemVerilog constructs.
// Designed for OpenMPW via OpenLane / SKY130B standard cells.
//
// Key design decisions for SKY130:
//   - INT8 × INT4 via shift-and-add (NO DSPs — SKY130 has none)
//   - Serialized: one output element per clock
//   - Q8.8 scale (FP16 dequant removed — use pre-converted scale_q88 input)
//   - 100 MHz target clock on SKY130HD std cells
//
// Interface:
//   - a_flat:    16×16 × 8-bit INT8 activations, row-major
//   - b_flat:    16×16 × 4-bit INT4 weights,    row-major (nibble packed)
//   - scale_q88: 16 × 16-bit Q8.8 scale per column
//   - c_flat:    16×16 × 32-bit INT32 output,   row-major
//
// Latency: ~514 cycles (256 COMPUTE + 256 SCALE + 2 overhead)
//
`timescale 1ns/1ps

module gemm_int4_sky130 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         busy,
    output reg         done,

    // A: INT8 activations [16][16] flat (16*16*8 = 2048 bits)
    input  wire [2047:0] a_flat,

    // B: INT4 weights [16][16] flat (16*16*4 = 1024 bits)
    input  wire [1023:0] b_flat,

    // Scale: Q8.8 per output column [16] (16*16 = 256 bits)
    input  wire [255:0]  scale_q88_flat,

    // C: INT32 output [16][16] flat (16*16*32 = 8192 bits)
    output wire [8191:0] c_flat
);

// ─── Parameters ─────────────────────────────────────────────────────────────
parameter TILE = 16;
parameter TILE_BITS = 4; // log2(16)

// ─── Unpack A (INT8) ─────────────────────────────────────────────────────────
// a_raw[i][j] = a_flat[(i*TILE+j)*8 +: 8]
wire signed [7:0] a_raw [0:15][0:15];
genvar gi, gj;
generate
    for (gi = 0; gi < 16; gi = gi+1) begin : UA_I
        for (gj = 0; gj < 16; gj = gj+1) begin : UA_J
            assign a_raw[gi][gj] = $signed(a_flat[(gi*16+gj)*8 +: 8]);
        end
    end
endgenerate

// ─── Unpack B (INT4 nibbles) ─────────────────────────────────────────────────
// b_nib[i][j] = b_flat[(i*TILE+j)*4 +: 4]
wire [3:0] b_nib [0:15][0:15];
genvar bi, bj;
generate
    for (bi = 0; bi < 16; bi = bi+1) begin : UB_I
        for (bj = 0; bj < 16; bj = bj+1) begin : UB_J
            assign b_nib[bi][bj] = b_flat[(bi*16+bj)*4 +: 4];
        end
    end
endgenerate

// ─── Unpack scale Q8.8 ──────────────────────────────────────────────────────
wire signed [15:0] scale_col [0:15];
genvar sk;
generate
    for (sk = 0; sk < 16; sk = sk+1) begin : GS
        assign scale_col[sk] = $signed(scale_q88_flat[sk*16 +: 16]);
    end
endgenerate

// ─── INT8 × INT4 shift-and-add ───────────────────────────────────────────────
// b is 4-bit two's complement: value = -8*b[3] + 4*b[2] + 2*b[1] + b[0]
// Result: 12-bit signed
function signed [11:0] mul_i8_i4;
    input signed [7:0] a;
    input        [3:0] b;
    reg signed [11:0] p0, p1, p2, p3;
    begin
        p0 = b[0] ? {{4{a[7]}}, a}       : 12'sd0;
        p1 = b[1] ? {{3{a[7]}}, a, 1'b0} : 12'sd0;
        p2 = b[2] ? {{2{a[7]}}, a, 2'b0} : 12'sd0;
        p3 = b[3] ? {{1{a[7]}}, a, 3'b0} : 12'sd0;
        mul_i8_i4 = p0 + p1 + p2 - p3;
    end
endfunction

// ─── Accumulators ────────────────────────────────────────────────────────────
reg signed [31:0] acc   [0:15][0:15];
reg signed [31:0] c_out [0:15][0:15];

// ─── State machine ───────────────────────────────────────────────────────────
localparam [1:0] S_IDLE    = 2'd0;
localparam [1:0] S_COMPUTE = 2'd1;
localparam [1:0] S_SCALE   = 2'd2;
localparam [1:0] S_DONE    = 2'd3;

reg [1:0] st;
reg [3:0] ci, cj;  // 4-bit counters for 0..15

// ── Dot product wires (unrolled loop for Verilog 2005) ───────────────────────
// Computed combinatorially from acc + current a/b row
wire signed [31:0] dot_ci_cj;
wire signed [11:0] dot_terms [0:15];

assign dot_terms[0]  = mul_i8_i4(a_raw[ci][0],  b_nib[0][cj]);
assign dot_terms[1]  = mul_i8_i4(a_raw[ci][1],  b_nib[1][cj]);
assign dot_terms[2]  = mul_i8_i4(a_raw[ci][2],  b_nib[2][cj]);
assign dot_terms[3]  = mul_i8_i4(a_raw[ci][3],  b_nib[3][cj]);
assign dot_terms[4]  = mul_i8_i4(a_raw[ci][4],  b_nib[4][cj]);
assign dot_terms[5]  = mul_i8_i4(a_raw[ci][5],  b_nib[5][cj]);
assign dot_terms[6]  = mul_i8_i4(a_raw[ci][6],  b_nib[6][cj]);
assign dot_terms[7]  = mul_i8_i4(a_raw[ci][7],  b_nib[7][cj]);
assign dot_terms[8]  = mul_i8_i4(a_raw[ci][8],  b_nib[8][cj]);
assign dot_terms[9]  = mul_i8_i4(a_raw[ci][9],  b_nib[9][cj]);
assign dot_terms[10] = mul_i8_i4(a_raw[ci][10], b_nib[10][cj]);
assign dot_terms[11] = mul_i8_i4(a_raw[ci][11], b_nib[11][cj]);
assign dot_terms[12] = mul_i8_i4(a_raw[ci][12], b_nib[12][cj]);
assign dot_terms[13] = mul_i8_i4(a_raw[ci][13], b_nib[13][cj]);
assign dot_terms[14] = mul_i8_i4(a_raw[ci][14], b_nib[14][cj]);
assign dot_terms[15] = mul_i8_i4(a_raw[ci][15], b_nib[15][cj]);

// Sign-extend each 12-bit term to 32-bit before summing
assign dot_ci_cj = {{20{dot_terms[0][11]}},  dot_terms[0]}
                 + {{20{dot_terms[1][11]}},  dot_terms[1]}
                 + {{20{dot_terms[2][11]}},  dot_terms[2]}
                 + {{20{dot_terms[3][11]}},  dot_terms[3]}
                 + {{20{dot_terms[4][11]}},  dot_terms[4]}
                 + {{20{dot_terms[5][11]}},  dot_terms[5]}
                 + {{20{dot_terms[6][11]}},  dot_terms[6]}
                 + {{20{dot_terms[7][11]}},  dot_terms[7]}
                 + {{20{dot_terms[8][11]}},  dot_terms[8]}
                 + {{20{dot_terms[9][11]}},  dot_terms[9]}
                 + {{20{dot_terms[10][11]}}, dot_terms[10]}
                 + {{20{dot_terms[11][11]}}, dot_terms[11]}
                 + {{20{dot_terms[12][11]}}, dot_terms[12]}
                 + {{20{dot_terms[13][11]}}, dot_terms[13]}
                 + {{20{dot_terms[14][11]}}, dot_terms[14]}
                 + {{20{dot_terms[15][11]}}, dot_terms[15]};

// ── Scale wire ───────────────────────────────────────────────────────────────
wire signed [47:0] scale_product;
assign scale_product = $signed({{16{acc[ci][cj][31]}}, acc[ci][cj]})
                     * $signed({{32{scale_col[cj][15]}}, scale_col[cj]});

// ─── FSM ─────────────────────────────────────────────────────────────────────
integer ii, jj;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        st   <= S_IDLE;
        ci   <= 4'd0;
        cj   <= 4'd0;
        busy <= 1'b0;
        done <= 1'b0;
        for (ii = 0; ii < 16; ii = ii+1) begin
            for (jj = 0; jj < 16; jj = jj+1) begin
                acc[ii][jj]   <= 32'sd0;
                c_out[ii][jj] <= 32'sd0;
            end
        end
    end else begin
        case (st)

            S_IDLE: begin
                done <= 1'b0;
                if (start) begin
                    busy <= 1'b1;
                    for (ii = 0; ii < 16; ii = ii+1)
                        for (jj = 0; jj < 16; jj = jj+1)
                            acc[ii][jj] <= 32'sd0;
                    ci <= 4'd0;
                    cj <= 4'd0;
                    st <= S_COMPUTE;
                end
            end

            S_COMPUTE: begin
                // Store dot product for (ci, cj)
                acc[ci][cj] <= dot_ci_cj;
                // Advance counters
                if (cj == 4'd15) begin
                    cj <= 4'd0;
                    if (ci == 4'd15) begin
                        ci <= 4'd0;
                        st <= S_SCALE;
                    end else begin
                        ci <= ci + 4'd1;
                    end
                end else begin
                    cj <= cj + 4'd1;
                end
            end

            S_SCALE: begin
                // Multiply acc[ci][cj] by scale_col[cj] (Q8.8)
                // product >> 8 to get scaled result
                c_out[ci][cj] <= scale_product[39:8];
                // Advance counters
                if (cj == 4'd15) begin
                    cj <= 4'd0;
                    if (ci == 4'd15) begin
                        ci <= 4'd0;
                        st <= S_DONE;
                    end else begin
                        ci <= ci + 4'd1;
                    end
                end else begin
                    cj <= cj + 4'd1;
                end
            end

            S_DONE: begin
                busy <= 1'b0;
                done <= 1'b1;
                st   <= S_IDLE;
            end

            default: st <= S_IDLE;

        endcase
    end
end

// ─── Pack output ─────────────────────────────────────────────────────────────
genvar oi, oj;
generate
    for (oi = 0; oi < 16; oi = oi+1) begin : CO_I
        for (oj = 0; oj < 16; oj = oj+1) begin : CO_J
            assign c_flat[(oi*16+oj)*32 +: 32] = c_out[oi][oj];
        end
    end
endgenerate

endmodule
