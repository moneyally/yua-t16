`timescale 1ns/1ps
`default_nettype none

// gemm_int4.sv — INT4 GEMM tile for YUA-T16 v2
// AWQ-style: INT4 weights + FP16 per-column scale -> INT32 accumulate
//
// Critical Icarus 12 rules followed:
//  - NO dynamic bit-selects inside always blocks
//  - Use always @(posedge clk) with BLOCKING assignments
//  - Flat packed arrays for cocotb
//  - generate/assign for all packed unpacking OUTSIDE always

module gemm_int4 #(
  parameter int TILE = 16
)(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  output logic        busy,
  output logic        done,

  // A matrix: INT8 activations [TILE][TILE], flat packed (8-bit each)
  input  logic [TILE*TILE*8-1:0]  a_flat,

  // B matrix: INT4 weights [TILE][TILE], flat packed (4-bit each)
  // Element (i,j) at bit offset (i*TILE+j)*4, 4-bit two's complement
  input  logic [TILE*TILE*4-1:0]  b_flat,

  // Scale factors: one FP16 scale per column j
  // scale_flat[j] occupies bits [j*16+15 : j*16]
  input  logic [TILE*16-1:0]      scale_flat,

  // Output: INT32 accumulator [TILE][TILE], flat packed (32-bit each)
  // Element (i,j) at bit offset (i*TILE+j)*32
  output logic [TILE*TILE*32-1:0] c_flat
);

  // ----------------------------------------------------------------
  // FP16 -> real conversion (combinational, outside always)
  // FP16: [15]=sign, [14:10]=exp(bias 15), [9:0]=mantissa
  // ----------------------------------------------------------------
  function automatic real f16r(input logic [15:0] fp16);
    logic        s;
    logic [4:0]  e;
    logic [9:0]  m;
    real         mant;
    int          exp_val;
    real         result;
    begin
      s = fp16[15];
      e = fp16[14:10];
      m = fp16[9:0];

      if (e == 5'b11111) begin
        // Inf or NaN -> treat as 1.0
        result = 1.0;
      end else if (e == 5'b00000) begin
        // Zero or denormal
        mant   = $itor(m) / 1024.0;
        result = mant * (2.0 ** (-14));
      end else begin
        mant    = 1.0 + $itor(m) / 1024.0;
        exp_val = int'(e) - 15;
        result  = mant * (2.0 ** exp_val);
      end

      if (s) result = -result;
      f16r = result;
    end
  endfunction

  // ----------------------------------------------------------------
  // Unpack a_flat -> a_raw[i][j]: 8-bit signed
  // Element (i,j) = bits [(i*TILE+j)*8 +: 8]
  // Use generate/assign OUTSIDE always (Icarus rule #1)
  // ----------------------------------------------------------------
  logic signed [7:0] a_raw [0:TILE-1][0:TILE-1];

  genvar gi, gj;
  generate
    for (gi = 0; gi < TILE; gi++) begin : GEN_A_UNPACK_I
      for (gj = 0; gj < TILE; gj++) begin : GEN_A_UNPACK_J
        assign a_raw[gi][gj] = signed'(a_flat[(gi*TILE+gj)*8 +: 8]);
      end
    end
  endgenerate

  // ----------------------------------------------------------------
  // Unpack b_flat -> b_raw[i][j]: 4-bit sign-extended to 8-bit signed
  // Element (i,j) at bits [(i*TILE+j)*4 +: 4]
  // Sign extension: if bit[3]=1 -> negative
  // ----------------------------------------------------------------
  logic signed [7:0] b_raw [0:TILE-1][0:TILE-1];

  genvar bi, bj;
  generate
    for (bi = 0; bi < TILE; bi++) begin : GEN_B_UNPACK_I
      for (bj = 0; bj < TILE; bj++) begin : GEN_B_UNPACK_J
        // Extract 4-bit nibble
        logic [3:0] nibble;
        assign nibble = b_flat[(bi*TILE+bj)*4 +: 4];
        // Sign-extend 4-bit to 8-bit: if bit[3] set, top 4 bits = 1111
        assign b_raw[bi][bj] = {{4{nibble[3]}}, nibble};
      end
    end
  endgenerate

  // ----------------------------------------------------------------
  // Unpack scale_flat -> scale_fp16[j]: 16-bit FP16
  // ----------------------------------------------------------------
  logic [15:0] scale_fp16 [0:TILE-1];

  genvar sj;
  generate
    for (sj = 0; sj < TILE; sj++) begin : GEN_SCALE_UNPACK
      assign scale_fp16[sj] = scale_flat[sj*16 +: 16];
    end
  endgenerate

  // ----------------------------------------------------------------
  // State machine: IDLE -> COMPUTE (iterate i,j,k) -> DONE
  // ----------------------------------------------------------------
  typedef enum logic [1:0] {
    S_IDLE    = 2'd0,
    S_COMPUTE = 2'd1,
    S_DONE    = 2'd2
  } state_t;

  state_t st;

  // Iteration counters
  logic [$clog2(TILE)-1:0] ci, cj, ck;

  // Accumulators: real arithmetic for dequantized computation
  real acc [0:TILE-1][0:TILE-1];

  // Output registers (INT32)
  logic signed [31:0] c_out [0:TILE-1][0:TILE-1];

  // ----------------------------------------------------------------
  // Main FSM + compute loop
  // All assignments BLOCKING inside always @(posedge clk)
  // (Icarus rule #2: use always @(posedge clk) with blocking)
  // ----------------------------------------------------------------
  integer ii, jj, kk_int;
  real    w_fp, prod, rounded;
  integer rounded_int;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st   = S_IDLE;
      ci   = '0;
      cj   = '0;
      ck   = '0;
      busy = 1'b0;
      done = 1'b0;
      // Clear accumulators
      for (ii = 0; ii < TILE; ii++) begin
        for (jj = 0; jj < TILE; jj++) begin
          acc[ii][jj]   = 0.0;
          c_out[ii][jj] = 32'sd0;
        end
      end
    end else begin
      case (st)

        S_IDLE: begin
          done = 1'b0;
          if (start) begin
            busy = 1'b1;
            // Clear accumulators on start
            for (ii = 0; ii < TILE; ii++) begin
              for (jj = 0; jj < TILE; jj++) begin
                acc[ii][jj] = 0.0;
              end
            end
            ci = '0;
            cj = '0;
            ck = '0;
            st = S_COMPUTE;
          end
        end

        S_COMPUTE: begin
          // One output element per clock: compute c[ci][cj] = sum_k(a[ci][k] * w_dq[k][cj])
          // We accumulate across all k for (ci, cj) in one cycle, then move to next (ci,cj)
          // w_dq[k][cj] = real(b_raw[k][cj]) * f16r(scale_fp16[cj])
          acc[ci][cj] = 0.0;
          for (kk_int = 0; kk_int < TILE; kk_int++) begin
            w_fp = $itor(b_raw[ci][kk_int]) * f16r(scale_fp16[kk_int]);
            // Wait, spec says: c[i][j] = sum_k( a[i][k] * w[k][j] )
            // w_dq[k][j] = real(b_raw[k][j]) * scale[j]
            // So for output (ci, cj): sum_k( a[ci][k] * b_raw[k][cj] * scale[cj] )
            // But b_raw is [k][j] not [i][k]
            // b_raw[k][cj]: k=row of B (same as k index), cj=col of B
            // In the generate block: b_raw[bi][bj] so b_raw[k][cj] is correct
            w_fp = $itor(b_raw[kk_int][cj]) * f16r(scale_fp16[cj]);
            acc[ci][cj] = acc[ci][cj] + ($itor(a_raw[ci][kk_int]) * w_fp);
          end

          // Round to INT32
          rounded = acc[ci][cj];
          if (rounded >= 0.0)
            rounded_int = int'(rounded + 0.5);
          else
            rounded_int = -int'(-rounded + 0.5);

          c_out[ci][cj] = rounded_int;

          // Advance (ci, cj) — iterate j inner, i outer
          if (cj == TILE-1) begin
            cj = '0;
            if (ci == TILE-1) begin
              ci = '0;
              st = S_DONE;
            end else begin
              ci = ci + 1'b1;
            end
          end else begin
            cj = cj + 1'b1;
          end
        end

        S_DONE: begin
          busy = 1'b0;
          done = 1'b1;
          st   = S_IDLE;
        end

        default: begin
          st = S_IDLE;
        end

      endcase
    end
  end

  // ----------------------------------------------------------------
  // Pack c_out -> c_flat via generate/assign OUTSIDE always
  // (Icarus rule #1)
  // ----------------------------------------------------------------
  genvar oi, oj;
  generate
    for (oi = 0; oi < TILE; oi++) begin : GEN_C_PACK_I
      for (oj = 0; oj < TILE; oj++) begin : GEN_C_PACK_J
        assign c_flat[(oi*TILE+oj)*32 +: 32] = c_out[oi][oj];
      end
    end
  endgenerate

endmodule

`default_nettype wire
