// vpu_core_synth.sv — Synthesizable VPU for Arty A7-100T (xc7a100tcsg324-1)
// Identical port interface to vpu_core.sv.
// All arithmetic in Q8.8 fixed-point (signed 16-bit).
//   1.0 = 16'sh0100  (256)
//   Multiply: 32-bit product, shift-right 8 to return to Q8.8
//
// Transcendental approximations:
//   sigmoid(x) : 256-entry LUT, x in [-8,+8], index = (x_q88 + 2048) >> 4
//   exp(delta) : 256-entry LUT, delta in [-8,0], index = (delta_q88 + 2048) >> 3
//   1/sqrt(ms) : 256-entry LUT, index = mean_sq[16:9]
//
// No `real`, no $exp/$sqrt/$itor/$rtoi — fully synthesizable.
//
`timescale 1ns/1ps
`default_nettype none

module vpu_core_synth #(
  parameter int DEPTH = 256
)(
  input  logic        clk,
  input  logic        rst_n,

  input  logic [3:0]  op_type,
  input  logic [15:0] vec_len,
  input  logic [15:0] imm_fp16_0,
  input  logic [15:0] imm_fp16_1,

  input  logic        start,
  output logic        busy,
  output logic        done,

  input  logic [16*DEPTH-1:0] src_flat,
  input  logic [16*DEPTH-1:0] aux_flat,
  output logic [16*DEPTH-1:0] dst_flat
);

  // ─── Unpack src/aux ──────────────────────────────────────────────────────
  wire signed [15:0] src_r [0:DEPTH-1];
  wire signed [15:0] aux_r [0:DEPTH-1];
  logic signed [15:0] dst_r [0:DEPTH-1];

  genvar gi;
  generate
    for (gi = 0; gi < DEPTH; gi = gi+1) begin : UNPACK
      assign src_r[gi]             = signed'(src_flat[gi*16 +: 16]);
      assign aux_r[gi]             = signed'(aux_flat[gi*16 +: 16]);
      assign dst_flat[gi*16 +: 16] = dst_r[gi];
    end
  endgenerate

  // ─── Q8.8 multiply helper (32->16 saturating) ────────────────────────────
  function automatic signed [15:0] q88_mul;
    input signed [15:0] a;
    input signed [15:0] b;
    logic signed [31:0] prod;
    logic signed [31:0] shifted;
    begin
      prod    = a * b;
      shifted = prod >>> 8;
      if (shifted > 32'sh00007FFF)
        q88_mul = 16'sh7FFF;
      else if (shifted < 32'shFFFF8000)
        q88_mul = 16'sh8000;
      else
        q88_mul = shifted[15:0];
    end
  endfunction

  // ─── Sigmoid LUT ──────────────────────────────────────────────────────────
  // x = -8.0 + i/16.0 for i=0..255
  // val = min(256, max(0, round(sigmoid(x) * 256)))
  // Verified: i=0->0, i=80->0x000C(12), i=128->0x0080(128),
  //           i=176->0x00F4(244), i=240->0x0100(256), i=255->0x0100(256)
  logic [15:0] lut_sigmoid [0:255];

  initial begin
    lut_sigmoid[  0] = 16'h0000; lut_sigmoid[  1] = 16'h0000;
    lut_sigmoid[  2] = 16'h0000; lut_sigmoid[  3] = 16'h0000;
    lut_sigmoid[  4] = 16'h0000; lut_sigmoid[  5] = 16'h0000;
    lut_sigmoid[  6] = 16'h0000; lut_sigmoid[  7] = 16'h0000;
    lut_sigmoid[  8] = 16'h0000; lut_sigmoid[  9] = 16'h0000;
    lut_sigmoid[ 10] = 16'h0000; lut_sigmoid[ 11] = 16'h0000;
    lut_sigmoid[ 12] = 16'h0000; lut_sigmoid[ 13] = 16'h0000;
    lut_sigmoid[ 14] = 16'h0000; lut_sigmoid[ 15] = 16'h0000;
    lut_sigmoid[ 16] = 16'h0000; lut_sigmoid[ 17] = 16'h0000;
    lut_sigmoid[ 18] = 16'h0000; lut_sigmoid[ 19] = 16'h0000;
    lut_sigmoid[ 20] = 16'h0000; lut_sigmoid[ 21] = 16'h0000;
    lut_sigmoid[ 22] = 16'h0000; lut_sigmoid[ 23] = 16'h0000;
    lut_sigmoid[ 24] = 16'h0000; lut_sigmoid[ 25] = 16'h0000;
    lut_sigmoid[ 26] = 16'h0000; lut_sigmoid[ 27] = 16'h0000;
    lut_sigmoid[ 28] = 16'h0000; lut_sigmoid[ 29] = 16'h0001;
    lut_sigmoid[ 30] = 16'h0001; lut_sigmoid[ 31] = 16'h0001;
    lut_sigmoid[ 32] = 16'h0001; lut_sigmoid[ 33] = 16'h0001;
    lut_sigmoid[ 34] = 16'h0001; lut_sigmoid[ 35] = 16'h0001;
    lut_sigmoid[ 36] = 16'h0001; lut_sigmoid[ 37] = 16'h0001;
    lut_sigmoid[ 38] = 16'h0001; lut_sigmoid[ 39] = 16'h0001;
    lut_sigmoid[ 40] = 16'h0001; lut_sigmoid[ 41] = 16'h0001;
    lut_sigmoid[ 42] = 16'h0001; lut_sigmoid[ 43] = 16'h0001;
    lut_sigmoid[ 44] = 16'h0001; lut_sigmoid[ 45] = 16'h0001;
    lut_sigmoid[ 46] = 16'h0002; lut_sigmoid[ 47] = 16'h0002;
    lut_sigmoid[ 48] = 16'h0002; lut_sigmoid[ 49] = 16'h0002;
    lut_sigmoid[ 50] = 16'h0002; lut_sigmoid[ 51] = 16'h0002;
    lut_sigmoid[ 52] = 16'h0002; lut_sigmoid[ 53] = 16'h0002;
    lut_sigmoid[ 54] = 16'h0002; lut_sigmoid[ 55] = 16'h0003;
    lut_sigmoid[ 56] = 16'h0003; lut_sigmoid[ 57] = 16'h0003;
    lut_sigmoid[ 58] = 16'h0003; lut_sigmoid[ 59] = 16'h0003;
    lut_sigmoid[ 60] = 16'h0004; lut_sigmoid[ 61] = 16'h0004;
    lut_sigmoid[ 62] = 16'h0004; lut_sigmoid[ 63] = 16'h0004;
    lut_sigmoid[ 64] = 16'h0005; lut_sigmoid[ 65] = 16'h0005;
    lut_sigmoid[ 66] = 16'h0005; lut_sigmoid[ 67] = 16'h0006;
    lut_sigmoid[ 68] = 16'h0006; lut_sigmoid[ 69] = 16'h0006;
    lut_sigmoid[ 70] = 16'h0007; lut_sigmoid[ 71] = 16'h0007;
    lut_sigmoid[ 72] = 16'h0008; lut_sigmoid[ 73] = 16'h0008;
    lut_sigmoid[ 74] = 16'h0008; lut_sigmoid[ 75] = 16'h0009;
    lut_sigmoid[ 76] = 16'h000A; lut_sigmoid[ 77] = 16'h000A;
    lut_sigmoid[ 78] = 16'h000B; lut_sigmoid[ 79] = 16'h000B;
    lut_sigmoid[ 80] = 16'h000C; lut_sigmoid[ 81] = 16'h000D;
    lut_sigmoid[ 82] = 16'h000E; lut_sigmoid[ 83] = 16'h000F;
    lut_sigmoid[ 84] = 16'h000F; lut_sigmoid[ 85] = 16'h0010;
    lut_sigmoid[ 86] = 16'h0011; lut_sigmoid[ 87] = 16'h0012;
    lut_sigmoid[ 88] = 16'h0013; lut_sigmoid[ 89] = 16'h0015;
    lut_sigmoid[ 90] = 16'h0016; lut_sigmoid[ 91] = 16'h0017;
    lut_sigmoid[ 92] = 16'h0018; lut_sigmoid[ 93] = 16'h001A;
    lut_sigmoid[ 94] = 16'h001B; lut_sigmoid[ 95] = 16'h001D;
    lut_sigmoid[ 96] = 16'h001F; lut_sigmoid[ 97] = 16'h0020;
    lut_sigmoid[ 98] = 16'h0022; lut_sigmoid[ 99] = 16'h0024;
    lut_sigmoid[100] = 16'h0026; lut_sigmoid[101] = 16'h0028;
    lut_sigmoid[102] = 16'h002A; lut_sigmoid[103] = 16'h002C;
    lut_sigmoid[104] = 16'h002F; lut_sigmoid[105] = 16'h0031;
    lut_sigmoid[106] = 16'h0034; lut_sigmoid[107] = 16'h0036;
    lut_sigmoid[108] = 16'h0039; lut_sigmoid[109] = 16'h003C;
    lut_sigmoid[110] = 16'h003F; lut_sigmoid[111] = 16'h0042;
    lut_sigmoid[112] = 16'h0045; lut_sigmoid[113] = 16'h0048;
    lut_sigmoid[114] = 16'h004B; lut_sigmoid[115] = 16'h004F;
    lut_sigmoid[116] = 16'h0052; lut_sigmoid[117] = 16'h0056;
    lut_sigmoid[118] = 16'h0059; lut_sigmoid[119] = 16'h005D;
    lut_sigmoid[120] = 16'h0061; lut_sigmoid[121] = 16'h0064;
    lut_sigmoid[122] = 16'h0068; lut_sigmoid[123] = 16'h006C;
    lut_sigmoid[124] = 16'h0070; lut_sigmoid[125] = 16'h0074;
    lut_sigmoid[126] = 16'h0078; lut_sigmoid[127] = 16'h007C;
    lut_sigmoid[128] = 16'h0080; lut_sigmoid[129] = 16'h0084;
    lut_sigmoid[130] = 16'h0088; lut_sigmoid[131] = 16'h008C;
    lut_sigmoid[132] = 16'h0090; lut_sigmoid[133] = 16'h0094;
    lut_sigmoid[134] = 16'h0098; lut_sigmoid[135] = 16'h009C;
    lut_sigmoid[136] = 16'h009F; lut_sigmoid[137] = 16'h00A3;
    lut_sigmoid[138] = 16'h00A7; lut_sigmoid[139] = 16'h00AA;
    lut_sigmoid[140] = 16'h00AE; lut_sigmoid[141] = 16'h00B1;
    lut_sigmoid[142] = 16'h00B5; lut_sigmoid[143] = 16'h00B8;
    lut_sigmoid[144] = 16'h00BB; lut_sigmoid[145] = 16'h00BE;
    lut_sigmoid[146] = 16'h00C1; lut_sigmoid[147] = 16'h00C4;
    lut_sigmoid[148] = 16'h00C7; lut_sigmoid[149] = 16'h00CA;
    lut_sigmoid[150] = 16'h00CC; lut_sigmoid[151] = 16'h00CF;
    lut_sigmoid[152] = 16'h00D1; lut_sigmoid[153] = 16'h00D4;
    lut_sigmoid[154] = 16'h00D6; lut_sigmoid[155] = 16'h00D8;
    lut_sigmoid[156] = 16'h00DA; lut_sigmoid[157] = 16'h00DC;
    lut_sigmoid[158] = 16'h00DE; lut_sigmoid[159] = 16'h00E0;
    lut_sigmoid[160] = 16'h00E1; lut_sigmoid[161] = 16'h00E3;
    lut_sigmoid[162] = 16'h00E5; lut_sigmoid[163] = 16'h00E6;
    lut_sigmoid[164] = 16'h00E8; lut_sigmoid[165] = 16'h00E9;
    lut_sigmoid[166] = 16'h00EA; lut_sigmoid[167] = 16'h00EB;
    lut_sigmoid[168] = 16'h00ED; lut_sigmoid[169] = 16'h00EE;
    lut_sigmoid[170] = 16'h00EF; lut_sigmoid[171] = 16'h00F0;
    lut_sigmoid[172] = 16'h00F1; lut_sigmoid[173] = 16'h00F1;
    lut_sigmoid[174] = 16'h00F2; lut_sigmoid[175] = 16'h00F3;
    lut_sigmoid[176] = 16'h00F4; lut_sigmoid[177] = 16'h00F5;
    lut_sigmoid[178] = 16'h00F5; lut_sigmoid[179] = 16'h00F6;
    lut_sigmoid[180] = 16'h00F6; lut_sigmoid[181] = 16'h00F7;
    lut_sigmoid[182] = 16'h00F8; lut_sigmoid[183] = 16'h00F8;
    lut_sigmoid[184] = 16'h00F8; lut_sigmoid[185] = 16'h00F9;
    lut_sigmoid[186] = 16'h00F9; lut_sigmoid[187] = 16'h00FA;
    lut_sigmoid[188] = 16'h00FA; lut_sigmoid[189] = 16'h00FA;
    lut_sigmoid[190] = 16'h00FB; lut_sigmoid[191] = 16'h00FB;
    lut_sigmoid[192] = 16'h00FB; lut_sigmoid[193] = 16'h00FC;
    lut_sigmoid[194] = 16'h00FC; lut_sigmoid[195] = 16'h00FC;
    lut_sigmoid[196] = 16'h00FC; lut_sigmoid[197] = 16'h00FD;
    lut_sigmoid[198] = 16'h00FD; lut_sigmoid[199] = 16'h00FD;
    lut_sigmoid[200] = 16'h00FD; lut_sigmoid[201] = 16'h00FD;
    lut_sigmoid[202] = 16'h00FE; lut_sigmoid[203] = 16'h00FE;
    lut_sigmoid[204] = 16'h00FE; lut_sigmoid[205] = 16'h00FE;
    lut_sigmoid[206] = 16'h00FE; lut_sigmoid[207] = 16'h00FE;
    lut_sigmoid[208] = 16'h00FE; lut_sigmoid[209] = 16'h00FE;
    lut_sigmoid[210] = 16'h00FE; lut_sigmoid[211] = 16'h00FF;
    lut_sigmoid[212] = 16'h00FF; lut_sigmoid[213] = 16'h00FF;
    lut_sigmoid[214] = 16'h00FF; lut_sigmoid[215] = 16'h00FF;
    lut_sigmoid[216] = 16'h00FF; lut_sigmoid[217] = 16'h00FF;
    lut_sigmoid[218] = 16'h00FF; lut_sigmoid[219] = 16'h00FF;
    lut_sigmoid[220] = 16'h00FF; lut_sigmoid[221] = 16'h00FF;
    lut_sigmoid[222] = 16'h00FF; lut_sigmoid[223] = 16'h00FF;
    lut_sigmoid[224] = 16'h00FF; lut_sigmoid[225] = 16'h00FF;
    lut_sigmoid[226] = 16'h00FF; lut_sigmoid[227] = 16'h00FF;
    lut_sigmoid[228] = 16'h0100; lut_sigmoid[229] = 16'h0100;
    lut_sigmoid[230] = 16'h0100; lut_sigmoid[231] = 16'h0100;
    lut_sigmoid[232] = 16'h0100; lut_sigmoid[233] = 16'h0100;
    lut_sigmoid[234] = 16'h0100; lut_sigmoid[235] = 16'h0100;
    lut_sigmoid[236] = 16'h0100; lut_sigmoid[237] = 16'h0100;
    lut_sigmoid[238] = 16'h0100; lut_sigmoid[239] = 16'h0100;
    lut_sigmoid[240] = 16'h0100; lut_sigmoid[241] = 16'h0100;
    lut_sigmoid[242] = 16'h0100; lut_sigmoid[243] = 16'h0100;
    lut_sigmoid[244] = 16'h0100; lut_sigmoid[245] = 16'h0100;
    lut_sigmoid[246] = 16'h0100; lut_sigmoid[247] = 16'h0100;
    lut_sigmoid[248] = 16'h0100; lut_sigmoid[249] = 16'h0100;
    lut_sigmoid[250] = 16'h0100; lut_sigmoid[251] = 16'h0100;
    lut_sigmoid[252] = 16'h0100; lut_sigmoid[253] = 16'h0100;
    lut_sigmoid[254] = 16'h0100; lut_sigmoid[255] = 16'h0100;  end

  // ─── Exp LUT ──────────────────────────────────────────────────────────────
  // delta = -8.0 + i/32.0 for i=0..254; i=255 -> exp(0)=1.0=256
  // index = (delta_q88 + 2048) >> 3
  // All values in [0, 256]; no overflow/wrap.
  // Verified: i=0->0, i=128->5(0x0005), i=254->240(0x00F0), i=255->256(0x0100)
  logic [15:0] lut_exp [0:255];

  initial begin
    lut_exp[  0] = 16'h0000; lut_exp[  1] = 16'h0000;
    lut_exp[  2] = 16'h0000; lut_exp[  3] = 16'h0000;
    lut_exp[  4] = 16'h0000; lut_exp[  5] = 16'h0000;
    lut_exp[  6] = 16'h0000; lut_exp[  7] = 16'h0000;
    lut_exp[  8] = 16'h0000; lut_exp[  9] = 16'h0000;
    lut_exp[ 10] = 16'h0000; lut_exp[ 11] = 16'h0000;
    lut_exp[ 12] = 16'h0000; lut_exp[ 13] = 16'h0000;
    lut_exp[ 14] = 16'h0000; lut_exp[ 15] = 16'h0000;
    lut_exp[ 16] = 16'h0000; lut_exp[ 17] = 16'h0000;
    lut_exp[ 18] = 16'h0000; lut_exp[ 19] = 16'h0000;
    lut_exp[ 20] = 16'h0000; lut_exp[ 21] = 16'h0000;
    lut_exp[ 22] = 16'h0000; lut_exp[ 23] = 16'h0000;
    lut_exp[ 24] = 16'h0000; lut_exp[ 25] = 16'h0000;
    lut_exp[ 26] = 16'h0000; lut_exp[ 27] = 16'h0000;
    lut_exp[ 28] = 16'h0000; lut_exp[ 29] = 16'h0000;
    lut_exp[ 30] = 16'h0000; lut_exp[ 31] = 16'h0000;
    lut_exp[ 32] = 16'h0000; lut_exp[ 33] = 16'h0000;
    lut_exp[ 34] = 16'h0000; lut_exp[ 35] = 16'h0000;
    lut_exp[ 36] = 16'h0000; lut_exp[ 37] = 16'h0000;
    lut_exp[ 38] = 16'h0000; lut_exp[ 39] = 16'h0000;
    lut_exp[ 40] = 16'h0000; lut_exp[ 41] = 16'h0000;
    lut_exp[ 42] = 16'h0000; lut_exp[ 43] = 16'h0000;
    lut_exp[ 44] = 16'h0000; lut_exp[ 45] = 16'h0000;
    lut_exp[ 46] = 16'h0000; lut_exp[ 47] = 16'h0000;
    lut_exp[ 48] = 16'h0000; lut_exp[ 49] = 16'h0000;
    lut_exp[ 50] = 16'h0000; lut_exp[ 51] = 16'h0000;
    lut_exp[ 52] = 16'h0000; lut_exp[ 53] = 16'h0000;
    lut_exp[ 54] = 16'h0000; lut_exp[ 55] = 16'h0000;
    lut_exp[ 56] = 16'h0000; lut_exp[ 57] = 16'h0001;
    lut_exp[ 58] = 16'h0001; lut_exp[ 59] = 16'h0001;
    lut_exp[ 60] = 16'h0001; lut_exp[ 61] = 16'h0001;
    lut_exp[ 62] = 16'h0001; lut_exp[ 63] = 16'h0001;
    lut_exp[ 64] = 16'h0001; lut_exp[ 65] = 16'h0001;
    lut_exp[ 66] = 16'h0001; lut_exp[ 67] = 16'h0001;
    lut_exp[ 68] = 16'h0001; lut_exp[ 69] = 16'h0001;
    lut_exp[ 70] = 16'h0001; lut_exp[ 71] = 16'h0001;
    lut_exp[ 72] = 16'h0001; lut_exp[ 73] = 16'h0001;
    lut_exp[ 74] = 16'h0001; lut_exp[ 75] = 16'h0001;
    lut_exp[ 76] = 16'h0001; lut_exp[ 77] = 16'h0001;
    lut_exp[ 78] = 16'h0001; lut_exp[ 79] = 16'h0001;
    lut_exp[ 80] = 16'h0001; lut_exp[ 81] = 16'h0001;
    lut_exp[ 82] = 16'h0001; lut_exp[ 83] = 16'h0001;
    lut_exp[ 84] = 16'h0001; lut_exp[ 85] = 16'h0001;
    lut_exp[ 86] = 16'h0001; lut_exp[ 87] = 16'h0001;
    lut_exp[ 88] = 16'h0001; lut_exp[ 89] = 16'h0001;
    lut_exp[ 90] = 16'h0001; lut_exp[ 91] = 16'h0001;
    lut_exp[ 92] = 16'h0002; lut_exp[ 93] = 16'h0002;
    lut_exp[ 94] = 16'h0002; lut_exp[ 95] = 16'h0002;
    lut_exp[ 96] = 16'h0002; lut_exp[ 97] = 16'h0002;
    lut_exp[ 98] = 16'h0002; lut_exp[ 99] = 16'h0002;
    lut_exp[100] = 16'h0002; lut_exp[101] = 16'h0002;
    lut_exp[102] = 16'h0002; lut_exp[103] = 16'h0002;
    lut_exp[104] = 16'h0002; lut_exp[105] = 16'h0002;
    lut_exp[106] = 16'h0002; lut_exp[107] = 16'h0002;
    lut_exp[108] = 16'h0003; lut_exp[109] = 16'h0003;
    lut_exp[110] = 16'h0003; lut_exp[111] = 16'h0003;
    lut_exp[112] = 16'h0003; lut_exp[113] = 16'h0003;
    lut_exp[114] = 16'h0003; lut_exp[115] = 16'h0003;
    lut_exp[116] = 16'h0003; lut_exp[117] = 16'h0003;
    lut_exp[118] = 16'h0003; lut_exp[119] = 16'h0004;
    lut_exp[120] = 16'h0004; lut_exp[121] = 16'h0004;
    lut_exp[122] = 16'h0004; lut_exp[123] = 16'h0004;
    lut_exp[124] = 16'h0004; lut_exp[125] = 16'h0004;
    lut_exp[126] = 16'h0004; lut_exp[127] = 16'h0005;
    lut_exp[128] = 16'h0005; lut_exp[129] = 16'h0005;
    lut_exp[130] = 16'h0005; lut_exp[131] = 16'h0005;
    lut_exp[132] = 16'h0005; lut_exp[133] = 16'h0005;
    lut_exp[134] = 16'h0006; lut_exp[135] = 16'h0006;
    lut_exp[136] = 16'h0006; lut_exp[137] = 16'h0006;
    lut_exp[138] = 16'h0006; lut_exp[139] = 16'h0007;
    lut_exp[140] = 16'h0007; lut_exp[141] = 16'h0007;
    lut_exp[142] = 16'h0007; lut_exp[143] = 16'h0007;
    lut_exp[144] = 16'h0008; lut_exp[145] = 16'h0008;
    lut_exp[146] = 16'h0008; lut_exp[147] = 16'h0008;
    lut_exp[148] = 16'h0009; lut_exp[149] = 16'h0009;
    lut_exp[150] = 16'h0009; lut_exp[151] = 16'h000A;
    lut_exp[152] = 16'h000A; lut_exp[153] = 16'h000A;
    lut_exp[154] = 16'h000B; lut_exp[155] = 16'h000B;
    lut_exp[156] = 16'h000B; lut_exp[157] = 16'h000C;
    lut_exp[158] = 16'h000C; lut_exp[159] = 16'h000C;
    lut_exp[160] = 16'h000D; lut_exp[161] = 16'h000D;
    lut_exp[162] = 16'h000E; lut_exp[163] = 16'h000E;
    lut_exp[164] = 16'h000E; lut_exp[165] = 16'h000F;
    lut_exp[166] = 16'h000F; lut_exp[167] = 16'h0010;
    lut_exp[168] = 16'h0010; lut_exp[169] = 16'h0011;
    lut_exp[170] = 16'h0011; lut_exp[171] = 16'h0012;
    lut_exp[172] = 16'h0013; lut_exp[173] = 16'h0013;
    lut_exp[174] = 16'h0014; lut_exp[175] = 16'h0014;
    lut_exp[176] = 16'h0015; lut_exp[177] = 16'h0016;
    lut_exp[178] = 16'h0016; lut_exp[179] = 16'h0017;
    lut_exp[180] = 16'h0018; lut_exp[181] = 16'h0019;
    lut_exp[182] = 16'h0019; lut_exp[183] = 16'h001A;
    lut_exp[184] = 16'h001B; lut_exp[185] = 16'h001C;
    lut_exp[186] = 16'h001D; lut_exp[187] = 16'h001E;
    lut_exp[188] = 16'h001F; lut_exp[189] = 16'h0020;
    lut_exp[190] = 16'h0021; lut_exp[191] = 16'h0022;
    lut_exp[192] = 16'h0023; lut_exp[193] = 16'h0024;
    lut_exp[194] = 16'h0025; lut_exp[195] = 16'h0026;
    lut_exp[196] = 16'h0027; lut_exp[197] = 16'h0029;
    lut_exp[198] = 16'h002A; lut_exp[199] = 16'h002B;
    lut_exp[200] = 16'h002C; lut_exp[201] = 16'h002E;
    lut_exp[202] = 16'h002F; lut_exp[203] = 16'h0031;
    lut_exp[204] = 16'h0032; lut_exp[205] = 16'h0034;
    lut_exp[206] = 16'h0036; lut_exp[207] = 16'h0037;
    lut_exp[208] = 16'h0039; lut_exp[209] = 16'h003B;
    lut_exp[210] = 16'h003D; lut_exp[211] = 16'h003F;
    lut_exp[212] = 16'h0041; lut_exp[213] = 16'h0043;
    lut_exp[214] = 16'h0045; lut_exp[215] = 16'h0047;
    lut_exp[216] = 16'h0049; lut_exp[217] = 16'h004C;
    lut_exp[218] = 16'h004E; lut_exp[219] = 16'h0051;
    lut_exp[220] = 16'h0053; lut_exp[221] = 16'h0056;
    lut_exp[222] = 16'h0058; lut_exp[223] = 16'h005B;
    lut_exp[224] = 16'h005E; lut_exp[225] = 16'h0061;
    lut_exp[226] = 16'h0064; lut_exp[227] = 16'h0067;
    lut_exp[228] = 16'h006B; lut_exp[229] = 16'h006E;
    lut_exp[230] = 16'h0072; lut_exp[231] = 16'h0075;
    lut_exp[232] = 16'h0079; lut_exp[233] = 16'h007D;
    lut_exp[234] = 16'h0081; lut_exp[235] = 16'h0085;
    lut_exp[236] = 16'h0089; lut_exp[237] = 16'h008D;
    lut_exp[238] = 16'h0092; lut_exp[239] = 16'h0096;
    lut_exp[240] = 16'h009B; lut_exp[241] = 16'h00A0;
    lut_exp[242] = 16'h00A5; lut_exp[243] = 16'h00AB;
    lut_exp[244] = 16'h00B0; lut_exp[245] = 16'h00B6;
    lut_exp[246] = 16'h00BB; lut_exp[247] = 16'h00C1;
    lut_exp[248] = 16'h00C7; lut_exp[249] = 16'h00CE;
    lut_exp[250] = 16'h00D4; lut_exp[251] = 16'h00DB;
    lut_exp[252] = 16'h00E2; lut_exp[253] = 16'h00E9;
    lut_exp[254] = 16'h00F0; lut_exp[255] = 16'h0100;  end

  // ─── Isqrt LUT ────────────────────────────────────────────────────────────
  // index = mean_sq[16:9]  (fixes bug: was [23:16], now [16:9])
  // For vs_real=1.0: vs_q88=256, mean_sq=65536, [16:9]=128, lut[128]=256 ✓
  // lut_isqrt[0]   = 0x7FFF (saturate for division by zero)
  // lut_isqrt[i]   = round(2896.31 / sqrt(i)) for i=1..255
  // Verified: i=1->2896(0x0B50), i=128->256(0x0100), i=170->222(0x00DE), i=255->181(0x00B5)
  logic [15:0] lut_isqrt [0:255];

  initial begin
    lut_isqrt[  0] = 16'h7FFF; lut_isqrt[  1] = 16'h0B50;
    lut_isqrt[  2] = 16'h0800; lut_isqrt[  3] = 16'h0688;
    lut_isqrt[  4] = 16'h05A8; lut_isqrt[  5] = 16'h050F;
    lut_isqrt[  6] = 16'h049E; lut_isqrt[  7] = 16'h0447;
    lut_isqrt[  8] = 16'h0400; lut_isqrt[  9] = 16'h03C5;
    lut_isqrt[ 10] = 16'h0394; lut_isqrt[ 11] = 16'h0369;
    lut_isqrt[ 12] = 16'h0344; lut_isqrt[ 13] = 16'h0323;
    lut_isqrt[ 14] = 16'h0306; lut_isqrt[ 15] = 16'h02EC;
    lut_isqrt[ 16] = 16'h02D4; lut_isqrt[ 17] = 16'h02BE;
    lut_isqrt[ 18] = 16'h02AB; lut_isqrt[ 19] = 16'h0298;
    lut_isqrt[ 20] = 16'h0288; lut_isqrt[ 21] = 16'h0278;
    lut_isqrt[ 22] = 16'h0269; lut_isqrt[ 23] = 16'h025C;
    lut_isqrt[ 24] = 16'h024F; lut_isqrt[ 25] = 16'h0243;
    lut_isqrt[ 26] = 16'h0238; lut_isqrt[ 27] = 16'h022D;
    lut_isqrt[ 28] = 16'h0223; lut_isqrt[ 29] = 16'h021A;
    lut_isqrt[ 30] = 16'h0211; lut_isqrt[ 31] = 16'h0208;
    lut_isqrt[ 32] = 16'h0200; lut_isqrt[ 33] = 16'h01F8;
    lut_isqrt[ 34] = 16'h01F1; lut_isqrt[ 35] = 16'h01EA;
    lut_isqrt[ 36] = 16'h01E3; lut_isqrt[ 37] = 16'h01DC;
    lut_isqrt[ 38] = 16'h01D6; lut_isqrt[ 39] = 16'h01D0;
    lut_isqrt[ 40] = 16'h01CA; lut_isqrt[ 41] = 16'h01C4;
    lut_isqrt[ 42] = 16'h01BF; lut_isqrt[ 43] = 16'h01BA;
    lut_isqrt[ 44] = 16'h01B5; lut_isqrt[ 45] = 16'h01B0;
    lut_isqrt[ 46] = 16'h01AB; lut_isqrt[ 47] = 16'h01A6;
    lut_isqrt[ 48] = 16'h01A2; lut_isqrt[ 49] = 16'h019E;
    lut_isqrt[ 50] = 16'h019A; lut_isqrt[ 51] = 16'h0196;
    lut_isqrt[ 52] = 16'h0192; lut_isqrt[ 53] = 16'h018E;
    lut_isqrt[ 54] = 16'h018A; lut_isqrt[ 55] = 16'h0187;
    lut_isqrt[ 56] = 16'h0183; lut_isqrt[ 57] = 16'h0180;
    lut_isqrt[ 58] = 16'h017C; lut_isqrt[ 59] = 16'h0179;
    lut_isqrt[ 60] = 16'h0176; lut_isqrt[ 61] = 16'h0173;
    lut_isqrt[ 62] = 16'h0170; lut_isqrt[ 63] = 16'h016D;
    lut_isqrt[ 64] = 16'h016A; lut_isqrt[ 65] = 16'h0167;
    lut_isqrt[ 66] = 16'h0165; lut_isqrt[ 67] = 16'h0162;
    lut_isqrt[ 68] = 16'h015F; lut_isqrt[ 69] = 16'h015D;
    lut_isqrt[ 70] = 16'h015A; lut_isqrt[ 71] = 16'h0158;
    lut_isqrt[ 72] = 16'h0155; lut_isqrt[ 73] = 16'h0153;
    lut_isqrt[ 74] = 16'h0151; lut_isqrt[ 75] = 16'h014E;
    lut_isqrt[ 76] = 16'h014C; lut_isqrt[ 77] = 16'h014A;
    lut_isqrt[ 78] = 16'h0148; lut_isqrt[ 79] = 16'h0146;
    lut_isqrt[ 80] = 16'h0144; lut_isqrt[ 81] = 16'h0142;
    lut_isqrt[ 82] = 16'h0140; lut_isqrt[ 83] = 16'h013E;
    lut_isqrt[ 84] = 16'h013C; lut_isqrt[ 85] = 16'h013A;
    lut_isqrt[ 86] = 16'h0138; lut_isqrt[ 87] = 16'h0137;
    lut_isqrt[ 88] = 16'h0135; lut_isqrt[ 89] = 16'h0133;
    lut_isqrt[ 90] = 16'h0131; lut_isqrt[ 91] = 16'h0130;
    lut_isqrt[ 92] = 16'h012E; lut_isqrt[ 93] = 16'h012C;
    lut_isqrt[ 94] = 16'h012B; lut_isqrt[ 95] = 16'h0129;
    lut_isqrt[ 96] = 16'h0128; lut_isqrt[ 97] = 16'h0126;
    lut_isqrt[ 98] = 16'h0125; lut_isqrt[ 99] = 16'h0123;
    lut_isqrt[100] = 16'h0122; lut_isqrt[101] = 16'h0120;
    lut_isqrt[102] = 16'h011F; lut_isqrt[103] = 16'h011D;
    lut_isqrt[104] = 16'h011C; lut_isqrt[105] = 16'h011B;
    lut_isqrt[106] = 16'h0119; lut_isqrt[107] = 16'h0118;
    lut_isqrt[108] = 16'h0117; lut_isqrt[109] = 16'h0115;
    lut_isqrt[110] = 16'h0114; lut_isqrt[111] = 16'h0113;
    lut_isqrt[112] = 16'h0112; lut_isqrt[113] = 16'h0110;
    lut_isqrt[114] = 16'h010F; lut_isqrt[115] = 16'h010E;
    lut_isqrt[116] = 16'h010D; lut_isqrt[117] = 16'h010C;
    lut_isqrt[118] = 16'h010B; lut_isqrt[119] = 16'h010A;
    lut_isqrt[120] = 16'h0108; lut_isqrt[121] = 16'h0107;
    lut_isqrt[122] = 16'h0106; lut_isqrt[123] = 16'h0105;
    lut_isqrt[124] = 16'h0104; lut_isqrt[125] = 16'h0103;
    lut_isqrt[126] = 16'h0102; lut_isqrt[127] = 16'h0101;
    lut_isqrt[128] = 16'h0100; lut_isqrt[129] = 16'h00FF;
    lut_isqrt[130] = 16'h00FE; lut_isqrt[131] = 16'h00FD;
    lut_isqrt[132] = 16'h00FC; lut_isqrt[133] = 16'h00FB;
    lut_isqrt[134] = 16'h00FA; lut_isqrt[135] = 16'h00F9;
    lut_isqrt[136] = 16'h00F8; lut_isqrt[137] = 16'h00F7;
    lut_isqrt[138] = 16'h00F7; lut_isqrt[139] = 16'h00F6;
    lut_isqrt[140] = 16'h00F5; lut_isqrt[141] = 16'h00F4;
    lut_isqrt[142] = 16'h00F3; lut_isqrt[143] = 16'h00F2;
    lut_isqrt[144] = 16'h00F1; lut_isqrt[145] = 16'h00F1;
    lut_isqrt[146] = 16'h00F0; lut_isqrt[147] = 16'h00EF;
    lut_isqrt[148] = 16'h00EE; lut_isqrt[149] = 16'h00ED;
    lut_isqrt[150] = 16'h00EC; lut_isqrt[151] = 16'h00EC;
    lut_isqrt[152] = 16'h00EB; lut_isqrt[153] = 16'h00EA;
    lut_isqrt[154] = 16'h00E9; lut_isqrt[155] = 16'h00E9;
    lut_isqrt[156] = 16'h00E8; lut_isqrt[157] = 16'h00E7;
    lut_isqrt[158] = 16'h00E6; lut_isqrt[159] = 16'h00E6;
    lut_isqrt[160] = 16'h00E5; lut_isqrt[161] = 16'h00E4;
    lut_isqrt[162] = 16'h00E4; lut_isqrt[163] = 16'h00E3;
    lut_isqrt[164] = 16'h00E2; lut_isqrt[165] = 16'h00E1;
    lut_isqrt[166] = 16'h00E1; lut_isqrt[167] = 16'h00E0;
    lut_isqrt[168] = 16'h00DF; lut_isqrt[169] = 16'h00DF;
    lut_isqrt[170] = 16'h00DE; lut_isqrt[171] = 16'h00DD;
    lut_isqrt[172] = 16'h00DD; lut_isqrt[173] = 16'h00DC;
    lut_isqrt[174] = 16'h00DC; lut_isqrt[175] = 16'h00DB;
    lut_isqrt[176] = 16'h00DA; lut_isqrt[177] = 16'h00DA;
    lut_isqrt[178] = 16'h00D9; lut_isqrt[179] = 16'h00D8;
    lut_isqrt[180] = 16'h00D8; lut_isqrt[181] = 16'h00D7;
    lut_isqrt[182] = 16'h00D7; lut_isqrt[183] = 16'h00D6;
    lut_isqrt[184] = 16'h00D6; lut_isqrt[185] = 16'h00D5;
    lut_isqrt[186] = 16'h00D4; lut_isqrt[187] = 16'h00D4;
    lut_isqrt[188] = 16'h00D3; lut_isqrt[189] = 16'h00D3;
    lut_isqrt[190] = 16'h00D2; lut_isqrt[191] = 16'h00D2;
    lut_isqrt[192] = 16'h00D1; lut_isqrt[193] = 16'h00D0;
    lut_isqrt[194] = 16'h00D0; lut_isqrt[195] = 16'h00CF;
    lut_isqrt[196] = 16'h00CF; lut_isqrt[197] = 16'h00CE;
    lut_isqrt[198] = 16'h00CE; lut_isqrt[199] = 16'h00CD;
    lut_isqrt[200] = 16'h00CD; lut_isqrt[201] = 16'h00CC;
    lut_isqrt[202] = 16'h00CC; lut_isqrt[203] = 16'h00CB;
    lut_isqrt[204] = 16'h00CB; lut_isqrt[205] = 16'h00CA;
    lut_isqrt[206] = 16'h00CA; lut_isqrt[207] = 16'h00C9;
    lut_isqrt[208] = 16'h00C9; lut_isqrt[209] = 16'h00C8;
    lut_isqrt[210] = 16'h00C8; lut_isqrt[211] = 16'h00C7;
    lut_isqrt[212] = 16'h00C7; lut_isqrt[213] = 16'h00C6;
    lut_isqrt[214] = 16'h00C6; lut_isqrt[215] = 16'h00C6;
    lut_isqrt[216] = 16'h00C5; lut_isqrt[217] = 16'h00C5;
    lut_isqrt[218] = 16'h00C4; lut_isqrt[219] = 16'h00C4;
    lut_isqrt[220] = 16'h00C3; lut_isqrt[221] = 16'h00C3;
    lut_isqrt[222] = 16'h00C2; lut_isqrt[223] = 16'h00C2;
    lut_isqrt[224] = 16'h00C2; lut_isqrt[225] = 16'h00C1;
    lut_isqrt[226] = 16'h00C1; lut_isqrt[227] = 16'h00C0;
    lut_isqrt[228] = 16'h00C0; lut_isqrt[229] = 16'h00BF;
    lut_isqrt[230] = 16'h00BF; lut_isqrt[231] = 16'h00BF;
    lut_isqrt[232] = 16'h00BE; lut_isqrt[233] = 16'h00BE;
    lut_isqrt[234] = 16'h00BD; lut_isqrt[235] = 16'h00BD;
    lut_isqrt[236] = 16'h00BD; lut_isqrt[237] = 16'h00BC;
    lut_isqrt[238] = 16'h00BC; lut_isqrt[239] = 16'h00BB;
    lut_isqrt[240] = 16'h00BB; lut_isqrt[241] = 16'h00BB;
    lut_isqrt[242] = 16'h00BA; lut_isqrt[243] = 16'h00BA;
    lut_isqrt[244] = 16'h00B9; lut_isqrt[245] = 16'h00B9;
    lut_isqrt[246] = 16'h00B9; lut_isqrt[247] = 16'h00B8;
    lut_isqrt[248] = 16'h00B8; lut_isqrt[249] = 16'h00B8;
    lut_isqrt[250] = 16'h00B7; lut_isqrt[251] = 16'h00B7;
    lut_isqrt[252] = 16'h00B6; lut_isqrt[253] = 16'h00B6;
    lut_isqrt[254] = 16'h00B6; lut_isqrt[255] = 16'h00B5;  end

  // ─── Index calculation helpers ────────────────────────────────────────────
  function automatic [7:0] sig_idx;
    input signed [15:0] x_q88;
    logic signed [15:0] clamped;
    logic [15:0]        shifted;
    begin
      if (x_q88 > 16'sh0800)       clamped = 16'sh0800;
      else if (x_q88 < -16'sh0800) clamped = -16'sh0800;
      else                          clamped = x_q88;
      shifted = (16'(signed'(clamped)) + 16'd2048) >> 4;
      sig_idx = (shifted > 16'd255) ? 8'd255 : shifted[7:0];
    end
  endfunction

  function automatic [7:0] exp_idx_fn;
    input signed [15:0] delta_q88;
    logic signed [15:0] clamped;
    logic [15:0]        shifted;
    begin
      if (delta_q88 > 16'sh0000)       clamped = 16'sh0000;
      else if (delta_q88 < -16'sh0800) clamped = -16'sh0800;
      else                              clamped = delta_q88;
      shifted = (16'(signed'(clamped)) + 16'd2048) >> 3;
      exp_idx_fn = (shifted > 16'd255) ? 8'd255 : shifted[7:0];
    end
  endfunction

  // ─── State machine ────────────────────────────────────────────────────────
  localparam logic [2:0] ST_IDLE  = 3'd0;
  localparam logic [2:0] ST_PASS1 = 3'd1;
  localparam logic [2:0] ST_PASS2 = 3'd2;
  localparam logic [2:0] ST_PASS3 = 3'd3;
  localparam logic [2:0] ST_DONE  = 3'd4;

  localparam logic [3:0] OP_ELEM_ADD   = 4'h0;
  localparam logic [3:0] OP_ELEM_MUL   = 4'h1;
  localparam logic [3:0] OP_SCALE      = 4'h2;
  localparam logic [3:0] OP_RESIDUAL   = 4'h3;
  localparam logic [3:0] OP_RMSNORM    = 4'h4;
  localparam logic [3:0] OP_SILU       = 4'h5;
  localparam logic [3:0] OP_ROPE       = 4'h6;
  localparam logic [3:0] OP_SOFTMAX    = 4'h7;
  localparam logic [3:0] OP_CLAMP      = 4'h8;
  localparam logic [3:0] OP_GELU_APPROX = 4'h9;

  logic [2:0]  state;
  logic [3:0]  op_r;
  logic [15:0] vlen_r;
  logic signed [15:0] imm0_r, imm1_r;
  logic [15:0] idx;

  logic [31:0] sum_sq;
  logic signed [15:0] vmax;
  logic [31:0] sum_exp;
  logic signed [15:0] rms_scale;

  integer ii;

  always @(posedge clk or negedge rst_n) begin : MAIN_FSM
    if (!rst_n) begin
      state     <= ST_IDLE;
      busy      <= 1'b0;
      done      <= 1'b0;
      idx       <= 16'd0;
      op_r      <= 4'd0;
      vlen_r    <= 16'd0;
      imm0_r    <= 16'sh0000;
      imm1_r    <= 16'sh0000;
      sum_sq    <= 32'd0;
      vmax      <= 16'sh8000;
      sum_exp   <= 32'd0;
      rms_scale <= 16'sh0100;
      for (ii = 0; ii < DEPTH; ii = ii+1)
        dst_r[ii] <= 16'sh0;
    end else begin
      done <= 1'b0;

      case (state)

        ST_IDLE: begin
          if (start) begin
            op_r     <= op_type;
            vlen_r   <= vec_len;
            imm0_r   <= signed'(imm_fp16_0);
            imm1_r   <= signed'(imm_fp16_1);
            idx      <= 16'd0;
            busy     <= 1'b1;
            sum_sq   <= 32'd0;
            vmax     <= 16'sh8000;
            sum_exp  <= 32'd0;
            state    <= ST_PASS1;
          end
        end

        ST_PASS1: begin
          if (idx < vlen_r) begin
            begin : P1_ELEM
              logic signed [15:0] vs, va;
              logic signed [31:0] sq_val;
              logic signed [15:0] xe, xo, cv, sv;
              vs = src_r[idx];
              va = aux_r[idx];

              case (op_r)
                OP_ELEM_ADD, OP_RESIDUAL: begin
                  begin : ADD_SAT
                    logic signed [16:0] sum17;
                    sum17 = {vs[15], vs} + {va[15], va};
                    if (sum17 > 17'sh0_7FFF)       dst_r[idx] <= 16'sh7FFF;
                    else if (sum17 < -17'sh0_8000) dst_r[idx] <= 16'sh8000;
                    else                            dst_r[idx] <= sum17[15:0];
                  end
                end

                OP_ELEM_MUL: begin
                  dst_r[idx] <= q88_mul(vs, va);
                end

                OP_SCALE: begin
                  dst_r[idx] <= q88_mul(vs, imm0_r);
                end

                OP_CLAMP: begin
                  if ($signed(vs) < $signed(imm0_r))      dst_r[idx] <= imm0_r;
                  else if ($signed(vs) > $signed(imm1_r)) dst_r[idx] <= imm1_r;
                  else                                     dst_r[idx] <= vs;
                end

                OP_SILU: begin
                  begin : SILU_BLOCK
                    logic [7:0] sidx;
                    logic signed [15:0] sig_val;
                    sidx    = sig_idx(vs);
                    sig_val = signed'(lut_sigmoid[sidx]);
                    dst_r[idx] <= q88_mul(vs, sig_val);
                  end
                end

                OP_GELU_APPROX: begin
                  // GELU(x) ≈ x * sigmoid(1.702 * x)
                  // 1.702 * 256 = 435.712 ≈ 436 = 0x01B4
                  begin : GELU_BLOCK
                    logic signed [15:0] gelu_arg, sig_val;
                    logic [7:0] sidx;
                    gelu_arg = q88_mul(vs, 16'sh01B4);
                    sidx     = sig_idx(gelu_arg);
                    sig_val  = signed'(lut_sigmoid[sidx]);
                    dst_r[idx] <= q88_mul(vs, sig_val);
                  end
                end

                OP_RMSNORM: begin
                  sq_val = vs * vs;
                  sum_sq <= sum_sq + sq_val[31:0];
                end

                OP_SOFTMAX: begin
                  if ($signed(vs) > $signed(vmax))
                    vmax <= vs;
                end

                OP_ROPE: begin
                  if (idx[0] == 1'b1) begin
                    xe = src_r[idx-1];
                    xo = vs;
                    cv = aux_r[idx-1];
                    sv = va;
                    dst_r[idx-1] <= q88_mul(xe, cv) - q88_mul(xo, sv);
                    dst_r[idx]   <= q88_mul(xo, cv) + q88_mul(xe, sv);
                  end
                end

                default: ;
              endcase
            end
            idx <= idx + 16'd1;
          end else begin
            case (op_r)
              OP_RMSNORM: begin
                begin : RMSNORM_END
                  logic [31:0] mean_sq;
                  logic [7:0]  lut_idx;
                  mean_sq   = sum_sq / {16'd0, vlen_r};
                  // FIX: use [16:9] not [23:16]
                  // mean_sq is Q16.16; [16:9] = floor(mean_sq/512)
                  // vs_real=1.0 -> vs_q88=256 -> sq=65536 -> mean_sq[16:9]=128 -> lut[128]=256(1.0) ✓
                  lut_idx   = mean_sq[16:9];
                  rms_scale <= signed'(lut_isqrt[lut_idx]);
                end
                idx   <= 16'd0;
                state <= ST_PASS2;
              end
              OP_SOFTMAX: begin
                sum_exp <= 32'd0;
                idx     <= 16'd0;
                state   <= ST_PASS2;
              end
              default: begin
                state <= ST_DONE;
                busy  <= 1'b0;
                done  <= 1'b1;
              end
            endcase
          end
        end

        ST_PASS2: begin
          if (idx < vlen_r) begin
            begin : P2_ELEM
              logic signed [15:0] vs, va;
              logic signed [15:0] delta;
              logic [7:0]         eidx;
              logic [15:0]        ev_val;
              logic signed [31:0] ev_ext;
              vs = src_r[idx];
              va = aux_r[idx];

              case (op_r)
                OP_RMSNORM: begin
                  dst_r[idx] <= q88_mul(q88_mul(vs, va), rms_scale);
                end

                OP_SOFTMAX: begin
                  delta  = vs - vmax;
                  eidx   = exp_idx_fn(delta);
                  ev_val = lut_exp[eidx];
                  dst_r[idx] <= signed'(ev_val);
                  ev_ext = {16'd0, ev_val};
                  sum_exp <= sum_exp + ev_ext;
                end

                default: ;
              endcase
            end
            idx <= idx + 16'd1;
          end else begin
            case (op_r)
              OP_SOFTMAX: begin
                idx   <= 16'd0;
                state <= ST_PASS3;
              end
              default: begin
                state <= ST_DONE;
                busy  <= 1'b0;
                done  <= 1'b1;
              end
            endcase
          end
        end

        ST_PASS3: begin
          if (idx < vlen_r) begin
            begin : P3_ELEM
              logic [15:0]        ev;
              logic [31:0]        ev_ext;
              logic [31:0]        norm;
              ev     = dst_r[idx];
              ev_ext = {16'd0, ev};
              if (sum_exp != 32'd0)
                norm = (ev_ext << 8) / sum_exp;
              else
                norm = 32'd0;
              dst_r[idx] <= (norm > 32'h0000_7FFF) ? 16'sh7FFF : signed'(norm[15:0]);
            end
            idx <= idx + 16'd1;
          end else begin
            state <= ST_DONE;
            busy  <= 1'b0;
            done  <= 1'b1;
          end
        end

        ST_DONE: begin
          state <= ST_IDLE;
        end

        default: begin
          state <= ST_IDLE;
        end

      endcase
    end
  end

endmodule
`default_nettype wire
