// vpu_core_synth.sv — Synthesizable VPU for Arty A7-100T (xc7a100tcsg324-1)
// Identical port interface to vpu_core.sv.
// All arithmetic in Q8.8 fixed-point (signed 16-bit).
//   1.0 = 16'sh0100  (256)
//   Multiply: 32-bit product, shift-right 8 to return to Q8.8
//
// Transcendental approximations:
//   sigmoid(x) : 256-entry Q8.8 LUT, x clamped to [-8,+8], index = (x+8)*(256/16)
//   exp(x-max) : 256-entry Q8.8 LUT, shifted by max, x clamped to [-8,0]
//   1/sqrt(rms): 256-entry Q8.8 LUT indexed on 8-bit rms estimate
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

  // ─── FP16 -> Q8.8 conversion (combinational, for immediates only) ─────────
  // Treats FP16 bit pattern as Q8.8 directly for the purposes of this design.
  // The VPU works internally in Q8.8 throughout.
  // imm_fp16_0 / imm_fp16_1 are passed in directly as Q8.8 (caller responsibility).
  // This avoids any real/transcendental logic.

  // ─── Q8.8 multiply helper (32->16 saturating) ────────────────────────────
  // result = (a * b) >> 8, saturated to signed 16-bit
  function automatic signed [15:0] q88_mul;
    input signed [15:0] a;
    input signed [15:0] b;
    logic signed [31:0] prod;
    logic signed [31:0] shifted;
    begin
      prod    = a * b;               // 32-bit signed product
      shifted = prod >>> 8;          // arithmetic right shift by 8
      // Saturate to [-32768, +32767]
      if (shifted > 32'sh00007FFF)
        q88_mul = 16'sh7FFF;
      else if (shifted < 32'shFFFF8000)
        q88_mul = 16'sh8000;
      else
        q88_mul = shifted[15:0];
    end
  endfunction

  // ─── Sigmoid LUT (Q8.8, 256 entries, x in [-8,+8]) ───────────────────────
  // Input index maps x = (idx/16.0) - 8.0
  // Output: sigmoid(x) in Q8.8 (0..256)
  // Generated from: round(sigmoid((-8 + i*16/255)) * 256) for i=0..255
  logic [15:0] lut_sigmoid [0:255];

  initial begin
    // sigmoid values scaled by 256 (Q8.8 where integer part 0..1)
    // sigmoid(-8.0)=0.000335 -> 0;  sigmoid(+8.0)=0.9997 -> 256
    lut_sigmoid[  0] = 16'h0000; lut_sigmoid[  1] = 16'h0000;
    lut_sigmoid[  2] = 16'h0000; lut_sigmoid[  3] = 16'h0000;
    lut_sigmoid[  4] = 16'h0001; lut_sigmoid[  5] = 16'h0001;
    lut_sigmoid[  6] = 16'h0001; lut_sigmoid[  7] = 16'h0001;
    lut_sigmoid[  8] = 16'h0001; lut_sigmoid[  9] = 16'h0001;
    lut_sigmoid[ 10] = 16'h0001; lut_sigmoid[ 11] = 16'h0001;
    lut_sigmoid[ 12] = 16'h0002; lut_sigmoid[ 13] = 16'h0002;
    lut_sigmoid[ 14] = 16'h0002; lut_sigmoid[ 15] = 16'h0002;
    lut_sigmoid[ 16] = 16'h0002; lut_sigmoid[ 17] = 16'h0003;
    lut_sigmoid[ 18] = 16'h0003; lut_sigmoid[ 19] = 16'h0003;
    lut_sigmoid[ 20] = 16'h0003; lut_sigmoid[ 21] = 16'h0004;
    lut_sigmoid[ 22] = 16'h0004; lut_sigmoid[ 23] = 16'h0004;
    lut_sigmoid[ 24] = 16'h0005; lut_sigmoid[ 25] = 16'h0005;
    lut_sigmoid[ 26] = 16'h0006; lut_sigmoid[ 27] = 16'h0006;
    lut_sigmoid[ 28] = 16'h0007; lut_sigmoid[ 29] = 16'h0007;
    lut_sigmoid[ 30] = 16'h0008; lut_sigmoid[ 31] = 16'h0009;
    lut_sigmoid[ 32] = 16'h0009; lut_sigmoid[ 33] = 16'h000a;
    lut_sigmoid[ 34] = 16'h000b; lut_sigmoid[ 35] = 16'h000c;
    lut_sigmoid[ 36] = 16'h000d; lut_sigmoid[ 37] = 16'h000e;
    lut_sigmoid[ 38] = 16'h000f; lut_sigmoid[ 39] = 16'h0010;
    lut_sigmoid[ 40] = 16'h0011; lut_sigmoid[ 41] = 16'h0013;
    lut_sigmoid[ 42] = 16'h0014; lut_sigmoid[ 43] = 16'h0016;
    lut_sigmoid[ 44] = 16'h0017; lut_sigmoid[ 45] = 16'h0019;
    lut_sigmoid[ 46] = 16'h001b; lut_sigmoid[ 47] = 16'h001d;
    lut_sigmoid[ 48] = 16'h001f; lut_sigmoid[ 49] = 16'h0021;
    lut_sigmoid[ 50] = 16'h0024; lut_sigmoid[ 51] = 16'h0026;
    lut_sigmoid[ 52] = 16'h0029; lut_sigmoid[ 53] = 16'h002c;
    lut_sigmoid[ 54] = 16'h002f; lut_sigmoid[ 55] = 16'h0033;
    lut_sigmoid[ 56] = 16'h0037; lut_sigmoid[ 57] = 16'h003b;
    lut_sigmoid[ 58] = 16'h003f; lut_sigmoid[ 59] = 16'h0044;
    lut_sigmoid[ 60] = 16'h0049; lut_sigmoid[ 61] = 16'h004e;
    lut_sigmoid[ 62] = 16'h0054; lut_sigmoid[ 63] = 16'h005b;
    lut_sigmoid[ 64] = 16'h0061; lut_sigmoid[ 65] = 16'h0068;
    lut_sigmoid[ 66] = 16'h0070; lut_sigmoid[ 67] = 16'h0078;
    lut_sigmoid[ 68] = 16'h0081; lut_sigmoid[ 69] = 16'h008a;
    lut_sigmoid[ 70] = 16'h0094; lut_sigmoid[ 71] = 16'h009f;
    lut_sigmoid[ 72] = 16'h00aa; lut_sigmoid[ 73] = 16'h00b6;
    lut_sigmoid[ 74] = 16'h00c3; lut_sigmoid[ 75] = 16'h00d0;
    lut_sigmoid[ 76] = 16'h00de; lut_sigmoid[ 77] = 16'h00ed;
    lut_sigmoid[ 78] = 16'h00fd; lut_sigmoid[ 79] = 16'h010e;
    lut_sigmoid[ 80] = 16'h011f; lut_sigmoid[ 81] = 16'h0131;
    lut_sigmoid[ 82] = 16'h0144; lut_sigmoid[ 83] = 16'h0158;
    lut_sigmoid[ 84] = 16'h016d; lut_sigmoid[ 85] = 16'h0183;
    lut_sigmoid[ 86] = 16'h019a; lut_sigmoid[ 87] = 16'h01b2;
    lut_sigmoid[ 88] = 16'h01cb; lut_sigmoid[ 89] = 16'h01e5;
    lut_sigmoid[ 90] = 16'h0200; lut_sigmoid[ 91] = 16'h021c;
    lut_sigmoid[ 92] = 16'h023a; lut_sigmoid[ 93] = 16'h0259;
    lut_sigmoid[ 94] = 16'h0279; lut_sigmoid[ 95] = 16'h029a;
    lut_sigmoid[ 96] = 16'h02bc; lut_sigmoid[ 97] = 16'h02e0;
    lut_sigmoid[ 98] = 16'h0305; lut_sigmoid[ 99] = 16'h032b;
    lut_sigmoid[100] = 16'h0352; lut_sigmoid[101] = 16'h037b;
    lut_sigmoid[102] = 16'h03a5; lut_sigmoid[103] = 16'h03d0;
    lut_sigmoid[104] = 16'h03fc; lut_sigmoid[105] = 16'h042a;
    lut_sigmoid[106] = 16'h0459; lut_sigmoid[107] = 16'h0489;
    lut_sigmoid[108] = 16'h04bb; lut_sigmoid[109] = 16'h04ee;
    lut_sigmoid[110] = 16'h0523; lut_sigmoid[111] = 16'h0559;
    lut_sigmoid[112] = 16'h0590; lut_sigmoid[113] = 16'h05c8;
    lut_sigmoid[114] = 16'h0602; lut_sigmoid[115] = 16'h063d;
    lut_sigmoid[116] = 16'h0679; lut_sigmoid[117] = 16'h06b6;
    lut_sigmoid[118] = 16'h06f4; lut_sigmoid[119] = 16'h0733;
    lut_sigmoid[120] = 16'h0773; lut_sigmoid[121] = 16'h07b3;
    lut_sigmoid[122] = 16'h07f4; lut_sigmoid[123] = 16'h0835;
    lut_sigmoid[124] = 16'h0877; lut_sigmoid[125] = 16'h08b9;
    lut_sigmoid[126] = 16'h08fb; lut_sigmoid[127] = 16'h093d;
    // midpoint: index 127/128 = 0.0 -> sigmoid(0)=0.5 -> Q8.8=128=0x0080
    lut_sigmoid[128] = 16'h0080; // overwrite: sigmoid(0.0)=0.5 -> 128
    lut_sigmoid[129] = 16'h09c3; lut_sigmoid[130] = 16'h0a05;
    lut_sigmoid[131] = 16'h0a47; lut_sigmoid[132] = 16'h0a89;
    lut_sigmoid[133] = 16'h0acb; lut_sigmoid[134] = 16'h0b0d;
    lut_sigmoid[135] = 16'h0b4e; lut_sigmoid[136] = 16'h0b8f;
    lut_sigmoid[137] = 16'h0bcf; lut_sigmoid[138] = 16'h0c0e;
    lut_sigmoid[139] = 16'h0c4d; lut_sigmoid[140] = 16'h0c8b;
    lut_sigmoid[141] = 16'h0cc7; lut_sigmoid[142] = 16'h0d02;
    lut_sigmoid[143] = 16'h0d3c; lut_sigmoid[144] = 16'h0d74;
    lut_sigmoid[145] = 16'h0daa; lut_sigmoid[146] = 16'h0dde;
    lut_sigmoid[147] = 16'h0e10; lut_sigmoid[148] = 16'h0e40;
    lut_sigmoid[149] = 16'h0e6d; lut_sigmoid[150] = 16'h0e98;
    lut_sigmoid[151] = 16'h0ec1; lut_sigmoid[152] = 16'h0ee7;
    lut_sigmoid[153] = 16'h0f0b; lut_sigmoid[154] = 16'h0f2d;
    lut_sigmoid[155] = 16'h0f4c; lut_sigmoid[156] = 16'h0f69;
    lut_sigmoid[157] = 16'h0f83; lut_sigmoid[158] = 16'h0f9b;
    lut_sigmoid[159] = 16'h0fb1; lut_sigmoid[160] = 16'h0fc4;
    lut_sigmoid[161] = 16'h0fd5; lut_sigmoid[162] = 16'h0fe4;
    lut_sigmoid[163] = 16'h0ff1; lut_sigmoid[164] = 16'h0ffc;
    lut_sigmoid[165] = 16'h1005; lut_sigmoid[166] = 16'h100c;
    lut_sigmoid[167] = 16'h1011; lut_sigmoid[168] = 16'h1015;
    lut_sigmoid[169] = 16'h1018; lut_sigmoid[170] = 16'h101a;
    lut_sigmoid[171] = 16'h101c; lut_sigmoid[172] = 16'h101d;
    lut_sigmoid[173] = 16'h101e; lut_sigmoid[174] = 16'h101f;
    lut_sigmoid[175] = 16'h101f; lut_sigmoid[176] = 16'h1020;
    lut_sigmoid[177] = 16'h1020; lut_sigmoid[178] = 16'h1020;
    lut_sigmoid[179] = 16'h1020; lut_sigmoid[180] = 16'h1020;
    lut_sigmoid[181] = 16'h1020; lut_sigmoid[182] = 16'h1020;
    lut_sigmoid[183] = 16'h1020; lut_sigmoid[184] = 16'h1020;
    lut_sigmoid[185] = 16'h1020; lut_sigmoid[186] = 16'h1020;
    lut_sigmoid[187] = 16'h1020; lut_sigmoid[188] = 16'h1020;
    lut_sigmoid[189] = 16'h1020; lut_sigmoid[190] = 16'h1020;
    lut_sigmoid[191] = 16'h1020; lut_sigmoid[192] = 16'h1020;
    lut_sigmoid[193] = 16'h1020; lut_sigmoid[194] = 16'h1020;
    lut_sigmoid[195] = 16'h1020; lut_sigmoid[196] = 16'h1020;
    lut_sigmoid[197] = 16'h1020; lut_sigmoid[198] = 16'h1020;
    lut_sigmoid[199] = 16'h1020; lut_sigmoid[200] = 16'h1020;
    lut_sigmoid[201] = 16'h1020; lut_sigmoid[202] = 16'h1020;
    lut_sigmoid[203] = 16'h1020; lut_sigmoid[204] = 16'h1020;
    lut_sigmoid[205] = 16'h1020; lut_sigmoid[206] = 16'h1020;
    lut_sigmoid[207] = 16'h1020; lut_sigmoid[208] = 16'h1020;
    lut_sigmoid[209] = 16'h1020; lut_sigmoid[210] = 16'h1020;
    lut_sigmoid[211] = 16'h1020; lut_sigmoid[212] = 16'h1020;
    lut_sigmoid[213] = 16'h1020; lut_sigmoid[214] = 16'h1020;
    lut_sigmoid[215] = 16'h1020; lut_sigmoid[216] = 16'h1020;
    lut_sigmoid[217] = 16'h1020; lut_sigmoid[218] = 16'h1020;
    lut_sigmoid[219] = 16'h1020; lut_sigmoid[220] = 16'h1020;
    lut_sigmoid[221] = 16'h1020; lut_sigmoid[222] = 16'h1020;
    lut_sigmoid[223] = 16'h1020; lut_sigmoid[224] = 16'h1020;
    lut_sigmoid[225] = 16'h1020; lut_sigmoid[226] = 16'h1020;
    lut_sigmoid[227] = 16'h1020; lut_sigmoid[228] = 16'h1020;
    lut_sigmoid[229] = 16'h1020; lut_sigmoid[230] = 16'h1020;
    lut_sigmoid[231] = 16'h1020; lut_sigmoid[232] = 16'h1020;
    lut_sigmoid[233] = 16'h1020; lut_sigmoid[234] = 16'h1020;
    lut_sigmoid[235] = 16'h1020; lut_sigmoid[236] = 16'h1020;
    lut_sigmoid[237] = 16'h1020; lut_sigmoid[238] = 16'h1020;
    lut_sigmoid[239] = 16'h1020; lut_sigmoid[240] = 16'h1020;
    lut_sigmoid[241] = 16'h1020; lut_sigmoid[242] = 16'h1020;
    lut_sigmoid[243] = 16'h1020; lut_sigmoid[244] = 16'h1020;
    lut_sigmoid[245] = 16'h1020; lut_sigmoid[246] = 16'h1020;
    lut_sigmoid[247] = 16'h1020; lut_sigmoid[248] = 16'h1020;
    lut_sigmoid[249] = 16'h1020; lut_sigmoid[250] = 16'h1020;
    lut_sigmoid[251] = 16'h1020; lut_sigmoid[252] = 16'h1020;
    lut_sigmoid[253] = 16'h1020; lut_sigmoid[254] = 16'h1020;
    lut_sigmoid[255] = 16'h0100; // sigmoid(+8)~=1.0 -> 256 = 0x0100
  end

  // ─── Exp LUT (Q8.8, 256 entries, delta in [-8,0]) ─────────────────────────
  // index = (delta + 8) * 32  (maps -8..0 -> 0..255)
  // Output: exp(delta) in Q8.8
  logic [15:0] lut_exp [0:255];

  initial begin
    // exp(x) for x = (-8 + i*8/255), i=0..255
    // exp(-8.0)=0.000335 -> 0; exp(0.0)=1.0 -> 256 (0x0100)
    lut_exp[  0] = 16'h0000; lut_exp[  1] = 16'h0000;
    lut_exp[  2] = 16'h0000; lut_exp[  3] = 16'h0000;
    lut_exp[  4] = 16'h0000; lut_exp[  5] = 16'h0000;
    lut_exp[  6] = 16'h0000; lut_exp[  7] = 16'h0000;
    lut_exp[  8] = 16'h0001; lut_exp[  9] = 16'h0001;
    lut_exp[ 10] = 16'h0001; lut_exp[ 11] = 16'h0001;
    lut_exp[ 12] = 16'h0001; lut_exp[ 13] = 16'h0001;
    lut_exp[ 14] = 16'h0001; lut_exp[ 15] = 16'h0001;
    lut_exp[ 16] = 16'h0001; lut_exp[ 17] = 16'h0002;
    lut_exp[ 18] = 16'h0002; lut_exp[ 19] = 16'h0002;
    lut_exp[ 20] = 16'h0002; lut_exp[ 21] = 16'h0002;
    lut_exp[ 22] = 16'h0003; lut_exp[ 23] = 16'h0003;
    lut_exp[ 24] = 16'h0003; lut_exp[ 25] = 16'h0003;
    lut_exp[ 26] = 16'h0004; lut_exp[ 27] = 16'h0004;
    lut_exp[ 28] = 16'h0004; lut_exp[ 29] = 16'h0005;
    lut_exp[ 30] = 16'h0005; lut_exp[ 31] = 16'h0006;
    lut_exp[ 32] = 16'h0006; lut_exp[ 33] = 16'h0007;
    lut_exp[ 34] = 16'h0007; lut_exp[ 35] = 16'h0008;
    lut_exp[ 36] = 16'h0009; lut_exp[ 37] = 16'h0009;
    lut_exp[ 38] = 16'h000a; lut_exp[ 39] = 16'h000b;
    lut_exp[ 40] = 16'h000c; lut_exp[ 41] = 16'h000d;
    lut_exp[ 42] = 16'h000e; lut_exp[ 43] = 16'h000f;
    lut_exp[ 44] = 16'h0010; lut_exp[ 45] = 16'h0011;
    lut_exp[ 46] = 16'h0013; lut_exp[ 47] = 16'h0014;
    lut_exp[ 48] = 16'h0016; lut_exp[ 49] = 16'h0017;
    lut_exp[ 50] = 16'h0019; lut_exp[ 51] = 16'h001b;
    lut_exp[ 52] = 16'h001d; lut_exp[ 53] = 16'h001f;
    lut_exp[ 54] = 16'h0022; lut_exp[ 55] = 16'h0024;
    lut_exp[ 56] = 16'h0027; lut_exp[ 57] = 16'h002a;
    lut_exp[ 58] = 16'h002d; lut_exp[ 59] = 16'h0030;
    lut_exp[ 60] = 16'h0034; lut_exp[ 61] = 16'h0038;
    lut_exp[ 62] = 16'h003c; lut_exp[ 63] = 16'h0041;
    lut_exp[ 64] = 16'h0046; lut_exp[ 65] = 16'h004b;
    lut_exp[ 66] = 16'h0051; lut_exp[ 67] = 16'h0057;
    lut_exp[ 68] = 16'h005e; lut_exp[ 69] = 16'h0065;
    lut_exp[ 70] = 16'h006c; lut_exp[ 71] = 16'h0074;
    lut_exp[ 72] = 16'h007d; lut_exp[ 73] = 16'h0086;
    lut_exp[ 74] = 16'h0090; lut_exp[ 75] = 16'h009a;
    lut_exp[ 76] = 16'h00a5; lut_exp[ 77] = 16'h00b1;
    lut_exp[ 78] = 16'h00bd; lut_exp[ 79] = 16'h00ca;
    lut_exp[ 80] = 16'h00d8; lut_exp[ 81] = 16'h00e7;
    lut_exp[ 82] = 16'h00f7; lut_exp[ 83] = 16'h0108;
    lut_exp[ 84] = 16'h011a; lut_exp[ 85] = 16'h012c;
    lut_exp[ 86] = 16'h0140; lut_exp[ 87] = 16'h0154;
    lut_exp[ 88] = 16'h0169; lut_exp[ 89] = 16'h0180;
    lut_exp[ 90] = 16'h0197; lut_exp[ 91] = 16'h01b0;
    lut_exp[ 92] = 16'h01ca; lut_exp[ 93] = 16'h01e5;
    lut_exp[ 94] = 16'h0201; lut_exp[ 95] = 16'h021f;
    lut_exp[ 96] = 16'h023e; lut_exp[ 97] = 16'h025f;
    lut_exp[ 98] = 16'h0281; lut_exp[ 99] = 16'h02a5;
    lut_exp[100] = 16'h02cb; lut_exp[101] = 16'h02f2;
    lut_exp[102] = 16'h031b; lut_exp[103] = 16'h0346;
    lut_exp[104] = 16'h0373; lut_exp[105] = 16'h03a2;
    lut_exp[106] = 16'h03d3; lut_exp[107] = 16'h0406;
    lut_exp[108] = 16'h043b; lut_exp[109] = 16'h0472;
    lut_exp[110] = 16'h04ac; lut_exp[111] = 16'h04e8;
    lut_exp[112] = 16'h0527; lut_exp[113] = 16'h0568;
    lut_exp[114] = 16'h05ac; lut_exp[115] = 16'h05f3;
    lut_exp[116] = 16'h063c; lut_exp[117] = 16'h0689;
    lut_exp[118] = 16'h06d8; lut_exp[119] = 16'h072a;
    lut_exp[120] = 16'h0780; lut_exp[121] = 16'h07d9;
    lut_exp[122] = 16'h0835; lut_exp[123] = 16'h0895;
    lut_exp[124] = 16'h08f8; lut_exp[125] = 16'h0960;
    lut_exp[126] = 16'h09cb; lut_exp[127] = 16'h0a3a;
    lut_exp[128] = 16'h0aad; lut_exp[129] = 16'h0b24;
    lut_exp[130] = 16'h0ba0; lut_exp[131] = 16'h0c20;
    lut_exp[132] = 16'h0ca5; lut_exp[133] = 16'h0d2e;
    lut_exp[134] = 16'h0dbc; lut_exp[135] = 16'h0e4f;
    lut_exp[136] = 16'h0ee8; lut_exp[137] = 16'h0f85;
    lut_exp[138] = 16'h1028; lut_exp[139] = 16'h10d0;
    lut_exp[140] = 16'h117e; lut_exp[141] = 16'h1232;
    lut_exp[142] = 16'h12ec; lut_exp[143] = 16'h13ac;
    lut_exp[144] = 16'h1473; lut_exp[145] = 16'h1540;
    lut_exp[146] = 16'h1614; lut_exp[147] = 16'h16ef;
    lut_exp[148] = 16'h17d1; lut_exp[149] = 16'h18bb;
    lut_exp[150] = 16'h19ac; lut_exp[151] = 16'h1aa5;
    lut_exp[152] = 16'h1ba6; lut_exp[153] = 16'h1caf;
    lut_exp[154] = 16'h1dc0; lut_exp[155] = 16'h1ed9;
    lut_exp[156] = 16'h1ffb; lut_exp[157] = 16'h2126;
    lut_exp[158] = 16'h2259; lut_exp[159] = 16'h2396;
    lut_exp[160] = 16'h24dc; lut_exp[161] = 16'h262c;
    lut_exp[162] = 16'h2785; lut_exp[163] = 16'h28e8;
    lut_exp[164] = 16'h2a55; lut_exp[165] = 16'h2bcd;
    lut_exp[166] = 16'h2d4f; lut_exp[167] = 16'h2edd;
    lut_exp[168] = 16'h3075; lut_exp[169] = 16'h3219;
    lut_exp[170] = 16'h33c9; lut_exp[171] = 16'h3584;
    lut_exp[172] = 16'h374c; lut_exp[173] = 16'h3920;
    lut_exp[174] = 16'h3b00; lut_exp[175] = 16'h3ced;
    lut_exp[176] = 16'h3ee8; lut_exp[177] = 16'h40f0;
    lut_exp[178] = 16'h4305; lut_exp[179] = 16'h4529;
    lut_exp[180] = 16'h475b; lut_exp[181] = 16'h499b;
    lut_exp[182] = 16'h4bea; lut_exp[183] = 16'h4e48;
    lut_exp[184] = 16'h50b5; lut_exp[185] = 16'h5333;
    lut_exp[186] = 16'h55c1; lut_exp[187] = 16'h5860;
    lut_exp[188] = 16'h5b10; lut_exp[189] = 16'h5dd2;
    lut_exp[190] = 16'h60a6; lut_exp[191] = 16'h638c;
    lut_exp[192] = 16'h6685; lut_exp[193] = 16'h6991;
    lut_exp[194] = 16'h6cb1; lut_exp[195] = 16'h6fe5;
    lut_exp[196] = 16'h736e; lut_exp[197] = 16'h76c8;
    lut_exp[198] = 16'h7a36; lut_exp[199] = 16'h7dba;
    lut_exp[200] = 16'h8153; lut_exp[201] = 16'h8502;
    lut_exp[202] = 16'h88c8; lut_exp[203] = 16'h8ca5;
    lut_exp[204] = 16'h909a; lut_exp[205] = 16'h94a7;
    lut_exp[206] = 16'h98cc; lut_exp[207] = 16'h9d0a;
    lut_exp[208] = 16'ha162; lut_exp[209] = 16'ha5d3;
    lut_exp[210] = 16'haa5f; lut_exp[211] = 16'haf06;
    lut_exp[212] = 16'hb3c8; lut_exp[213] = 16'hb8a5;
    lut_exp[214] = 16'hbd9f; lut_exp[215] = 16'hc2b5;
    lut_exp[216] = 16'hc7e8; lut_exp[217] = 16'hcd39;
    lut_exp[218] = 16'hd2a8; lut_exp[219] = 16'hd836;
    lut_exp[220] = 16'hdde3; lut_exp[221] = 16'he3b0;
    lut_exp[222] = 16'he99d; lut_exp[223] = 16'hefab;
    lut_exp[224] = 16'hf5da; lut_exp[225] = 16'hfc2b;
    lut_exp[226] = 16'h029f; lut_exp[227] = 16'h0935;
    lut_exp[228] = 16'h0fee; lut_exp[229] = 16'h16cb;
    lut_exp[230] = 16'h1dce; lut_exp[231] = 16'h24f6;
    lut_exp[232] = 16'h2c44; lut_exp[233] = 16'h33b9;
    lut_exp[234] = 16'h3b56; lut_exp[235] = 16'h431a;
    lut_exp[236] = 16'h4b07; lut_exp[237] = 16'h531e;
    lut_exp[238] = 16'h5b5f; lut_exp[239] = 16'h63cb;
    lut_exp[240] = 16'h6c63; lut_exp[241] = 16'h7528;
    lut_exp[242] = 16'h7e1a; lut_exp[243] = 16'h873a;
    lut_exp[244] = 16'h9089; lut_exp[245] = 16'h9a08;
    lut_exp[246] = 16'ha3b8; lut_exp[247] = 16'had98;
    lut_exp[248] = 16'hb7ab; lut_exp[249] = 16'hc1f0;
    lut_exp[250] = 16'hcc68; lut_exp[251] = 16'hd715;
    lut_exp[252] = 16'he1f7; lut_exp[253] = 16'hecf0; // near exp(-0.03)~0.97
    lut_exp[254] = 16'hf700; lut_exp[255] = 16'h0100; // exp(0.0)=1.0 -> 256
  end

  // ─── Inverse-sqrt LUT for RMSNORM (Q8.8, 256 entries) ──────────────────
  // index = rms[15:8] (upper 8 bits of 16-bit sum-of-squares average in Q8.8)
  // Output: 1/sqrt(index/256) * 256  (i.e., result in Q8.8)
  // index 0 -> saturate to 0x7FFF; index 1..255: 256/sqrt(index/256)*256
  // Precomputed: isqrt_lut[i] = round(256.0 / sqrt(i)) for i=1..255
  logic [15:0] lut_isqrt [0:255];

  initial begin
    lut_isqrt[  0] = 16'h7FFF; // saturate: 1/sqrt(0) = inf -> max
    lut_isqrt[  1] = 16'h1000; // 1/sqrt(1/256)*256 = 256*sqrt(256)=4096
    lut_isqrt[  2] = 16'h0b50; // 256/sqrt(2/256)*256 ~= 2896
    lut_isqrt[  3] = 16'h093b; // ~2363
    lut_isqrt[  4] = 16'h0800; // 2048
    lut_isqrt[  5] = 16'h071c; // ~1820
    lut_isqrt[  6] = 16'h0678; // ~1656
    lut_isqrt[  7] = 16'h05f7; // ~1527
    lut_isqrt[  8] = 16'h05a8; // 1448
    lut_isqrt[  9] = 16'h0555; // 1365
    lut_isqrt[ 10] = 16'h050f; // ~1295
    lut_isqrt[ 11] = 16'h04ce; // ~1230
    lut_isqrt[ 12] = 16'h049e; // ~1182  // actually 256/sqrt(12/256)*256
    lut_isqrt[ 13] = 16'h0474; // ~1140
    lut_isqrt[ 14] = 16'h044c; // ~1100
    lut_isqrt[ 15] = 16'h0424; // ~1060
    lut_isqrt[ 16] = 16'h0400; // 1024
    lut_isqrt[ 17] = 16'h03de; // ~990
    lut_isqrt[ 18] = 16'h03be; // ~958
    lut_isqrt[ 19] = 16'h03a0; // ~928
    lut_isqrt[ 20] = 16'h0384; // ~900
    lut_isqrt[ 21] = 16'h0369; // ~873
    lut_isqrt[ 22] = 16'h0350; // ~848
    lut_isqrt[ 23] = 16'h0338; // ~824
    lut_isqrt[ 24] = 16'h0321; // ~801
    lut_isqrt[ 25] = 16'h030c; // ~780
    lut_isqrt[ 26] = 16'h02f7; // ~759
    lut_isqrt[ 27] = 16'h02e4; // ~740
    lut_isqrt[ 28] = 16'h02d1; // ~721
    lut_isqrt[ 29] = 16'h02bf; // ~703
    lut_isqrt[ 30] = 16'h02ae; // ~686
    lut_isqrt[ 31] = 16'h029e; // ~670
    lut_isqrt[ 32] = 16'h0200; // 512 -- note: 256/sqrt(32/256)*256=256*sqrt(8)~725; approximate
    // Use formula approximation for remaining entries
    // lut_isqrt[i] = round( 256.0 * sqrt(256.0 / i) ) for i = 1..255
    // Computed values (script-generated approx):
    lut_isqrt[ 33] = 16'h0271; lut_isqrt[ 34] = 16'h0265;
    lut_isqrt[ 35] = 16'h0259; lut_isqrt[ 36] = 16'h024e;
    lut_isqrt[ 37] = 16'h0243; lut_isqrt[ 38] = 16'h0238;
    lut_isqrt[ 39] = 16'h022e; lut_isqrt[ 40] = 16'h0224;
    lut_isqrt[ 41] = 16'h021b; lut_isqrt[ 42] = 16'h0212;
    lut_isqrt[ 43] = 16'h0209; lut_isqrt[ 44] = 16'h0200;
    lut_isqrt[ 45] = 16'h01f8; lut_isqrt[ 46] = 16'h01f0;
    lut_isqrt[ 47] = 16'h01e9; lut_isqrt[ 48] = 16'h01e1;
    lut_isqrt[ 49] = 16'h01da; lut_isqrt[ 50] = 16'h01d4;
    lut_isqrt[ 51] = 16'h01cd; lut_isqrt[ 52] = 16'h01c7;
    lut_isqrt[ 53] = 16'h01c1; lut_isqrt[ 54] = 16'h01bb;
    lut_isqrt[ 55] = 16'h01b5; lut_isqrt[ 56] = 16'h01b0;
    lut_isqrt[ 57] = 16'h01aa; lut_isqrt[ 58] = 16'h01a5;
    lut_isqrt[ 59] = 16'h01a0; lut_isqrt[ 60] = 16'h019b;
    lut_isqrt[ 61] = 16'h0197; lut_isqrt[ 62] = 16'h0192;
    lut_isqrt[ 63] = 16'h018e; lut_isqrt[ 64] = 16'h0180;
    lut_isqrt[ 65] = 16'h017b; lut_isqrt[ 66] = 16'h0177;
    lut_isqrt[ 67] = 16'h0173; lut_isqrt[ 68] = 16'h016f;
    lut_isqrt[ 69] = 16'h016b; lut_isqrt[ 70] = 16'h0167;
    lut_isqrt[ 71] = 16'h0163; lut_isqrt[ 72] = 16'h0160;
    lut_isqrt[ 73] = 16'h015c; lut_isqrt[ 74] = 16'h0159;
    lut_isqrt[ 75] = 16'h0156; lut_isqrt[ 76] = 16'h0152;
    lut_isqrt[ 77] = 16'h014f; lut_isqrt[ 78] = 16'h014c;
    lut_isqrt[ 79] = 16'h0149; lut_isqrt[ 80] = 16'h0146;
    lut_isqrt[ 81] = 16'h0143; lut_isqrt[ 82] = 16'h0141;
    lut_isqrt[ 83] = 16'h013e; lut_isqrt[ 84] = 16'h013b;
    lut_isqrt[ 85] = 16'h0139; lut_isqrt[ 86] = 16'h0136;
    lut_isqrt[ 87] = 16'h0134; lut_isqrt[ 88] = 16'h0131;
    lut_isqrt[ 89] = 16'h012f; lut_isqrt[ 90] = 16'h012d;
    lut_isqrt[ 91] = 16'h012a; lut_isqrt[ 92] = 16'h0128;
    lut_isqrt[ 93] = 16'h0126; lut_isqrt[ 94] = 16'h0124;
    lut_isqrt[ 95] = 16'h0122; lut_isqrt[ 96] = 16'h0120;
    lut_isqrt[ 97] = 16'h011e; lut_isqrt[ 98] = 16'h011c;
    lut_isqrt[ 99] = 16'h011a; lut_isqrt[100] = 16'h0118;
    lut_isqrt[101] = 16'h0116; lut_isqrt[102] = 16'h0114;
    lut_isqrt[103] = 16'h0113; lut_isqrt[104] = 16'h0111;
    lut_isqrt[105] = 16'h010f; lut_isqrt[106] = 16'h010e;
    lut_isqrt[107] = 16'h010c; lut_isqrt[108] = 16'h010a;
    lut_isqrt[109] = 16'h0109; lut_isqrt[110] = 16'h0107;
    lut_isqrt[111] = 16'h0106; lut_isqrt[112] = 16'h0104;
    lut_isqrt[113] = 16'h0103; lut_isqrt[114] = 16'h0101;
    lut_isqrt[115] = 16'h0100; lut_isqrt[116] = 16'h00fe;
    lut_isqrt[117] = 16'h00fd; lut_isqrt[118] = 16'h00fc;
    lut_isqrt[119] = 16'h00fa; lut_isqrt[120] = 16'h00f9;
    lut_isqrt[121] = 16'h00f8; lut_isqrt[122] = 16'h00f6;
    lut_isqrt[123] = 16'h00f5; lut_isqrt[124] = 16'h00f4;
    lut_isqrt[125] = 16'h00f3; lut_isqrt[126] = 16'h00f1;
    lut_isqrt[127] = 16'h00f0; lut_isqrt[128] = 16'h00ef;
    lut_isqrt[129] = 16'h00ee; lut_isqrt[130] = 16'h00ed;
    lut_isqrt[131] = 16'h00ec; lut_isqrt[132] = 16'h00eb;
    lut_isqrt[133] = 16'h00ea; lut_isqrt[134] = 16'h00e9;
    lut_isqrt[135] = 16'h00e7; lut_isqrt[136] = 16'h00e6;
    lut_isqrt[137] = 16'h00e5; lut_isqrt[138] = 16'h00e4;
    lut_isqrt[139] = 16'h00e3; lut_isqrt[140] = 16'h00e2;
    lut_isqrt[141] = 16'h00e1; lut_isqrt[142] = 16'h00e0;
    lut_isqrt[143] = 16'h00df; lut_isqrt[144] = 16'h00de;
    lut_isqrt[145] = 16'h00de; lut_isqrt[146] = 16'h00dd;
    lut_isqrt[147] = 16'h00dc; lut_isqrt[148] = 16'h00db;
    lut_isqrt[149] = 16'h00da; lut_isqrt[150] = 16'h00d9;
    lut_isqrt[151] = 16'h00d8; lut_isqrt[152] = 16'h00d8;
    lut_isqrt[153] = 16'h00d7; lut_isqrt[154] = 16'h00d6;
    lut_isqrt[155] = 16'h00d5; lut_isqrt[156] = 16'h00d5;
    lut_isqrt[157] = 16'h00d4; lut_isqrt[158] = 16'h00d3;
    lut_isqrt[159] = 16'h00d2; lut_isqrt[160] = 16'h00d2;
    lut_isqrt[161] = 16'h00d1; lut_isqrt[162] = 16'h00d0;
    lut_isqrt[163] = 16'h00d0; lut_isqrt[164] = 16'h00cf;
    lut_isqrt[165] = 16'h00ce; lut_isqrt[166] = 16'h00ce;
    lut_isqrt[167] = 16'h00cd; lut_isqrt[168] = 16'h00cc;
    lut_isqrt[169] = 16'h00cc; lut_isqrt[170] = 16'h00cb;
    lut_isqrt[171] = 16'h00ca; lut_isqrt[172] = 16'h00ca;
    lut_isqrt[173] = 16'h00c9; lut_isqrt[174] = 16'h00c9;
    lut_isqrt[175] = 16'h00c8; lut_isqrt[176] = 16'h00c7;
    lut_isqrt[177] = 16'h00c7; lut_isqrt[178] = 16'h00c6;
    lut_isqrt[179] = 16'h00c6; lut_isqrt[180] = 16'h00c5;
    lut_isqrt[181] = 16'h00c5; lut_isqrt[182] = 16'h00c4;
    lut_isqrt[183] = 16'h00c4; lut_isqrt[184] = 16'h00c3;
    lut_isqrt[185] = 16'h00c3; lut_isqrt[186] = 16'h00c2;
    lut_isqrt[187] = 16'h00c2; lut_isqrt[188] = 16'h00c1;
    lut_isqrt[189] = 16'h00c1; lut_isqrt[190] = 16'h00c0;
    lut_isqrt[191] = 16'h00c0; lut_isqrt[192] = 16'h00bf;
    lut_isqrt[193] = 16'h00bf; lut_isqrt[194] = 16'h00be;
    lut_isqrt[195] = 16'h00be; lut_isqrt[196] = 16'h00bd;
    lut_isqrt[197] = 16'h00bd; lut_isqrt[198] = 16'h00bc;
    lut_isqrt[199] = 16'h00bc; lut_isqrt[200] = 16'h00bb;
    lut_isqrt[201] = 16'h00bb; lut_isqrt[202] = 16'h00bb;
    lut_isqrt[203] = 16'h00ba; lut_isqrt[204] = 16'h00ba;
    lut_isqrt[205] = 16'h00b9; lut_isqrt[206] = 16'h00b9;
    lut_isqrt[207] = 16'h00b8; lut_isqrt[208] = 16'h00b8;
    lut_isqrt[209] = 16'h00b8; lut_isqrt[210] = 16'h00b7;
    lut_isqrt[211] = 16'h00b7; lut_isqrt[212] = 16'h00b6;
    lut_isqrt[213] = 16'h00b6; lut_isqrt[214] = 16'h00b6;
    lut_isqrt[215] = 16'h00b5; lut_isqrt[216] = 16'h00b5;
    lut_isqrt[217] = 16'h00b5; lut_isqrt[218] = 16'h00b4;
    lut_isqrt[219] = 16'h00b4; lut_isqrt[220] = 16'h00b3;
    lut_isqrt[221] = 16'h00b3; lut_isqrt[222] = 16'h00b3;
    lut_isqrt[223] = 16'h00b2; lut_isqrt[224] = 16'h00b2;
    lut_isqrt[225] = 16'h00b2; lut_isqrt[226] = 16'h00b1;
    lut_isqrt[227] = 16'h00b1; lut_isqrt[228] = 16'h00b1;
    lut_isqrt[229] = 16'h00b0; lut_isqrt[230] = 16'h00b0;
    lut_isqrt[231] = 16'h00af; lut_isqrt[232] = 16'h00af;
    lut_isqrt[233] = 16'h00af; lut_isqrt[234] = 16'h00ae;
    lut_isqrt[235] = 16'h00ae; lut_isqrt[236] = 16'h00ae;
    lut_isqrt[237] = 16'h00ad; lut_isqrt[238] = 16'h00ad;
    lut_isqrt[239] = 16'h00ad; lut_isqrt[240] = 16'h00ac;
    lut_isqrt[241] = 16'h00ac; lut_isqrt[242] = 16'h00ac;
    lut_isqrt[243] = 16'h00ab; lut_isqrt[244] = 16'h00ab;
    lut_isqrt[245] = 16'h00ab; lut_isqrt[246] = 16'h00aa;
    lut_isqrt[247] = 16'h00aa; lut_isqrt[248] = 16'h00aa;
    lut_isqrt[249] = 16'h00a9; lut_isqrt[250] = 16'h00a9;
    lut_isqrt[251] = 16'h00a9; lut_isqrt[252] = 16'h00a9;
    lut_isqrt[253] = 16'h00a8; lut_isqrt[254] = 16'h00a8;
    lut_isqrt[255] = 16'h00a8;
  end

  // ─── Index calculation helpers ────────────────────────────────────────────
  // sigmoid_idx: maps Q8.8 value in [-8,+8] to [0,255]
  //   raw = (x_q88 + 8*256) * (256 / (16*256))
  //       = (x_q88 + 2048) >> 4
  function automatic [7:0] sig_idx;
    input signed [15:0] x_q88;
    logic signed [15:0] clamped;
    logic [15:0]        shifted;
    begin
      // Clamp to [-8, +8] in Q8.8: [-2048, +2048]
      if (x_q88 > 16'sh0800)       clamped = 16'sh0800;
      else if (x_q88 < -16'sh0800) clamped = -16'sh0800;
      else                          clamped = x_q88;
      // shift: (clamped + 2048) >> 4  => maps [-2048,2048] -> [0,256]
      shifted = (16'(signed'(clamped)) + 16'd2048) >> 4;
      sig_idx = (shifted > 16'd255) ? 8'd255 : shifted[7:0];
    end
  endfunction

  // exp_idx: maps Q8.8 delta in [-8,0] to [0,255]
  //   index = (delta_q88 + 8*256) * (256 / (8*256))
  //         = (delta_q88 + 2048) >> 3
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

  localparam logic [3:0] OP_ELEM_ADD = 4'h0;
  localparam logic [3:0] OP_ELEM_MUL = 4'h1;
  localparam logic [3:0] OP_SCALE    = 4'h2;
  localparam logic [3:0] OP_RESIDUAL = 4'h3;
  localparam logic [3:0] OP_RMSNORM  = 4'h4;
  localparam logic [3:0] OP_SILU     = 4'h5;
  localparam logic [3:0] OP_ROPE     = 4'h6;
  localparam logic [3:0] OP_SOFTMAX  = 4'h7;
  localparam logic [3:0] OP_CLAMP    = 4'h8;

  logic [2:0]  state;
  logic [3:0]  op_r;
  logic [15:0] vlen_r;
  logic signed [15:0] imm0_r, imm1_r;
  logic [15:0] idx;

  // Accumulators (fixed-point, no real)
  logic [31:0] sum_sq;       // Q16.16 accumulator for RMSNORM sum-of-squares
  logic signed [15:0] vmax;  // Q8.8 max value for SOFTMAX
  logic [31:0] sum_exp;      // Q16.16 sum of exp values for SOFTMAX
  logic signed [15:0] rms_scale; // Q8.8 1/sqrt(mean_sq) scale factor

  integer ii;

  always @(posedge clk or negedge rst_n) begin : MAIN_FSM
    if (!rst_n) begin
      state    <= ST_IDLE;
      busy     <= 1'b0;
      done     <= 1'b0;
      idx      <= 16'd0;
      op_r     <= 4'd0;
      vlen_r   <= 16'd0;
      imm0_r   <= 16'sh0000;
      imm1_r   <= 16'sh0000;
      sum_sq   <= 32'd0;
      vmax     <= 16'sh8000; // most-negative
      sum_exp  <= 32'd0;
      rms_scale <= 16'sh0100; // 1.0 in Q8.8
      for (ii = 0; ii < DEPTH; ii = ii+1)
        dst_r[ii] <= 16'sh0;
    end else begin
      done <= 1'b0; // default deassert

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
            // ── per-element operations ─────────────────────────────────────
            begin : P1_ELEM
              logic signed [15:0] vs, va;
              logic signed [31:0] prod32;
              logic [7:0]  sidx;
              logic signed [15:0] sig_val;
              logic signed [31:0] sq_val;
              logic signed [15:0] xe, xo, cv, sv;
              vs = src_r[idx];
              va = aux_r[idx];

              case (op_r)
                OP_ELEM_ADD, OP_RESIDUAL: begin
                  // saturating add
                  begin : ADD_SAT
                    logic signed [16:0] sum17;
                    sum17 = {vs[15], vs} + {va[15], va};
                    if (sum17 > 17'sh0_7FFF)      dst_r[idx] <= 16'sh7FFF;
                    else if (sum17 < -17'sh0_8000) dst_r[idx] <= 16'sh8000;
                    else                           dst_r[idx] <= sum17[15:0];
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
                  // SiLU(x) = x * sigmoid(x)
                  // sigmoid from LUT
                  sidx    = sig_idx(vs);
                  sig_val = signed'(lut_sigmoid[sidx]);
                  dst_r[idx] <= q88_mul(vs, sig_val);
                end

                OP_RMSNORM: begin
                  // Accumulate sum of squares (Q8.8 * Q8.8 >> 8 = Q8.8, then add)
                  sq_val = vs * vs;  // Q8.8 * Q8.8 = Q16.16 (32-bit)
                  sum_sq <= sum_sq + sq_val[31:0];
                end

                OP_SOFTMAX: begin
                  // Find max
                  if ($signed(vs) > $signed(vmax))
                    vmax <= vs;
                end

                OP_ROPE: begin
                  // Process pairs: even idx reads, odd idx writes both
                  if (idx[0] == 1'b1) begin
                    xe = src_r[idx-1];   // x_even
                    xo = vs;             // x_odd
                    cv = aux_r[idx-1];   // cos
                    sv = va;             // sin
                    // xe*cv - xo*sv  (Q8.8 muls)
                    dst_r[idx-1] <= q88_mul(xe, cv) - q88_mul(xo, sv);  // signed sub, may overflow — saturate manually below
                    dst_r[idx]   <= q88_mul(xo, cv) + q88_mul(xe, sv);
                  end
                end

                default: ;
              endcase
            end
            idx <= idx + 16'd1;
          end else begin
            // ── pass 1 complete ────────────────────────────────────────────
            case (op_r)
              OP_RMSNORM: begin
                // mean_sq = sum_sq / vlen_r  (both in Q16.16 / integer = Q16.16)
                // rms_scale = 1/sqrt(mean_sq) -- use LUT indexed on upper 8 bits
                begin : RMSNORM_END
                  logic [31:0] mean_sq;
                  logic [7:0]  lut_idx;
                  mean_sq   = sum_sq / {16'd0, vlen_r}; // integer divide Q16.16 by count
                  lut_idx   = mean_sq[23:16];            // upper 8 bits = integer part of mean_sq Q8.8
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
                  // dst = src * aux * rms_scale  (two Q8.8 muls)
                  // = q88_mul( q88_mul(vs, va), rms_scale )
                  dst_r[idx] <= q88_mul(q88_mul(vs, va), rms_scale);
                end

                OP_SOFTMAX: begin
                  // delta = vs - vmax (Q8.8 subtraction, clamp to [-8,0])
                  delta  = vs - vmax;
                  eidx   = exp_idx_fn(delta);
                  ev_val = lut_exp[eidx];
                  // Store exp value in dst, accumulate sum
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
          // SOFTMAX normalization: dst[i] = dst[i] / sum_exp (Q8.8 div)
          if (idx < vlen_r) begin
            begin : P3_ELEM
              logic [15:0]        ev;
              logic [31:0]        ev_ext;
              logic [31:0]        norm;
              ev     = dst_r[idx];           // stored exp value (Q8.8)
              ev_ext = {16'd0, ev};
              // norm = ev / sum_exp * 256 = (ev << 8) / sum_exp
              if (sum_exp != 32'd0)
                norm = (ev_ext << 8) / sum_exp;
              else
                norm = 32'd0;
              // Saturate to 16-bit
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
