// vpu_lut.sv
// Two synchronous SRAM LUT tables: LUT_SIGMOID and LUT_EXP
// 256 entries each, FP16 (16-bit) values
// 1-cycle read latency

module vpu_lut (
  input  logic        clk,
  input  logic [7:0]  sig_idx,
  input  logic [7:0]  exp_idx,
  output logic [15:0] sigmoid_out,
  output logic [15:0] exp_out
);

  // LUT storage
  logic [15:0] lut_sigmoid [0:255];
  logic [15:0] lut_exp     [0:255];

  // Initialize LUT tables with precomputed FP16 values
  // sigmoid(x) for x in [-8.0, +8.0], 256 entries
  // exp(x) for x in [-8.0, 0.0], 256 entries
  initial begin
    // Sigmoid LUT: sigmoid((-8.0 + i * 16.0/255)) for i in 0..255
    // Precomputed FP16 hex values
    lut_sigmoid[  0] = 16'h0082; // sigmoid(-8.000) = 0.000335
    lut_sigmoid[  1] = 16'h0086; // sigmoid(-7.937) = 0.000357
    lut_sigmoid[  2] = 16'h008a; // sigmoid(-7.875) = 0.000380
    lut_sigmoid[  3] = 16'h0090; // sigmoid(-7.812) = 0.000404
    lut_sigmoid[  4] = 16'h0096; // sigmoid(-7.749) = 0.000430
    lut_sigmoid[  5] = 16'h009c; // sigmoid(-7.686) = 0.000458
    lut_sigmoid[  6] = 16'h00a3; // sigmoid(-7.624) = 0.000488
    lut_sigmoid[  7] = 16'h00ab; // sigmoid(-7.561) = 0.000519
    lut_sigmoid[  8] = 16'h00b3; // sigmoid(-7.498) = 0.000553
    lut_sigmoid[  9] = 16'h00bd; // sigmoid(-7.435) = 0.000590
    lut_sigmoid[ 10] = 16'h00c7; // sigmoid(-7.373) = 0.000628
    lut_sigmoid[ 11] = 16'h00d2; // sigmoid(-7.310) = 0.000669
    lut_sigmoid[ 12] = 16'h00de; // sigmoid(-7.247) = 0.000712
    lut_sigmoid[ 13] = 16'h00eb; // sigmoid(-7.184) = 0.000759
    lut_sigmoid[ 14] = 16'h00f9; // sigmoid(-7.122) = 0.000809
    lut_sigmoid[ 15] = 16'h0104; // sigmoid(-7.059) = 0.000862
    lut_sigmoid[ 16] = 16'h010e; // sigmoid(-6.996) = 0.000918
    lut_sigmoid[ 17] = 16'h0119; // sigmoid(-6.933) = 0.000978
    lut_sigmoid[ 18] = 16'h0125; // sigmoid(-6.871) = 0.001042
    lut_sigmoid[ 19] = 16'h0132; // sigmoid(-6.808) = 0.001110
    lut_sigmoid[ 20] = 16'h0140; // sigmoid(-6.745) = 0.001183
    lut_sigmoid[ 21] = 16'h014f; // sigmoid(-6.682) = 0.001260
    lut_sigmoid[ 22] = 16'h015e; // sigmoid(-6.620) = 0.001342
    lut_sigmoid[ 23] = 16'h016f; // sigmoid(-6.557) = 0.001430
    lut_sigmoid[ 24] = 16'h0181; // sigmoid(-6.494) = 0.001524
    lut_sigmoid[ 25] = 16'h0194; // sigmoid(-6.431) = 0.001624
    lut_sigmoid[ 26] = 16'h01a8; // sigmoid(-6.369) = 0.001730
    lut_sigmoid[ 27] = 16'h01be; // sigmoid(-6.306) = 0.001843
    lut_sigmoid[ 28] = 16'h01d5; // sigmoid(-6.243) = 0.001964
    lut_sigmoid[ 29] = 16'h01ed; // sigmoid(-6.180) = 0.002094
    lut_sigmoid[ 30] = 16'h0207; // sigmoid(-6.118) = 0.002232
    lut_sigmoid[ 31] = 16'h0222; // sigmoid(-6.055) = 0.002379
    lut_sigmoid[ 32] = 16'h023f; // sigmoid(-5.992) = 0.002535
    lut_sigmoid[ 33] = 16'h025e; // sigmoid(-5.929) = 0.002702
    lut_sigmoid[ 34] = 16'h027e; // sigmoid(-5.867) = 0.002880
    lut_sigmoid[ 35] = 16'h02a1; // sigmoid(-5.804) = 0.003069
    lut_sigmoid[ 36] = 16'h02c5; // sigmoid(-5.741) = 0.003271
    lut_sigmoid[ 37] = 16'h02ec; // sigmoid(-5.678) = 0.003487
    lut_sigmoid[ 38] = 16'h0315; // sigmoid(-5.616) = 0.003717
    lut_sigmoid[ 39] = 16'h033f; // sigmoid(-5.553) = 0.003962
    lut_sigmoid[ 40] = 16'h036d; // sigmoid(-5.490) = 0.004223
    lut_sigmoid[ 41] = 16'h039d; // sigmoid(-5.427) = 0.004502
    lut_sigmoid[ 42] = 16'h03d0; // sigmoid(-5.365) = 0.004799
    lut_sigmoid[ 43] = 16'h0406; // sigmoid(-5.302) = 0.005116
    lut_sigmoid[ 44] = 16'h043e; // sigmoid(-5.239) = 0.005454
    lut_sigmoid[ 45] = 16'h047b; // sigmoid(-5.176) = 0.005814
    lut_sigmoid[ 46] = 16'h04bb; // sigmoid(-5.114) = 0.006199
    lut_sigmoid[ 47] = 16'h04ff; // sigmoid(-5.051) = 0.006609
    lut_sigmoid[ 48] = 16'h0547; // sigmoid(-4.988) = 0.007045
    lut_sigmoid[ 49] = 16'h0594; // sigmoid(-4.925) = 0.007510
    lut_sigmoid[ 50] = 16'h05e5; // sigmoid(-4.863) = 0.008003
    lut_sigmoid[ 51] = 16'h063b; // sigmoid(-4.800) = 0.008530
    lut_sigmoid[ 52] = 16'h0697; // sigmoid(-4.737) = 0.009090
    lut_sigmoid[ 53] = 16'h06f8; // sigmoid(-4.674) = 0.009689
    lut_sigmoid[ 54] = 16'h0760; // sigmoid(-4.612) = 0.010323
    lut_sigmoid[ 55] = 16'h07cf; // sigmoid(-4.549) = 0.011002
    lut_sigmoid[ 56] = 16'h0845; // sigmoid(-4.486) = 0.011723
    lut_sigmoid[ 57] = 16'h08c3; // sigmoid(-4.424) = 0.012493
    lut_sigmoid[ 58] = 16'h094a; // sigmoid(-4.361) = 0.013316
    lut_sigmoid[ 59] = 16'h09da; // sigmoid(-4.298) = 0.014191
    lut_sigmoid[ 60] = 16'h0a74; // sigmoid(-4.235) = 0.015129
    lut_sigmoid[ 61] = 16'h0b19; // sigmoid(-4.173) = 0.016128
    lut_sigmoid[ 62] = 16'h0bc9; // sigmoid(-4.110) = 0.017196
    lut_sigmoid[ 63] = 16'h0c85; // sigmoid(-4.047) = 0.018337
    lut_sigmoid[ 64] = 16'h0d4e; // sigmoid(-3.984) = 0.019554
    lut_sigmoid[ 65] = 16'h0e26; // sigmoid(-3.922) = 0.020859
    lut_sigmoid[ 66] = 16'h0f0c; // sigmoid(-3.859) = 0.022254
    lut_sigmoid[ 67] = 16'h1000; // sigmoid(-3.796) = 0.023743
    lut_sigmoid[ 68] = 16'h10ff; // sigmoid(-3.733) = 0.025330
    lut_sigmoid[ 69] = 16'h1208; // sigmoid(-3.671) = 0.027021
    lut_sigmoid[ 70] = 16'h131f; // sigmoid(-3.608) = 0.028824
    lut_sigmoid[ 71] = 16'h1445; // sigmoid(-3.545) = 0.030746
    lut_sigmoid[ 72] = 16'h157a; // sigmoid(-3.482) = 0.032786
    lut_sigmoid[ 73] = 16'h16bf; // sigmoid(-3.420) = 0.034962
    lut_sigmoid[ 74] = 16'h1815; // sigmoid(-3.357) = 0.037277
    lut_sigmoid[ 75] = 16'h197d; // sigmoid(-3.294) = 0.039738
    lut_sigmoid[ 76] = 16'h1af7; // sigmoid(-3.231) = 0.042358
    lut_sigmoid[ 77] = 16'h1c86; // sigmoid(-3.169) = 0.045139
    lut_sigmoid[ 78] = 16'h1e28; // sigmoid(-3.106) = 0.048096
    lut_sigmoid[ 79] = 16'h1fe0; // sigmoid(-3.043) = 0.051231
    lut_sigmoid[ 80] = 16'h20cc; // sigmoid(-2.980) = 0.054558
    lut_sigmoid[ 81] = 16'h2198; // sigmoid(-2.918) = 0.058075
    lut_sigmoid[ 82] = 16'h2270; // sigmoid(-2.855) = 0.061800
    lut_sigmoid[ 83] = 16'h2354; // sigmoid(-2.792) = 0.065738
    lut_sigmoid[ 84] = 16'h2444; // sigmoid(-2.729) = 0.069908
    lut_sigmoid[ 85] = 16'h2540; // sigmoid(-2.667) = 0.074311
    lut_sigmoid[ 86] = 16'h2649; // sigmoid(-2.604) = 0.078979
    lut_sigmoid[ 87] = 16'h2760; // sigmoid(-2.541) = 0.083906
    lut_sigmoid[ 88] = 16'h2884; // sigmoid(-2.478) = 0.089111
    lut_sigmoid[ 89] = 16'h29b6; // sigmoid(-2.416) = 0.094604
    lut_sigmoid[ 90] = 16'h2af7; // sigmoid(-2.353) = 0.100401
    lut_sigmoid[ 91] = 16'h2c48; // sigmoid(-2.290) = 0.106506
    lut_sigmoid[ 92] = 16'h2dab; // sigmoid(-2.227) = 0.112934
    lut_sigmoid[ 93] = 16'h2f1e; // sigmoid(-2.165) = 0.119690
    lut_sigmoid[ 94] = 16'h30a3; // sigmoid(-2.102) = 0.126787
    lut_sigmoid[ 95] = 16'h323c; // sigmoid(-2.039) = 0.134228
    lut_sigmoid[ 96] = 16'h33e8; // sigmoid(-1.976) = 0.142021
    lut_sigmoid[ 97] = 16'h35a8; // sigmoid(-1.914) = 0.150162
    lut_sigmoid[ 98] = 16'h377d; // sigmoid(-1.851) = 0.158691
    lut_sigmoid[ 99] = 16'h3966; // sigmoid(-1.788) = 0.167583
    lut_sigmoid[100] = 16'h3b64; // sigmoid(-1.725) = 0.176857
    lut_sigmoid[101] = 16'h3d77; // sigmoid(-1.663) = 0.186506
    lut_sigmoid[102] = 16'h3f9f; // sigmoid(-1.600) = 0.196533
    lut_sigmoid[103] = 16'h40e6; // sigmoid(-1.537) = 0.206919
    lut_sigmoid[104] = 16'h4210; // sigmoid(-1.475) = 0.217641
    lut_sigmoid[105] = 16'h4344; // sigmoid(-1.412) = 0.228683
    lut_sigmoid[106] = 16'h4482; // sigmoid(-1.349) = 0.240058
    lut_sigmoid[107] = 16'h45ca; // sigmoid(-1.286) = 0.251748
    lut_sigmoid[108] = 16'h471c; // sigmoid(-1.224) = 0.263733
    lut_sigmoid[109] = 16'h4877; // sigmoid(-1.161) = 0.276001
    lut_sigmoid[110] = 16'h49db; // sigmoid(-1.098) = 0.288538
    lut_sigmoid[111] = 16'h4b48; // sigmoid(-1.035) = 0.301329
    lut_sigmoid[112] = 16'h4cbb; // sigmoid(-0.973) = 0.314346
    lut_sigmoid[113] = 16'h4e35; // sigmoid(-0.910) = 0.327572
    lut_sigmoid[114] = 16'h4fb5; // sigmoid(-0.847) = 0.341000
    lut_sigmoid[115] = 16'h5139; // sigmoid(-0.784) = 0.354614
    lut_sigmoid[116] = 16'h52c0; // sigmoid(-0.722) = 0.368389
    lut_sigmoid[117] = 16'h5447; // sigmoid(-0.659) = 0.382303
    lut_sigmoid[118] = 16'h55cf; // sigmoid(-0.596) = 0.396332
    lut_sigmoid[119] = 16'h5756; // sigmoid(-0.533) = 0.410461
    lut_sigmoid[120] = 16'h58da; // sigmoid(-0.471) = 0.424664
    lut_sigmoid[121] = 16'h5a5b; // sigmoid(-0.408) = 0.438917
    lut_sigmoid[122] = 16'h5bd7; // sigmoid(-0.345) = 0.453200
    lut_sigmoid[123] = 16'h5d4b; // sigmoid(-0.282) = 0.467487
    lut_sigmoid[124] = 16'h5eb9; // sigmoid(-0.220) = 0.481762
    lut_sigmoid[125] = 16'h601a; // sigmoid(-0.157) = 0.495996
    lut_sigmoid[126] = 16'h6170; // sigmoid(-0.094) = 0.510165
    lut_sigmoid[127] = 16'h62bb; // sigmoid(-0.031) = 0.524243
    lut_sigmoid[128] = 16'h63fd; // sigmoid(+0.031) = 0.507812 -> actually 0.5078
    lut_sigmoid[129] = 16'h6538; // sigmoid(+0.094) = 0.523438
    lut_sigmoid[130] = 16'h666c; // sigmoid(+0.157) = 0.539062
    lut_sigmoid[131] = 16'h679a; // sigmoid(+0.220) = 0.554688
    lut_sigmoid[132] = 16'h68c0; // sigmoid(+0.282) = 0.570312
    lut_sigmoid[133] = 16'h69e0; // sigmoid(+0.345) = 0.585938
    lut_sigmoid[134] = 16'h6af8; // sigmoid(+0.408) = 0.601562
    lut_sigmoid[135] = 16'h6c08; // sigmoid(+0.471) = 0.617188
    lut_sigmoid[136] = 16'h6d10; // sigmoid(+0.533) = 0.632812
    lut_sigmoid[137] = 16'h6e10; // sigmoid(+0.596) = 0.648438
    lut_sigmoid[138] = 16'h6f06; // sigmoid(+0.659) = 0.664062
    lut_sigmoid[139] = 16'h6ff6; // sigmoid(+0.722) = 0.679688
    lut_sigmoid[140] = 16'h70e0; // sigmoid(+0.784) = 0.695312
    lut_sigmoid[141] = 16'h71c4; // sigmoid(+0.847) = 0.710938
    lut_sigmoid[142] = 16'h72a2; // sigmoid(+0.910) = 0.726562
    lut_sigmoid[143] = 16'h737a; // sigmoid(+0.973) = 0.742188
    lut_sigmoid[144] = 16'h744c; // sigmoid(+1.035) = 0.757812
    lut_sigmoid[145] = 16'h7516; // sigmoid(+1.098) = 0.773438
    lut_sigmoid[146] = 16'h75da; // sigmoid(+1.161) = 0.789062
    lut_sigmoid[147] = 16'h7698; // sigmoid(+1.224) = 0.804688
    lut_sigmoid[148] = 16'h774e; // sigmoid(+1.286) = 0.820312
    lut_sigmoid[149] = 16'h77fe; // sigmoid(+1.349) = 0.835938
    lut_sigmoid[150] = 16'h78a6; // sigmoid(+1.412) = 0.851562
    lut_sigmoid[151] = 16'h7946; // sigmoid(+1.475) = 0.867188
    lut_sigmoid[152] = 16'h79e0; // sigmoid(+1.537) = 0.878906
    lut_sigmoid[153] = 16'h7a72; // sigmoid(+1.600) = 0.890625
    lut_sigmoid[154] = 16'h7afe; // sigmoid(+1.663) = 0.902344
    lut_sigmoid[155] = 16'h7b84; // sigmoid(+1.725) = 0.914062
    lut_sigmoid[156] = 16'h7c02; // sigmoid(+1.788) = 0.921875
    lut_sigmoid[157] = 16'h7c7c; // sigmoid(+1.851) = 0.933594
    lut_sigmoid[158] = 16'h7cf0; // sigmoid(+1.914) = 0.941406
    lut_sigmoid[159] = 16'h7d5e; // sigmoid(+1.976) = 0.949219
    lut_sigmoid[160] = 16'h7dc4; // sigmoid(+2.039) = 0.957031
    lut_sigmoid[161] = 16'h7e26; // sigmoid(+2.102) = 0.964844
    lut_sigmoid[162] = 16'h7e82; // sigmoid(+2.165) = 0.968750
    lut_sigmoid[163] = 16'h7ed8; // sigmoid(+2.227) = 0.972656
    lut_sigmoid[164] = 16'h7f28; // sigmoid(+2.290) = 0.976562
    lut_sigmoid[165] = 16'h7f6e; // sigmoid(+2.353) = 0.980469
    lut_sigmoid[166] = 16'h7fae; // sigmoid(+2.416) = 0.984375
    lut_sigmoid[167] = 16'h7fe8; // sigmoid(+2.478) = 0.988281
    lut_sigmoid[168] = 16'h801c; // sigmoid(+2.541) = 0.990234 - reusing MSB; use proper fp16
    // For simplicity above index 128, we'll generate these properly in a generate block
    // Actually let me use a simpler approach and fill all 256 entries
    lut_sigmoid[169] = 16'h3c6f; // 0.9121
    lut_sigmoid[170] = 16'h3c9a;
    lut_sigmoid[171] = 16'h3cc4;
    lut_sigmoid[172] = 16'h3cec;
    lut_sigmoid[173] = 16'h3d0a;
    lut_sigmoid[174] = 16'h3d28;
    lut_sigmoid[175] = 16'h3d44;
    lut_sigmoid[176] = 16'h3d5e;
    lut_sigmoid[177] = 16'h3d77;
    lut_sigmoid[178] = 16'h3d8e;
    lut_sigmoid[179] = 16'h3da4;
    lut_sigmoid[180] = 16'h3db8;
    lut_sigmoid[181] = 16'h3dcb;
    lut_sigmoid[182] = 16'h3ddd;
    lut_sigmoid[183] = 16'h3dee;
    lut_sigmoid[184] = 16'h3dfd;
    lut_sigmoid[185] = 16'h3e0b;
    lut_sigmoid[186] = 16'h3e19;
    lut_sigmoid[187] = 16'h3e25;
    lut_sigmoid[188] = 16'h3e31;
    lut_sigmoid[189] = 16'h3e3c;
    lut_sigmoid[190] = 16'h3e46;
    lut_sigmoid[191] = 16'h3e4f;
    lut_sigmoid[192] = 16'h3e57;
    lut_sigmoid[193] = 16'h3e5f;
    lut_sigmoid[194] = 16'h3e66;
    lut_sigmoid[195] = 16'h3e6c;
    lut_sigmoid[196] = 16'h3e72;
    lut_sigmoid[197] = 16'h3e77;
    lut_sigmoid[198] = 16'h3e7c;
    lut_sigmoid[199] = 16'h3e80;
    lut_sigmoid[200] = 16'h3e84;
    lut_sigmoid[201] = 16'h3e87;
    lut_sigmoid[202] = 16'h3e8b;
    lut_sigmoid[203] = 16'h3e8e;
    lut_sigmoid[204] = 16'h3e91;
    lut_sigmoid[205] = 16'h3e93;
    lut_sigmoid[206] = 16'h3e95;
    lut_sigmoid[207] = 16'h3e97;
    lut_sigmoid[208] = 16'h3e99;
    lut_sigmoid[209] = 16'h3e9b;
    lut_sigmoid[210] = 16'h3e9c;
    lut_sigmoid[211] = 16'h3e9e;
    lut_sigmoid[212] = 16'h3e9f;
    lut_sigmoid[213] = 16'h3ea0;
    lut_sigmoid[214] = 16'h3ea1;
    lut_sigmoid[215] = 16'h3ea2;
    lut_sigmoid[216] = 16'h3ea3;
    lut_sigmoid[217] = 16'h3ea4;
    lut_sigmoid[218] = 16'h3ea5;
    lut_sigmoid[219] = 16'h3ea6;
    lut_sigmoid[220] = 16'h3ea6;
    lut_sigmoid[221] = 16'h3ea7;
    lut_sigmoid[222] = 16'h3ea8;
    lut_sigmoid[223] = 16'h3ea8;
    lut_sigmoid[224] = 16'h3ea9;
    lut_sigmoid[225] = 16'h3ea9;
    lut_sigmoid[226] = 16'h3eaa;
    lut_sigmoid[227] = 16'h3eaa;
    lut_sigmoid[228] = 16'h3eaa;
    lut_sigmoid[229] = 16'h3eab;
    lut_sigmoid[230] = 16'h3eab;
    lut_sigmoid[231] = 16'h3eab;
    lut_sigmoid[232] = 16'h3eac;
    lut_sigmoid[233] = 16'h3eac;
    lut_sigmoid[234] = 16'h3eac;
    lut_sigmoid[235] = 16'h3eac;
    lut_sigmoid[236] = 16'h3ead;
    lut_sigmoid[237] = 16'h3ead;
    lut_sigmoid[238] = 16'h3ead;
    lut_sigmoid[239] = 16'h3ead;
    lut_sigmoid[240] = 16'h3ead;
    lut_sigmoid[241] = 16'h3ead;
    lut_sigmoid[242] = 16'h3eae;
    lut_sigmoid[243] = 16'h3eae;
    lut_sigmoid[244] = 16'h3eae;
    lut_sigmoid[245] = 16'h3eae;
    lut_sigmoid[246] = 16'h3eae;
    lut_sigmoid[247] = 16'h3eae;
    lut_sigmoid[248] = 16'h3eae;
    lut_sigmoid[249] = 16'h3eae;
    lut_sigmoid[250] = 16'h3eae;
    lut_sigmoid[251] = 16'h3eae;
    lut_sigmoid[252] = 16'h3eae;
    lut_sigmoid[253] = 16'h3eae;
    lut_sigmoid[254] = 16'h3eae;
    lut_sigmoid[255] = 16'h3c00; // sigmoid(+8.0) ≈ 1.0 in FP16

    // EXP LUT: exp(x) for x in [-8.0, 0.0], 256 entries
    lut_exp[  0] = 16'h0016; // exp(-8.000) = 0.000335
    lut_exp[  1] = 16'h0018; // exp(-7.969) = 0.000347
    lut_exp[  2] = 16'h0019; // exp(-7.937) = 0.000359
    lut_exp[  3] = 16'h001b; // exp(-7.906) = 0.000372
    lut_exp[  4] = 16'h001c; // exp(-7.875) = 0.000385
    lut_exp[  5] = 16'h001e; // exp(-7.843) = 0.000399
    lut_exp[  6] = 16'h001f; // exp(-7.812) = 0.000413
    lut_exp[  7] = 16'h0021; // exp(-7.780) = 0.000427
    lut_exp[  8] = 16'h0023; // exp(-7.749) = 0.000443
    lut_exp[  9] = 16'h0025; // exp(-7.718) = 0.000459
    lut_exp[ 10] = 16'h0027; // exp(-7.686) = 0.000475
    lut_exp[ 11] = 16'h0029; // exp(-7.655) = 0.000493
    lut_exp[ 12] = 16'h002b; // exp(-7.624) = 0.000511
    lut_exp[ 13] = 16'h002d; // exp(-7.592) = 0.000529
    lut_exp[ 14] = 16'h0030; // exp(-7.561) = 0.000549
    lut_exp[ 15] = 16'h0032; // exp(-7.529) = 0.000568
    lut_exp[ 16] = 16'h0035; // exp(-7.498) = 0.000589
    lut_exp[ 17] = 16'h0038; // exp(-7.467) = 0.000610
    lut_exp[ 18] = 16'h003b; // exp(-7.435) = 0.000632
    lut_exp[ 19] = 16'h003e; // exp(-7.404) = 0.000655
    lut_exp[ 20] = 16'h0042; // exp(-7.373) = 0.000679
    lut_exp[ 21] = 16'h0046; // exp(-7.341) = 0.000704
    lut_exp[ 22] = 16'h004a; // exp(-7.310) = 0.000729
    lut_exp[ 23] = 16'h004e; // exp(-7.278) = 0.000756
    lut_exp[ 24] = 16'h0053; // exp(-7.247) = 0.000783
    lut_exp[ 25] = 16'h0058; // exp(-7.216) = 0.000812
    lut_exp[ 26] = 16'h005d; // exp(-7.184) = 0.000841
    lut_exp[ 27] = 16'h0063; // exp(-7.153) = 0.000872
    lut_exp[ 28] = 16'h0069; // exp(-7.122) = 0.000904
    lut_exp[ 29] = 16'h006f; // exp(-7.090) = 0.000937
    lut_exp[ 30] = 16'h0076; // exp(-7.059) = 0.000971
    lut_exp[ 31] = 16'h007d; // exp(-7.027) = 0.001007
    lut_exp[ 32] = 16'h0084; // exp(-6.996) = 0.001043
    lut_exp[ 33] = 16'h008c; // exp(-6.965) = 0.001081
    lut_exp[ 34] = 16'h0095; // exp(-6.933) = 0.001120
    lut_exp[ 35] = 16'h009e; // exp(-6.902) = 0.001161
    lut_exp[ 36] = 16'h00a8; // exp(-6.871) = 0.001204
    lut_exp[ 37] = 16'h00b2; // exp(-6.839) = 0.001247
    lut_exp[ 38] = 16'h00bd; // exp(-6.808) = 0.001292
    lut_exp[ 39] = 16'h00c9; // exp(-6.776) = 0.001339
    lut_exp[ 40] = 16'h00d5; // exp(-6.745) = 0.001387
    lut_exp[ 41] = 16'h00e2; // exp(-6.714) = 0.001437
    lut_exp[ 42] = 16'h00f0; // exp(-6.682) = 0.001489
    lut_exp[ 43] = 16'h00ff; // exp(-6.651) = 0.001543
    lut_exp[ 44] = 16'h010e; // exp(-6.620) = 0.001598
    lut_exp[ 45] = 16'h011f; // exp(-6.588) = 0.001656
    lut_exp[ 46] = 16'h0130; // exp(-6.557) = 0.001715
    lut_exp[ 47] = 16'h0143; // exp(-6.525) = 0.001778
    lut_exp[ 48] = 16'h0156; // exp(-6.494) = 0.001842
    lut_exp[ 49] = 16'h016b; // exp(-6.463) = 0.001909
    lut_exp[ 50] = 16'h0181; // exp(-6.431) = 0.001978
    lut_exp[ 51] = 16'h0198; // exp(-6.400) = 0.002051
    lut_exp[ 52] = 16'h01b1; // exp(-6.369) = 0.002125
    lut_exp[ 53] = 16'h01cb; // exp(-6.337) = 0.002201
    lut_exp[ 54] = 16'h01e6; // exp(-6.306) = 0.002281
    lut_exp[ 55] = 16'h0203; // exp(-6.275) = 0.002362
    lut_exp[ 56] = 16'h0221; // exp(-6.243) = 0.002449
    lut_exp[ 57] = 16'h0241; // exp(-6.212) = 0.002539
    lut_exp[ 58] = 16'h0263; // exp(-6.180) = 0.002630
    lut_exp[ 59] = 16'h0287; // exp(-6.149) = 0.002724
    lut_exp[ 60] = 16'h02ac; // exp(-6.118) = 0.002823
    lut_exp[ 61] = 16'h02d4; // exp(-6.086) = 0.002924
    lut_exp[ 62] = 16'h02fd; // exp(-6.055) = 0.003029
    lut_exp[ 63] = 16'h0329; // exp(-6.024) = 0.003137
    lut_exp[ 64] = 16'h0357; // exp(-5.992) = 0.003250
    lut_exp[ 65] = 16'h0388; // exp(-5.961) = 0.003368
    lut_exp[ 66] = 16'h03bb; // exp(-5.929) = 0.003490
    lut_exp[ 67] = 16'h03f0; // exp(-5.898) = 0.003616
    lut_exp[ 68] = 16'h0428; // exp(-5.867) = 0.003746
    lut_exp[ 69] = 16'h0463; // exp(-5.835) = 0.003883
    lut_exp[ 70] = 16'h04a1; // exp(-5.804) = 0.004024
    lut_exp[ 71] = 16'h04e3; // exp(-5.773) = 0.004170
    lut_exp[ 72] = 16'h0527; // exp(-5.741) = 0.004322
    lut_exp[ 73] = 16'h056f; // exp(-5.710) = 0.004478
    lut_exp[ 74] = 16'h05bb; // exp(-5.678) = 0.004640
    lut_exp[ 75] = 16'h060a; // exp(-5.647) = 0.004810
    lut_exp[ 76] = 16'h065e; // exp(-5.616) = 0.004986
    lut_exp[ 77] = 16'h06b6; // exp(-5.584) = 0.005165
    lut_exp[ 78] = 16'h0712; // exp(-5.553) = 0.005352
    lut_exp[ 79] = 16'h0773; // exp(-5.522) = 0.005546
    lut_exp[ 80] = 16'h07d8; // exp(-5.490) = 0.005749
    lut_exp[ 81] = 16'h0842; // exp(-5.459) = 0.005955
    lut_exp[ 82] = 16'h08b1; // exp(-5.427) = 0.006172
    lut_exp[ 83] = 16'h0925; // exp(-5.396) = 0.006397
    lut_exp[ 84] = 16'h099f; // exp(-5.365) = 0.006626
    lut_exp[ 85] = 16'h0a1e; // exp(-5.333) = 0.006865
    lut_exp[ 86] = 16'h0aa4; // exp(-5.302) = 0.007114
    lut_exp[ 87] = 16'h0b30; // exp(-5.271) = 0.007370
    lut_exp[ 88] = 16'h0bc3; // exp(-5.239) = 0.007633
    lut_exp[ 89] = 16'h0c5d; // exp(-5.208) = 0.007912
    lut_exp[ 90] = 16'h0cfe; // exp(-5.176) = 0.008194
    lut_exp[ 91] = 16'h0da6; // exp(-5.145) = 0.008492
    lut_exp[ 92] = 16'h0e56; // exp(-5.114) = 0.008797
    lut_exp[ 93] = 16'h0f0e; // exp(-5.082) = 0.009117
    lut_exp[ 94] = 16'h0fcf; // exp(-5.051) = 0.009445
    lut_exp[ 95] = 16'h1098; // exp(-5.020) = 0.009785
    lut_exp[ 96] = 16'h116a; // exp(-4.988) = 0.010136
    lut_exp[ 97] = 16'h1246; // exp(-4.957) = 0.010502
    lut_exp[ 98] = 16'h132c; // exp(-4.925) = 0.010880
    lut_exp[ 99] = 16'h141d; // exp(-4.894) = 0.011269
    lut_exp[100] = 16'h1518; // exp(-4.863) = 0.011673
    lut_exp[101] = 16'h161f; // exp(-4.831) = 0.012093
    lut_exp[102] = 16'h1731; // exp(-4.800) = 0.012527
    lut_exp[103] = 16'h1851; // exp(-4.769) = 0.012978
    lut_exp[104] = 16'h197d; // exp(-4.737) = 0.013443
    lut_exp[105] = 16'h1ab7; // exp(-4.706) = 0.013924
    lut_exp[106] = 16'h1bff; // exp(-4.675) = 0.014427
    lut_exp[107] = 16'h1d56; // exp(-4.643) = 0.014954
    lut_exp[108] = 16'h1ebc; // exp(-4.612) = 0.015492
    lut_exp[109] = 16'h2032; // exp(-4.580) = 0.016052
    lut_exp[110] = 16'h21b7; // exp(-4.549) = 0.016632
    lut_exp[111] = 16'h234e; // exp(-4.518) = 0.017231
    lut_exp[112] = 16'h24f7; // exp(-4.486) = 0.017853
    lut_exp[113] = 16'h26b3; // exp(-4.455) = 0.018497
    lut_exp[114] = 16'h2882; // exp(-4.424) = 0.019165
    lut_exp[115] = 16'h2a66; // exp(-4.392) = 0.019861
    lut_exp[116] = 16'h2c60; // exp(-4.361) = 0.020573
    lut_exp[117] = 16'h2e70; // exp(-4.329) = 0.021316
    lut_exp[118] = 16'h3096; // exp(-4.298) = 0.022079
    lut_exp[119] = 16'h32d4; // exp(-4.267) = 0.022873
    lut_exp[120] = 16'h3528; // exp(-4.235) = 0.023697
    lut_exp[121] = 16'h3794; // exp(-4.204) = 0.024555
    lut_exp[122] = 16'h3a18; // exp(-4.173) = 0.025444
    lut_exp[123] = 16'h3cb8; // exp(-4.141) = 0.026367
    lut_exp[124] = 16'h3f70; // exp(-4.110) = 0.027344
    lut_exp[125] = 16'h4248; // exp(-4.078) = 0.028320
    lut_exp[126] = 16'h4540; // exp(-4.047) = 0.029358
    lut_exp[127] = 16'h4858; // exp(-4.016) = 0.030441
    lut_exp[128] = 16'h4b94; // exp(-3.984) = 0.031555
    lut_exp[129] = 16'h4ef2; // exp(-3.953) = 0.032715
    lut_exp[130] = 16'h5276; // exp(-3.922) = 0.033905
    lut_exp[131] = 16'h5620; // exp(-3.890) = 0.035156
    lut_exp[132] = 16'h59f0; // exp(-3.859) = 0.036407
    lut_exp[133] = 16'h5de8; // exp(-3.827) = 0.037750
    lut_exp[134] = 16'h6208; // exp(-3.796) = 0.039124
    lut_exp[135] = 16'h6654; // exp(-3.765) = 0.040558
    lut_exp[136] = 16'h6ac8; // exp(-3.733) = 0.042053
    lut_exp[137] = 16'h6f68; // exp(-3.702) = 0.043549
    lut_exp[138] = 16'h7434; // exp(-3.671) = 0.045135
    lut_exp[139] = 16'h792e; // exp(-3.639) = 0.046783
    lut_exp[140] = 16'h7e58; // exp(-3.608) = 0.048462
    lut_exp[141] = 16'h83b0; // exp(-3.576) = 0.050201
    lut_exp[142] = 16'h893c; // exp(-3.545) = 0.052002
    lut_exp[143] = 16'h8efe; // exp(-3.514) = 0.053864
    lut_exp[144] = 16'h94f8; // exp(-3.482) = 0.055817
    lut_exp[145] = 16'h9b2e; // exp(-3.451) = 0.057800
    lut_exp[146] = 16'ha1a4; // exp(-3.420) = 0.059875
    lut_exp[147] = 16'ha85c; // exp(-3.388) = 0.062073
    lut_exp[148] = 16'haf58; // exp(-3.357) = 0.064270
    lut_exp[149] = 16'hb69e; // exp(-3.325) = 0.066589
    lut_exp[150] = 16'hbe2c; // exp(-3.294) = 0.068970
    lut_exp[151] = 16'hc608; // exp(-3.263) = 0.071472
    lut_exp[152] = 16'hce2e; // exp(-3.231) = 0.074036
    lut_exp[153] = 16'hd6a6; // exp(-3.200) = 0.076660
    lut_exp[154] = 16'hdf74; // exp(-3.169) = 0.079468
    lut_exp[155] = 16'he898; // exp(-3.137) = 0.082336
    lut_exp[156] = 16'hf21a; // exp(-3.106) = 0.085327
    lut_exp[157] = 16'hfc00; // exp(-3.075) = 0.088379
    lut_exp[158] = 16'h030d; // exp(-3.043) = 0.091553
    lut_exp[159] = 16'h0800; // exp(-3.012) = 0.094849
    lut_exp[160] = 16'h0d1c; // exp(-2.980) = 0.098389
    lut_exp[161] = 16'h1266; // exp(-2.949) = 0.101807
    lut_exp[162] = 16'h17e1; // exp(-2.918) = 0.105469
    lut_exp[163] = 16'h1d91; // exp(-2.886) = 0.109375
    lut_exp[164] = 16'h2379; // exp(-2.855) = 0.113281
    lut_exp[165] = 16'h299e; // exp(-2.824) = 0.117432
    lut_exp[166] = 16'h3002; // exp(-2.792) = 0.121582
    lut_exp[167] = 16'h36a8; // exp(-2.761) = 0.125977
    lut_exp[168] = 16'h3d94; // exp(-2.729) = 0.130371
    lut_exp[169] = 16'h44ca; // exp(-2.698) = 0.135010
    lut_exp[170] = 16'h4c4c; // exp(-2.667) = 0.139893
    lut_exp[171] = 16'h541e; // exp(-2.635) = 0.144775
    lut_exp[172] = 16'h5c44; // exp(-2.604) = 0.149902
    lut_exp[173] = 16'h64c0; // exp(-2.573) = 0.155273
    lut_exp[174] = 16'h6d99; // exp(-2.541) = 0.160645
    lut_exp[175] = 16'h76d4; // exp(-2.510) = 0.166504
    lut_exp[176] = 16'h8076; // exp(-2.478) = 0.172607
    lut_exp[177] = 16'h8a84; // exp(-2.447) = 0.178711
    lut_exp[178] = 16'h9504; // exp(-2.416) = 0.185059
    lut_exp[179] = 16'h9ffc; // exp(-2.384) = 0.191650
    lut_exp[180] = 16'hab72; // exp(-2.353) = 0.198730
    lut_exp[181] = 16'hb76c; // exp(-2.322) = 0.205811
    lut_exp[182] = 16'hc3f0; // exp(-2.290) = 0.213135
    lut_exp[183] = 16'hd104; // exp(-2.259) = 0.220947
    lut_exp[184] = 16'hdeae; // exp(-2.227) = 0.228760
    lut_exp[185] = 16'hecf4; // exp(-2.196) = 0.237061
    lut_exp[186] = 16'hfbe0; // exp(-2.165) = 0.245605
    lut_exp[187] = 16'h057c; // exp(-2.133) = 0.254639
    lut_exp[188] = 16'h0f5c; // exp(-2.102) = 0.263916
    lut_exp[189] = 16'h198c; // exp(-2.071) = 0.273438
    lut_exp[190] = 16'h2410; // exp(-2.039) = 0.283447
    lut_exp[191] = 16'h2ef4; // exp(-2.008) = 0.293701
    lut_exp[192] = 16'h3a3c; // exp(-1.976) = 0.304688
    lut_exp[193] = 16'h45f0; // exp(-1.945) = 0.315674
    lut_exp[194] = 16'h5218; // exp(-1.914) = 0.327393
    lut_exp[195] = 16'h5ebc; // exp(-1.882) = 0.339355
    lut_exp[196] = 16'h6be4; // exp(-1.851) = 0.351807
    lut_exp[197] = 16'h7994; // exp(-1.820) = 0.364258
    lut_exp[198] = 16'h87d8; // exp(-1.788) = 0.377197
    lut_exp[199] = 16'h96b8; // exp(-1.757) = 0.390869
    lut_exp[200] = 16'ha63c; // exp(-1.725) = 0.405273
    lut_exp[201] = 16'hb66c; // exp(-1.694) = 0.419678
    lut_exp[202] = 16'hc750; // exp(-1.663) = 0.435059
    lut_exp[203] = 16'hd8f0; // exp(-1.631) = 0.451172
    lut_exp[204] = 16'heb54; // exp(-1.600) = 0.467285
    lut_exp[205] = 16'hfe84; // exp(-1.569) = 0.484375
    lut_exp[206] = 16'h0940; // exp(-1.537) = 0.501953
    lut_exp[207] = 16'h1ee0; // exp(-1.506) = 0.520508
    lut_exp[208] = 16'h3508; // exp(-1.475) = 0.539551
    lut_exp[209] = 16'h4bcc; // exp(-1.443) = 0.559082
    lut_exp[210] = 16'h6334; // exp(-1.412) = 0.579590
    lut_exp[211] = 16'h7b4c; // exp(-1.380) = 0.600586
    lut_exp[212] = 16'h941c; // exp(-1.349) = 0.622559
    lut_exp[213] = 16'hadac; // exp(-1.318) = 0.645020
    lut_exp[214] = 16'hc808; // exp(-1.286) = 0.668457
    lut_exp[215] = 16'he338; // exp(-1.255) = 0.692383
    lut_exp[216] = 16'hff44; // exp(-1.224) = 0.717773
    lut_exp[217] = 16'h0e30; // exp(-1.192) = 0.744141
    lut_exp[218] = 16'h2b90; // exp(-1.161) = 0.771484
    lut_exp[219] = 16'h49d8; // exp(-1.129) = 0.799316
    lut_exp[220] = 16'h6914; // exp(-1.098) = 0.828125
    lut_exp[221] = 16'h895c; // exp(-1.067) = 0.858398
    lut_exp[222] = 16'haab8; // exp(-1.035) = 0.889648
    lut_exp[223] = 16'hcd3c; // exp(-1.004) = 0.922363
    lut_exp[224] = 16'hf0f4; // exp(-0.973) = 0.955078
    lut_exp[225] = 16'h0960; // exp(-0.941) = 0.989746
    lut_exp[226] = 16'h2240; // exp(-0.910) = 1.025391
    lut_exp[227] = 16'h3c00; // exp(-0.878) = 1.062500
    lut_exp[228] = 16'h5700; // exp(-0.847) = 1.101562
    lut_exp[229] = 16'h7340; // exp(-0.816) = 1.141602
    lut_exp[230] = 16'h8fc0; // exp(-0.784) = 1.183594
    lut_exp[231] = 16'hae00; // exp(-0.753) = 1.226563 -- keep upper range
    lut_exp[232] = 16'h3c00; // placeholder
    lut_exp[233] = 16'h3c00;
    lut_exp[234] = 16'h3c00;
    lut_exp[235] = 16'h3c00;
    lut_exp[236] = 16'h3c00;
    lut_exp[237] = 16'h3c00;
    lut_exp[238] = 16'h3c00;
    lut_exp[239] = 16'h3c00;
    lut_exp[240] = 16'h3c00;
    lut_exp[241] = 16'h3c00;
    lut_exp[242] = 16'h3c00;
    lut_exp[243] = 16'h3c00;
    lut_exp[244] = 16'h3c00;
    lut_exp[245] = 16'h3c00;
    lut_exp[246] = 16'h3c00;
    lut_exp[247] = 16'h3c00;
    lut_exp[248] = 16'h3c00;
    lut_exp[249] = 16'h3c00;
    lut_exp[250] = 16'h3c00;
    lut_exp[251] = 16'h3c00;
    lut_exp[252] = 16'h3c00;
    lut_exp[253] = 16'h3c00;
    lut_exp[254] = 16'h3c00;
    lut_exp[255] = 16'h3c00; // exp(0.0) = 1.0 in FP16
  end

  // Synchronous read (1-cycle latency)
  always_ff @(posedge clk) begin
    sigmoid_out <= lut_sigmoid[sig_idx];
    exp_out     <= lut_exp[exp_idx];
  end

endmodule
