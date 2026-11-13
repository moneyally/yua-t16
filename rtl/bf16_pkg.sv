// bf16_pkg.sv — BF16/FP32 산술의 단일 출처 (SSOT)
//
// 배경: bf16_mul 과 fp32_add 가 mxu_bf16_16x16.sv 안에만 있었다.
// mxu_bf16_128x128.sv 는 이 함수들에 접근할 수 없어서 누적을 직접 구현했고,
// FP32 비트패턴을 정수로 더하는 버그가 생겼다 (K>1 에서 결과가 틀림).
// 산술은 여기 한 곳에만 둔다.
`ifndef BF16_PKG_SV
`define BF16_PKG_SV

package bf16_pkg;

  // ─────────────────────────────────────────────────────────
  // BF16 × BF16 → FP32 (조합 논리)
  //   BF16: {s[15], e[14:7], m[6:0]}, 지수 바이어스 127
  // ─────────────────────────────────────────────────────────
  function automatic logic [31:0] bf16_mul(input logic [15:0] a, input logic [15:0] b);
    logic        a_sign, b_sign, r_sign;
    logic [7:0]  a_exp, b_exp, r_exp;
    logic [6:0]  a_man, b_man;
    logic [8:0]  r_exp_raw;
    logic [15:0] man_prod;
    logic [22:0] r_man;
    begin
      a_sign = a[15]; b_sign = b[15];
      a_exp  = a[14:7]; b_exp = b[14:7];
      a_man  = a[6:0];  b_man = b[6:0];
      r_sign = a_sign ^ b_sign;

      if (a_exp == 8'd0 || b_exp == 8'd0) begin
        bf16_mul = 32'd0;                                   // 0 입력 → 0
      end else if (a_exp == 8'hFF || b_exp == 8'hFF) begin
        bf16_mul = {r_sign, 8'hFE, 23'h7FFFFF};             // Inf/NaN → 최대 유한값으로 클램프
      end else begin
        man_prod  = {1'b1, a_man} * {1'b1, b_man};
        r_exp_raw = {1'b0, a_exp} + {1'b0, b_exp} - 9'd127;

        if (man_prod[15]) begin
          r_man     = {man_prod[14:0], 8'd0};
          r_exp_raw = r_exp_raw + 9'd1;
        end else begin
          r_man     = {man_prod[13:0], 9'd0};
        end

        if (r_exp_raw[8] || r_exp_raw == 9'd0) begin
          bf16_mul = 32'd0;                                 // 언더플로
        end else if (r_exp_raw >= 9'd255) begin
          bf16_mul = {r_sign, 8'hFE, 23'h7FFFFF};           // 오버플로 클램프
        end else begin
          r_exp    = r_exp_raw[7:0];
          bf16_mul = {r_sign, r_exp, r_man};
        end
      end
    end
  endfunction

  // ─────────────────────────────────────────────────────────
  // FP32 + FP32 → FP32 (누산용)
  //
  // 원본 대비 수정 2건:
  //  1) sum==0 분기가 결과를 낸 뒤에도 아래 정규화 코드로 떨어져
  //     {r_s, r_e, r_m} 로 덮어쓰이던 fall-through 버그를 막았다.
  //  2) 정규화 while 루프를 26비트 폭에 맞춰 25회로 상한을 둔 for 루프로
  //     바꿨다. 동작은 동일하고 합성 가능해진다 (원본은 r_e 가 최대 255라
  //     합성 도구가 255회까지 펼쳐야 했다).
  // ─────────────────────────────────────────────────────────
  function automatic logic [31:0] fp32_add(input logic [31:0] a, input logic [31:0] b);
    logic        a_s, b_s, r_s;
    logic [7:0]  a_e, b_e, r_e, exp_diff;
    logic [22:0] a_m, b_m, r_m;
    logic [24:0] a_full, b_full;
    logic [25:0] sum;
    logic        done;
    begin
      done     = 1'b0;
      fp32_add = 32'd0;
      r_s = 1'b0; r_e = 8'd0; r_m = 23'd0; sum = 26'd0;

      a_s = a[31]; a_e = a[30:23]; a_m = a[22:0];
      b_s = b[31]; b_e = b[30:23]; b_m = b[22:0];

      if (a_e == 8'd0 && a_m == 23'd0) begin
        fp32_add = b; done = 1'b1;
      end else if (b_e == 8'd0 && b_m == 23'd0) begin
        fp32_add = a; done = 1'b1;
      end

      if (!done) begin
        a_full = {1'b1, a_m, 1'b0};
        b_full = {1'b1, b_m, 1'b0};

        if (a_e >= b_e) begin
          exp_diff = a_e - b_e;
          b_full   = (exp_diff > 8'd25) ? 25'd0 : (b_full >> exp_diff);
          r_e      = a_e;
          if (a_s == b_s) begin
            sum = {1'b0, a_full} + {1'b0, b_full};
            r_s = a_s;
          end else if (a_full >= b_full) begin
            sum = {1'b0, a_full} - {1'b0, b_full};
            r_s = a_s;
          end else begin
            sum = {1'b0, b_full} - {1'b0, a_full};
            r_s = b_s;
          end
        end else begin
          exp_diff = b_e - a_e;
          a_full   = (exp_diff > 8'd25) ? 25'd0 : (a_full >> exp_diff);
          r_e      = b_e;
          if (a_s == b_s) begin
            sum = {1'b0, b_full} + {1'b0, a_full};
            r_s = a_s;
          end else if (b_full >= a_full) begin
            sum = {1'b0, b_full} - {1'b0, a_full};
            r_s = b_s;
          end else begin
            sum = {1'b0, a_full} - {1'b0, b_full};
            r_s = a_s;
          end
        end

        if (sum == 26'd0) begin
          fp32_add = 32'd0;                      // ← 수정 1: 여기서 끝낸다
        end else begin
          if (sum[25]) begin
            sum = sum >> 1;
            r_e = r_e + 8'd1;
          end else begin
            // ← 수정 2: 상한 25회 정규화 (sum 은 26비트)
            for (int unsigned s = 0; s < 25; s++) begin
              if (!sum[24] && r_e > 8'd0) begin
                sum = sum << 1;
                r_e = r_e - 8'd1;
              end
            end
          end
          r_m = sum[23:1];
          fp32_add = (r_e >= 8'hFF) ? {r_s, 8'hFE, 23'h7FFFFF} : {r_s, r_e, r_m};
        end
      end
    end
  endfunction

  // ═════════════════════════════════════════════════════════════════
  // 고정소수점 누산 (PE 재설계)
  // ═════════════════════════════════════════════════════════════════
  //
  // 왜 바꾸나
  //   fp32_add 를 PE 마다 두면 PE 하나가 합성 후 4,369 셀이다.
  //   16×16 타일 = 1,118,464 셀, PAR=8 (64 타일) = 71,618,048 셀.
  //   정렬 시프터 + 정규화 + 반올림을 매 사이클 PE 마다 도는 구조라
  //   면적도 임계 경로도 감당이 안 된다.
  //
  // 어떻게 바꾸나
  //   곱을 고정소수점으로 정렬해 더하기만 한다 (시프트 + 덧셈).
  //   정규화와 반올림은 누산 루프에서 빠지고 출력에서 한 번만 한다.
  //   누산이 정수 영역이라 K 에 대해 "정확"하다 — 중간 반올림이 없다.
  //
  // 설계점
  //   ACC_W = 48, ACC_F = 16
  //     최대 크기  2^(48-1-16) = 2^31 ≈ 2.1e9
  //     분해능     2^-16 ≈ 1.5e-5
  //   BF16 입력은 가수가 8비트뿐이라 이 분해능이면 입력 정밀도를 넘는다.
  //   범위를 벗어나면 포화시킨다 (조용한 랩어라운드 금지).
  //
  // 전체 FP32 범위가 필요하면 ACC_W 를 키워야 한다. BF16 지수 전 범위를
  // 손실 없이 담으려면 약 520비트가 필요하고, 그건 PE 마다 둘 수 없다.

  localparam int ACC_W = 48;   // 누산기 전체 폭 (2의 보수)
  localparam int ACC_F = 16;   // 소수부 비트 수

  localparam logic signed [ACC_W-1:0] FXP_MAX =  {1'b0, {(ACC_W-1){1'b1}}};
  localparam logic signed [ACC_W-1:0] FXP_MIN =  {1'b1, {(ACC_W-1){1'b0}}};

  // BF16 × BF16 → 고정소수점 증분.
  //
  //   a = (-1)^sa × Ma × 2^(ea-134),  Ma = {1'b1, a[6:0]}  (8비트)
  //   a×b = ±(Ma·Mb) × 2^(ea+eb-268)
  //   고정소수점(소수부 F) = ±P × 2^(ea+eb-268+F),  P = Ma·Mb (16비트)
  function automatic logic signed [ACC_W-1:0] bf16_mul_fxp(
      input logic [15:0] a, input logic [15:0] b);
    logic        sa, sb, sgn;
    logic [7:0]  ea, eb;
    logic [15:0] prod;
    logic signed [9:0] sh;          // ea+eb 최대 508, -268+F 를 더해도 10비트로 충분
    logic [ACC_W-1:0] mag, wide;
    begin
      sa = a[15]; ea = a[14:7];
      sb = b[15]; eb = b[14:7];
      sgn = sa ^ sb;

      if (ea == 8'd0 || eb == 8'd0) begin
        bf16_mul_fxp = '0;                                  // 0 / 비정규수 → 0
      end else if (ea == 8'hFF || eb == 8'hFF) begin
        bf16_mul_fxp = sgn ? FXP_MIN : FXP_MAX;             // Inf/NaN → 포화
      end else begin
        prod = {1'b1, a[6:0]} * {1'b1, b[6:0]};
        sh   = $signed({2'b0, ea}) + $signed({2'b0, eb})
             - $signed(10'd268) + $signed(10'(ACC_F));

        wide = {{(ACC_W-16){1'b0}}, prod};

        if (sh >= $signed(10'(ACC_W - 16))) begin
          mag = FXP_MAX;                                    // 상위 포화
        end else if (sh >= 0) begin
          mag = wide << sh;
        end else if (sh <= -$signed(10'd16)) begin
          mag = '0;                                         // 전부 밀려나감
        end else begin
          mag = wide >> (-sh);
        end

        bf16_mul_fxp = sgn ? -$signed(mag) : $signed(mag);
      end
    end
  endfunction

  // 포화 덧셈. 조용히 랩어라운드하면 틀린 값이 그대로 흘러간다.
  function automatic logic signed [ACC_W-1:0] fxp_add_sat(
      input logic signed [ACC_W-1:0] x, input logic signed [ACC_W-1:0] y);
    logic signed [ACC_W-1:0] s;
    begin
      s = x + y;
      // 같은 부호끼리 더했는데 부호가 바뀌면 넘친 것이다
      if ((x[ACC_W-1] == y[ACC_W-1]) && (s[ACC_W-1] != x[ACC_W-1]))
        fxp_add_sat = x[ACC_W-1] ? FXP_MIN : FXP_MAX;
      else
        fxp_add_sat = s;
    end
  endfunction

  // 고정소수점 → FP32. 누산 루프 밖, 출력에서 한 번만 돈다.
  function automatic logic [31:0] fxp_to_fp32(input logic signed [ACC_W-1:0] v);
    logic              sgn;
    logic [ACC_W-1:0]  mag;
    int                msb;
    logic signed [9:0] e_unb;
    logic [8:0]        e_fp;
    logic [22:0]       mant;
    begin
      if (v == '0) begin
        fxp_to_fp32 = 32'd0;
      end else begin
        sgn = v[ACC_W-1];
        mag = sgn ? (-v) : v;       // FXP_MIN 은 -v 가 자기 자신이지만
                                    // 아래 msb 탐색이 최상위 비트를 잡아 처리된다
        // 최상위 1 의 위치를 이진 탐색으로 찾는다 (48비트를 6단계).
        // 선형 48회 루프는 합성 시 48단 우선순위 체인이 되고,
        // 시뮬레이터 엘라보레이션 비용도 PE 수에 비례해 폭증한다.
        // (주석을 'Verilator' 로 시작하면 그 툴이 프라그마로 읽는다.)
        begin
          logic [ACC_W-1:0] x;
          x = mag; msb = 0;
          if (|x[ACC_W-1:32]) begin msb = msb + 32; x = x >> 32; end
          if (|x[31:16])      begin msb = msb + 16; x = x >> 16; end
          if (|x[15:8])       begin msb = msb + 8;  x = x >> 8;  end
          if (|x[7:4])        begin msb = msb + 4;  x = x >> 4;  end
          if (|x[3:2])        begin msb = msb + 2;  x = x >> 2;  end
          if (x[1])                 msb = msb + 1;
        end

        e_unb = $signed(10'(msb)) - $signed(10'(ACC_F));
        e_fp  = 9'(e_unb + $signed(10'd127));

        // msb 아래 23비트가 가수
        if (msb >= 23)
          mant = 23'(mag >> (msb - 23));
        else
          mant = 23'(mag << (23 - msb));
        // 최상위 1 은 암묵 비트이므로 가수에서 빠진다

        if (e_fp >= 9'd255)
          fxp_to_fp32 = {sgn, 8'hFE, 23'h7FFFFF};   // 상위 포화
        else if ($signed(e_fp) <= $signed(9'd0))
          fxp_to_fp32 = 32'd0;                      // 언더플로
        else
          fxp_to_fp32 = {sgn, e_fp[7:0], mant};
      end
    end
  endfunction

endpackage

`endif
