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

endpackage

`endif
