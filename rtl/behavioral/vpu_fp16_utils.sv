// vpu_fp16_utils.sv
// Behavioral FP16 utility package for VPU simulation
// Purely behavioral / simulation-only functions

package vpu_fp16_utils;

  // Convert FP16 bits to real
  function automatic real fp16_to_real(input logic [15:0] fp16_bits);
    logic        sign;
    logic [4:0]  exp;
    logic [9:0]  mant;
    real         result;
    int          exp_int;

    sign = fp16_bits[15];
    exp  = fp16_bits[14:10];
    mant = fp16_bits[9:0];

    if (exp == 5'h1F) begin
      // Inf or NaN
      if (mant == 10'h0)
        result = sign ? -1e30 : 1e30;  // treat as large finite for sim
      else
        result = 0.0;  // NaN -> 0
    end else if (exp == 5'h00) begin
      // Subnormal or zero
      result = $itor(mant) * (2.0 ** (-14)) * (2.0 ** (-10));
      if (sign) result = -result;
    end else begin
      exp_int = $signed({1'b0, exp}) - 15;
      result = (1.0 + $itor(mant) / 1024.0) * (2.0 ** exp_int);
      if (sign) result = -result;
    end

    return result;
  endfunction

  // Convert real to FP16 bits
  function automatic logic [15:0] real_to_fp16(input real val);
    logic        sign;
    real         abs_val;
    int          exp_int;
    real         mant_val;
    logic [9:0]  mant_bits;
    logic [4:0]  exp_bits;
    logic [15:0] result;
    real         TWO_14;
    real         MAX_FP16;

    TWO_14  = 16384.0;    // 2^14
    MAX_FP16 = 65504.0;

    if (val != val) begin  // NaN check
      result = 16'h7FFF;
      return result;
    end

    sign = (val < 0.0) ? 1'b1 : 1'b0;
    abs_val = sign ? -val : val;

    if (abs_val == 0.0) begin
      result = {sign, 15'h0};
      return result;
    end

    if (abs_val > MAX_FP16) begin
      result = {sign, 5'h1E, 10'h3FF};  // max finite FP16
      return result;
    end

    if (abs_val < 5.96e-8) begin  // smaller than smallest normal
      // subnormal
      mant_val = abs_val / (TWO_14 * (2.0**(-10)));
      if (mant_val > 1023.0) mant_val = 1023.0;
      mant_bits = logic'($rtoi(mant_val));
      result = {sign, 5'h00, mant_bits};
      return result;
    end

    // Normal number
    // find exponent: abs_val = 2^exp_int * (1 + f)
    exp_int = 0;
    mant_val = abs_val;
    while (mant_val >= 2.0) begin
      mant_val = mant_val / 2.0;
      exp_int = exp_int + 1;
    end
    while (mant_val < 1.0) begin
      mant_val = mant_val * 2.0;
      exp_int = exp_int - 1;
    end

    // mant_val is now in [1.0, 2.0)
    // fractional part * 1024
    mant_bits = logic'($rtoi((mant_val - 1.0) * 1024.0));
    exp_int = exp_int + 15;  // add bias

    if (exp_int <= 0) begin
      // underflow to zero
      result = {sign, 15'h0};
    end else if (exp_int >= 31) begin
      // overflow to max
      result = {sign, 5'h1E, 10'h3FF};
    end else begin
      exp_bits = logic'(exp_int);
      result = {sign, exp_bits, mant_bits};
    end

    return result;
  endfunction

  // Behavioral FP16 add (returns real)
  function automatic real fp16_add_real(
    input logic [15:0] a,
    input logic [15:0] b
  );
    return fp16_to_real(a) + fp16_to_real(b);
  endfunction

  // Behavioral FP16 mul (returns real)
  function automatic real fp16_mul_real(
    input logic [15:0] a,
    input logic [15:0] b
  );
    return fp16_to_real(a) * fp16_to_real(b);
  endfunction

  // Behavioral reciprocal square root (real)
  function automatic real fp16_rsqrt_real(input real x);
    if (x <= 0.0) return 0.0;
    return 1.0 / $sqrt(x);
  endfunction

  // Sigmoid function (real)
  function automatic real sigmoid_real(input real x);
    return 1.0 / (1.0 + $exp(-x));
  endfunction

  // Exp function (real)
  function automatic real exp_real(input real x);
    return $exp(x);
  endfunction

endpackage
