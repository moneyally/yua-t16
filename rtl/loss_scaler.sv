// loss_scaler.sv — ORBIT-G3 Dynamic Loss Scaling Controller
// SSOT: ORBIT_G3_RTL_ISSUES.md (G3-RTL-012)
//
// Mixed precision training safety: controls loss scale factor.
// No tensor datapath — this is a scalar control primitive.
//
// Policy:
//   overflow_detect=1 → scale = max(scale * backoff_factor, min_scale)
//   overflow_detect=0, success_count==growth_interval
//                     → scale = min(scale * growth_factor, max_scale)
//   overflow_detect=0, success_count<growth_interval
//                     → scale unchanged, success_count++
`timescale 1ns/1ps
`default_nettype none

module loss_scaler (
  input  logic        clk,
  input  logic        rst_n,

  // Control
  input  logic        step_done,       // pulse: one training step completed
  input  logic        overflow_detect, // 1 = gradient overflow detected this step

  // Configuration (FP32 bit patterns)
  input  logic [31:0] init_scale_fp32,     // default: 32768.0 (2^15)
  input  logic [31:0] growth_factor_fp32,  // default: 2.0
  input  logic [31:0] backoff_factor_fp32, // default: 0.5
  input  logic [15:0] growth_interval,     // steps of no-overflow before scale-up
  input  logic [31:0] min_scale_fp32,      // default: 1.0
  input  logic [31:0] max_scale_fp32,      // default: 2^24

  // Output
  output logic [31:0] current_scale_fp32,  // current loss scale (FP32 bits)
  output logic        scale_valid,         // scale has been updated
  output logic        busy,
  output logic        done_pulse,
  output logic [7:0]  err_code             // 0=ok, 1=invalid config
);

  // ═══════════════════════════════════════════════════════════
  // FP32 ↔ real helpers (behavioral, correctness-first)
  // ═══════════════════════════════════════════════════════════
  function automatic real fp32_to_real(input logic [31:0] bits);
    logic [7:0] e; logic [22:0] m; real r; integer ev;
    e = bits[30:23]; m = bits[22:0];
    if (e==0 && m==0) begin fp32_to_real = 0.0; end
    else if (e==8'hFF) begin fp32_to_real = bits[31] ? -3.4e38 : 3.4e38; end
    else begin
      ev = e - 127; r = 1.0 + $itor(m)/8388608.0;
      if (ev>=0) r = r * (2.0**ev); else r = r / (2.0**(-ev));
      if (bits[31]) r = -r; fp32_to_real = r;
    end
  endfunction

  function automatic logic [31:0] real_to_fp32(input real v);
    logic s; real a, mf; integer ei, iter, mb_raw;
    logic [22:0] mb; logic [7:0] eb;
    s = (v<0.0); a = s ? -v : v;
    if (a==0.0) begin real_to_fp32 = 32'd0; end
    else if (a>=3.4e38) begin real_to_fp32 = {s,8'hFE,23'h7FFFFF}; end
    else begin
      ei=0; mf=a; iter=0;
      while(mf>=2.0 && iter<200) begin mf=mf/2.0; ei=ei+1; iter=iter+1; end
      while(mf<1.0 && iter<200) begin mf=mf*2.0; ei=ei-1; iter=iter+1; end
      mb_raw=$rtoi((mf-1.0)*8388608.0+0.5);
      if(mb_raw>=8388608) begin mb=23'd0; ei=ei+1; end else mb=mb_raw[22:0];
      ei=ei+127;
      if(ei<=0) real_to_fp32=32'd0;
      else if(ei>=255) real_to_fp32={s,8'hFE,23'h7FFFFF};
      else begin eb=ei[7:0]; real_to_fp32={s,eb,mb}; end
    end
  endfunction

  // ═══════════════════════════════════════════════════════════
  // State
  // ═══════════════════════════════════════════════════════════
  typedef enum logic [2:0] {
    ST_IDLE,
    ST_VALIDATE,
    ST_EVAL,
    ST_DONE,
    ST_ERROR
  } state_t;

  state_t st;

  real scale_r;            // current scale as real
  logic [15:0] success_cnt; // consecutive no-overflow steps
  logic initialized;

  assign busy = (st != ST_IDLE && st != ST_DONE && st != ST_ERROR);
  assign current_scale_fp32 = real_to_fp32(scale_r);

  // ═══════════════════════════════════════════════════════════
  // FSM
  // ═══════════════════════════════════════════════════════════
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st             <= ST_IDLE;
      scale_r        <= 32768.0;  // default init
      success_cnt    <= 16'd0;
      initialized    <= 1'b0;
      scale_valid    <= 1'b0;
      done_pulse     <= 1'b0;
      err_code       <= 8'd0;
    end else begin
      done_pulse  <= 1'b0;
      scale_valid <= 1'b0;

      case (st)
        ST_IDLE: begin
          err_code <= 8'd0;
          if (!initialized) begin
            scale_r     <= fp32_to_real(init_scale_fp32);
            initialized <= 1'b1;
            scale_valid <= 1'b1;
          end
          if (step_done) begin
            st <= ST_VALIDATE;
          end
        end

        ST_VALIDATE: begin
          // Check config validity
          automatic real mn, mx, gf, bf;
          mn = fp32_to_real(min_scale_fp32);
          mx = fp32_to_real(max_scale_fp32);
          gf = fp32_to_real(growth_factor_fp32);
          bf = fp32_to_real(backoff_factor_fp32);

          if (mn > mx || gf <= 1.0 || bf >= 1.0 || bf <= 0.0) begin
            err_code <= 8'd1;
            st       <= ST_ERROR;
          end else begin
            st <= ST_EVAL;
          end
        end

        ST_EVAL: begin
          automatic real new_scale, mn, mx, gf, bf;
          mn = fp32_to_real(min_scale_fp32);
          mx = fp32_to_real(max_scale_fp32);
          gf = fp32_to_real(growth_factor_fp32);
          bf = fp32_to_real(backoff_factor_fp32);

          if (overflow_detect) begin
            // Overflow: scale down immediately, reset success counter
            new_scale = scale_r * bf;
            if (new_scale < mn) new_scale = mn;
            scale_r     <= new_scale;
            success_cnt <= 16'd0;
          end else begin
            // No overflow: increment success counter
            if (success_cnt + 1 >= growth_interval) begin
              // Scale up
              new_scale = scale_r * gf;
              if (new_scale > mx) new_scale = mx;
              scale_r     <= new_scale;
              success_cnt <= 16'd0;
            end else begin
              success_cnt <= success_cnt + 1;
            end
          end

          scale_valid <= 1'b1;
          st          <= ST_DONE;
        end

        ST_DONE: begin
          done_pulse <= 1'b1;
          st         <= ST_IDLE;
        end

        ST_ERROR: begin
          done_pulse <= 1'b1;
          st         <= ST_IDLE;
        end

        default: st <= ST_IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
