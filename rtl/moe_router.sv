// moe_router.sv — Behavioral MoE Router for Icarus Verilog 12
// Follows Icarus 12 rules:
//   - No dynamic bit-selects inside always blocks
//   - Blocking assignments with always @(posedge clk)
//   - Unpack flat arrays via generate/assign outside always
//   - rf16 mantissa overflow check before 10-bit truncation
`timescale 1ns/1ps
`default_nettype none

module moe_router #(
  parameter int NUM_EXPERTS = 8,
  parameter int TOP_K       = 2,
  parameter int D_MODEL     = 16,
  parameter int MAX_TOKENS  = 4
)(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  output logic        busy,
  output logic        done,

  input  logic [7:0]  num_tokens,

  // hidden: [MAX_TOKENS][D_MODEL] FP16, flat packed
  input  logic [MAX_TOKENS*D_MODEL*16-1:0] hidden_flat,

  // router_weight: [D_MODEL][NUM_EXPERTS] FP16, flat packed
  input  logic [D_MODEL*NUM_EXPERTS*16-1:0] weight_flat,

  // indices: [MAX_TOKENS][TOP_K] uint8, flat packed
  output logic [MAX_TOKENS*TOP_K*8-1:0]  indices_flat,
  // scores: [MAX_TOKENS][TOP_K] FP16, flat packed
  output logic [MAX_TOKENS*TOP_K*16-1:0] scores_flat
);

  // ── Unpack hidden and weight via generate/assign outside always ──────────
  wire [15:0] hidden_r  [0:MAX_TOKENS-1][0:D_MODEL-1];
  wire [15:0] weight_r  [0:D_MODEL-1][0:NUM_EXPERTS-1];

  reg  [7:0]  idx_out_r [0:MAX_TOKENS-1][0:TOP_K-1];
  reg  [15:0] scr_out_r [0:MAX_TOKENS-1][0:TOP_K-1];

  genvar gt, gd, ge, gk;

  generate
    for (gt = 0; gt < MAX_TOKENS; gt = gt+1) begin : UNPACK_H_T
      for (gd = 0; gd < D_MODEL; gd = gd+1) begin : UNPACK_H_D
        // hidden_flat layout: token0[d0..dD-1], token1[d0..dD-1], ...
        assign hidden_r[gt][gd] = hidden_flat[(gt*D_MODEL + gd)*16 +: 16];
      end
    end
  endgenerate

  generate
    for (gd = 0; gd < D_MODEL; gd = gd+1) begin : UNPACK_W_D
      for (ge = 0; ge < NUM_EXPERTS; ge = ge+1) begin : UNPACK_W_E
        // weight_flat layout: [d0][e0..eE-1], [d1][e0..eE-1], ...
        assign weight_r[gd][ge] = weight_flat[(gd*NUM_EXPERTS + ge)*16 +: 16];
      end
    end
  endgenerate

  generate
    for (gt = 0; gt < MAX_TOKENS; gt = gt+1) begin : PACK_OUT_T
      for (gk = 0; gk < TOP_K; gk = gk+1) begin : PACK_OUT_K
        assign indices_flat[(gt*TOP_K + gk)*8  +: 8]  = idx_out_r[gt][gk];
        assign scores_flat [(gt*TOP_K + gk)*16 +: 16] = scr_out_r[gt][gk];
      end
    end
  endgenerate

  // ── FP16 helpers (identical to vpu_core.sv) ──────────────────────────────
  function automatic real f16r;
    input [15:0] b;
    reg [4:0] e5; reg [9:0] m; reg s;
    real r; integer ei;
    begin
      s = b[15]; e5 = b[14:10]; m = b[9:0];
      if (e5 == 5'h1F) begin
        r = s ? -65504.0 : 65504.0;
      end else if (e5 == 5'h00) begin
        r = $itor(m) * 5.960464e-8;
        if (s) r = -r;
      end else begin
        ei = e5 - 15;
        r  = 1.0 + $itor(m) / 1024.0;
        if (ei >= 0) r = r * (2.0 ** ei);
        else         r = r / (2.0 ** (-ei));
        if (s) r = -r;
      end
      f16r = r;
    end
  endfunction

  function automatic [15:0] rf16;
    input real v;
    reg s; real a, mf; integer ei, iter;
    reg [9:0] mb; reg [4:0] eb;
    begin
      s = (v < 0.0) ? 1'b1 : 1'b0;
      a = s ? -v : v;
      if (a == 0.0)          begin rf16 = {s, 15'h0};           end
      else if (a >= 65504.0) begin rf16 = {s, 5'h1E, 10'h3FF}; end
      else if (a < 5.96e-8)  begin rf16 = {s, 15'h0};           end
      else begin
        if (a < 6.104e-5) begin
          mb = $rtoi(a / 5.960464e-8 + 0.5);
          if (mb > 1023) mb = 1023;
          rf16 = {s, 5'h00, mb};
        end else begin
          ei = 0; mf = a; iter = 0;
          while (mf >= 2.0 && iter < 64) begin mf = mf/2.0; ei = ei+1; iter = iter+1; end
          while (mf <  1.0 && iter < 64) begin mf = mf*2.0; ei = ei-1; iter = iter+1; end
          begin : RF16_NORM
            integer mb_raw;
            mb_raw = $rtoi((mf - 1.0) * 1024.0 + 0.5);
            if (mb_raw >= 1024) begin mb = 10'h0; ei = ei+1; end
            else                begin mb = mb_raw; end
          end
          ei = ei + 15;
          if (ei <= 0)       rf16 = {s, 15'h0};
          else if (ei >= 31) rf16 = {s, 5'h1E, 10'h3FF};
          else begin eb = ei; rf16 = {s, eb, mb}; end
        end
      end
    end
  endfunction

  function automatic real sigmoid_r;
    input real x;
    begin
      if (x >  16.0) sigmoid_r = 1.0;
      else if (x < -16.0) sigmoid_r = 0.0;
      else sigmoid_r = 1.0 / (1.0 + $exp(-x));
    end
  endfunction

  // ── State machine ────────────────────────────────────────────────────────
  localparam ST_IDLE           = 3'd0;
  localparam ST_COMPUTE_LOGITS = 3'd1;
  localparam ST_SOFTMAX        = 3'd2;
  localparam ST_TOPK           = 3'd3;
  localparam ST_DONE           = 3'd4;

  reg [2:0]  state;
  reg [7:0]  ntok_r;      // actual token count
  reg [7:0]  cur_tok;     // current token being processed
  reg [7:0]  cur_exp;     // current expert being processed

  // Logits and probs storage (real arrays — behavioral)
  real logits [0:MAX_TOKENS-1][0:NUM_EXPERTS-1];
  real probs  [0:MAX_TOKENS-1][0:NUM_EXPERTS-1];

  integer ii, jj;

  // ── Main FSM ─────────────────────────────────────────────────────────────
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state    = ST_IDLE;
      busy     = 1'b0;
      done     = 1'b0;
      ntok_r   = 8'd0;
      cur_tok  = 8'd0;
      cur_exp  = 8'd0;
      for (ii = 0; ii < MAX_TOKENS; ii = ii+1) begin
        for (jj = 0; jj < NUM_EXPERTS; jj = jj+1) begin
          logits[ii][jj] = 0.0;
          probs[ii][jj]  = 0.0;
        end
      end
      for (ii = 0; ii < MAX_TOKENS; ii = ii+1) begin
        for (jj = 0; jj < TOP_K; jj = jj+1) begin
          idx_out_r[ii][jj] = 8'd0;
          scr_out_r[ii][jj] = 16'h0;
        end
      end
    end else begin
      done = 1'b0;  // default

      case (state)

        // ── IDLE: wait for start ────────────────────────────────────────
        ST_IDLE: begin
          if (start) begin
            ntok_r  = num_tokens;
            cur_tok = 8'd0;
            cur_exp = 8'd0;
            busy    = 1'b1;
            state   = ST_COMPUTE_LOGITS;
          end
        end

        // ── COMPUTE_LOGITS: one expert dot-product per clock ────────────
        // For cur_tok, cur_exp: logits[cur_tok][cur_exp] = dot(hidden[cur_tok], weight[:,cur_exp])
        ST_COMPUTE_LOGITS: begin
          begin : LOGIT_CALC
            real acc;
            integer d;
            acc = 0.0;
            for (d = 0; d < D_MODEL; d = d+1) begin
              acc = acc + f16r(hidden_r[cur_tok][d]) * f16r(weight_r[d][cur_exp]);
            end
            logits[cur_tok][cur_exp] = acc;
          end

          // Advance to next expert / token
          if (cur_exp == NUM_EXPERTS - 1) begin
            cur_exp = 8'd0;
            if (cur_tok == ntok_r - 1) begin
              // All logits computed — move to softmax
              cur_tok = 8'd0;
              state   = ST_SOFTMAX;
            end else begin
              cur_tok = cur_tok + 8'd1;
            end
          end else begin
            cur_exp = cur_exp + 8'd1;
          end
        end

        // ── SOFTMAX: compute softmax over experts for each token ─────────
        // Done in one cycle per token (behavioral loop)
        ST_SOFTMAX: begin
          begin : SOFTMAX_CALC
            real mx, sum_exp, shifted, ev;
            integer e;

            // Find max logit for numerical stability
            mx = logits[cur_tok][0];
            for (e = 1; e < NUM_EXPERTS; e = e+1) begin
              if (logits[cur_tok][e] > mx) mx = logits[cur_tok][e];
            end

            // Compute exp(logit - max) and sum
            sum_exp = 0.0;
            for (e = 0; e < NUM_EXPERTS; e = e+1) begin
              shifted = logits[cur_tok][e] - mx;
              if (shifted < -87.3) shifted = -87.3;
              ev = $exp(shifted);
              probs[cur_tok][e] = ev;
              sum_exp = sum_exp + ev;
            end

            // Normalize
            for (e = 0; e < NUM_EXPERTS; e = e+1) begin
              if (sum_exp > 0.0)
                probs[cur_tok][e] = probs[cur_tok][e] / sum_exp;
              else
                probs[cur_tok][e] = 0.0;
            end
          end

          if (cur_tok == ntok_r - 1) begin
            cur_tok = 8'd0;
            state   = ST_TOPK;
          end else begin
            cur_tok = cur_tok + 8'd1;
          end
        end

        // ── TOPK: find top-k experts per token ───────────────────────────
        // Simple selection sort for TOP_K (small K, behavioral)
        ST_TOPK: begin
          begin : TOPK_CALC
            real best_prob;
            integer best_idx, k, e;
            // Local copy of probs for this token (to zero out selected)
            real local_probs [0:NUM_EXPERTS-1];

            for (e = 0; e < NUM_EXPERTS; e = e+1)
              local_probs[e] = probs[cur_tok][e];

            for (k = 0; k < TOP_K; k = k+1) begin
              best_prob = -1.0;
              best_idx  = 0;
              for (e = 0; e < NUM_EXPERTS; e = e+1) begin
                if (local_probs[e] > best_prob) begin
                  best_prob = local_probs[e];
                  best_idx  = e;
                end
              end
              idx_out_r[cur_tok][k] = best_idx[7:0];
              scr_out_r[cur_tok][k] = rf16(best_prob);
              // Zero out selected so next iteration picks different expert
              local_probs[best_idx] = -1.0;
            end
          end

          if (cur_tok == ntok_r - 1) begin
            state = ST_DONE;
            busy  = 1'b0;
            done  = 1'b1;
          end else begin
            cur_tok = cur_tok + 8'd1;
          end
        end

        // ── DONE: one-cycle pulse, back to IDLE ──────────────────────────
        ST_DONE:
          state = ST_IDLE;

        default:
          state = ST_IDLE;

      endcase
    end
  end

endmodule
`default_nettype wire
