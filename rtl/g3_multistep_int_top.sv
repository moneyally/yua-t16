// g3_multistep_int_top.sv — ORBIT-G3 Multi-Step Training Loop
// SSOT: ORBIT_G3_RTL_ISSUES.md (G3-INT-003)
//
// Runs N training steps sequentially on single layer:
//   step k: backward(dW_k) → optimizer(param_k → param_{k+1}) → loss_scaler
// State feedback: param/m/v carry across steps. scale accumulates.
//
// Forward is omitted — loop proof focuses on update recurrence.
// Active region: 16×16.
`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off UNUSEDSIGNAL */

module g3_multistep_int_top #(
  parameter int DIM = 16
)(
  input  logic        clk,
  input  logic        rst_n,
  input  logic        start,
  input  logic [7:0]  num_steps,

  // Hyperparams
  input  logic [31:0] lr_fp32, beta1_fp32, beta2_fp32, epsilon_fp32, weight_decay_fp32,
  input  logic        adamw_enable,

  // Loss scaler config
  input  logic [31:0] init_scale_fp32, growth_factor_fp32, backoff_factor_fp32,
  input  logic [15:0] growth_interval,
  input  logic [31:0] min_scale_fp32, max_scale_fp32,

  // Per-step inputs (TB drives each step)
  input  logic [15:0] x_bf16   [0:DIM-1][0:DIM-1],
  input  logic [15:0] w_bf16   [0:DIM-1][0:DIM-1],
  input  logic [15:0] dy_bf16  [0:DIM-1][0:DIM-1],
  input  logic        overflow_inject,  // TB controls overflow per step

  // Initial state
  input  logic [31:0] param_init [0:DIM-1][0:DIM-1],
  input  logic [31:0] m_init     [0:DIM-1][0:DIM-1],
  input  logic [31:0] v_init     [0:DIM-1][0:DIM-1],

  // Observable state (current after each step)
  output logic [31:0] cur_param [0:DIM-1][0:DIM-1],
  output logic [31:0] cur_m     [0:DIM-1][0:DIM-1],
  output logic [31:0] cur_v     [0:DIM-1][0:DIM-1],
  output logic [31:0] cur_scale_fp32,
  output logic [7:0]  cur_step,

  output logic        busy,
  output logic        step_done_pulse,
  output logic        loop_done_pulse,
  output logic [7:0]  err_code
);

  // ═══════════════════════════════════════════════════════════
  // State registers (feedback across steps)
  // ═══════════════════════════════════════════════════════════
  logic [31:0] st_param [0:DIM-1][0:DIM-1];
  logic [31:0] st_m     [0:DIM-1][0:DIM-1];
  logic [31:0] st_v     [0:DIM-1][0:DIM-1];

  // Wire outputs
  genvar gi, gj;
  generate for (gi=0;gi<DIM;gi++) for (gj=0;gj<DIM;gj++) begin : OUT
    assign cur_param[gi][gj] = st_param[gi][gj];
    assign cur_m[gi][gj] = st_m[gi][gj];
    assign cur_v[gi][gj] = st_v[gi][gj];
  end endgenerate

  // ═══════════════════════════════════════════════════════════
  // Backward engine
  // ═══════════════════════════════════════════════════════════
  logic bk_start, bk_done;
  logic [7:0] bk_err;
  logic [31:0] dw [0:DIM-1][0:DIM-1];

  backward_engine #(.DIM(DIM)) u_bk (
    .clk(clk),.rst_n(rst_n),.start(bk_start),.mode(2'd1),.acc_clr(1'b0),
    .x_in(x_bf16),.w_in(w_bf16),.dy_in(dy_bf16),
    .result(dw),.busy(),.done_pulse(bk_done),.err_code(bk_err));

  // ═══════════════════════════════════════════════════════════
  // Optimizer
  // ═══════════════════════════════════════════════════════════
  logic opt_start, opt_done;
  logic [7:0] opt_err;
  logic [31:0] opt_param_out [0:DIM-1][0:DIM-1];
  logic [31:0] opt_m_out [0:DIM-1][0:DIM-1];
  logic [31:0] opt_v_out [0:DIM-1][0:DIM-1];

  optimizer_unit #(.DIM(DIM)) u_opt (
    .clk(clk),.rst_n(rst_n),.start(opt_start),.adamw_enable(adamw_enable),
    .lr_fp32(lr_fp32),.beta1_fp32(beta1_fp32),.beta2_fp32(beta2_fp32),
    .epsilon_fp32(epsilon_fp32),.weight_decay_fp32(weight_decay_fp32),
    .param_in(st_param),.grad_in(dw),.m_in(st_m),.v_in(st_v),
    .param_out(opt_param_out),.m_out(opt_m_out),.v_out(opt_v_out),
    .busy(),.done_pulse(opt_done),.err_code(opt_err));

  // ═══════════════════════════════════════════════════════════
  // Loss scaler
  // ═══════════════════════════════════════════════════════════
  logic ls_step_done, ls_done;
  logic [7:0] ls_err;

  loss_scaler u_ls (
    .clk(clk),.rst_n(rst_n),
    .step_done(ls_step_done),.overflow_detect(overflow_inject),
    .init_scale_fp32(init_scale_fp32),.growth_factor_fp32(growth_factor_fp32),
    .backoff_factor_fp32(backoff_factor_fp32),.growth_interval(growth_interval),
    .min_scale_fp32(min_scale_fp32),.max_scale_fp32(max_scale_fp32),
    .current_scale_fp32(cur_scale_fp32),.scale_valid(),
    .busy(),.done_pulse(ls_done),.err_code(ls_err));

  // ═══════════════════════════════════════════════════════════
  // Done capture
  // ═══════════════════════════════════════════════════════════
  logic bk_done_r, opt_done_r, ls_done_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin bk_done_r<=0; opt_done_r<=0; ls_done_r<=0; end
    else begin
      if (tst==ST_BKWD) bk_done_r<=0; else if (bk_done) bk_done_r<=1;
      if (tst==ST_OPT)  opt_done_r<=0; else if (opt_done) opt_done_r<=1;
      if (tst==ST_SCALE) ls_done_r<=0; else if (ls_done) ls_done_r<=1;
    end
  end

  // ═══════════════════════════════════════════════════════════
  // Top FSM
  // ═══════════════════════════════════════════════════════════
  typedef enum logic [3:0] {
    ST_IDLE,
    ST_INIT,        // copy init state
    ST_BKWD,
    ST_WAIT_BKWD,
    ST_OPT,
    ST_WAIT_OPT,
    ST_COMMIT,      // copy optimizer output to state registers
    ST_SCALE,
    ST_WAIT_SCALE,
    ST_NEXT_STEP,
    ST_DONE,
    ST_FAULT
  } state_t;

  state_t tst;
  logic [7:0] step_cnt, total_steps;

  assign busy = (tst != ST_IDLE && tst != ST_DONE && tst != ST_FAULT);

  integer si, sj;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      tst <= ST_IDLE;
      step_cnt <= 0; total_steps <= 0;
      bk_start<=0; opt_start<=0; ls_step_done<=0;
      step_done_pulse<=0; loop_done_pulse<=0; err_code<=0;
      cur_step<=0;
      for(si=0;si<DIM;si++) for(sj=0;sj<DIM;sj++) begin
        st_param[si][sj]<=32'd0; st_m[si][sj]<=32'd0; st_v[si][sj]<=32'd0;
      end
    end else begin
      step_done_pulse<=0; loop_done_pulse<=0;
      bk_start<=0; opt_start<=0; ls_step_done<=0;

      case (tst)
        ST_IDLE: begin
          err_code<=0;
          if (start) begin total_steps<=num_steps; step_cnt<=0; tst<=ST_INIT; end
        end

        ST_INIT: begin
          for(si=0;si<DIM;si++) for(sj=0;sj<DIM;sj++) begin
            st_param[si][sj]<=param_init[si][sj];
            st_m[si][sj]<=m_init[si][sj];
            st_v[si][sj]<=v_init[si][sj];
          end
          tst<=ST_BKWD;
        end

        ST_BKWD: begin bk_start<=1; tst<=ST_WAIT_BKWD; end
        ST_WAIT_BKWD: if (bk_done_r) begin
          if(bk_err) begin err_code<=8'h10; tst<=ST_FAULT; end
          else tst<=ST_OPT;
        end

        ST_OPT: begin opt_start<=1; tst<=ST_WAIT_OPT; end
        ST_WAIT_OPT: if (opt_done_r) begin
          if(opt_err) begin err_code<=8'h20; tst<=ST_FAULT; end
          else tst<=ST_COMMIT;
        end

        ST_COMMIT: begin
          // Copy optimizer output to state registers for next step
          for(si=0;si<DIM;si++) for(sj=0;sj<DIM;sj++) begin
            st_param[si][sj] <= opt_param_out[si][sj];
            st_m[si][sj]     <= opt_m_out[si][sj];
            st_v[si][sj]     <= opt_v_out[si][sj];
          end
          tst<=ST_SCALE;
        end

        ST_SCALE: begin ls_step_done<=1; tst<=ST_WAIT_SCALE; end
        ST_WAIT_SCALE: if (ls_done_r) begin
          if(ls_err) begin err_code<=8'h30; tst<=ST_FAULT; end
          else tst<=ST_NEXT_STEP;
        end

        ST_NEXT_STEP: begin
          step_done_pulse <= 1;
          cur_step <= step_cnt[7:0];
          step_cnt <= step_cnt + 1;
          if (step_cnt + 1 >= total_steps)
            tst <= ST_DONE;
          else
            tst <= ST_BKWD;  // next step
        end

        ST_DONE: begin loop_done_pulse<=1; tst<=ST_IDLE; end
        ST_FAULT: begin loop_done_pulse<=1; tst<=ST_IDLE; end
        default: tst<=ST_IDLE;
      endcase
    end
  end

endmodule

`default_nettype wire
