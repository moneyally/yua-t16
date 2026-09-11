// ============================================================================
// dr1_top.sv — ORBIT-DR1 헤드 최상위 (PLAN W7: **DELTA_STEP 전체 경로**)
//
// SSOT: docs/DESIGN.md 2절(수학), 5절(블록), 5.1절(완료 신호 계약), 6절(사이클)
//       spec/deltarule.md 2절(opcode), 3절(디스크립터), 4절(실패 조건)
//
// 대응 골든: sim/golden/deltarule.py  step(S, q, k, v, alpha, beta)
//            연산 **순서**가 골든 docstring 1~10 단계와 같아야 비트 일치가 난다.
//
// ----------------------------------------------------------------------------
// 동작하는 경로 (W7 기준 전부)
// ----------------------------------------------------------------------------
//   DELTA_INIT (0x50) : state_sram 을 0 으로
//   DELTA_STEP (0x51) : 토큰 1개. 스크래치에서 q/k/v 를 읽고 o 를 쓴다
//   DELTA_DUMP (0x52) : 상태 S 전체를 스크래치 dst_addr 로 쓴다 (+ 스트림 포트)
//
// ----------------------------------------------------------------------------
// DELTA_STEP 의 단계와 골든 대응
// ----------------------------------------------------------------------------
//   ST_LOAD    q,k,v 를 스크래치 → vec_regs        (골든: 인자 그 자체)
//   ST_MV_K    p = S·k                             (골든: matvec)
//   ST_ERR     err = v − α·p                       (골든: compute_err)
//   ST_UPD_*   행마다 S[i] ← α·S[i] + β·err[i]·kᵀ  (골든: update_row × d)
//   ST_MV_Q    o = S_next·q                        (골든: matvec)
//   ST_WR_O    o → 스크래치
//
// **순서가 계약이다.** 예를 들어 o 를 갱신 전 S 로 계산하면 골든과 어긋난다
// (골든은 9~10단계에서 S_next 를 쓴다).
//
// ----------------------------------------------------------------------------
// 완료 신호 계약 (docs/DESIGN.md 5.1, BUG-001 재발 방지)
// ----------------------------------------------------------------------------
//   done_ok / done_err 중 **정확히 하나**, 각각 정확히 1사이클.
//   done_pulse = ok | err (리타이어). 완료 IRQ 는 done_ok 에만 걸린다.
//
// ----------------------------------------------------------------------------
// α, β 클램프 (spec/deltarule.md 1절)
// ----------------------------------------------------------------------------
//   0x8000(=1.0) 초과는 미정의다. **여기서 클램프하고 DR1_CLAMP_COUNT 를 올린다.**
//   클램프는 이 한 곳에만 있다 — update_unit/err_unit 은 이미 정리된 값을 받는다.
// ============================================================================
`timescale 1ns/1ps
`default_nettype none

module dr1_top #(
  parameter int D         = 16,
  parameter int W         = 16,
  parameter int NUM_SLOTS = 1,     // spec/deltarule.md: v1 은 슬롯 0 하나
  parameter int SCRATCH   = 1024   // 스크래치 원소 개수 (S 덤프 d*d + 벡터 4d)
)(
  input  wire         clk,
  input  wire         rst_n,

  // 디스크립터 명령 (desc_fsm_v2 의 cmd 인터페이스 + 원시 디스크립터 바이트)
  input  wire         cmd_valid,
  output logic        cmd_ready,
  input  wire [7:0]   cmd_opcode,
  input  wire [7:0]   cmd_slot,       // 바이트 1
  input  wire [63:0]  cmd_q_addr,     // 바이트 16 (act_addr 자리)
  input  wire [63:0]  cmd_k_addr,     // 바이트 24 (wgt_addr 자리)
  input  wire [63:0]  cmd_dst_addr,   // 바이트 32 (out_addr 자리) = o_addr / dump dst
  input  wire [63:0]  cmd_v_addr,     // 바이트 44
  input  wire [15:0]  cmd_alpha,      // 바이트 52, UQ1.15
  input  wire [15:0]  cmd_beta,       // 바이트 54, UQ1.15

  // 스크래치 메모리 포트 (dr1_scratch 의 DR1 쪽)
  output logic                       scr_rd_en,
  output logic [$clog2(SCRATCH)-1:0] scr_rd_addr,
  input  wire  [W-1:0]               scr_rd_data,
  output logic                       scr_wr_en,
  output logic [$clog2(SCRATCH)-1:0] scr_wr_addr,
  output logic [W-1:0]               scr_wr_data,

  // 상태 덤프 스트림 (행 우선, spec/deltarule.md 3.5절). 메모리 쓰기와 **병행**한다
  output logic                 dump_valid,
  output logic [$clog2(D)-1:0] dump_row,
  output logic [D*W-1:0]       dump_data,

  // 완료 신호 (docs/DESIGN.md 5.1)
  output logic        busy,
  output logic        done_ok,
  output logic        done_err,
  output logic        done_pulse,
  output logic [7:0]  fault_code,

  // 트레이스 이벤트 (spec/deltarule.md 5.2). 각각 1사이클 펄스
  output logic        sat_event,
  output logic        clamp_event,

  // 레지스터 (spec/deltarule.md 5절). W1C 는 set-wins.
  output logic [31:0] dr1_status,
  output logic [31:0] dr1_sat_count,
  output logic [31:0] dr1_clamp_count,
  output logic [31:0] dr1_cycles,
  input  wire         sat_count_clr,
  input  wire         clamp_count_clr
);

  localparam int AW  = $clog2(D);        // 상태 행 인덱스
  localparam int SAW = $clog2(SCRATCH);  // 스크래치 주소
  localparam int PW  = AW + 1;           // D 까지 세는 포인터
  localparam logic [PW-1:0] D_CNT = PW'(D);

  // opcode (spec/deltarule.md 2절)
  localparam logic [7:0] OPC_DELTA_INIT = 8'h50;
  localparam logic [7:0] OPC_DELTA_STEP = 8'h51;
  localparam logic [7:0] OPC_DELTA_DUMP = 8'h52;

  // fault code (spec/deltarule.md 4절)
  localparam logic [7:0] FC_NONE           = 8'h00;
  localparam logic [7:0] FC_ILLEGAL_OPCODE = 8'h01;
  localparam logic [7:0] FC_DR1_BAD_SLOT   = 8'h05;
  localparam logic [7:0] FC_DR1_UNALIGNED  = 8'h06;
  // 0x07 DR1_UNIMPL 은 W6 에서 DELTA_STEP 이 없을 때 쓰던 코드다. W7 에서 STEP 이
  // 구현되어 **더 이상 내지 않는다.** spec/deltarule.md 4절에는 남는다 (이 빌드에
  // 없는 경로를 가리키는 코드이고, 다음에 또 쓸 자리가 생긴다).
  localparam logic [7:0] FC_DR1_ADDR_RANGE = 8'h08;

  localparam int ADDR_ALIGN_BITS = 4;   // 16바이트 정렬 (spec 4절)

  localparam logic [1:0] SEL_Q = 2'd0;  // vec_regs 와 같은 인코딩
  localparam logic [1:0] SEL_K = 2'd1;
  localparam logic [1:0] SEL_V = 2'd2;

  typedef enum logic [3:0] {
    ST_IDLE      = 4'd0,
    ST_INIT      = 4'd1,
    ST_DUMP_RD   = 4'd2,   // 상태 행 읽기 요청
    ST_DUMP_WR   = 4'd3,   // 행의 원소를 스크래치로 (D사이클)
    ST_LOAD      = 4'd4,   // q,k,v 를 스크래치 → vec_regs
    ST_MV_K      = 4'd5,   // p = S·k
    ST_ERR       = 4'd6,   // err = v − α·p
    ST_UPD_RD    = 4'd7,   // S[i] 읽기 요청
    ST_UPD_START = 4'd8,   // update_unit 기동
    ST_UPD_WAIT  = 4'd9,   // update_unit 완료 대기 + 쓰기
    ST_MV_Q      = 4'd10,  // o = S_next·q
    ST_WR_O      = 4'd11,  // o → 스크래치
    ST_DONE_OK   = 4'd12,
    ST_DONE_ERR  = 4'd13
  } state_t;

  state_t state, state_n;

  logic [7:0]  opcode_r, slot_r, fault_r;
  logic [15:0] alpha_r, beta_r;
  logic [SAW-1:0] q_base, k_base, v_base, o_base, dump_base;

  // ==========================================================================
  // 하위 블록
  // ==========================================================================
  logic           sram_rd_en;
  logic [AW-1:0]  sram_rd_row;
  logic [D*W-1:0] sram_rd_data;
  logic           sram_wr_en;
  logic [AW-1:0]  sram_wr_row;
  logic [D*W-1:0] sram_wr_data;
  logic           sram_clr_start, sram_clr_busy;

  state_sram #(.D(D), .W(W)) u_state (
    .clk(clk), .rst_n(rst_n),
    .rd_en(sram_rd_en), .rd_row(sram_rd_row), .rd_data(sram_rd_data),
    .wr_en(sram_wr_en), .wr_row(sram_wr_row), .wr_data(sram_wr_data),
    .clr_start(sram_clr_start), .clr_busy(sram_clr_busy)
  );

  logic           vr_ld_en;
  logic [1:0]     vr_ld_sel;
  logic [AW-1:0]  vr_ld_idx;
  logic [W-1:0]   vr_ld_data;
  logic [D*W-1:0] q_flat, k_flat, v_flat;

  vec_regs #(.D(D), .W(W)) u_vec (
    .clk(clk), .rst_n(rst_n),
    .ld_en(vr_ld_en), .ld_sel(vr_ld_sel), .ld_idx(vr_ld_idx), .ld_data(vr_ld_data),
    .q_flat(q_flat), .k_flat(k_flat), .v_flat(v_flat)
  );

  // matvec_unit — p 와 o 를 **같은 유닛**으로 두 번 돈다 (골든도 같은 함수다)
  logic           mv_start;
  logic [D*W-1:0] mv_x_flat, mv_y_flat;
  logic           mv_rd_en;
  logic [AW-1:0]  mv_rd_row;
  logic [31:0]    mv_sat, mv_cycles;
  logic           mv_busy, mv_done;

  matvec_unit #(.D(D), .W(W)) u_mv (
    .clk(clk), .rst_n(rst_n),
    .start(mv_start), .x_flat(mv_x_flat),
    .sram_rd_en(mv_rd_en), .sram_rd_row(mv_rd_row), .sram_rd_data(sram_rd_data),
    .y_flat(mv_y_flat), .sat_count(mv_sat), .cycles(mv_cycles),
    .busy(mv_busy), .done(mv_done)
  );

  // 갱신 중인 행의 err 원소. **포트 연결식에 $signed() 를 직접 쓰지 않는다** —
  // sv2v→yosys 0.33 이 부호 붙은 부분선택을 포트에 물리면 내부 assert 로 죽는다
  // (`arg->is_signed == sig.as_wire()->is_signed`). 중간 wire 로 빼면 통과한다.
  wire signed [W-1:0] err_sel = $signed(err_r[upd_idx[AW-1:0]*W +: W]);

  logic           up_start;
  logic [D*W-1:0] up_row_flat;
  logic [31:0]    up_sat, up_cycles;
  logic           up_busy, up_done;

  update_unit #(.D(D), .W(W)) u_upd (
    .clk(clk), .rst_n(rst_n),
    .start(up_start),
    .alpha_uq15(alpha_r), .beta_uq15(beta_r),
    .err_i(err_sel),
    // **state_sram 의 읽기 데이터를 그대로 준다.** 여기에 레지스터를 하나 더 두면
    // update_unit 이 start 를 받는 사이클에 그 레지스터는 아직 **이전 행**을 들고
    // 있다 (등록이 같은 edge 에 일어나기 때문). 실제로 그 버그를 냈고, 상태가 0 인
    // 첫 토큰만 통과해서 10토큰 테스트에서 잡혔다 — 1토큰 테스트로는 못 잡는다.
    .s_row_flat(sram_rd_data), .k_flat(k_flat),
    .row_flat(up_row_flat), .sat_count(up_sat), .cycles(up_cycles),
    .busy(up_busy), .done(up_done)
  );

  // err_unit — 순수 조합. p 는 레지스터, v 는 vec_regs.
  logic [D*W-1:0] p_r, err_r, o_r;
  logic [D*W-1:0] err_comb;
  logic [31:0]    err_sat;

  err_unit #(.D(D), .W(W)) u_err (
    .alpha_uq15(alpha_r), .v_flat(v_flat), .p_flat(p_r),
    .err_flat(err_comb), .sat_count(err_sat)
  );

  // ==========================================================================
  // 디코드 — cmd 를 받는 사이클에 전부 판정한다
  // ==========================================================================
  wire slot_bad = (cmd_slot >= 8'(NUM_SLOTS));

  // 정렬: spec 4절. STEP 은 q/k/v/o 넷 다, DUMP 는 dst 만.
  wire step_unalign = (cmd_q_addr[ADDR_ALIGN_BITS-1:0]   != '0)
                   || (cmd_k_addr[ADDR_ALIGN_BITS-1:0]   != '0)
                   || (cmd_v_addr[ADDR_ALIGN_BITS-1:0]   != '0)
                   || (cmd_dst_addr[ADDR_ALIGN_BITS-1:0] != '0);
  wire dump_unalign = (cmd_dst_addr[ADDR_ALIGN_BITS-1:0] != '0);

  // 범위: 원소 인덱스 = 바이트 주소 / 2. 벡터는 D개, 덤프는 D*D개 들어간다.
  localparam int unsigned VEC_LIMIT  = SCRATCH - D;
  localparam int unsigned DUMP_LIMIT = SCRATCH - D * D;

  // 인자는 **바이트 주소를 1비트 내린 것**(= 원소 인덱스)이다. 비트 0 은 정렬
  // 검사가 이미 본다 — 여기서 또 보면 같은 규칙이 두 곳에 생긴다.
  function automatic logic addr_oor(input logic [62:0] elem, input int unsigned limit);
    addr_oor = (elem[62:SAW] != '0) || (64'(elem[SAW-1:0]) > 64'(limit));
  endfunction

  wire step_oor = addr_oor(cmd_q_addr[63:1],   VEC_LIMIT)
               || addr_oor(cmd_k_addr[63:1],   VEC_LIMIT)
               || addr_oor(cmd_v_addr[63:1],   VEC_LIMIT)
               || addr_oor(cmd_dst_addr[63:1], VEC_LIMIT);
  wire dump_oor = addr_oor(cmd_dst_addr[63:1], DUMP_LIMIT);

  logic [7:0] decode_fault;
  logic       decode_is_init, decode_is_step, decode_is_dump;

  always_comb begin
    decode_fault   = FC_NONE;
    decode_is_init = 1'b0;
    decode_is_step = 1'b0;
    decode_is_dump = 1'b0;

    if (slot_bad) begin
      // 슬롯 검사가 먼저다 — opcode 가 무엇이든 슬롯이 틀리면 그것부터 잘못이다
      decode_fault = FC_DR1_BAD_SLOT;
    end else begin
      case (cmd_opcode)
        OPC_DELTA_INIT: decode_is_init = 1'b1;
        OPC_DELTA_STEP: begin
          if      (step_unalign) decode_fault = FC_DR1_UNALIGNED;
          else if (step_oor)     decode_fault = FC_DR1_ADDR_RANGE;
          else                   decode_is_step = 1'b1;
        end
        OPC_DELTA_DUMP: begin
          if      (dump_unalign) decode_fault = FC_DR1_UNALIGNED;
          else if (dump_oor)     decode_fault = FC_DR1_ADDR_RANGE;
          else                   decode_is_dump = 1'b1;
        end
        default: decode_fault = FC_ILLEGAL_OPCODE;
      endcase
    end
  end

  // α/β 클램프 (spec 1절). 여기 한 곳에서만 한다.
  wire alpha_clamped = (cmd_alpha > 16'h8000);
  wire beta_clamped  = (cmd_beta  > 16'h8000);
  wire [15:0] alpha_eff = alpha_clamped ? 16'h8000 : cmd_alpha;
  wire [15:0] beta_eff  = beta_clamped  ? 16'h8000 : cmd_beta;

  // ==========================================================================
  // 카운터·포인터
  // ==========================================================================
  logic [PW-1:0]  issue_ptr, emit_ptr;   // DUMP 행
  logic [PW-1:0]  elem_ptr;              // DUMP 행 안의 열 / WR_O 인덱스
  logic           rd_pending;
  logic [AW+2:0]  ld_cnt;                // 0 .. 3D  (q,k,v 연속)
  logic           ld_pending;
  logic [1:0]     ld_sel_d;
  logic [AW-1:0]  ld_idx_d;
  logic [PW-1:0]  upd_idx;               // 갱신 중인 행
  logic [D*W-1:0] dump_row_hold;

  localparam logic [AW+2:0] LD_TOTAL = (AW+3)'(3 * D);

  // 적재 카운터를 벡터별 오프셋으로 쪼갠 것. 식에 바로 비트 선택을 붙일 수 없어
  // (SystemVerilog 문법) 중간 신호로 둔다. AW 비트로 캐스팅해서 잘라 두면
  // 상위 비트가 남지 않는다 (verilator UNUSEDSIGNAL 방지).
  wire [AW-1:0] ld_off_k = AW'(ld_cnt - (AW+3)'(D));
  wire [AW-1:0] ld_off_v = AW'(ld_cnt - (AW+3)'(2 * D));

  // DUMP 행 우선 오프셋: (행 번호) × D. emit_ptr 은 이미 1 증가한 뒤라 1을 뺀다.
  wire [PW-1:0]  dump_row_idx = emit_ptr - PW'(1);
  wire [SAW-1:0] dump_row_off = SAW'(dump_row_idx) * SAW'(D);

  // ==========================================================================
  // FSM (조합)
  // ==========================================================================
  always_comb begin
    state_n = state;
    case (state)
      ST_IDLE: begin
        if (cmd_valid) begin
          if      (decode_fault != FC_NONE) state_n = ST_DONE_ERR;
          else if (decode_is_init)          state_n = ST_INIT;
          else if (decode_is_step)          state_n = ST_LOAD;
          else if (decode_is_dump)          state_n = ST_DUMP_RD;
          else                              state_n = ST_DONE_ERR;  // 방어
        end
      end

      ST_INIT: if (!sram_clr_busy && !sram_clr_start) state_n = ST_DONE_OK;

      // ── DUMP ──
      ST_DUMP_RD: if (rd_pending) state_n = ST_DUMP_WR;
      ST_DUMP_WR: if (elem_ptr >= D_CNT)
                    state_n = (issue_ptr >= D_CNT) ? ST_DONE_OK : ST_DUMP_RD;

      // ── STEP ──
      ST_LOAD:      if (ld_cnt >= LD_TOTAL && !ld_pending) state_n = ST_MV_K;
      ST_MV_K:      if (mv_done)  state_n = ST_ERR;
      ST_ERR:       state_n = ST_UPD_RD;
      ST_UPD_RD:    state_n = ST_UPD_START;
      ST_UPD_START: state_n = ST_UPD_WAIT;
      ST_UPD_WAIT:  if (up_done) state_n = (upd_idx + PW'(1) >= D_CNT) ? ST_MV_Q : ST_UPD_RD;
      ST_MV_Q:      if (mv_done) state_n = ST_WR_O;
      ST_WR_O:      if (elem_ptr >= D_CNT) state_n = ST_DONE_OK;

      ST_DONE_OK:  state_n = ST_IDLE;
      ST_DONE_ERR: state_n = ST_IDLE;
      default:     state_n = ST_IDLE;
    endcase
  end

  // ==========================================================================
  // 출력·포트 먹스
  // ==========================================================================
  assign cmd_ready  = (state == ST_IDLE);
  assign busy       = (state != ST_IDLE);
  assign done_ok    = (state == ST_DONE_OK);
  assign done_err   = (state == ST_DONE_ERR);
  assign done_pulse = done_ok | done_err;
  assign fault_code = done_err ? fault_r : FC_NONE;

  wire mv_phase = (state == ST_MV_K) || (state == ST_MV_Q);

  // state_sram 읽기 포트: DUMP / UPDATE / matvec 가 나눠 쓴다
  always_comb begin
    if (state == ST_DUMP_RD) begin
      sram_rd_en  = (issue_ptr < D_CNT) && !rd_pending;
      sram_rd_row = issue_ptr[AW-1:0];
    end else if (state == ST_UPD_RD) begin
      sram_rd_en  = 1'b1;
      sram_rd_row = upd_idx[AW-1:0];
    end else if (mv_phase) begin
      sram_rd_en  = mv_rd_en;
      sram_rd_row = mv_rd_row;
    end else begin
      sram_rd_en  = 1'b0;
      sram_rd_row = '0;
    end
  end

  // state_sram 쓰기 포트: update_unit 결과만 쓴다
  assign sram_wr_en   = (state == ST_UPD_WAIT) && up_done;
  assign sram_wr_row  = upd_idx[AW-1:0];
  assign sram_wr_data = up_row_flat;

  // matvec 입력: MV_K 는 k, MV_Q 는 q
  assign mv_x_flat = (state == ST_MV_Q) ? q_flat : k_flat;
  assign mv_start  = ((state == ST_MV_K) || (state == ST_MV_Q)) && !mv_busy && !mv_done;

  assign up_start = (state == ST_UPD_START);

  // 스크래치 포트
  always_comb begin
    scr_rd_en   = 1'b0;
    scr_rd_addr = '0;
    scr_wr_en   = 1'b0;
    scr_wr_addr = '0;
    scr_wr_data = '0;

    if (state == ST_LOAD && ld_cnt < LD_TOTAL) begin
      scr_rd_en = 1'b1;
      // ld_cnt 0..D-1 = q, D..2D-1 = k, 2D..3D-1 = v
      if      (ld_cnt < (AW+3)'(D))     scr_rd_addr = q_base + SAW'(ld_cnt);
      else if (ld_cnt < (AW+3)'(2 * D)) scr_rd_addr = k_base + SAW'(ld_cnt - (AW+3)'(D));
      else                              scr_rd_addr = v_base + SAW'(ld_cnt - (AW+3)'(2 * D));
    end

    if (state == ST_WR_O && elem_ptr < D_CNT) begin
      scr_wr_en   = 1'b1;
      scr_wr_addr = o_base + SAW'(elem_ptr);
      scr_wr_data = o_r[elem_ptr[AW-1:0]*W +: W];
    end

    if (state == ST_DUMP_WR && elem_ptr < D_CNT) begin
      scr_wr_en   = 1'b1;
      // 행 우선: 행 r 의 열 c → dump_base + r*D + c   (spec 3.5절)
      scr_wr_addr = dump_base + dump_row_off + SAW'(elem_ptr);
      scr_wr_data = dump_row_hold[elem_ptr[AW-1:0]*W +: W];
    end
  end

  // vec_regs 적재 — 스크래치 읽기 지연 1사이클 뒤에 쓴다
  assign vr_ld_en   = ld_pending;
  assign vr_ld_sel  = ld_sel_d;
  assign vr_ld_idx  = ld_idx_d;
  assign vr_ld_data = scr_rd_data;

  // DUMP 스트림 (W6 호환 — 테스트벤치가 이걸로 본다)
  assign dump_valid = (state == ST_DUMP_RD) && rd_pending;
  assign dump_row   = emit_ptr[AW-1:0];
  assign dump_data  = sram_rd_data;

  assign dr1_status = {20'd0, slot_r, 3'd0, busy};

  // ==========================================================================
  // 포화·클램프 카운트
  //   골든 step() 의 sat_count 는 matvec(p) + err + update×D + matvec(o) 합이다.
  //   같은 지점에서 같은 수를 더한다.
  // ==========================================================================
  logic [31:0] sat_add;
  always_comb begin
    sat_add = 32'd0;
    if (mv_phase && mv_done)                     sat_add = mv_sat;
    else if (state == ST_ERR)                    sat_add = err_sat;
    else if (state == ST_UPD_WAIT && up_done)    sat_add = up_sat;
  end

  wire [31:0] clamp_add = (state == ST_IDLE && cmd_valid && decode_is_step)
                        ? (32'(alpha_clamped) + 32'(beta_clamped)) : 32'd0;

  assign sat_event   = (sat_add   != 32'd0);
  assign clamp_event = (clamp_add != 32'd0);

  // ==========================================================================
  // 순차 논리
  // ==========================================================================
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state           <= ST_IDLE;
      opcode_r        <= 8'd0;
      slot_r          <= 8'd0;
      fault_r         <= FC_NONE;
      alpha_r         <= 16'h8000;
      beta_r          <= 16'd0;
      q_base          <= '0;
      k_base          <= '0;
      v_base          <= '0;
      o_base          <= '0;
      dump_base       <= '0;
      issue_ptr       <= '0;
      emit_ptr        <= '0;
      elem_ptr        <= '0;
      rd_pending      <= 1'b0;
      ld_cnt          <= '0;
      ld_pending      <= 1'b0;
      ld_sel_d        <= SEL_Q;
      ld_idx_d        <= '0;
      upd_idx         <= '0;
      p_r             <= '0;
      err_r           <= '0;
      o_r             <= '0;
      dump_row_hold   <= '0;
      sram_clr_start  <= 1'b0;
      dr1_sat_count   <= 32'd0;
      dr1_clamp_count <= 32'd0;
      dr1_cycles      <= 32'd0;
    end else begin
      state          <= state_n;
      sram_clr_start <= 1'b0;

      // 사이클 카운터 — matvec_unit / update_unit 과 같은 규약
      if (state == ST_IDLE && cmd_valid) dr1_cycles <= 32'd1;
      else if (state != ST_IDLE)         dr1_cycles <= dr1_cycles + 32'd1;

      // W1C: 하드웨어 증가와 클리어가 겹치면 증가가 이긴다 (irq_ctrl 과 같은 규칙)
      if (sat_add != 32'd0)     dr1_sat_count   <= dr1_sat_count + sat_add;
      else if (sat_count_clr)   dr1_sat_count   <= 32'd0;

      if (clamp_add != 32'd0)   dr1_clamp_count <= dr1_clamp_count + clamp_add;
      else if (clamp_count_clr) dr1_clamp_count <= 32'd0;

      case (state)
        ST_IDLE: begin
          if (cmd_valid) begin
            opcode_r   <= cmd_opcode;
            slot_r     <= cmd_slot;
            fault_r    <= decode_fault;
            alpha_r    <= alpha_eff;
            beta_r     <= beta_eff;
            q_base     <= cmd_q_addr[SAW:1];
            k_base     <= cmd_k_addr[SAW:1];
            v_base     <= cmd_v_addr[SAW:1];
            o_base     <= cmd_dst_addr[SAW:1];
            dump_base  <= cmd_dst_addr[SAW:1];
            issue_ptr  <= '0;
            emit_ptr   <= '0;
            elem_ptr   <= '0;
            rd_pending <= 1'b0;
            ld_cnt     <= '0;
            ld_pending <= 1'b0;
            upd_idx    <= '0;
            if (decode_is_init) sram_clr_start <= 1'b1;
          end
        end

        // ── DUMP: 행을 읽고(1사이클 지연) 원소를 하나씩 스크래치로 ──
        ST_DUMP_RD: begin
          if (sram_rd_en) issue_ptr <= issue_ptr + PW'(1);
          rd_pending <= sram_rd_en;
          if (rd_pending) begin
            emit_ptr      <= emit_ptr + PW'(1);
            dump_row_hold <= sram_rd_data;
            elem_ptr      <= '0;
          end
        end

        ST_DUMP_WR: begin
          if (elem_ptr < D_CNT) elem_ptr <= elem_ptr + PW'(1);
          else                  rd_pending <= 1'b0;
        end

        // ── STEP ──
        ST_LOAD: begin
          // 요청을 흘리고, 1사이클 뒤 도착분을 vec_regs 에 쓴다
          if (ld_cnt < LD_TOTAL) ld_cnt <= ld_cnt + (AW+3)'(1);
          ld_pending <= (ld_cnt < LD_TOTAL);
          if (ld_cnt < (AW+3)'(D)) begin
            ld_sel_d <= SEL_Q;
            ld_idx_d <= ld_cnt[AW-1:0];
          end else if (ld_cnt < (AW+3)'(2 * D)) begin
            ld_sel_d <= SEL_K;
            ld_idx_d <= ld_off_k[AW-1:0];
          end else begin
            ld_sel_d <= SEL_V;
            ld_idx_d <= ld_off_v[AW-1:0];
          end
        end

        ST_MV_K: if (mv_done) p_r <= mv_y_flat;

        ST_ERR:  err_r <= err_comb;

        ST_UPD_RD: ;   // 읽기 요청만 (조합 출력)

        ST_UPD_START: ;   // update_unit 이 sram_rd_data 를 직접 래치한다

        ST_UPD_WAIT: if (up_done) upd_idx <= upd_idx + PW'(1);

        ST_MV_Q: if (mv_done) begin
          o_r      <= mv_y_flat;
          elem_ptr <= '0;
        end

        ST_WR_O: if (elem_ptr < D_CNT) elem_ptr <= elem_ptr + PW'(1);

        default: ;
      endcase
    end
  end

  // opcode_r 은 파형 디버깅용이다. up_cycles / mv_cycles 는 유닛별 실측이라
  // dr1_top 은 쓰지 않는다 (DR1_CYCLES 는 트랜잭션 전체를 센다).
  /* verilator lint_off UNUSEDSIGNAL */
  wire _unused_ok = &{1'b0, opcode_r, up_cycles, mv_cycles, up_busy, 1'b0};
  /* verilator lint_on UNUSEDSIGNAL */

endmodule

`default_nettype wire
