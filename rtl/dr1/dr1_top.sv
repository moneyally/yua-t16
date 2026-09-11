// ============================================================================
// dr1_top.sv — ORBIT-DR1 헤드 최상위 (PLAN W6: **INIT/DUMP 골격만**)
//
// SSOT: docs/DESIGN.md 5절(블록), 5.1절(완료 신호 계약)
//       spec/deltarule.md 2절(opcode), 3.4절(DUMP), 4절(실패 조건)
//
// ----------------------------------------------------------------------------
// 이번 주(W6)에 **동작하는** 경로는 두 개뿐이다:
//   DELTA_INIT (0x50) : state_sram 을 0 으로 (clr_start → clr_busy 가 내려갈 때까지)
//   DELTA_DUMP (0x52) : 상태를 행 단위로 흘려보낸다 (spec 3.5절 행 우선)
//
// DELTA_STEP (0x51) 은 **일부러 구현하지 않는다.** 받으면 즉시
//   done_err + fault_code = 0x07 (DR1_UNIMPL)
// 을 낸다. 조용히 성공하는 것보다 명시적으로 실패하는 것이 낫다 —
// 계산 경로는 W7 이다 (PLAN "한 번에 하나").
//
// ----------------------------------------------------------------------------
// 완료 신호 계약 (docs/DESIGN.md 5.1, BUG-001 재발 방지)
// ----------------------------------------------------------------------------
//   done_ok    : 성공. 정확히 1사이클. **DESC_DONE IRQ 는 이것에만 걸린다.**
//   done_err   : 실패. 정확히 1사이클. fault_code 가 같이 유효하다.
//   done_pulse : 리타이어 = done_ok | done_err. 자원 회수(OOM 감소)용.
//   한 트랜잭션은 ok 와 err 중 **정확히 하나**만 낸다. 둘 다 내는 경로는 없다
//   (ST_DONE_OK 와 ST_DONE_ERR 이 서로 다른 상태이고, 각각에서 IDLE 로 간다).
//
// ----------------------------------------------------------------------------
// DUMP 출력은 지금 **스트림 포트**다 (dump_valid/dump_row/dump_data).
// 스크래치 메모리 쓰기 경로에 붙이는 것은 W7~W8 이다. 지금 이것을 "메모리에
// 썼다" 고 말하지 않는다 — 테스트벤치가 받아서 골든과 대조할 뿐이다.
// ============================================================================
`timescale 1ns/1ps
`default_nettype none

module dr1_top #(
  parameter int D         = 16,
  parameter int W         = 16,
  parameter int NUM_SLOTS = 1       // spec/deltarule.md: v1 은 슬롯 0 하나
)(
  input  wire         clk,
  input  wire         rst_n,

  // 디스크립터 명령 (desc_fsm_v2 의 cmd 인터페이스에서 온다)
  input  wire         cmd_valid,
  output logic        cmd_ready,
  input  wire [7:0]   cmd_opcode,
  input  wire [7:0]   cmd_slot,       // 디스크립터 바이트 1
  input  wire [63:0]  cmd_dst_addr,   // DUMP 목적지 (= out_addr, 바이트 32)

  // 상태 덤프 스트림 (행 우선, spec/deltarule.md 3.5절)
  output logic                 dump_valid,
  output logic [$clog2(D)-1:0] dump_row,
  output logic [D*W-1:0]       dump_data,

  // 완료 신호 (docs/DESIGN.md 5.1)
  output logic        busy,
  output logic        done_ok,
  output logic        done_err,
  output logic        done_pulse,
  output logic [7:0]  fault_code,

  // 레지스터 (spec/deltarule.md 5절). W1C 는 set-wins.
  output logic [31:0] dr1_status,
  output logic [31:0] dr1_sat_count,
  output logic [31:0] dr1_clamp_count,
  output logic [31:0] dr1_cycles,
  input  wire         sat_count_clr,     // W1C 쓰기 스트로브
  input  wire         clamp_count_clr
);

  localparam int AW = $clog2(D);

  // opcode (spec/deltarule.md 2절)
  localparam logic [7:0] OPC_DELTA_INIT = 8'h50;
  localparam logic [7:0] OPC_DELTA_STEP = 8'h51;
  localparam logic [7:0] OPC_DELTA_DUMP = 8'h52;

  // fault code (spec/deltarule.md 4절)
  localparam logic [7:0] FC_NONE           = 8'h00;
  localparam logic [7:0] FC_ILLEGAL_OPCODE = 8'h01;
  localparam logic [7:0] FC_DR1_BAD_SLOT   = 8'h05;
  localparam logic [7:0] FC_DR1_UNALIGNED  = 8'h06;
  localparam logic [7:0] FC_DR1_UNIMPL     = 8'h07;

  localparam int ADDR_ALIGN_BITS = 4;   // 16바이트 정렬 (act_sram 데이터 폭 128비트)

  typedef enum logic [2:0] {
    ST_IDLE     = 3'd0,
    ST_INIT     = 3'd1,   // state_sram 클리어 대기
    ST_DUMP     = 3'd2,   // 행 스트리밍
    ST_DONE_OK  = 3'd3,
    ST_DONE_ERR = 3'd4
  } state_t;

  state_t state, state_n;

  logic [7:0]  opcode_r;
  logic [7:0]  slot_r;
  logic [7:0]  fault_r;
  logic [63:0] dst_addr_r;

  // --------------------------------------------------------------------------
  // state_sram — 상태가 사는 곳. INIT 과 DUMP 가 각각 clr/read 포트를 쓴다.
  // --------------------------------------------------------------------------
  logic            sram_rd_en;
  logic [AW-1:0]   sram_rd_row;
  logic [D*W-1:0]  sram_rd_data;
  logic            sram_clr_start;
  logic            sram_clr_busy;

  state_sram #(.D(D), .W(W)) u_state (
    .clk       (clk),
    .rst_n     (rst_n),
    .rd_en     (sram_rd_en),
    .rd_row    (sram_rd_row),
    .rd_data   (sram_rd_data),
    // 쓰기 포트는 W7(STEP)에서 update_unit 이 쓴다. W6 에서는 놀린다.
    .wr_en     (1'b0),
    .wr_row    ({AW{1'b0}}),
    .wr_data   ({(D*W){1'b0}}),
    .clr_start (sram_clr_start),
    .clr_busy  (sram_clr_busy)
  );

  // --------------------------------------------------------------------------
  // 디코드 — cmd 를 받는 사이클에 결정한다
  // --------------------------------------------------------------------------
  wire slot_bad    = (cmd_slot >= 8'(NUM_SLOTS));
  wire dst_unalign = (cmd_dst_addr[ADDR_ALIGN_BITS-1:0] != '0);

  logic [7:0] decode_fault;
  logic       decode_is_init, decode_is_dump;

  always_comb begin
    decode_fault   = FC_NONE;
    decode_is_init = 1'b0;
    decode_is_dump = 1'b0;

    if (slot_bad) begin
      // 슬롯 검사가 먼저다 — 어떤 opcode 든 슬롯이 틀리면 그것부터 잘못이다
      decode_fault = FC_DR1_BAD_SLOT;
    end else begin
      case (cmd_opcode)
        OPC_DELTA_INIT: decode_is_init = 1'b1;
        OPC_DELTA_DUMP: begin
          if (dst_unalign) decode_fault = FC_DR1_UNALIGNED;
          else             decode_is_dump = 1'b1;
        end
        OPC_DELTA_STEP: decode_fault = FC_DR1_UNIMPL;   // W7 에서 구현
        default:        decode_fault = FC_ILLEGAL_OPCODE;
      endcase
    end
  end

  // --------------------------------------------------------------------------
  // DUMP 스트리밍 포인터
  //   state_sram 읽기 지연이 1사이클이므로 matvec_unit 과 같은 겹침 구조다:
  //     사이클 c   : rd_row = r 요청
  //     사이클 c+1 : rd_data = S[r] 도착 → dump_valid
  // --------------------------------------------------------------------------
  localparam int PW = AW + 1;
  localparam logic [PW-1:0] D_CNT = PW'(D);

  logic [PW-1:0] issue_ptr, emit_ptr;
  logic          rd_pending;

  // --------------------------------------------------------------------------
  // FSM
  // --------------------------------------------------------------------------
  always_comb begin
    state_n = state;
    case (state)
      ST_IDLE: begin
        if (cmd_valid) begin
          if (decode_fault != FC_NONE) state_n = ST_DONE_ERR;
          else if (decode_is_init)     state_n = ST_INIT;
          else if (decode_is_dump)     state_n = ST_DUMP;
          else                         state_n = ST_DONE_ERR;  // 도달 불가 (방어)
        end
      end
      ST_INIT:     if (!sram_clr_busy && !sram_clr_start) state_n = ST_DONE_OK;
      ST_DUMP:     if (emit_ptr >= D_CNT)                 state_n = ST_DONE_OK;
      ST_DONE_OK:  state_n = ST_IDLE;
      ST_DONE_ERR: state_n = ST_IDLE;
      default:     state_n = ST_IDLE;
    endcase
  end

  assign cmd_ready  = (state == ST_IDLE);
  assign busy       = (state != ST_IDLE);
  assign done_ok    = (state == ST_DONE_OK);
  assign done_err   = (state == ST_DONE_ERR);
  assign done_pulse = done_ok | done_err;          // 리타이어 = ok | err
  assign fault_code = done_err ? fault_r : FC_NONE;

  assign sram_rd_en  = (state == ST_DUMP) && (issue_ptr < D_CNT);
  assign sram_rd_row = issue_ptr[AW-1:0];

  assign dump_valid = (state == ST_DUMP) && rd_pending;
  assign dump_row   = emit_ptr[AW-1:0];
  assign dump_data  = sram_rd_data;

  // --------------------------------------------------------------------------
  // 레지스터 (spec/deltarule.md 5절)
  //   DR1_SAT_COUNT / DR1_CLAMP_COUNT 는 **W7 의 STEP 경로가** 올린다.
  //   W6 에는 올리는 쪽이 없으므로 값이 0 에 머문다 — 합성기가 접는 것이 정상이다.
  //   W1C 규칙(set-wins)은 지금 배선해 둔다. 나중에 붙이면 그때 틀린다.
  // --------------------------------------------------------------------------
  logic sat_bump, clamp_bump;
  assign sat_bump   = 1'b0;   // W7: requant 포화 이벤트
  assign clamp_bump = 1'b0;   // W7: α/β > 0x8000 클램프 이벤트

  assign dr1_status = {20'd0, slot_r, 3'd0, busy};

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state         <= ST_IDLE;
      opcode_r      <= 8'd0;
      slot_r        <= 8'd0;
      fault_r       <= FC_NONE;
      dst_addr_r    <= 64'd0;
      issue_ptr     <= '0;
      emit_ptr      <= '0;
      rd_pending    <= 1'b0;
      sram_clr_start <= 1'b0;
      dr1_sat_count   <= 32'd0;
      dr1_clamp_count <= 32'd0;
      dr1_cycles      <= 32'd0;
    end else begin
      state <= state_n;
      sram_clr_start <= 1'b0;

      // 사이클 카운터 — matvec_unit / update_unit 과 같은 규약
      if (state == ST_IDLE && cmd_valid) dr1_cycles <= 32'd1;
      else if (state != ST_IDLE)         dr1_cycles <= dr1_cycles + 32'd1;

      // W1C: 하드웨어 증가와 클리어가 겹치면 증가가 이긴다 (irq_ctrl 과 같은 규칙)
      if (sat_bump)            dr1_sat_count   <= dr1_sat_count + 32'd1;
      else if (sat_count_clr)  dr1_sat_count   <= 32'd0;

      if (clamp_bump)          dr1_clamp_count <= dr1_clamp_count + 32'd1;
      else if (clamp_count_clr) dr1_clamp_count <= 32'd0;

      case (state)
        ST_IDLE: begin
          if (cmd_valid) begin
            opcode_r   <= cmd_opcode;
            slot_r     <= cmd_slot;
            dst_addr_r <= cmd_dst_addr;
            fault_r    <= decode_fault;
            issue_ptr  <= '0;
            emit_ptr   <= '0;
            rd_pending <= 1'b0;
            if (decode_is_init) sram_clr_start <= 1'b1;
          end
        end

        ST_DUMP: begin
          if (sram_rd_en) issue_ptr <= issue_ptr + PW'(1);
          rd_pending <= sram_rd_en;
          if (rd_pending) emit_ptr <= emit_ptr + PW'(1);
        end

        default: ;
      endcase
    end
  end

  // opcode_r / dst_addr_r 은 W7(STEP)에서 쓴다. 지금은 파형 디버깅용으로만 남긴다.
  /* verilator lint_off UNUSEDSIGNAL */
  wire _unused_ok = &{1'b0, opcode_r, dst_addr_r, 1'b0};
  /* verilator lint_on UNUSEDSIGNAL */

endmodule

`default_nettype wire
