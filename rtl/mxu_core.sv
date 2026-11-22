`timescale 1ns/1ps
`default_nettype none

// mxu_core.sv — BF16 MXU 에 메모리 경로를 붙인 코어
//
// ── 왜 ───────────────────────────────────────────────────────────
//
// `mxu_bf16_128x128` 은 **연산 블록**이다. 피연산자를 `a_col`/`b_row`
// 포트 배열로 받고 DMA 가 없다. `rtl/g3_int_top.sv` 의 주석이
// `TB drives a_row/b_col` 이라고 적고 있듯, 테스트벤치가 먹여 주는
// 구조였다. 즉 누산 정확도는 검증됐지만 실제 데이터를 스스로 가져오지
// 못했다.
//
// 이 모듈이 그 공백을 메운다. `gemm_core` (INT8) 에 들어간 메모리
// 아키텍처를 그대로 가져온다.
//
//   로드 FSM (lst)  DMA 읽기 → 피연산자 SRAM (요청 쪼개기 포함)
//   연산 FSM (cst)  SRAM → MXU (start/data_valid/done_pulse 프로토콜)
//   쓰기 FSM (wst)  누산기 → DMA 쓰기
//
// 활성값 SRAM 은 2 뱅크 이중 버퍼, 가중치는 weight-stationary 로 상주
// 한다. 세부 근거는 `spec/memory-bandwidth-analysis.md` 에 있다.
//
// ── INT8 코어와 다른 점 ──────────────────────────────────────────
//
// 1. 원소가 BF16 (16비트) 다. K-step 하나가 `DIM*16` 비트다.
//    DIM=16 → 256비트, DIM=32 → 512비트.
// 2. MXU 는 핸드셰이크가 있다. `acc_clr` 한 사이클 → `start` 와
//    `k_steps` → `data_valid` 로 K-step 을 흘림 → `done_pulse`.
// 3. **피연산자 타이밍.** MXU 안에서 `tile_en` 이 레지스터라 타일은
//    `data_valid` 보다 **한 사이클 뒤**에 피연산자를 먹는다. 동기 SRAM
//    은 주소를 낸 다음 사이클에 데이터를 내므로, `data_valid` 와 SRAM
//    주소를 **같은 사이클**에 내면 정확히 맞는다.
//    (INT8 코어에서 `mac_en` 정렬로 똑같이 당했다. 거기서는 발행 3사이클
//    뒤에 먹어서 앞쪽 K-step 을 놓쳤다.)
// 4. 출력이 `DIM*DIM` FP32 다. DIM=16 이면 1,024바이트로 INT8 16×16 과
//    같고, DIM=32 면 4,096바이트다.
//
// ── 아직 안 한 것 ────────────────────────────────────────────────
//
// `FP32_OUT=1` (조합 `acc_flat`) 경로만 쓴다. `STREAM_RD=1` 은 타일당
// 변환기 하나를 공유해 타일당 143,394 셀을 아끼지만(20.8%,
// `spec/pe-fixed-point-redesign.md`), `rd_fp32` 가 사이클당 하나씩
// 나오므로 쓰기가 `OUT_BEATS` 대신 `ACCS` 사이클이 된다. 그 교환을
// 재려면 먼저 이 경로가 맞아야 한다.
module mxu_core #(
  // 축소 차원 상한 (K-step 수). 실제 레이어 깊이까지 올릴 수 있다 —
  // 요청 쪼개기가 16비트 길이 필드 한계를 처리한다.
  parameter int MAX_KT = 256,

  // MXU 한 변. 16 의 배수여야 한다 (TILE_DIM=16).
  parameter int DIM = 16,

  // DMA 읽기/쓰기 폭. K-step 하나가 DIM*16 비트이므로 읽기 폭이 그것의
  // 배수여야 한다.
  parameter int RD_DATA_W = 256,
  parameter int WR_DATA_W = 128,

  // 피연산자 뱅크 수 (2의 거듭제곱). 활성값만 회전한다.
  parameter int BANKS = 2
)(
  input  logic clk,
  input  logic rst_n,

  input  logic        cmd_valid,
  output logic        cmd_ready,
  input  logic [63:0] act_addr,
  input  logic [63:0] wgt_addr,
  input  logic [63:0] out_addr,
  input  logic [31:0] Kt,
  // 가중치 재적재 생략 (SRAM 상주분 사용)
  input  logic        wgt_keep,

  output logic        done,

  // DMA Read
  output logic         rd_req_valid,
  input  logic         rd_req_ready,
  output logic [63:0]  rd_req_addr,
  output logic [15:0]  rd_req_len_bytes,
  input  logic         rd_done,

  input  logic         rd_data_valid,
  input  logic [RD_DATA_W-1:0] rd_data,
  output logic         rd_data_ready,
  input  logic         rd_data_last,

  // DMA Write
  output logic         wr_req_valid,
  input  logic         wr_req_ready,
  output logic [63:0]  wr_req_addr,
  output logic [15:0]  wr_req_len_bytes,
  input  logic         wr_done,

  output logic         wr_data_valid,
  input  logic         wr_data_ready,
  output logic [WR_DATA_W-1:0] wr_data,
  output logic         wr_data_last,

  output logic [31:0]  perf_cycles,
  output logic [31:0]  perf_bytes,
  output logic [31:0]  perf_wgt_reuse,
  output logic [31:0]  perf_wgt_load,

  output logic         busy
);

  // ----------------------------
  // 치수
  // ----------------------------
  localparam int ADDR_W   = $clog2(MAX_KT);
  localparam int ACCS     = DIM * DIM;             // 누산기 수
  localparam int KSTEP_W  = DIM * 16;              // K-step 하나의 비트 수
  localparam int WPB      = WR_DATA_W / 32;        // 쓰기 비트당 누산기
  localparam int OUT_BEATS = ACCS / WPB;
  localparam int OUT_BYTES = ACCS * 4;
  localparam int BEAT_W   = $clog2(OUT_BEATS + 1);
  localparam int WR_BEAT_B = WR_DATA_W / 8;
  localparam int RD_BEAT_B = RD_DATA_W / 8;

  // 읽기 워드 하나에 들어가는 K-step 수
  localparam int KPW   = RD_DATA_W / KSTEP_W;
  localparam int KPW_W = (KPW > 1) ? $clog2(KPW) : 1;

  // 뱅크
  localparam int BANK_W     = $clog2(BANKS);
  localparam int BANK_DEPTH = (MAX_KT + KPW - 1) / KPW;
  localparam int BADDR_W    = $clog2(BANK_DEPTH);
  localparam int SRAM_DEPTH = BANK_DEPTH * BANKS;
  localparam int SADDR_W    = $clog2(SRAM_DEPTH);

  // 요청 길이는 16비트다. 긴 전송은 쪼갠다 (gemm_core 와 같은 이유).
  localparam int LEN_W      = 32;
  localparam int RD_REQ_MAX = (65535 / RD_BEAT_B) * RD_BEAT_B;

  initial begin
    if (DIM % 16 != 0 || DIM < 16)
      $fatal(1, "mxu_core: DIM must be a positive multiple of 16 (got %0d)", DIM);
    if (KPW < 1 || KPW * KSTEP_W != RD_DATA_W)
      $fatal(1, "mxu_core: RD_DATA_W (%0d) must be a positive multiple of DIM*16 (%0d)",
             RD_DATA_W, KSTEP_W);
    if (WR_DATA_W != 128 && WR_DATA_W != 256 && WR_DATA_W != 512)
      $fatal(1, "mxu_core: WR_DATA_W must be 128, 256 or 512 (got %0d)", WR_DATA_W);
    if (OUT_BEATS * WPB != ACCS)
      $fatal(1, "mxu_core: DIM^2 (%0d) must be a multiple of WR_DATA_W/32 (%0d)",
             ACCS, WPB);
    if (OUT_BYTES > 65535)
      $fatal(1, "mxu_core: OUT_BYTES %0d exceeds the 16-bit wr_req_len_bytes field",
             OUT_BYTES);
    if (BANKS < 2 || (BANKS & (BANKS - 1)) != 0)
      $fatal(1, "mxu_core: BANKS must be a power of two >= 2 (got %0d)", BANKS);
  end

  // 뱅크 b 의 idx 번째 워드. BANK_DEPTH 가 2의 거듭제곱이 아니어도 맞는다.
  function automatic logic [SADDR_W-1:0] bank_addr(
      input logic [BANK_W-1:0]  b,
      input logic [BADDR_W-1:0] idx);
    bank_addr = SADDR_W'(b) * SADDR_W'(BANK_DEPTH) + SADDR_W'(idx);
  endfunction

  function automatic logic [31:0] clamp_kt(input logic [31:0] in);
    logic [31:0] mx;
    begin
      mx = MAX_KT;
      clamp_kt = (in > mx) ? mx : in;
    end
  endfunction

  // ----------------------------
  // FSM
  // ----------------------------
  typedef enum logic [2:0] {
    L_IDLE, L_ACT_REQ, L_ACT_RECV, L_WGT_REQ, L_WGT_RECV
  } lstate_t;

  // MXU 프로토콜에 맞춘 연산 FSM.
  //   C_CLR    acc_clr 한 사이클 (MXU 는 P_IDLE 에서만 받는다)
  //   C_START  start + k_steps
  //   C_FEED   data_valid 로 Kt 개를 흘린다
  //   C_WAIT   done_pulse 를 기다린다
  //   C_LATCH  acc_flat 을 떠서 쓰기 FSM 기동
  typedef enum logic [2:0] {
    C_IDLE, C_CLR, C_START, C_FEED, C_WAIT, C_LATCH
  } cstate_t;

  typedef enum logic [1:0] { W_IDLE, W_REQ, W_SEND, W_DONE } wstate_t;

  lstate_t lst, lst_n;
  cstate_t cst, cst_n;
  wstate_t wst, wst_n;

  logic ld_mark, wgt_load_done, cm_free, wr_start;

  // ----------------------------
  // 뱅크 소유권
  // ----------------------------
  logic [BANKS-1:0]  bank_full;
  logic [BANK_W-1:0] ld_bank, cm_bank;

  logic [31:0] Kt_bank  [0:BANKS-1];
  logic [63:0] out_bank [0:BANKS-1];
  wire  [31:0] Kt_cm  = Kt_bank[cm_bank];
  wire  [63:0] out_cm = out_bank[cm_bank];

  logic [63:0]      act_base_r, wgt_base_r;
  logic [LEN_W-1:0] rd_len_r, wgt_len_r;

  // weight-stationary
  logic wgt_valid_r;
  wire  wgt_skip = wgt_keep & wgt_valid_r;
  logic wgt_skip_r;
  logic [BANK_W-1:0] wld_bank, wres_bank;
  logic [BANK_W-1:0] wgt_bank_of [0:BANKS-1];
  wire  [BANK_W-1:0] wgt_bank_cm = wgt_bank_of[cm_bank];

  assign cmd_ready = (lst == L_IDLE) && !bank_full[ld_bank];
  wire   cmd_accept = cmd_valid & cmd_ready;

  logic [31:0] wgt_reuse_r, wgt_load_r;
  assign perf_wgt_reuse = wgt_reuse_r;
  assign perf_wgt_load  = wgt_load_r;

  integer bi;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      act_base_r <= 64'd0;
      wgt_base_r <= 64'd0;
      rd_len_r   <= '0;
      wgt_len_r  <= '0;
      for (bi = 0; bi < BANKS; bi = bi + 1) begin
        Kt_bank[bi]  <= 32'd0;
        out_bank[bi] <= 64'd0;
      end
    end else if (cmd_accept) begin
      act_base_r <= act_addr;
      wgt_base_r <= wgt_addr;
      // 읽기 길이는 워드 단위로 올림한다 (남는 부분은 쓰이지 않는다).
      rd_len_r   <= LEN_W'(((clamp_kt(Kt) + KPW - 1) / KPW) * RD_BEAT_B);
      wgt_len_r  <= LEN_W'(((clamp_kt(Kt) + KPW - 1) / KPW) * RD_BEAT_B);
      Kt_bank[ld_bank]  <= clamp_kt(Kt);
      out_bank[ld_bank] <= out_addr;
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      bank_full <= '0;
      ld_bank   <= '0;
      cm_bank   <= '0;
    end else begin
      // ld_mark 와 cm_free 는 같은 사이클에 올 수 있지만 같은 뱅크에는
      // 겹치지 않는다 (gemm_core 와 같은 논증: set 은 full=0 에서만,
      // clear 는 full=1 에서만).
      if (ld_mark) begin
        bank_full[ld_bank] <= 1'b1;
        ld_bank            <= ld_bank + 1'b1;
      end
      if (cm_free) begin
        bank_full[cm_bank] <= 1'b0;
        cm_bank            <= cm_bank + 1'b1;
      end
    end
  end

  integer wi;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wgt_valid_r <= 1'b0;
      wgt_skip_r  <= 1'b0;
      wgt_reuse_r <= 32'd0;
      wgt_load_r  <= 32'd0;
      wld_bank    <= '0;
      wres_bank   <= '0;
      for (wi = 0; wi < BANKS; wi = wi + 1)
        wgt_bank_of[wi] <= '0;
    end else begin
      if (cmd_accept) begin
        wgt_skip_r <= wgt_skip;
        if (wgt_skip) begin
          wgt_bank_of[ld_bank] <= wres_bank;
          wgt_reuse_r          <= wgt_reuse_r + 1;
        end else begin
          wgt_bank_of[ld_bank] <= wld_bank;
          wgt_valid_r          <= 1'b1;
          wgt_load_r           <= wgt_load_r + 1;
        end
      end
      // 포인터는 적재가 **끝난** 시점에 전진한다. 수락 시점에 올리면
      // 로드 FSM 이 다음 뱅크에 써 버린다 (gemm_core 에서 그렇게 깨졌다).
      if (wgt_load_done) begin
        wres_bank <= wld_bank;
        wld_bank  <= wld_bank + 1'b1;
      end
    end
  end

  // ----------------------------
  // Perf
  // ----------------------------
  logic [31:0] cyc_r, bytes_r;
  assign perf_cycles = cyc_r;
  assign perf_bytes  = bytes_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cyc_r   <= 0;
      bytes_r <= 0;
    end else begin
      if (busy) cyc_r <= cyc_r + 1;
      bytes_r <= bytes_r
        + ((rd_data_valid && rd_data_ready) ? RD_BEAT_B : 0)
        + ((wr_data_valid && wr_data_ready) ? WR_BEAT_B : 0);
    end
  end

  // ----------------------------
  // 피연산자 SRAM (1W1R, 뱅크)
  // ----------------------------
  logic act_we, wgt_we, act_re, wgt_re;
  logic [SADDR_W-1:0] act_waddr, wgt_waddr, act_raddr, wgt_raddr;
  logic [RD_DATA_W-1:0] act_word, wgt_word;
  logic [KSTEP_W-1:0]   act_kstep, wgt_kstep;

  act_sram #(.MAX_KT(MAX_KT), .DATA_W(RD_DATA_W), .DEPTH(SRAM_DEPTH)) u_act (
    .clk(clk), .rst_n(rst_n),
    .we(act_we), .re(act_re),
    .waddr(act_waddr), .wdata(rd_data),
    .raddr(act_raddr), .rdata(act_word)
  );

  wgt_sram #(.MAX_KT(MAX_KT), .DATA_W(RD_DATA_W), .DEPTH(SRAM_DEPTH)) u_wgt (
    .clk(clk), .rst_n(rst_n),
    .we(wgt_we), .re(wgt_re),
    .waddr(wgt_waddr), .wdata(rd_data),
    .raddr(wgt_raddr), .rdata(wgt_word)
  );

  logic [BADDR_W-1:0] load_idx;
  assign act_waddr = bank_addr(ld_bank,  load_idx);
  assign wgt_waddr = bank_addr(wld_bank, load_idx);

  // K 인덱스 → (워드 주소, 워드 내 위치)
  logic [ADDR_W-1:0]  feed_k;         // 지금 내는 K-step
  logic [KPW_W-1:0]   rd_sub, rd_sub_q;

  generate
    if (KPW == 1) begin : G_ONE
      assign act_raddr = bank_addr(cm_bank,     feed_k[BADDR_W-1:0]);
      assign wgt_raddr = bank_addr(wgt_bank_cm, feed_k[BADDR_W-1:0]);
      assign rd_sub    = '0;
      assign act_kstep = act_word;
      assign wgt_kstep = wgt_word;
    end else begin : G_WIDE
      assign act_raddr = bank_addr(cm_bank,     feed_k[KPW_W +: BADDR_W]);
      assign wgt_raddr = bank_addr(wgt_bank_cm, feed_k[KPW_W +: BADDR_W]);
      assign rd_sub    = feed_k[KPW_W-1:0];
      assign act_kstep = act_word[rd_sub_q*KSTEP_W +: KSTEP_W];
      assign wgt_kstep = wgt_word[rd_sub_q*KSTEP_W +: KSTEP_W];
    end
  endgenerate

  // SRAM 은 re=0 일 때 rdata 를 유지하므로 슬라이스 위치도 같이 유지한다.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)      rd_sub_q <= '0;
    else if (act_re) rd_sub_q <= rd_sub;
  end

  // ----------------------------
  // MXU
  // ----------------------------
  logic [15:0] a_col [0:DIM-1];
  logic [15:0] b_row [0:DIM-1];
  logic        mxu_start, mxu_acc_clr, mxu_data_valid;
  logic        mxu_busy, mxu_done, mxu_ready;
  logic [31:0] acc_flat [0:ACCS-1];
  logic [$clog2(ACCS)-1:0] mxu_rd_idx;
  logic [31:0]             mxu_rd_fp32;

  genvar li;
  generate
    for (li = 0; li < DIM; li++) begin : G_LANES
      assign a_col[li] = act_kstep[16*li +: 16];
      assign b_row[li] = wgt_kstep[16*li +: 16];
    end
  endgenerate

  mxu_bf16_128x128 #(
    .DIM(DIM), .TILE_DIM(16), .TILES(DIM/16), .PAR(DIM/16),
    .FP32_OUT(1), .STREAM_RD(0)
  ) u_mxu (
    .clk(clk), .rst_n(rst_n),
    .start(mxu_start), .acc_clr(mxu_acc_clr), .k_steps(16'(Kt_cm)),
    .a_col(a_col), .b_row(b_row), .data_valid(mxu_data_valid),
    .busy(mxu_busy), .done_pulse(mxu_done), .ready(mxu_ready),
    .acc_flat(acc_flat),
    .rd_idx(mxu_rd_idx), .rd_fp32(mxu_rd_fp32)
  );

  assign mxu_rd_idx = '0;   // STREAM_RD=0 — 쓰지 않는다

  // ----------------------------
  // 누산기 래치
  // ----------------------------
  logic [31:0] acc_lat [0:ACCS-1];
  logic [63:0] wr_addr_r;

  wire cm_latch = (cst == C_LATCH) && (wst == W_IDLE);

  genvar ai;
  generate
    for (ai = 0; ai < ACCS; ai++) begin : G_ACC_LAT
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)        acc_lat[ai] <= 32'd0;
        else if (cm_latch) acc_lat[ai] <= (Kt_cm != 0) ? acc_flat[ai] : 32'd0;
      end
    end
  endgenerate

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)        wr_addr_r <= 64'd0;
    else if (cm_latch) wr_addr_r <= out_cm;
  end

  // ----------------------------
  // 쓰기 데이터패스
  // ----------------------------
  logic [BEAT_W-1:0]    out_beat;
  logic [WR_DATA_W-1:0] wr_data_r;
  logic                 wr_valid_r;

  function automatic logic [31:0] x32(input logic [31:0] v);
    begin
      if ((^v === 1'bx)) x32 = 32'd0;
      else               x32 = v;
    end
  endfunction

  function automatic logic [WR_DATA_W-1:0] pack_beat(input int base);
    int w;
    begin
      pack_beat = '0;
      for (w = 0; w < WPB; w = w + 1)
        pack_beat[w*32 +: 32] = x32(acc_lat[base + w]);
    end
  endfunction

`ifdef COCOTB_SIM
  initial begin
    out_beat   = '0;
    wr_data_r  = '0;
    wr_valid_r = 1'b0;
  end
`endif

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_beat   <= '0;
      wr_data_r  <= '0;
      wr_valid_r <= 1'b0;
    end else begin
      if (wst == W_REQ && wr_req_ready) begin
        out_beat   <= '0;
        wr_data_r  <= pack_beat(0);
        wr_valid_r <= 1'b1;
      end else if (wst == W_SEND) begin
        if (wr_valid_r && wr_data_ready) begin
          if (out_beat == OUT_BEATS-1) begin
            wr_valid_r <= 1'b0;
          end else begin
            out_beat   <= out_beat + 1'b1;
            wr_data_r  <= pack_beat((out_beat + 1) * WPB);
            wr_valid_r <= 1'b1;
          end
        end
      end else begin
        wr_valid_r <= 1'b0;
      end
    end
  end

  assign wr_data       = wr_data_r;
  assign wr_data_valid = (wst == W_SEND) && wr_valid_r;
  assign wr_data_last  = (wst == W_SEND) && wr_valid_r && (out_beat == OUT_BEATS-1);
  assign wr_req_addr      = wr_addr_r;
  assign wr_req_len_bytes = 16'(OUT_BYTES);

  // ----------------------------
  // outputs
  // ----------------------------
  assign rd_data_ready = 1'b1;
  assign busy = (lst != L_IDLE) || (cst != C_IDLE) || (wst != W_IDLE)
             || (bank_full != '0);
  assign done = (wst == W_DONE);

  // ----------------------------
  // 로드 FSM (요청 쪼개기 포함)
  // ----------------------------
  logic [LEN_W-1:0] rd_left;
  logic [63:0]      rd_cur_addr;
  wire  [LEN_W-1:0] rd_this_len =
          (rd_left > LEN_W'(RD_REQ_MAX)) ? LEN_W'(RD_REQ_MAX) : rd_left;
  logic phase_begin;

  always_comb begin
    lst_n = lst;

    rd_req_valid     = 1'b0;
    rd_req_addr      = 64'd0;
    rd_req_len_bytes = 16'(rd_this_len);

    act_we        = 1'b0;
    wgt_we        = 1'b0;
    ld_mark       = 1'b0;
    wgt_load_done = 1'b0;

    case (lst)
      L_IDLE: if (cmd_accept) lst_n = L_ACT_REQ;

      L_ACT_REQ: begin
        rd_req_valid = 1'b1;
        rd_req_addr  = rd_cur_addr;
        if (rd_req_ready) lst_n = L_ACT_RECV;
      end

      L_ACT_RECV: begin
        if (rd_data_valid && rd_data_ready) act_we = 1'b1;
        if (rd_data_valid && rd_data_ready && rd_data_last) begin
          if (rd_left != '0)        lst_n = L_ACT_REQ;   // 아직 남았다
          else if (wgt_skip_r) begin
            ld_mark = 1'b1;
            lst_n   = L_IDLE;
          end else                  lst_n = L_WGT_REQ;
        end
      end

      L_WGT_REQ: begin
        rd_req_valid = 1'b1;
        rd_req_addr  = rd_cur_addr;
        if (rd_req_ready) lst_n = L_WGT_RECV;
      end

      L_WGT_RECV: begin
        if (rd_data_valid && rd_data_ready) wgt_we = 1'b1;
        if (rd_data_valid && rd_data_ready && rd_data_last) begin
          if (rd_left != '0) lst_n = L_WGT_REQ;
          else begin
            wgt_load_done = 1'b1;
            ld_mark       = 1'b1;
            lst_n         = L_IDLE;
          end
        end
      end

      default: lst_n = L_IDLE;
    endcase

    // 단계의 첫 요청에서만 load_idx 를 되돌린다 (쪼개진 요청에서
    // 되돌리면 앞 조각을 덮어쓴다).
    phase_begin = (lst == L_IDLE && cmd_accept)
               || (lst == L_ACT_RECV && rd_data_valid && rd_data_ready
                   && rd_data_last && rd_left == '0 && !wgt_skip_r);
  end

  wire rd_req_fire = rd_req_valid & rd_req_ready;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      lst         <= L_IDLE;
      load_idx    <= '0;
      rd_left     <= '0;
      rd_cur_addr <= 64'd0;
    end else begin
      lst <= lst_n;

      if (lst == L_IDLE && cmd_accept) begin
        rd_left     <= LEN_W'(((clamp_kt(Kt) + KPW - 1) / KPW) * RD_BEAT_B);
        rd_cur_addr <= act_addr;
      end else if (lst_n == L_WGT_REQ && lst == L_ACT_RECV) begin
        rd_left     <= wgt_len_r;
        rd_cur_addr <= wgt_base_r;
      end else if (rd_req_fire) begin
        rd_left     <= rd_left - rd_this_len;
        rd_cur_addr <= rd_cur_addr + 64'(rd_this_len);
      end

      if (phase_begin)
        load_idx <= '0;
      else if ((lst == L_ACT_RECV || lst == L_WGT_RECV)
               && rd_data_valid && rd_data_ready)
        load_idx <= load_idx + 1'b1;
    end
  end

  // ----------------------------
  // 연산 FSM (MXU 프로토콜)
  // ----------------------------
  // 지금까지 낸 K-step 수.
  //
  // **ADDR_W 비트로는 안 된다.** ADDR_W = $clog2(MAX_KT) 는 주소
  // 0..MAX_KT-1 을 표현하는 폭이고, Kt = MAX_KT 까지 **세려면** 한 비트
  // 더 필요하다. 8비트로 두면 `feed_cnt < 256` 이 항상 참이 되어
  // K-step 을 영원히 흘린다 (Kt=MAX_KT 에서만 드러난다).
  localparam int CNT_W = $clog2(MAX_KT + 1);
  logic [CNT_W-1:0] feed_cnt;

  always_comb begin
    cst_n = cst;

    mxu_start      = 1'b0;
    mxu_acc_clr    = 1'b0;
    mxu_data_valid = 1'b0;
    act_re         = 1'b0;
    wgt_re         = 1'b0;
    cm_free        = 1'b0;
    wr_start       = 1'b0;
    feed_k         = feed_cnt[ADDR_W-1:0];   // 주소는 ADDR_W 만 쓴다

    case (cst)
      C_IDLE: if (bank_full[cm_bank]) cst_n = C_CLR;

      // MXU 는 P_IDLE 에서만 acc_clr 을 받고, 같은 사이클에 start 를
      // 받지 않는다 (else-if). 그래서 한 사이클 떼어 놓는다.
      C_CLR: begin
        mxu_acc_clr = 1'b1;
        cst_n       = C_START;
      end

      C_START: begin
        if (Kt_cm == 0) begin
          // 할 일이 없다. 누산기는 방금 지웠으므로 0 을 쓴다.
          cst_n = C_LATCH;
        end else begin
          mxu_start = 1'b1;
          cst_n     = C_FEED;
        end
      end

      // 핵심 타이밍: data_valid 와 SRAM 주소를 **같은 사이클**에 낸다.
      // SRAM 은 다음 사이클에 데이터를 내고, MXU 의 tile_en 도 그때
      // 서므로 정확히 맞는다.
      C_FEED: begin
        if (feed_cnt < Kt_cm) begin
          mxu_data_valid = 1'b1;
          act_re         = 1'b1;
          wgt_re         = 1'b1;
        end else begin
          cst_n = C_WAIT;
        end
      end

      C_WAIT: if (mxu_done) cst_n = C_LATCH;

      C_LATCH: begin
        // acc_lat 은 단일 버퍼다. 쓰기 FSM 이 비어야 덮어쓸 수 있다.
        if (wst == W_IDLE) begin
          wr_start = 1'b1;
          cm_free  = 1'b1;
          cst_n    = C_IDLE;
        end
      end

      default: cst_n = C_IDLE;
    endcase
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cst      <= C_IDLE;
      feed_cnt <= '0;
    end else begin
      cst <= cst_n;
      if (cst == C_CLR)
        feed_cnt <= '0;
      else if (cst == C_FEED && mxu_data_valid)
        feed_cnt <= feed_cnt + 1'b1;
    end
  end

  // ----------------------------
  // 쓰기 FSM
  // ----------------------------
  always_comb begin
    wst_n        = wst;
    wr_req_valid = 1'b0;
    case (wst)
      W_IDLE: if (wr_start) wst_n = W_REQ;
      W_REQ: begin
        wr_req_valid = 1'b1;
        if (wr_req_ready) wst_n = W_SEND;
      end
      W_SEND:
        if (wr_data_ready && wr_valid_r && (out_beat == OUT_BEATS-1))
          wst_n = W_DONE;
      W_DONE: wst_n = W_IDLE;
      default: wst_n = W_IDLE;
    endcase
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) wst <= W_IDLE;
    else        wst <= wst_n;
  end

`ifdef COCOTB_SIM
  // MXU 가 받을 준비가 안 됐는데 밀어 넣으면 K-step 을 잃는다.
  // 전부-병렬 구성에서 ready 는 항상 1 이지만, 가정이 깨지면 조용히
  // 틀리는 대신 터뜨린다.
  always_ff @(posedge clk) begin
    if (rst_n && mxu_data_valid && !mxu_ready)
      $fatal(1, "mxu_core: fed the MXU while it was not ready");
  end
`endif

endmodule

`default_nettype wire
