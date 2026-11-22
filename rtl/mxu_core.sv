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
// `STREAM_RD` 을 재 봤고 **쓰지 않기로 했다.** 측정값:
//
//   DIM=16 WR=128  71.0 →  392.0 사이클/op  (가동률 90.1% → 16.3%)
//   DIM=32 WR=512  71.0 → 1160.0 사이클/op  (가동률 90.1% →  5.5%)
//
// 타일당 143,394 셀(20.8%) 을 아끼지만 처리량을 5.5~16.3배 잃는다.
// 손실은 ACCS/OUT_BEATS 에 비례해 커지므로 DIM=128 에서는 더 나쁘다.
// 자세한 이유와 대안 설계점은 `STREAM_RD` 파라미터 주석에 있다.
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
  parameter int BANKS = 2,

  // 누산기 읽기 방식.
  //   0 (기본): MXU 가 `acc_flat` 에 DIM² FP32 를 조합으로 내보낸다.
  //             PE 마다 고정소수점→FP32 변환기가 붙는다.
  //   1       : 타일당 변환기 하나를 공유하고 `rd_idx`/`rd_fp32` 로
  //             순차로 읽는다. **재 보고 기본값으로 쓰지 않기로 했다.**
  //
  // 측정 (둘 다 결과는 일치한다 — 느릴 뿐이다):
  //
  //   DIM=16 WR=128  71.0 →  392.0 사이클/op   5.52배
  //   DIM=32 WR=512  71.0 → 1160.0 사이클/op  16.34배
  //
  // 두 가지가 겹쳐서 이렇게 된다.
  //
  // 1) 비트 하나를 모으는 데 WPB 사이클이 걸린다. 출력이 행 우선이라
  //    한 비트의 WPB 개는 같은 행의 연속된 열이고, 그건 **같은 타일**
  //    안에 있다. 타일마다 변환기가 하나씩 있어 사이클당 TILES 개를
  //    읽을 수는 있지만, 그 TILES 개는 서로 다른 행/열에 흩어져 있어
  //    같은 비트에 들어가지 않는다. 그래서 쓰기가 OUT_BEATS 대신
  //    ACCS + OUT_BEATS 사이클이다.
  // 2) 겹치기가 사라진다. FP32_OUT=1 은 결과를 acc_lat 에 떠 놓으므로
  //    다음 디스크립터의 연산과 쓰기가 겹친다. STREAM_RD=1 은 누산기를
  //    타일에 둔 채 읽으므로 읽는 동안 `mxu_acc_clr` 을 세울 수 없다
  //    (`rd_hold`). 주기가 max(연산, 쓰기) 가 아니라 합이 된다.
  //
  // 래치를 두면 2) 는 없어지지만 ACCS×32 비트 플롭이 생겨 변환기로
  // 아낀 것을 되돌려준다. 그래서 래치를 두지 않았다 (`LAT_N`).
  //
  // ── 측정이 가리킨 대안 → `RD_LANES` 로 구현했다 ──────────────────
  //
  // 문제는 "공유" 가 아니라 "하나" 였다. 타일당 변환기를 **WPB 개**
  // 두면 비트 하나를 한 사이클에 채운다 (같은 타일의 인접한 WPB
  // 열이므로 가능하다). `RD_LANES` 가 그것이다.
  //
  // 파라미터를 남긴 것은 비용이 0 (기본 0 에서 경로가 elaborate 되지
  // 않는다) 이고, 대안 설계점의 기준선이 되기 때문이다.
  parameter int STREAM_RD = 0,

  // 타일마다 둘 공유 변환기 수 (`STREAM_RD=1` 일 때만 쓰인다).
  //
  //   0 (기본): 쓰기 비트 폭에 맞춘다 (WPB = WR_DATA_W/32). 비트
  //             하나를 한 사이클에 채우므로 비트당 사이클 손실이 0 이다.
  //   1       : 타일당 하나. 가장 작고 가장 느리다 (위 측정값).
  //   그 사이 : WPB 를 나누는 2의 거듭제곱.
  //
  // 겹치기 상실은 이것으로 해결되지 않는다. 누산기를 타일에 둔 채
  // 읽으므로 읽는 동안 다음 디스크립터를 시작할 수 없다 (`rd_hold`).
  // 그래서 주기가 max(연산, 쓰기) 가 아니라 합이고, 그 비용은
  // OUT_BEATS 만큼이다 — 비트당 손실과는 별개다.
  parameter int RD_LANES = 0,

  // 누산기 뱅크 수 (`STREAM_RD=1` 일 때만 쓰인다).
  //
  // 1 이면 누산기를 타일에 둔 채로 읽는다. 읽는 동안 다음 디스크립터가
  // `acc_clr` 을 세울 수 없으므로(`rd_hold`) 연산과 쓰기가 겹치지
  // 못하고, 주기가 max(연산, 쓰기) 가 아니라 합이 된다. 그 비용은
  // 디스크립터당 OUT_BEATS 사이클이다 — Kt=64 에서 93%, Kt=4096 에서
  // 1.6% 다.
  //
  // 2 면 타일 안에서 누산기를 핑퐁한다. `C_LATCH` 에서 `snap` 을 한
  // 사이클 쳐서 방금 끝난 뱅크를 읽기 쪽으로 넘기고, 연산은 반대쪽
  // 뱅크에서 바로 다음 디스크립터를 시작한다. `rd_hold` 가 사라진다.
  // 비용은 타일당 누산기 한 벌의 플롭 (DIM² × ACC_W / 타일 수).
  parameter int ACC_BUFS = 1
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
  localparam int RDP_W    = $clog2(ACCS + 1);       // 순차 읽기 포인터
  // 실제 레인 수. 0 은 "쓰기 비트 폭에 맞춘다" 는 뜻이다.
  localparam int RDL      = (STREAM_RD == 0) ? 1
                          : (RD_LANES == 0)  ? WPB : RD_LANES;
  // 비트 하나를 채우는 데 걸리는 사이클.
  localparam int GSTEPS   = WPB / RDL;
  localparam int GS_W     = (GSTEPS > 1) ? $clog2(GSTEPS) : 1;
  // 실제 뱅크 수. FP32_OUT 경로는 acc_lat 으로 이미 겹치기를 얻으므로
  // 핑퐁이 필요 없고, 타일이 FP32_OUT + 핑퐁 조합을 금지한다.
  localparam int ABUF     = (STREAM_RD == 0) ? 1 : ACC_BUFS;
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
    if (ACC_BUFS != 1 && ACC_BUFS != 2)
      $fatal(1, "mxu_core: ACC_BUFS(%0d) 는 1 또는 2 여야 한다", ACC_BUFS);
    if (ACC_BUFS > 1 && STREAM_RD == 0)
      $fatal(1, "mxu_core: ACC_BUFS>1 은 STREAM_RD=1 에서만 의미가 있다 (FP32_OUT 은 acc_lat 을 쓴다)");
    if (STREAM_RD != 0 && (WPB % RDL) != 0)
      $fatal(1, "mxu_core: RD_LANES(%0d) 가 WPB(%0d) 를 나누지 않는다",
             RDL, WPB);
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

  // Kt 를 MAX_KT 로 자른 값. 전에는 함수였다.
  //
  // 함수 안의 지역 변수를 비동기 리셋이 있는 always_ff 에서 부르면
  // Yosys 가 그 지역 변수에 레지스터를 만들려 하고 "Multiple edge
  // sensitive events found for this signal" 로 멈춘다. 순수 식이므로
  // wire 하나로 두면 끝이고, 그래야 합성된다.
  wire [31:0] kt_clamped = (Kt > 32'(MAX_KT)) ? 32'(MAX_KT) : Kt;

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
      rd_len_r   <= LEN_W'(((kt_clamped + KPW - 1) / KPW) * RD_BEAT_B);
      wgt_len_r  <= LEN_W'(((kt_clamped + KPW - 1) / KPW) * RD_BEAT_B);
      Kt_bank[ld_bank]  <= kt_clamped;
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
  logic [RDL*32-1:0]       mxu_rd_fp32;

  genvar li;
  generate
    for (li = 0; li < DIM; li++) begin : G_LANES
      assign a_col[li] = act_kstep[16*li +: 16];
      assign b_row[li] = wgt_kstep[16*li +: 16];
    end
  endgenerate

  mxu_bf16_128x128 #(
    .DIM(DIM), .TILE_DIM(16), .TILES(DIM/16), .PAR(DIM/16),
    .FP32_OUT((STREAM_RD != 0) ? 0 : 1), .STREAM_RD(STREAM_RD),
    .RD_LANES(RDL), .ACC_BUFS(ABUF)
  ) u_mxu (
    .clk(clk), .rst_n(rst_n),
    .start(mxu_start), .acc_clr(mxu_acc_clr), .k_steps(16'(Kt_cm)),
    .a_col(a_col), .b_row(b_row), .data_valid(mxu_data_valid),
    .busy(mxu_busy), .done_pulse(mxu_done), .ready(mxu_ready),
    .acc_flat(acc_flat),
    .snap(cm_latch),
    .rd_idx(mxu_rd_idx), .rd_fp32(mxu_rd_fp32)
  );

  // ----------------------------
  // 순차 읽기 포인터 (STREAM_RD=1 일 때만 움직인다)
  // ----------------------------
  // 비트 하나에 WPB 개가 들어간다. 공유 변환기는 사이클당 하나를 내므로
  // 비트 하나를 모으는 데 WPB 사이클이 걸린다. 행 우선 출력이라 한 비트의
  // WPB 개는 같은 행의 연속된 열이고, 그건 같은 타일 안에 있다 — 타일을
  // 병렬로 읽어도 한 비트를 더 빨리 채울 수 없다.
  logic [RDP_W-1:0] rd_ptr;        // 다음에 읽을 누산기 (레인 0)
  logic [GS_W-1:0]  gstep;         // 비트 안에서 몇 번째 묶음인가
  logic [WR_DATA_W-1:0] gather;    // 모으는 중인 비트
  wire  gth_last = (gstep == GS_W'(GSTEPS-1));

  // 이번 사이클에 들어오는 RDL 개 워드 (X 는 0 으로 클립).
  logic [RDL*32-1:0] rd_chunk;
  // 그 묶음을 끼워 넣은 다음 상태. 완성되면 그대로 비트가 된다.
  logic [WR_DATA_W-1:0] gather_next;

  generate
    if (STREAM_RD != 0) begin : G_RD_PTR
      assign mxu_rd_idx = rd_ptr;
    end else begin : G_RD_TIE
      assign mxu_rd_idx = '0;   // 쓰지 않는다
    end
  endgenerate

  // ----------------------------
  // 누산기 래치
  // ----------------------------
  // STREAM_RD=1 이면 래치를 두지 않는다. ACCS×32 비트 플롭을 두는 순간
  // 변환기를 아껴 번 것을 플롭으로 되돌려주는 셈이기 때문이다. 대신
  // 누산기를 타일 안에 둔 채로 읽어 비트로 바로 모은다. 그래서 읽는
  // 동안 연산 FSM 이 다음 디스크립터를 시작할 수 없다 — 그게 비용이다.
  localparam int LAT_N = (STREAM_RD != 0) ? 1 : ACCS;
  logic [31:0] acc_lat [0:LAT_N-1];
  logic [63:0] wr_addr_r;

  wire cm_latch = (cst == C_LATCH) && (wst == W_IDLE);

  // STREAM_RD=1 에서 읽기가 진행 중인 동안 연산 FSM 을 세우는 신호.
  // 누산기를 핑퐁하면(ABUF=2) 읽히는 뱅크와 누산되는 뱅크가 다르므로
  // 세울 필요가 없다 — 그게 OUT_BEATS 사이클을 돌려받는 길이다.
  wire rd_hold = (STREAM_RD != 0) && (ABUF == 1) && (wst != W_IDLE);

  genvar ai;
  generate
    if (STREAM_RD == 0) begin : G_ACC_LAT_EN
      for (ai = 0; ai < ACCS; ai++) begin : G_ACC_LAT
        always_ff @(posedge clk or negedge rst_n) begin
          if (!rst_n)        acc_lat[ai] <= 32'd0;
          else if (cm_latch) acc_lat[ai] <= (Kt_cm != 0) ? acc_flat[ai] : 32'd0;
        end
      end
    end else begin : G_ACC_LAT_OFF
      always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) acc_lat[0] <= 32'd0;
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

  // 시뮬레이션에서 X 를 0 으로 클립한다. 하드웨어에는 X 가 없으므로
  // 합성에서는 그냥 통과다 — `===` 는 합성 구문이 아니고, 이 함수가
  // 데이터패스 안에 있어서 **`mxu_core` 전체가 합성되지 않았다**
  // ("Non-constant function call in constant expression"). 그래서
  // 이 설계의 클럭 주파수를 한 번도 측정할 수 없었다.
  function automatic logic [31:0] x32(input logic [31:0] v);
    begin
`ifdef COCOTB_SIM
      if ((^v === 1'bx)) x32 = 32'd0;
      else               x32 = v;
`else
      x32 = v;
`endif
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

  // 이번 사이클에 읽힌 RDL 개 워드를 X 클립해 묶는다.
  genvar gl;
  generate
    for (gl = 0; gl < RDL; gl++) begin : G_CHUNK
      assign rd_chunk[32*gl +: 32] = x32(mxu_rd_fp32[32*gl +: 32]);
    end
  endgenerate

  // 묶음을 비트의 gstep 번째 자리에 끼운 결과. gstep 이 마지막이면
  // 이것이 곧 완성된 비트다 — 연결 연산으로 손으로 자르지 않는다.
  always_comb begin
    gather_next = gather;
    gather_next[32*RDL*gstep +: 32*RDL] = rd_chunk;
  end

  generate
  if (STREAM_RD == 0) begin : G_WR_PARALLEL
    // 래치에서 비트를 통째로 떠낸다. 비트당 1 사이클.
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
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        rd_ptr <= '0;
        gstep  <= '0;
        gather <= '0;
      end
    end
  end else begin : G_WR_STREAM
    // 공유 변환기에서 사이클당 RDL 개를 받아 비트를 모은다. 비트
    // 하나에 GSTEPS = WPB/RDL 사이클이 걸린다. RDL == WPB 면
    // GSTEPS == 1 이라 **한 사이클에 비트가 완성된다** — 그때는
    // FP32_OUT=1 과 비트당 비용이 같다.
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n) begin
        out_beat   <= '0;
        wr_data_r  <= '0;
        wr_valid_r <= 1'b0;
        rd_ptr     <= '0;
        gstep      <= '0;
        gather     <= '0;
      end else if (wst == W_REQ) begin
        // 요청이 수락되기 전부터 첫 비트를 모아 둔다. rd_ptr 은
        // W_DONE 에서 0 으로 돌아간다.
        out_beat   <= '0;
        wr_valid_r <= 1'b0;
        if (rd_ptr < RDP_W'(WPB)) begin
          gather <= gather_next;
          rd_ptr <= rd_ptr + RDP_W'(RDL);
          gstep  <= gth_last ? '0 : gstep + 1'b1;
          if (gth_last) begin
            wr_data_r  <= gather_next;
            wr_valid_r <= 1'b1;
          end
        end else begin
          wr_valid_r <= 1'b1;   // 첫 비트는 이미 다 모았다
        end
      end else if (wst == W_SEND) begin
        if (wr_valid_r && wr_data_ready) begin
          if (out_beat == OUT_BEATS-1) begin
            wr_valid_r <= 1'b0;
          end else begin
            out_beat <= out_beat + 1'b1;
            if (GSTEPS == 1) begin
              // 비트가 한 사이클에 완성되는 구성이면, 수락되는 **그
              // 사이클에** 다음 비트를 집는다. rd_ptr 이 이미 다음
              // 묶음을 가리키고 변환기 출력은 조합이므로 가능하다.
              //
              // 이걸 안 하면 비트마다 valid 를 한 사이클 내려 모으는
              // 셈이 되어 **비트당 2 사이클**이 된다. 처음에 그렇게
              // 썼고, DIM=16/WR=128 에서 200 사이클/op 이 나왔다.
              gather     <= gather_next;
              wr_data_r  <= gather_next;
              rd_ptr     <= rd_ptr + RDP_W'(RDL);
              wr_valid_r <= 1'b1;
            end else begin
              wr_valid_r <= 1'b0;   // 다음 비트를 모으는 동안 내린다
            end
          end
        end else if (!wr_valid_r) begin
          // 다음 비트를 모은다: 사이클당 RDL 개.
          gather <= gather_next;
          rd_ptr <= rd_ptr + RDP_W'(RDL);
          gstep  <= gth_last ? '0 : gstep + 1'b1;
          if (gth_last) begin
            wr_data_r  <= gather_next;
            wr_valid_r <= 1'b1;
          end
        end
      end else begin
        wr_valid_r <= 1'b0;
        if (wst == W_DONE) begin
          rd_ptr <= '0;
          gstep  <= '0;
        end
      end
    end
  end
  endgenerate

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
        rd_left     <= LEN_W'(((kt_clamped + KPW - 1) / KPW) * RD_BEAT_B);
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
      // STREAM_RD=1 이면 누산기가 타일 안에 남아 있는 상태로 읽히는
      // 중이므로, 읽기가 끝날 때까지 다음 디스크립터를 시작할 수 없다.
      // C_CLR 이 `mxu_acc_clr` 을 세우면 읽는 중인 값을 지워 버린다.
      // STREAM_RD=0 이면 결과가 acc_lat 에 떠 있으므로 기다릴 필요가 없다.
      C_IDLE: if (bank_full[cm_bank] && (rd_hold == 1'b0)) cst_n = C_CLR;

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
        // (STREAM_RD=1 이면 래치가 없고, 쓰기 FSM 이 타일에서 직접
        //  읽으므로 역시 비어 있어야 한다.)
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
