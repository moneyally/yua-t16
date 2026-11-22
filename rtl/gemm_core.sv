`timescale 1ns/1ps
`default_nettype none

// gemm_core.sv — 16x16 MAC 타일 하나를 Kt 스텝 돌리는 GEMM 코어
//
// ── 3단 파이프라인 ────────────────────────────────────────────────
//
// 원래 FSM 은 페치 → 연산 → 쓰기가 한 줄로 직렬이었다. 페치 동안
// MAC 배열이 통째로 놀고, 연산 동안 DMA 읽기 채널이 논다. 세 단계는
// 서로 다른 자원을 쓰므로 (읽기 채널 / MAC 배열 / 쓰기 채널) 겹칠 수
// 있고, 그러면 디스크립터 주기가 세 단계의 "합" 이 아니라 "최대" 가
// 된다.
//
//   로드 FSM (lst)  : DMA 읽기 채널, SRAM 쓰기 포트
//   연산 FSM (cst)  : SRAM 읽기 포트, MAC 배열, 누산기
//   쓰기 FSM (wst)  : DMA 쓰기 채널
//
// 세 FSM 은 독립적으로 돌고 자원으로만 묶인다. 묶이는 지점은 둘뿐이다.
//
//   1) 피연산자 SRAM: 2 뱅크 이중 버퍼. 로드 FSM 이 뱅크 B 를 채우는
//      동안 연산 FSM 이 뱅크 A 를 읽는다. bank_full 로 주고받는다.
//   2) acc_out_lat: 단일 버퍼. 연산 FSM 은 C_LATCH 에서 쓰기 FSM 이
//      비기를 기다린다.
//
// Kt=64 / RD_DATA_W=256 기준 각 단계의 사이클:
//   로드  1 + 32 + 1 + 32 = 66
//   연산  1 + 64 + 1 + 1  = 67
//   쓰기  1 + 64 + 1      = 66
// 주기는 67 이고, 직렬일 때의 199 와 비교된다.
//
// ── Weight-stationary ────────────────────────────────────────────
//
// 디스크립터마다 ACT 와 WGT 를 둘 다 읽으면 DRAM 트래픽의 절반이
// 가중치 재읽기다. LLM 추론에서 한 레이어의 가중치는 그 레이어를 지나는
// 모든 활성값 타일에 대해 똑같다 — 한 번 읽으면 된다.
//
// `wgt_keep` 이 서면 로드 FSM 이 WGT 단계를 건너뛰고 SRAM 에 남아 있는
// 가중치를 그대로 쓴다. 가중치 타일 하나를 N 개 디스크립터가 재사용하면
// 읽기 트래픽이 (N+1)/(2N) 로 줄고, N 이 크면 **절반**이 된다.
//
// 가중치 SRAM 도 2 뱅크지만 **포인터가 실제 적재 때만 전진한다.**
// 건너뛴 디스크립터는 직전에 적재된 뱅크를 가리킨다. 활성값 뱅크가
// 2개라 로드 FSM 은 연산 FSM 보다 최대 한 디스크립터 앞설 수 있고,
// 그 조건에서 "적재 중인 가중치 뱅크 != 연산 중인 가중치 뱅크" 가
// 항상 성립한다 (아래 증명). 그래서 드레인이 필요 없다 — wgt_keep=0
// 경로도 완전히 파이프라인된 채로 남는다.
//
// Eyeriss (Chen/Emer/Sze, ISCA 2016) 의 weight-stationary 가 같은
// 원리다. 그 논문은 AlexNet DRAM 접근을 2,896M → 61M 으로 줄였다.
// spec/memory-bandwidth-analysis.md 참조.
module gemm_core #(
  parameter int MAX_KT = 256,

  // DMA 읽기 데이터 폭. 128, 256 또는 512.
  //
  // 한 K-step 은 INT8 16개 = 128비트다. 128 이면 비트 하나가 K-step
  // 하나를 나르고, 256 이면 두 개를 나른다. 전송 바이트 수는 같고
  // 비트 수가 절반이 되므로 읽기 채널 병목이 완화된다.
  //
  // 이중 버퍼링 뒤에는 이 값이 주기를 결정한다. 128 이면 로드가
  // 1+Kt+1+Kt = 2Kt+2 사이클로 연산(Kt+3)의 두 배라 로드가 묶는 쪽이
  // 되고, 256 이면 Kt+2 로 연산과 비슷해져 연산이 묶는 쪽이 된다.
  //
  // 쓰기 채널은 건드리지 않는다 (독립 채널이고 병목이 아니다).
  parameter int RD_DATA_W = 128,

  // 피연산자 스테이징 뱅크 수. 2의 거듭제곱 (2 또는 4).
  //
  // 2 면 로드와 연산이 서로를 기다린다. 로드가 뱅크 A 를 채우고 나면
  // 연산이 뱅크 B 를 비울 때까지 할 일이 없고, 그 반대도 같다. 측정하면
  // 세 단계가 전부 76~80% 만 바쁘고 아무도 포화되지 않는다 — 대역폭이
  // 아니라 뱅크 깊이가 묶는 쪽이다.
  //
  // 4 면 로드가 두 디스크립터만큼 앞서갈 수 있어 핸드셰이크 지연이
  // 흡수된다. 값은 SRAM 깊이만 늘린다 (플립플롭이 아니라 BRAM).
  // spec/memory-bandwidth-analysis.md 참조.
  parameter int BANKS = 2,

  // DMA 쓰기 데이터 폭. 128, 256 또는 512.
  //
  // 디스크립터당 출력은 타일 크기의 제곱(N²)이고 연산은 Kt 사이클이다.
  // N=16/Kt=64 에서는 쓰기 64 비트 vs 연산 64 사이클로 균형이 맞지만,
  // 타일을 키우면 출력이 N² 로 커져 쓰기가 묶는 쪽이 된다.
  // 이 파라미터로 그쪽도 맞춘다.
  parameter int WR_DATA_W = 128,

  // MAC 타일 한 변의 길이. MAC 은 N², 피연산자 바이트는 N 이므로
  // 산술 강도가 N 에 비례한다.
  //
  // N=16 / RD=256 / WS 에서 로드 채널이 45.5% 만 바빠 2.19× 여유가
  // 있었다. N=32 는 MAC 4배, 피연산자 2배이므로 그 여유를 정확히
  // 소진한다. 다만 출력이 N² 로 커지므로 WR_DATA_W 도 맞춰야 한다.
  //
  // 제약: RD_DATA_W 가 한 K-step (N 바이트) 의 배수여야 한다.
  // 즉 N=32 는 RD_DATA_W >= 256 을 요구한다.
  parameter int TILE_N = 16
)(
  input  logic clk,
  input  logic rst_n,

  input  logic        cmd_valid,
  output logic        cmd_ready,
  input  logic [63:0] act_addr,
  input  logic [63:0] wgt_addr,
  input  logic [63:0] out_addr,
  input  logic [31:0] Kt,
  // 1 이면 WGT 페치를 건너뛰고 SRAM 에 상주한 가중치를 쓴다.
  // 가중치를 한 번도 적재하지 않았으면 무시된다 (아래 wgt_valid_r).
  input  logic        wgt_keep,
  // 1 이면 이 디스크립터의 가중치가 INT4 로 패킹돼 있다.
  // K-step 하나가 N 니블 = N/2 바이트이므로 가중치 트래픽이 절반이 된다.
  // MAC 배열은 그대로 8비트를 받고, 니블을 부호확장해 넣는다.
  // 포맷은 **상주 가중치에 딸린 속성**이다 — 건너뛴 디스크립터는 직전에
  // 적재된 가중치의 포맷을 쓴다.
  input  logic        wgt_int4,
  // ── K 분할 누산 ─────────────────────────────────────────────────
  // 축소 차원은 디스크립터당 MAX_KT 로 제한된다. 실제 LLM 레이어는
  // K=4096 이므로 여러 디스크립터로 쪼개 누산해야 한다.
  //
  //   acc_keep=0, no_wb=1   첫 조각 (누산기 클리어, 쓰기 없음)
  //   acc_keep=1, no_wb=1   중간 조각
  //   acc_keep=1, no_wb=0   마지막 조각 (누산 이어받고 쓰기)
  //
  // 조각이 하나뿐이면 acc_keep=0, no_wb=0 이다 (기존 동작).
  input  logic        acc_keep,   // 누산기를 지우지 않는다
  input  logic        no_wb,      // 결과를 쓰지 않는다

  output logic        done,
  // 상주 가중치를 실제로 쓴 디스크립터 수 / 적재 횟수 (호스트 계측용)
  output logic [31:0] perf_wgt_reuse,
  output logic [31:0] perf_wgt_load,

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

  output logic         busy
);

  // ----------------------------
  // 치수
  // ----------------------------
  localparam int ADDR_W    = $clog2(MAX_KT);

  localparam int N          = TILE_N;
  localparam int ACCS       = N * N;                  // 누산기 수
  localparam int KSTEP_W    = N * 8;                  // K-step 하나의 비트 수
  // 쓰기 비트 하나에 실리는 누산기 개수와 디스크립터당 비트 수
  localparam int WPB        = WR_DATA_W / 32;         // 비트당 누산기
  localparam int OUT_BEATS  = ACCS / WPB;
  localparam int OUT_BYTES  = ACCS * 4;
  localparam int BEAT_W     = $clog2(OUT_BEATS + 1);
  localparam int WR_BEAT_B  = WR_DATA_W / 8;          // 비트당 바이트
  localparam int RD_BEAT_B  = RD_DATA_W / 8;

  // ── DMA 요청 길이의 한계 ────────────────────────────────────────
  //
  // rd_req_len_bytes 는 16비트다. 256비트 읽기에서 한 요청은 최대
  // 65,504바이트(비트 2,047개)까지다. Kt 가 그보다 깊으면 길이가
  // **조용히 잘린다** — Kt=2048 은 0 이 되어 멈추고, Kt=11008 은
  // 0x6000(24,576B)이라는 그럴듯한 짧은 요청이 되어 틀린 결과를 낸다.
  //
  // 실제 LLM 레이어는 K=4096 / 11008 이라 이 한계에 바로 걸린다.
  // 그래서 긴 전송을 여러 요청으로 쪼갠다 — 실제 DMA 엔진도 PCIe MPS 와
  // AXI 4KB 경계 때문에 같은 일을 한다.
  //
  // 길이 카운터는 32비트로 둔다. MAX_KT * RD_BEAT_B 가 16비트를 넘으므로
  // 16비트로 세면 같은 잘림이 카운터에서 되살아난다.
  localparam int LEN_W      = 32;
  localparam int RD_REQ_MAX = (65535 / RD_BEAT_B) * RD_BEAT_B;

  // 워드 하나에 들어가는 K-step 수
  localparam int KPW   = RD_DATA_W / KSTEP_W;
  localparam int KPW_W = (KPW > 1) ? $clog2(KPW) : 1;

  // INT4 가중치: K-step 하나가 N 니블 = KSTEP_W/2 비트다. 따라서 워드에
  // 두 배로 들어간다. KPW >= 1 이므로 W4PW >= 2 가 항상 성립하고,
  // 주소/부분워드 분할이 비트 슬라이싱만으로 끝난다.
  localparam int NIB_W   = KSTEP_W / 2;      // INT4 K-step 하나의 비트 수
  localparam int W4PW    = KPW * 2;
  localparam int W4PW_W  = $clog2(W4PW);

  // 뱅크 하나가 디스크립터 하나의 피연산자를 담는다.
  localparam int BANK_W      = $clog2(BANKS);              // 뱅크 인덱스 폭
  localparam int BANK_DEPTH  = (MAX_KT + KPW - 1) / KPW;   // 뱅크당 워드 수
  localparam int BADDR_W     = $clog2(BANK_DEPTH);         // 뱅크 내 주소
  localparam int SRAM_DEPTH  = BANK_DEPTH * BANKS;
  // 뱅크 주소를 `{bank, idx}` 로 이어 붙이면 BANK_DEPTH 가 2의 거듭제곱
  // 이어야 한다. 실제 LLM 레이어 깊이는 그렇지 않다 (K=11008).
  // 11008 이면 뱅크 1 이 16,384..27,391 을 가리켜 SRAM 범위 0..22,015 를
  // 넘는다 — 첫 디스크립터는 뱅크 0 만 쓰므로 통과하고, **두 번째부터
  // 조용히 틀린다.** 그래서 뱅크 기점을 곱셈으로 계산한다.
  localparam int SADDR_W     = $clog2(SRAM_DEPTH);

  initial begin
    if (RD_DATA_W != 128 && RD_DATA_W != 256 && RD_DATA_W != 512)
      $fatal(1, "gemm_core: RD_DATA_W must be 128, 256 or 512 (got %0d)", RD_DATA_W);
    // 뱅크 인덱스를 주소 상위에 그냥 붙이므로 2의 거듭제곱이어야 한다.
    if (BANKS < 2 || (BANKS & (BANKS - 1)) != 0)
      $fatal(1, "gemm_core: BANKS must be a power of two >= 2 (got %0d)", BANKS);
    if (WR_DATA_W != 128 && WR_DATA_W != 256 && WR_DATA_W != 512)
      $fatal(1, "gemm_core: WR_DATA_W must be 128, 256 or 512 (got %0d)", WR_DATA_W);
    // K-step 하나(N 바이트)가 읽기 비트에 정수 개 들어가야 한다.
    if (KPW < 1 || KPW * KSTEP_W != RD_DATA_W)
      $fatal(1, "gemm_core: RD_DATA_W (%0d) must be a positive multiple of TILE_N*8 (%0d)",
             RD_DATA_W, KSTEP_W);
    // 누산기 수가 쓰기 비트로 정수 개 나뉘어야 한다.
    // INT4 니블이 바이트 경계에 맞아야 한다 (N 이 짝수).
    if (N % 2 != 0)
      $fatal(1, "gemm_core: TILE_N must be even for INT4 weights (got %0d)", N);
    // 쓰기 요청 길이도 16비트다. TILE_N=128 이면 N²×4 = 65,536 으로 넘친다.
    if (OUT_BYTES > 65535)
      $fatal(1, "gemm_core: OUT_BYTES %0d exceeds the 16-bit wr_req_len_bytes field",
             OUT_BYTES);
    if (OUT_BEATS * WPB != ACCS)
      $fatal(1, "gemm_core: TILE_N^2 (%0d) must be a multiple of WR_DATA_W/32 (%0d)",
             ACCS, WPB);
  end

  // 뱅크 b 의 idx 번째 워드. BANK_DEPTH 가 2의 거듭제곱이 아니어도 맞는다.
  // BANKS 가 작은 2의 거듭제곱이므로 합성에서는 덧셈 몇 개가 된다.
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
  // FSM 선언
  // ----------------------------
  // 로드: DMA 읽기 → 피연산자 SRAM 뱅크
  typedef enum logic [2:0] {
    L_IDLE,
    L_ACT_REQ,
    L_ACT_RECV,
    L_WGT_REQ,
    L_WGT_RECV
  } lstate_t;

  // 연산: 뱅크 → MAC 배열 → acc_out_lat
  typedef enum logic [2:0] {
    C_IDLE,
    C_CLR,
    C_COMPUTE,
    C_POST,     // drain 1cycle (마지막 MAC 이 진입 에지에서 일어난다)
    C_LATCH     // acc_out_mac -> acc_out_lat, 쓰기 FSM 기동
  } cstate_t;

  // 쓰기: acc_out_lat → DMA 쓰기 채널
  typedef enum logic [1:0] { W_IDLE, W_REQ, W_SEND, W_DONE } wstate_t;

  lstate_t lst, lst_n;
  cstate_t cst, cst_n;
  wstate_t wst, wst_n;

  logic ld_mark;        // 로드 완료: 뱅크를 연산 FSM 에 넘긴다 (1 사이클)
  logic wgt_load_done;  // 가중치 수신의 마지막 비트 (1 사이클)
  logic cm_free;     // 연산 완료: 뱅크를 로드 FSM 에 돌려준다 (1 사이클)
  logic wr_start;    // 쓰기 FSM 기동 (1 사이클)

  // ----------------------------
  // 뱅크 소유권
  // ----------------------------
  logic [BANKS-1:0]  bank_full;  // 로드 완료, 연산 대기
  logic [BANK_W-1:0] ld_bank;    // 로드 FSM 이 채우는 뱅크
  logic [BANK_W-1:0] cm_bank;    // 연산 FSM 이 읽는 뱅크

  // 디스크립터별 값 중 단계를 건너 쓰이는 것만 뱅크마다 둔다.
  //   Kt      : 연산 FSM 이 쓴다
  //   out_addr: 쓰기 FSM 이 쓴다
  // act/wgt 주소와 읽기 길이는 로드 FSM 안에서만 쓰이고, 로드 FSM 은
  // 한 번에 하나만 처리하므로 단일 레지스터로 족하다.
  logic [31:0] Kt_bank  [0:BANKS-1];
  logic [63:0] out_bank [0:BANKS-1];
  logic        ak_bank  [0:BANKS-1];   // acc_keep
  logic        nw_bank  [0:BANKS-1];   // no_wb

  wire [31:0] Kt_cm  = Kt_bank[cm_bank];
  wire [63:0] out_cm = out_bank[cm_bank];
  wire        ak_cm  = ak_bank[cm_bank];
  wire        nw_cm  = nw_bank[cm_bank];

  logic [63:0] act_base_r, wgt_base_r;
  // 전체 전송 길이. 요청 하나의 길이가 아니다 (위 RD_REQ_MAX 참조).
  logic [LEN_W-1:0] rd_len_r;    // 활성값 (INT8)
  logic [LEN_W-1:0] wgt_len_r;   // 가중치 (INT8 또는 INT4)

  // ── Weight-stationary 상태 ──────────────────────────────────────
  // 가중치를 한 번이라도 적재했는가. 적재 전에는 wgt_keep 을 무시해야
  // 한다 — 그러지 않으면 0 으로 채워진 SRAM 으로 계산한다.
  logic wgt_valid_r;
  // 이 디스크립터가 실제로 가중치를 건너뛰는가.
  wire  wgt_skip = wgt_keep & wgt_valid_r;
  // ── 가중치 뱅크 포인터 ──────────────────────────────────────────
  //
  //   wld_bank      로드 FSM 이 적재할 가중치 뱅크
  //   wres_bank     가장 최근에 적재가 끝난 뱅크 (= 상주 가중치)
  //   wgt_bank_of[] 대기 중인 각 디스크립터가 읽어야 할 가중치 뱅크
  //
  // 안전성: 활성값 뱅크가 2개이므로 로드 FSM 은 연산 FSM 보다 최대
  // 한 디스크립터 앞선다. 디스크립터 N 을 연산하는 동안 로드가 N+1 을
  // 적재한다고 하자.
  //   - N 이 적재했다면 wgt_bank_of[N]=w, 그 적재가 끝나며 wld=~w 이다.
  //   - N 이 건너뛰었다면 wgt_bank_of[N]=wres=w 이고 wld 는 그대로 ~w 다.
  // 두 경우 모두 N+1 이 적재할 뱅크(~w)가 N 이 읽는 뱅크(w)와 다르다. ∎
  logic [BANK_W-1:0] wld_bank, wres_bank;
  logic [BANK_W-1:0] wgt_bank_of [0:BANKS-1];
  wire  [BANK_W-1:0] wgt_bank_cm = wgt_bank_of[cm_bank];

  // INT4 여부는 상주 가중치의 속성이므로 가중치 뱅크마다 보관한다.
  // 건너뛴 디스크립터는 wgt_bank_of 가 가리키는 뱅크의 포맷을 쓴다.
  logic w4_of_wbank [0:BANKS-1];
  wire  w4_cm = w4_of_wbank[wgt_bank_cm];
  // 로드 FSM 이 지금 적재 중인 가중치의 포맷
  logic w4_ld_r;

  // 로드 FSM 은 자기 뱅크가 비어 있을 때만 명령을 받는다.
  // 가중치 교체는 드레인을 요구하지 않는다 (위 증명).
  assign cmd_ready = (lst == L_IDLE) && !bank_full[ld_bank];
  wire   cmd_accept = cmd_valid & cmd_ready;

  // 이 디스크립터가 가중치를 건너뛰는지 래치한다 (로드 FSM 이 쓴다).
  logic wgt_skip_r;

  // 호스트 계측
  logic [31:0] wgt_reuse_r, wgt_load_r;
  assign perf_wgt_reuse = wgt_reuse_r;
  assign perf_wgt_load  = wgt_load_r;

  integer wi;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wgt_valid_r <= 1'b0;
      wgt_skip_r  <= 1'b0;
      wgt_reuse_r <= 32'd0;
      wgt_load_r  <= 32'd0;
      wld_bank    <= '0;
      wres_bank   <= '0;
      w4_ld_r     <= 1'b0;
      for (wi = 0; wi < BANKS; wi = wi + 1) begin
        wgt_bank_of[wi]  <= '0;
        w4_of_wbank[wi]  <= 1'b0;
      end
    end else if (cmd_accept) begin
      wgt_skip_r <= wgt_skip;
      if (wgt_skip) begin
        // 상주 가중치를 그대로 쓴다. 포인터는 움직이지 않는다.
        wgt_bank_of[ld_bank] <= wres_bank;
        wgt_reuse_r          <= wgt_reuse_r + 1;
      end else begin
        // 이 디스크립터가 wld_bank 에 적재한다. 포인터는 여기서 올리지
        // 않는다 — 올리면 로드 FSM 이 다음 뱅크에 써 버린다.
        // wgt_valid_r 은 지금 세워도 된다. 이것을 보고 건너뛸 수 있는
        // 디스크립터는 lst==L_IDLE 을 기다리므로 이 적재가 끝난 뒤다.
        wgt_bank_of[ld_bank] <= wld_bank;
        w4_of_wbank[wld_bank] <= wgt_int4;
        w4_ld_r              <= wgt_int4;
        wgt_valid_r          <= 1'b1;
        wgt_load_r           <= wgt_load_r + 1;
      end
    end
    // 가중치 적재가 실제로 끝난 시점에 포인터를 전진시킨다.
    if (wgt_load_done) begin
      wres_bank <= wld_bank;
      wld_bank  <= wld_bank + 1'b1;
    end
  end

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
        ak_bank[bi]  <= 1'b0;
        nw_bank[bi]  <= 1'b0;
      end
    end else if (cmd_accept) begin
      act_base_r <= act_addr;
      wgt_base_r <= wgt_addr;
      // 읽기 길이는 비트 크기의 배수여야 한다.
      // RD_DATA_W=256 이면 비트 하나가 K-step 두 개를 나르므로, Kt 가
      // 홀수면 Kt*16 바이트가 32 의 배수가 아니다. 워드 단위로 올림해
      // 요청한다. 남는 절반은 뱅크 마지막 워드의 상위에 들어가고
      // 연산은 k < Kt 만 읽으므로 사용되지 않는다.
      rd_len_r   <= LEN_W'(((clamp_kt(Kt) + KPW - 1) / KPW) * RD_BEAT_B);
      // INT4 면 워드에 K-step 이 두 배 들어가므로 요청 길이가 절반이 된다.
      wgt_len_r  <= wgt_int4
                  ? LEN_W'(((clamp_kt(Kt) + W4PW - 1) / W4PW) * RD_BEAT_B)
                  : LEN_W'(((clamp_kt(Kt) + KPW  - 1) / KPW ) * RD_BEAT_B);

      Kt_bank[ld_bank]  <= clamp_kt(Kt);
      out_bank[ld_bank] <= out_addr;
      ak_bank[ld_bank]  <= acc_keep;
      nw_bank[ld_bank]  <= no_wb;
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      bank_full <= '0;
      ld_bank   <= 1'b0;
      cm_bank   <= 1'b0;
    end else begin
      // ld_mark 와 cm_free 가 같은 사이클에 겹칠 수 있지만, 같은 뱅크에는
      // 겹치지 않는다. ld_mark 는 bank_full=0 인 뱅크에서만 (cmd_ready 조건),
      // cm_free 는 bank_full=1 인 뱅크에서만 일어나므로 두 뱅크는 다르다.
      // 뱅크는 순환 사용한다. BANKS 가 2의 거듭제곱이라 자연히 감싼다.
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

`ifdef COCOTB_SIM
  // 위 논증이 틀리면 bank_full 의 두 대입이 같은 비트에서 충돌한다.
  // 조용히 틀리는 대신 터뜨린다.
  always_ff @(posedge clk) begin
    if (rst_n && ld_mark && cm_free && (ld_bank == cm_bank))
      $fatal(1, "gemm_core: bank %0d set and cleared in the same cycle", ld_bank);
  end
`endif

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
        + ((rd_data_valid && rd_data_ready) ? (RD_DATA_W/8) : 0)
        + ((wr_data_valid && wr_data_ready) ? WR_BEAT_B : 0);
    end
  end

  // ----------------------------
  // 피연산자 SRAM (1W1R, 2 뱅크)
  // ----------------------------
  logic act_we, wgt_we, act_re, wgt_re;
  logic [SADDR_W-1:0] act_waddr;
  logic [SADDR_W-1:0] act_raddr;
  logic [SADDR_W-1:0] wgt_waddr;
  logic [SADDR_W-1:0] wgt_raddr;
  logic [RD_DATA_W-1:0] act_word, wgt_word;   // SRAM 출력 (워드 전체)
  logic [KSTEP_W-1:0] act_rdata, wgt_rdata;   // 선택된 K-step 하나

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

  // 수신 비트 = 뱅크 내 워드 인덱스. 상위에 뱅크 비트를 붙인다.
  // 현재 단계(ACT 또는 WGT)에서 남은 바이트와 다음 요청 주소.
  // 요청 하나가 RD_REQ_MAX 를 넘지 못하므로 여러 번 나눠 낸다.
  logic [LEN_W-1:0] rd_left;
  logic [63:0]      rd_cur_addr;
  // 이번 요청의 길이 (수락 시 남은 양에서 빼기 위해 조합으로 계산)
  wire  [LEN_W-1:0] rd_this_len =
          (rd_left > LEN_W'(RD_REQ_MAX)) ? LEN_W'(RD_REQ_MAX) : rd_left;
  // 이 단계의 첫 요청인가 (load_idx 리셋 시점)
  logic phase_begin;

  logic [BADDR_W-1:0] load_idx;
  assign act_waddr = bank_addr(ld_bank,  load_idx);
  assign wgt_waddr = bank_addr(wld_bank, load_idx);

  // ✅ hold read address at "last issued" k to avoid reading past Kt
  logic [ADDR_W-1:0] rd_k_r;
  // ✅ IMPORTANT: SRAM is sync-read. If we update rd_k_r at posedge, the same
  // posedge read would still see OLD rd_k_r. So, when issuing a new k, drive
  // SRAM raddr directly from issue_k_addr in that cycle.
  logic [ADDR_W-1:0] rd_k_mux;

  // K 인덱스를 (뱅크 내 워드 주소, 워드 내 위치) 로 나눈다.
  // SRAM 은 동기 읽기라 rdata 가 한 사이클 뒤에 나온다. 따라서 슬라이스에
  // 쓸 위치도 주소를 낸 그 사이클의 값을 한 사이클 지연시켜 써야 한다.
  logic [KPW_W-1:0]  rd_half, rd_half_q;      // INT8 부분워드 위치
  logic [W4PW_W-1:0] w4_sub,  w4_sub_q;       // INT4 부분워드 위치

  logic [BADDR_W-1:0]  wgt_wa8;               // INT8 가중치 워드 주소
  logic [KSTEP_W-1:0]  wgt_rdata_i8;          // INT8 경로 결과
  logic [KSTEP_W-1:0]  wgt_rdata_i4;          // INT4 경로 결과 (부호확장 후)

  generate
    if (KPW == 1) begin : G_W128
      assign act_raddr    = bank_addr(cm_bank, rd_k_mux[BADDR_W-1:0]);
      assign wgt_wa8      = rd_k_mux[BADDR_W-1:0];
      assign rd_half      = '0;
      assign act_rdata    = act_word;
      assign wgt_rdata_i8 = wgt_word;
    end else begin : G_WIDE
      assign act_raddr    = bank_addr(cm_bank, rd_k_mux[KPW_W +: BADDR_W]);
      assign wgt_wa8      = rd_k_mux[KPW_W +: BADDR_W];
      assign rd_half      = rd_k_mux[KPW_W-1:0];
      assign act_rdata    = act_word[rd_half_q*KSTEP_W +: KSTEP_W];
      assign wgt_rdata_i8 = wgt_word[rd_half_q*KSTEP_W +: KSTEP_W];
    end
  endgenerate

  // INT4 경로. W4PW >= 2 가 항상 성립하므로 분기가 필요 없다.
  assign w4_sub = rd_k_mux[W4PW_W-1:0];

  // 가중치 주소는 포맷에 따라 다르다 — INT4 면 K-step 두 개가 한 워드에
  // 더 들어가므로 K 인덱스를 한 비트 더 오른쪽으로 민다.
  //
  // 부분선택 `rd_k_mux[W4PW_W +: BADDR_W]` 로 쓰면 범위를 넘는다:
  // N=16/RD=128 에서 rd_k_mux 는 8비트인데 BADDR_W 도 8 이라 [8:1] 이
  // 된다. 시프트 후 하위 BADDR_W 비트를 취하면 폭에 무관하게 맞는다.
  // Kt <= MAX_KT 이므로 k/W4PW 는 BANK_DEPTH 를 넘지 않는다.
  wire [ADDR_W-1:0]   w4_word_idx = rd_k_mux >> W4PW_W;
  wire [BADDR_W-1:0]  wgt_wa4     = w4_word_idx[BADDR_W-1:0];

  assign wgt_raddr = bank_addr(wgt_bank_cm, w4_cm ? wgt_wa4 : wgt_wa8);

  // 니블 N 개를 꺼내 8비트로 부호확장한다. INT4 는 2의 보수 [-8, 7] 이다.
  wire [NIB_W-1:0] w4_nibbles = wgt_word[w4_sub_q*NIB_W +: NIB_W];

  genvar nb;
  generate
    for (nb = 0; nb < N; nb++) begin : NIB_SEXT
      assign wgt_rdata_i4[8*nb +: 8] =
        {{4{w4_nibbles[4*nb + 3]}}, w4_nibbles[4*nb +: 4]};
    end
  endgenerate

  assign wgt_rdata = w4_cm ? wgt_rdata_i4 : wgt_rdata_i8;

  // SRAM 이 re=0 일 때 rdata 를 유지하므로, 슬라이스 위치도 같이 유지한다.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_half_q <= '0;
      w4_sub_q  <= '0;
    end else if (act_re) begin
      rd_half_q <= rd_half;
      w4_sub_q  <= w4_sub;
    end
  end

  // ----------------------------
  // MAC array
  // ----------------------------
  logic mac_acc_clr, mac_en;
  logic act_issue, wgt_issue;        // issue_fire exposed (debug)
  logic act_issue_d, wgt_issue_d;    // ✅ 1-cycle delayed valid (SRAM->MAC align)
  logic issue_fire;

  logic signed [7:0]  a_row [0:N-1];
  logic signed [7:0]  b_col [0:N-1];

// MAC internal result as packed flat vector (Icarus-safe — no unpacked array output port)
logic [32*ACCS-1:0] acc_out_mac_flat;

// Unpacked views of MAC result and latched outputs for cocotb/FSM access
logic signed [31:0] acc_out_mac [0:ACCS-1];  // continuous unpack from acc_out_mac_flat
logic signed [31:0] acc_out_flat [0:ACCS-1]; // latch of mac result (reset-safe, Icarus-safe)
logic signed [31:0] acc_out_lat  [0:ACCS-1]; // write-out latch

  genvar i;
  generate
    for (i=0;i<N;i++) begin : ROW_UNPACK
      assign a_row[i] = act_rdata[8*i +: 8];
      assign b_col[i] = wgt_rdata[8*i +: 8];
    end
  endgenerate

mac_array #(.N(N)) u_mac (
  .clk(clk),
  .rst_n(rst_n),
  .en(mac_en),
  .acc_clr(mac_acc_clr),
  .a_row(a_row),
  .b_col(b_col),
  .acc_out_flat(acc_out_mac_flat)
);

// Unpack acc_out_mac_flat -> acc_out_mac[] via continuous assigns
genvar ui;
generate
  for (ui = 0; ui < ACCS; ui++) begin : UNPACK_MAC
    assign acc_out_mac[ui] = acc_out_mac_flat[ui*32 +: 32];
  end
endgenerate

  // 연산 FSM 이 실제로 래치하고 넘어가는 사이클. C_LATCH 에서 쓰기 FSM 을
  // 기다리며 멈춰 있을 수 있으므로 상태만으로는 안 된다.
  // no_wb 면 래치하지 않는다 — 누산기에 남겨 두고 다음 조각이 이어받는다.
  wire cm_latch = (cst == C_LATCH) && !nw_cm && (wst == W_IDLE);
  // 중간 조각이 통과하는 사이클 (쓰기 없이 완료로 센다).
  wire cm_pass  = (cst == C_LATCH) &&  nw_cm;

// --------------------------------------------------
// cocotb / Icarus-safe observable accumulators
// Use generate so each element gets its own always_ff — avoids Icarus
// "constant selects in always_* processes" bug with loop-indexed arrays.
// --------------------------------------------------
genvar zz;
generate
  for (zz = 0; zz < ACCS; zz++) begin : GEN_ACC_FLAT
    always_ff @(posedge clk or negedge rst_n) begin
      if (!rst_n)
        acc_out_flat[zz] <= 32'sd0;
      else if (cst == C_CLR)
        acc_out_flat[zz] <= 32'sd0;
      else if (cm_latch)
        acc_out_flat[zz] <= acc_out_mac[zz];
    end
  end
endgenerate

genvar kk;
generate
  for (kk = 0; kk < ACCS; kk++) begin : GEN_ACC_LAT
    always_ff @(posedge clk or negedge rst_n) begin
      // C_CLR 에서 지우지 않는다. 쓰기 FSM 이 분리되면서 다음 디스크립터가
      // C_CLR 에 도달할 때 아직 전송 중인 acc_out_lat 을 지워버리는
      // 해저드가 생기기 때문이다. Kt=0 이면 0 을 래치하므로 동작은 같다.
      if (!rst_n)
        acc_out_lat[kk] <= 32'sd0;
      else if (cm_latch)
        acc_out_lat[kk] <= (Kt_cm != 0) ? acc_out_mac[kk] : 32'sd0;
    end
  end
endgenerate

  // 쓰기 주소도 래치 시점에 떠 둔다. cm_free 가 같은 에지에 cm_bank 를
  // 돌리므로 out_cm 은 그 다음 사이클에 이미 다음 뱅크를 가리킨다.
  logic [63:0] wr_addr_r;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)        wr_addr_r <= 64'd0;
    else if (cm_latch) wr_addr_r <= out_cm;
  end

  // ----------------------------
  // Compute issue counter
  // ----------------------------
  logic [31:0] issue_k_addr;
  logic [31:0] issue_k_mac;
  // Icarus-safe truncation: continuous assign outside always_* avoids
  // "constant selects in always_* processes" warning on part-select.
  logic [ADDR_W-1:0] issue_k_addr_trunc;
  assign issue_k_addr_trunc = issue_k_addr[ADDR_W-1:0];

  // 피연산자는 발행 1사이클 뒤에 SRAM 출력에 나온다. MAC 은 그 사이클에
  // 먹어야 한다.
  //
  // 원래는 여기에 mac_en_d1/d2 두 단을 더 거쳐 발행 3사이클 뒤였다.
  // SRAM 출력은 다음 읽기 전까지 "유지" 되므로, 연속 발행하면 피연산자는
  // t+1 부터 매 사이클 넘어가는데 MAC 은 t+3 부터 먹기 시작한다.
  // 결과적으로 앞쪽 K-step 을 놓치고 마지막 것을 중복 누산한다.
  // Kt=1 에서는 피연산자가 하나뿐이라 우연히 맞아서 드러나지 않았다
  // (test_gemm_top.py 는 Kt=1 만 돌린다).
  assign mac_en = act_issue_d & wgt_issue_d;

always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    issue_k_addr  <= 0;
    issue_k_mac   <= 0;
    act_issue_d   <= 0;
    wgt_issue_d   <= 0;
    rd_k_r        <= '0;
  end else begin
    // 1-cycle delay: SRAM data valid
    act_issue_d <= issue_fire;
    wgt_issue_d <= issue_fire;

    if (cst == C_CLR) begin
      issue_k_addr <= 0;
      issue_k_mac  <= 0;
      rd_k_r       <= '0;
    end else begin
      if (issue_fire) begin
        rd_k_r       <= issue_k_addr[ADDR_W-1:0];
        issue_k_addr <= issue_k_addr + 1;
      end

      if (mac_en)
        issue_k_mac <= issue_k_mac + 1;
    end
  end
end

  // ----------------------------
  // Write datapath (X-safe + small)
  // ----------------------------
  logic [BEAT_W-1:0]      out_beat;
  logic [WR_DATA_W-1:0]   wr_data_r;
  logic                   wr_valid_r;

  function automatic logic [31:0] x32(input logic signed [31:0] v);
    logic [31:0] t;
    begin
      t = v;
// FIX v0.2: X는 항상 0으로 클리핑
      if ((^t === 1'bx)) x32 = 32'd0;
      else              x32 = t;

    end
  endfunction

  // 누산기 WPB 개를 한 비트로 묶는다. 하위 워드가 낮은 인덱스다
  // (리틀엔디안 메모리 순서와 맞춘다).
  function automatic logic [WR_DATA_W-1:0] pack_beat(input int base);
    int w;
    begin
      pack_beat = '0;
      for (w = 0; w < WPB; w = w + 1)
        pack_beat[w*32 +: 32] = x32(acc_out_lat[base + w]);
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

  // ----------------------------
  // outputs
  // ----------------------------
  assign rd_data_ready = 1'b1;
  // 세 FSM 중 하나라도 돌고 있거나, 채워진 뱅크가 남아 있으면 바쁘다.
  assign busy = (lst != L_IDLE) || (cst != C_IDLE) || (wst != W_IDLE) || (bank_full != '0);
  // done 은 쓰기가 끝난 시점에 울린다 (결과가 메모리에 실제로 도착한 때).
  // 쓰기가 없는 중간 조각은 연산이 끝난 시점에 울린다. 둘 다 한 사이클
  // 펄스이고 같은 사이클에 겹치지 않는다 (cm_pass 는 nw_cm 일 때만,
  // W_DONE 은 쓰기를 시작한 디스크립터에서만 일어난다).
  assign done = (wst == W_DONE) || cm_pass;

  // ----------------------------
  // 로드 FSM (DMA 읽기 채널 + SRAM 쓰기 포트)
  // ----------------------------
  always_comb begin
    lst_n = lst;

    rd_req_valid     = 1'b0;
    rd_req_addr      = 64'd0;
    // 기본값은 이번 조각의 길이다. rd_len_r 은 전체 길이이므로 그대로
    // 쓰면 16비트로 잘린다 — 그것이 바로 이 작업이 고친 버그다.
    rd_req_len_bytes = 16'(rd_this_len);

    act_we        = 1'b0;
    wgt_we        = 1'b0;
    ld_mark       = 1'b0;
    wgt_load_done = 1'b0;

    case (lst)
      L_IDLE: if (cmd_accept) lst_n = L_ACT_REQ;

      // 요청 하나는 최대 RD_REQ_MAX 바이트다. 남은 양이 더 많으면
      // 수신이 끝난 뒤 다시 REQ 로 돌아와 이어서 요청한다.
      L_ACT_REQ: begin
        rd_req_valid     = 1'b1;
        rd_req_addr      = rd_cur_addr;
        rd_req_len_bytes = 16'(rd_this_len);
        if (rd_req_ready) lst_n = L_ACT_RECV;
      end

      L_ACT_RECV: begin
        if (rd_data_valid && rd_data_ready) act_we = 1'b1;
        // rd_data_last 는 **이 요청의** 마지막 비트다. 전체 전송의
        // 마지막인지는 rd_left 로 따로 판단한다.
        if (rd_data_valid && rd_data_ready && rd_data_last) begin
          if (rd_left != '0) begin
            lst_n = L_ACT_REQ;          // 아직 남았다 — 이어서 요청
          end else if (wgt_skip_r) begin
            // 가중치는 SRAM 에 상주한다. 읽을 것이 없으니 바로 넘긴다.
            ld_mark = 1'b1;
            lst_n   = L_IDLE;
          end else begin
            lst_n = L_WGT_REQ;
          end
        end
      end

      L_WGT_REQ: begin
        rd_req_valid     = 1'b1;
        rd_req_addr      = rd_cur_addr;
        rd_req_len_bytes = 16'(rd_this_len);
        if (rd_req_ready) lst_n = L_WGT_RECV;
      end

      L_WGT_RECV: begin
        if (rd_data_valid && rd_data_ready) wgt_we = 1'b1;
        if (rd_data_valid && rd_data_ready && rd_data_last) begin
          if (rd_left != '0) begin
            lst_n = L_WGT_REQ;          // 아직 남았다
          end else begin
            wgt_load_done = 1'b1;
            // 마지막 워드는 이 에지에 SRAM 에 들어가고 bank_full 도 같은
            // 에지에 선다. 연산 FSM 은 다음 사이클에야 그것을 보므로
            // 첫 읽기는 그 뒤가 된다. 해저드 없음.
            ld_mark = 1'b1;
            lst_n   = L_IDLE;
          end
        end
      end

      default: lst_n = L_IDLE;
    endcase

    // 단계의 첫 요청에서만 load_idx 를 0 으로 되돌린다. 쪼개진 두 번째
    // 요청에서 되돌리면 SRAM 주소가 앞으로 되감겨 앞 조각을 덮어쓴다.
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

      // 단계 시작: 남은 양과 주소를 세팅한다.
      if (lst == L_IDLE && cmd_accept) begin
        rd_left     <= LEN_W'(((clamp_kt(Kt) + KPW - 1) / KPW) * RD_BEAT_B);
        rd_cur_addr <= act_addr;
      end else if (lst_n == L_WGT_REQ && lst == L_ACT_RECV) begin
        rd_left     <= wgt_len_r;
        rd_cur_addr <= wgt_base_r;
      end else if (rd_req_fire) begin
        // 요청을 냈다 — 남은 양에서 빼고 주소를 전진시킨다.
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
  // 연산 FSM (SRAM 읽기 포트 + MAC 배열)
  // ----------------------------
  always_comb begin
    cst_n = cst;

    act_re      = 1'b0;
    wgt_re      = 1'b0;
    mac_acc_clr = 1'b0;
    act_issue   = 1'b0;
    wgt_issue   = 1'b0;
    issue_fire  = 1'b0;
    wr_start    = 1'b0;
    cm_free     = 1'b0;
    // default raddr = last issued
    rd_k_mux    = rd_k_r;

    case (cst)
      C_IDLE: if (bank_full[cm_bank]) cst_n = C_CLR;

      C_CLR: begin
        // K 분할의 중간/마지막 조각이면 앞 조각의 누산을 이어받는다.
        // 발행 카운터는 cst==C_CLR 로 따로 리셋되므로 영향이 없다.
        mac_acc_clr = ~ak_cm;
        if (Kt_cm == 0) cst_n = C_POST;
        else            cst_n = C_COMPUTE;
      end

      C_COMPUTE: begin
        // ✅ issue_fire: exactly one pulse per k (bounded)
        issue_fire = (issue_k_addr < Kt_cm);
        act_issue  = issue_fire;
        wgt_issue  = issue_fire;

        // ✅ drive SRAM address directly from issue_k_addr in the issue cycle
        rd_k_mux = issue_fire ? issue_k_addr_trunc : rd_k_r;

        // ✅ one-cycle SRAM read enable per issue is enough (rdata holds in SRAM model)
        act_re = issue_fire;
        wgt_re = issue_fire;

        // ✅ exit after MAC count has reached Kt (robust: >=)
        if (issue_k_mac >= Kt_cm)
          cst_n = C_POST;
      end

      C_POST: begin
        // ✅ fixed 1-cycle drain: always move forward
        cst_n = C_LATCH;
      end

      C_LATCH: begin
        if (nw_cm) begin
          // K 분할 중간 조각 — 누산기에 남겨 두고 쓰지 않는다.
          // acc_out_lat 은 건드리지 않는다 (앞 디스크립터의 쓰기가
          // 아직 진행 중일 수 있다). 쓰기 FSM 을 기다릴 필요도 없다.
          cm_free = 1'b1;
          cst_n   = C_IDLE;
        end else if (wst == W_IDLE) begin
          // acc_out_lat 은 단일 버퍼다. 쓰기 FSM 이 비어야 덮어쓸 수 있다.
          wr_start = 1'b1;
          cm_free  = 1'b1;    // 뱅크를 로드 FSM 에 돌려준다
          cst_n    = C_IDLE;
        end
      end

      default: cst_n = C_IDLE;
    endcase
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) cst <= C_IDLE;
    else        cst <= cst_n;
  end

  // ----------------------------
  // 쓰기 FSM (DMA 쓰기 채널)
  // ----------------------------
  assign wr_req_addr      = wr_addr_r;
  assign wr_req_len_bytes = 16'(OUT_BYTES);   // N² × 4, 폭과 무관

  always_comb begin
    wst_n        = wst;
    wr_req_valid = 1'b0;      // 이 신호는 쓰기 FSM 이 단독으로 구동한다
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

endmodule

`default_nettype wire
