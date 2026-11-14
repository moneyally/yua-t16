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
// spec/memory-bandwidth-analysis.md 참조.
module gemm_core #(
  parameter int MAX_KT = 256,
  parameter int DMA_BEAT_BYTES = 16,

  // DMA 읽기 데이터 폭. 128 또는 256.
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
  output logic [127:0] wr_data,
  output logic         wr_data_last,

  output logic [31:0]  perf_cycles,
  output logic [31:0]  perf_bytes,

  output logic         busy
);

  // ----------------------------
  // 치수
  // ----------------------------
  localparam int ADDR_W    = $clog2(MAX_KT);
  localparam int OUT_BEATS = 64;      // 1024 / 16
  localparam int OUT_BYTES = 1024;

  // 워드 하나에 들어가는 K-step 수
  localparam int KPW   = RD_DATA_W / 128;              // 1 또는 2
  localparam int KPW_W = (KPW > 1) ? $clog2(KPW) : 1;

  // 뱅크 하나가 디스크립터 하나의 피연산자를 담는다.
  localparam int BANK_W      = $clog2(BANKS);              // 뱅크 인덱스 폭
  localparam int BANK_DEPTH  = (MAX_KT + KPW - 1) / KPW;   // 뱅크당 워드 수
  localparam int BADDR_W     = $clog2(BANK_DEPTH);         // 뱅크 내 주소
  localparam int SRAM_DEPTH  = BANK_DEPTH * BANKS;
  localparam int SADDR_W     = BADDR_W + BANK_W;

  initial begin
    if (RD_DATA_W != 128 && RD_DATA_W != 256)
      $fatal(1, "gemm_core: RD_DATA_W must be 128 or 256 (got %0d)", RD_DATA_W);
    // 뱅크 인덱스를 주소 상위에 그냥 붙이므로 2의 거듭제곱이어야 한다.
    if (BANKS < 2 || (BANKS & (BANKS - 1)) != 0)
      $fatal(1, "gemm_core: BANKS must be a power of two >= 2 (got %0d)", BANKS);
  end

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

  logic ld_mark;     // 로드 완료: 뱅크를 연산 FSM 에 넘긴다 (1 사이클)
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

  wire [31:0] Kt_cm  = Kt_bank[cm_bank];
  wire [63:0] out_cm = out_bank[cm_bank];

  logic [63:0] act_base_r, wgt_base_r;
  logic [15:0] rd_len_r;

  // 로드 FSM 은 자기 뱅크가 비어 있을 때만 명령을 받는다.
  assign cmd_ready = (lst == L_IDLE) && !bank_full[ld_bank];
  wire   cmd_accept = cmd_valid & cmd_ready;

  integer bi;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      act_base_r <= 64'd0;
      wgt_base_r <= 64'd0;
      rd_len_r   <= 16'd0;
      for (bi = 0; bi < BANKS; bi = bi + 1) begin
        Kt_bank[bi]  <= 32'd0;
        out_bank[bi] <= 64'd0;
      end
    end else if (cmd_accept) begin
      act_base_r <= act_addr;
      wgt_base_r <= wgt_addr;
      // 읽기 길이는 비트 크기의 배수여야 한다.
      // RD_DATA_W=256 이면 비트 하나가 K-step 두 개를 나르므로, Kt 가
      // 홀수면 Kt*16 바이트가 32 의 배수가 아니다. 워드 단위로 올림해
      // 요청한다. 남는 절반은 뱅크 마지막 워드의 상위에 들어가고
      // 연산은 k < Kt 만 읽으므로 사용되지 않는다.
      rd_len_r   <= 16'(((clamp_kt(Kt) + KPW - 1) / KPW) * KPW * DMA_BEAT_BYTES);

      Kt_bank[ld_bank]  <= clamp_kt(Kt);
      out_bank[ld_bank] <= out_addr;
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
        + ((wr_data_valid && wr_data_ready) ? DMA_BEAT_BYTES : 0);
    end
  end

  // ----------------------------
  // 피연산자 SRAM (1W1R, 2 뱅크)
  // ----------------------------
  logic act_we, wgt_we, act_re, wgt_re;
  logic [SADDR_W-1:0] act_waddr, wgt_waddr;
  logic [SADDR_W-1:0] act_raddr, wgt_raddr;
  logic [RD_DATA_W-1:0] act_word, wgt_word;   // SRAM 출력 (워드 전체)
  logic [127:0] act_rdata, wgt_rdata;         // 선택된 K-step 하나

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
  logic [BADDR_W-1:0] load_idx;
  assign act_waddr = {ld_bank, load_idx};
  assign wgt_waddr = {ld_bank, load_idx};

  // ✅ hold read address at "last issued" k to avoid reading past Kt
  logic [ADDR_W-1:0] rd_k_r;
  // ✅ IMPORTANT: SRAM is sync-read. If we update rd_k_r at posedge, the same
  // posedge read would still see OLD rd_k_r. So, when issuing a new k, drive
  // SRAM raddr directly from issue_k_addr in that cycle.
  logic [ADDR_W-1:0] rd_k_mux;

  // K 인덱스를 (뱅크 내 워드 주소, 워드 내 위치) 로 나눈다.
  // SRAM 은 동기 읽기라 rdata 가 한 사이클 뒤에 나온다. 따라서 슬라이스에
  // 쓸 위치도 주소를 낸 그 사이클의 값을 한 사이클 지연시켜 써야 한다.
  logic [KPW_W-1:0] rd_half, rd_half_q;

  generate
    if (KPW == 1) begin : G_W128
      assign act_raddr = {cm_bank, rd_k_mux[BADDR_W-1:0]};
      assign wgt_raddr = {cm_bank, rd_k_mux[BADDR_W-1:0]};
      assign rd_half   = '0;
      assign act_rdata = act_word;
      assign wgt_rdata = wgt_word;
    end else begin : G_WIDE
      assign act_raddr = {cm_bank, rd_k_mux[KPW_W +: BADDR_W]};
      assign wgt_raddr = {cm_bank, rd_k_mux[KPW_W +: BADDR_W]};
      assign rd_half   = rd_k_mux[KPW_W-1:0];
      assign act_rdata = act_word[rd_half_q*128 +: 128];
      assign wgt_rdata = wgt_word[rd_half_q*128 +: 128];
    end
  endgenerate

  // SRAM 이 re=0 일 때 rdata 를 유지하므로, 슬라이스 위치도 같이 유지한다.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)       rd_half_q <= '0;
    else if (act_re)  rd_half_q <= rd_half;
  end

  // ----------------------------
  // MAC array
  // ----------------------------
  logic mac_acc_clr, mac_en;
  logic act_issue, wgt_issue;        // issue_fire exposed (debug)
  logic act_issue_d, wgt_issue_d;    // ✅ 1-cycle delayed valid (SRAM->MAC align)
  logic issue_fire;

  logic signed [7:0]  a_row [0:15];
  logic signed [7:0]  b_col [0:15];

// MAC internal result as packed flat vector (Icarus-safe — no unpacked array output port)
logic [32*256-1:0] acc_out_mac_flat;

// Unpacked views of MAC result and latched outputs for cocotb/FSM access
logic signed [31:0] acc_out_mac [0:255];  // continuous unpack from acc_out_mac_flat
logic signed [31:0] acc_out_flat [0:255]; // latch of mac result (reset-safe, Icarus-safe)
logic signed [31:0] acc_out_lat  [0:255]; // write-out latch

  genvar i;
  generate
    for (i=0;i<16;i++) begin : ROW_UNPACK
      assign a_row[i] = act_rdata[8*i +: 8];
      assign b_col[i] = wgt_rdata[8*i +: 8];
    end
  endgenerate

mac_array u_mac (
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
  for (ui = 0; ui < 256; ui++) begin : UNPACK_MAC
    assign acc_out_mac[ui] = acc_out_mac_flat[ui*32 +: 32];
  end
endgenerate

  // 연산 FSM 이 실제로 래치하고 넘어가는 사이클. C_LATCH 에서 쓰기 FSM 을
  // 기다리며 멈춰 있을 수 있으므로 상태만으로는 안 된다.
  wire cm_latch = (cst == C_LATCH) && (wst == W_IDLE);

// --------------------------------------------------
// cocotb / Icarus-safe observable accumulators
// Use generate so each element gets its own always_ff — avoids Icarus
// "constant selects in always_* processes" bug with loop-indexed arrays.
// --------------------------------------------------
genvar zz;
generate
  for (zz = 0; zz < 256; zz++) begin : GEN_ACC_FLAT
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
  for (kk = 0; kk < 256; kk++) begin : GEN_ACC_LAT
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
  logic [6:0]   out_beat;
  logic [127:0] wr_data_r;
  logic         wr_valid_r;

  function automatic logic [31:0] x32(input logic signed [31:0] v);
    logic [31:0] t;
    begin
      t = v;
// FIX v0.2: X는 항상 0으로 클리핑
      if ((^t === 1'bx)) x32 = 32'd0;
      else              x32 = t;

    end
  endfunction

  function automatic logic [127:0] pack4(input int base);
    begin
      pack4 = {
        x32(acc_out_lat[base+3]),
        x32(acc_out_lat[base+2]),
        x32(acc_out_lat[base+1]),
        x32(acc_out_lat[base+0])
      };
    end
  endfunction

`ifdef COCOTB_SIM
  initial begin
    out_beat   = 7'd0;
    wr_data_r  = 128'd0;
    wr_valid_r = 1'b0;
  end
`endif

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_beat   <= 7'd0;
      wr_data_r  <= 128'd0;
      wr_valid_r <= 1'b0;
    end else begin
      if (wst == W_REQ && wr_req_ready) begin
        out_beat   <= 7'd0;
        wr_data_r  <= pack4(0);
        wr_valid_r <= 1'b1;
      end else if (wst == W_SEND) begin
        if (wr_valid_r && wr_data_ready) begin
          if (out_beat == OUT_BEATS-1) begin
            wr_valid_r <= 1'b0;
          end else begin
            out_beat   <= out_beat + 1'b1;
            wr_data_r  <= pack4((out_beat + 1) * 4);
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
  assign done = (wst == W_DONE);

  // ----------------------------
  // 로드 FSM (DMA 읽기 채널 + SRAM 쓰기 포트)
  // ----------------------------
  always_comb begin
    lst_n = lst;

    rd_req_valid     = 1'b0;
    rd_req_addr      = 64'd0;
    rd_req_len_bytes = rd_len_r;

    act_we  = 1'b0;
    wgt_we  = 1'b0;
    ld_mark = 1'b0;

    case (lst)
      L_IDLE: if (cmd_accept) lst_n = L_ACT_REQ;

      L_ACT_REQ: begin
        rd_req_valid = 1'b1;
        rd_req_addr  = act_base_r;
        if (rd_req_ready) lst_n = L_ACT_RECV;
      end

      L_ACT_RECV: begin
        if (rd_data_valid && rd_data_ready) act_we = 1'b1;
        // ✅ "진짜 마지막 beat" 기준으로 수신 완료 확정
        if (rd_data_valid && rd_data_ready && rd_data_last)
          lst_n = L_WGT_REQ;
      end

      L_WGT_REQ: begin
        rd_req_valid = 1'b1;
        rd_req_addr  = wgt_base_r;
        if (rd_req_ready) lst_n = L_WGT_RECV;
      end

      L_WGT_RECV: begin
        if (rd_data_valid && rd_data_ready) wgt_we = 1'b1;
        if (rd_data_valid && rd_data_ready && rd_data_last) begin
          // 마지막 워드는 이 에지에 SRAM 에 들어가고 bank_full 도 같은
          // 에지에 선다. 연산 FSM 은 다음 사이클에야 그것을 보므로
          // 첫 읽기는 그 뒤가 된다. 해저드 없음.
          ld_mark = 1'b1;
          lst_n   = L_IDLE;
        end
      end

      default: lst_n = L_IDLE;
    endcase
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      lst      <= L_IDLE;
      load_idx <= '0;
    end else begin
      lst <= lst_n;

      if (lst == L_ACT_REQ && rd_req_valid && rd_req_ready) load_idx <= '0;
      else if (lst == L_WGT_REQ && rd_req_valid && rd_req_ready) load_idx <= '0;
      else if ((lst == L_ACT_RECV || lst == L_WGT_RECV) && rd_data_valid && rd_data_ready)
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
        mac_acc_clr = 1'b1;
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
        // acc_out_lat 은 단일 버퍼다. 쓰기 FSM 이 비어야 덮어쓸 수 있다.
        if (wst == W_IDLE) begin
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
  assign wr_req_len_bytes = 16'(OUT_BYTES);

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
