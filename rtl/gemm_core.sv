`timescale 1ns/1ps
`default_nettype none

module gemm_core #(
  parameter int MAX_KT = 256,
  parameter int DMA_BEAT_BYTES = 16,

  // DMA 읽기 데이터 폭. 128 또는 256.
  //
  // 한 K-step 은 INT8 16개 = 128비트다. 128 이면 비트 하나가 K-step
  // 하나를 나르고, 256 이면 두 개를 나른다. 전송 바이트 수는 같고
  // 비트 수가 절반이 되므로 읽기 채널 병목이 완화된다.
  //
  // Kt=64 기준: 읽기 128 비트 vs 연산 68 사이클 → 읽기가 1.9배 모자라
  // 완전히 겹쳐도 가동률 상한이 53% 다. 256비트면 읽기 64 비트가 되어
  // 연산 한계에 닿는다. spec/memory-bandwidth-analysis.md 참조.
  //
  // 쓰기 채널은 건드리지 않는다 (독립 채널이고 병목이 아니다).
  parameter int RD_DATA_W = 128
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

  localparam int ADDR_W    = $clog2(MAX_KT);
  localparam int OUT_BEATS = 64;      // 1024 / 16
  localparam int OUT_BYTES = 1024;

  function automatic logic [31:0] clamp_kt(input logic [31:0] in);
    logic [31:0] mx;
    begin
      mx = MAX_KT;
      clamp_kt = (in > mx) ? mx : in;
    end
  endfunction

  // ----------------------------
  // Command latch
  // ----------------------------
  logic [63:0] act_base_r, wgt_base_r, out_base_r;
  logic [31:0] Kt_r;
  logic [15:0] rd_len_r;

  typedef enum logic [3:0] {
    S_IDLE,
    S_RD_ACT_REQ,
    S_RD_ACT_RECV,
    S_RD_WGT_REQ,
    S_RD_WGT_RECV,
    S_ACC_CLR,
    S_COMPUTE,
    S_POST_MAC,   // drain 1cycle (last MAC happens on entry edge)
    S_LATCH       // acc_out_flat -> acc_out_lat, 쓰기 FSM 기동
  } state_t;

  state_t st, st_n;

  // ── 쓰기 백엔드 FSM ────────────────────────────────────────────
  //
  // 쓰기는 Kt 와 무관하게 64 비트(1,024바이트) 고정이다. 메인 FSM 안에
  // 두면 그 동안 MAC 배열과 DMA 읽기 채널이 통째로 논다 (Kt=1 에서
  // 79 사이클 중 65 가 쓰기였다). 읽기와 쓰기는 독립 채널이므로
  // 분리하면 다음 디스크립터의 페치/연산과 겹칠 수 있다.
  //
  // acc_out_lat 은 단일 버퍼라 메인 FSM 은 S_LATCH 에서 쓰기 FSM 이
  // 비기를 기다린다. 그 지점만 지키면 나머지는 자유롭게 겹친다.
  typedef enum logic [1:0] { W_IDLE, W_REQ, W_SEND, W_DONE } wstate_t;
  wstate_t wst, wst_n;

  logic wr_start;      // S_LATCH 에서 1 사이클 펄스


  assign cmd_ready = (st == S_IDLE);
  wire   cmd_accept = cmd_valid & cmd_ready;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      act_base_r <= 64'd0;
      wgt_base_r <= 64'd0;
      out_base_r <= 64'd0;
      Kt_r       <= 32'd0;
      rd_len_r   <= 16'd0;
    end else if (cmd_accept) begin
      act_base_r <= act_addr;
      wgt_base_r <= wgt_addr;
      out_base_r <= out_addr;
      Kt_r       <= clamp_kt(Kt);
      // 읽기 길이는 비트 크기의 배수여야 한다.
      // RD_DATA_W=256 이면 비트 하나가 K-step 두 개를 나르므로, Kt 가
      // 홀수면 Kt*16 바이트가 32 의 배수가 아니다. 워드 단위로 올림해
      // 요청한다. 남는 절반은 SRAM 마지막 워드의 상위에 들어가고
      // 연산은 k < Kt 만 읽으므로 사용되지 않는다.
      rd_len_r   <= 16'(((clamp_kt(Kt) + KPW - 1) / KPW) * KPW * DMA_BEAT_BYTES);
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
        + ((rd_data_valid && rd_data_ready) ? (RD_DATA_W/8) : 0)
        + ((wr_data_valid && wr_data_ready) ? DMA_BEAT_BYTES : 0);
    end
  end

  // ----------------------------
  // SRAM interfaces
  // ----------------------------
  // 워드 하나에 들어가는 K-step 수와 SRAM 워드 수
  localparam int KPW        = RD_DATA_W / 128;              // 1 또는 2
  localparam int KPW_W      = (KPW > 1) ? $clog2(KPW) : 1;
  localparam int SRAM_DEPTH = (MAX_KT + KPW - 1) / KPW;
  localparam int SADDR_W    = $clog2(SRAM_DEPTH);

  initial begin
    if (RD_DATA_W != 128 && RD_DATA_W != 256)
      $fatal(1, "gemm_core: RD_DATA_W must be 128 or 256 (got %0d)", RD_DATA_W);
  end

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

  logic [SADDR_W-1:0] load_idx;   // 수신 비트 = SRAM 워드 인덱스

  // ✅ hold read address at "last issued" k to avoid reading past Kt (Kt=1 → addr would become 1 and go X)
  logic [ADDR_W-1:0] rd_k_r;
  // ✅ IMPORTANT: SRAM is sync-read. If we update rd_k_r at posedge, the same posedge read would still see OLD rd_k_r.
  // So, when issuing a new k, drive SRAM raddr directly from issue_k_addr in that cycle.
  logic [ADDR_W-1:0] rd_k_mux;

  // K 인덱스를 (워드 주소, 워드 내 위치) 로 나눈다.
  // SRAM 은 동기 읽기라 rdata 가 한 사이클 뒤에 나온다. 따라서 슬라이스에
  // 쓸 위치도 주소를 낸 그 사이클의 값을 한 사이클 지연시켜 써야 한다.
  logic [KPW_W-1:0] rd_half, rd_half_q;

  generate
    if (KPW == 1) begin : G_W128
      assign act_raddr = rd_k_mux[SADDR_W-1:0];
      assign wgt_raddr = rd_k_mux[SADDR_W-1:0];
      assign rd_half   = '0;
      assign act_rdata = act_word;
      assign wgt_rdata = wgt_word;
    end else begin : G_WIDE
      assign act_raddr = rd_k_mux[KPW_W +: SADDR_W];
      assign wgt_raddr = rd_k_mux[KPW_W +: SADDR_W];
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
  logic mac_en_d1;
  logic mac_en_d2;    

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

// Unpack acc_out_mac_flat -> acc_out_mac[] via continuous assigns (no always_* needed)
genvar ui;
generate
  for (ui = 0; ui < 256; ui++) begin : UNPACK_MAC
    assign acc_out_mac[ui] = acc_out_mac_flat[ui*32 +: 32];
  end
endgenerate

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
      else if (st == S_ACC_CLR)
        acc_out_flat[zz] <= 32'sd0;
      else if (st == S_LATCH)
        acc_out_flat[zz] <= acc_out_mac[zz];
    end
  end
endgenerate

genvar kk;
generate
  for (kk = 0; kk < 256; kk++) begin : GEN_ACC_LAT
    always_ff @(posedge clk or negedge rst_n) begin
      // S_ACC_CLR 에서 지우던 것을 래치 시점으로 접었다.
      // 쓰기 FSM 이 분리되면서 다음 디스크립터가 S_ACC_CLR 에 도달할 때
      // 아직 전송 중인 acc_out_lat 을 지워버리는 해저드가 생기기 때문이다.
      // Kt=0 이면 0 을 래치하므로 동작은 같다.
      if (!rst_n)
        acc_out_lat[kk] <= 32'sd0;
      else if (st == S_LATCH)
        acc_out_lat[kk] <= (Kt_r != 0) ? acc_out_mac[kk] : 32'sd0;
    end
  end
endgenerate

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
  // 결과적으로 앞쪽 K-step 을 놓치고 마지막 것을 중복 누적한다.
  // Kt=1 에서는 피연산자가 하나뿐이라 우연히 맞아서 드러나지 않았다
  // (test_gemm_top.py 는 Kt=1 만 돌린다).
  assign mac_en = act_issue_d & wgt_issue_d;

always_ff @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    issue_k_addr  <= 0;
    issue_k_mac   <= 0;
    act_issue_d   <= 0;
    wgt_issue_d   <= 0;
    mac_en_d1     <= 0;
    mac_en_d2     <= 0;
    rd_k_r        <= '0;
  end else begin
    // 1-cycle delay: SRAM data valid
    act_issue_d <= issue_fire;
    wgt_issue_d <= issue_fire;

    // mac_en_d1/d2 는 더 이상 쓰지 않는다 (위 주석 참조).
    mac_en_d1 <= act_issue_d & wgt_issue_d;
    mac_en_d2 <= mac_en_d1;

    if (st == S_ACC_CLR) begin
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
  // 메인 FSM 이 S_IDLE 로 돌아와도 쓰기가 남아 있으면 아직 바쁘다.
  assign busy = (st != S_IDLE) || (wst != W_IDLE);
  // done 은 쓰기가 끝난 시점에 울린다 (결과가 메모리에 실제로 도착한 때).
  assign done = (wst == W_DONE);

  // ----------------------------
  // 쓰기 FSM
  // ----------------------------
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

  // ----------------------------
  // FSM sequential
  // ----------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st       <= S_IDLE;
      load_idx <= '0;
    end else begin
      st <= st_n;


      if (st == S_RD_ACT_REQ && rd_req_valid && rd_req_ready) load_idx <= '0;
      else if (st == S_RD_WGT_REQ && rd_req_valid && rd_req_ready) load_idx <= '0;
      else if ((st == S_RD_ACT_RECV || st == S_RD_WGT_RECV) && rd_data_valid && rd_data_ready)
        load_idx <= load_idx + 1'b1;
    end
  end

  // ----------------------------
  // FSM combinational
  // ----------------------------
  always_comb begin
    st_n = st;

    rd_req_valid     = 1'b0;
    rd_req_addr      = 64'd0;
    rd_req_len_bytes = rd_len_r;

    // wr_req_valid 는 쓰기 FSM 소유다. 여기서 건드리면 다중 구동이 된다.
    wr_req_addr      = out_base_r;
    wr_req_len_bytes = OUT_BYTES;

    act_we    = 1'b0;
    wgt_we    = 1'b0;
    act_re    = 1'b0;
    wgt_re    = 1'b0;

    act_waddr = load_idx;
    wgt_waddr = load_idx;

    mac_acc_clr = 1'b0;
    act_issue   = 1'b0;
    wgt_issue   = 1'b0;
    wr_start    = 1'b0;
   issue_fire  = 1'b0;
    // default raddr = last issued
    rd_k_mux    = rd_k_r;

    case (st)
      S_IDLE: begin
        if (cmd_valid) st_n = S_RD_ACT_REQ;
      end

      S_RD_ACT_REQ: begin
        rd_req_valid = 1'b1;
        rd_req_addr  = act_base_r;
        if (rd_req_ready) st_n = S_RD_ACT_RECV;
      end

      S_RD_ACT_RECV: begin
        if (rd_data_valid && rd_data_ready) act_we = 1'b1;
        // ✅ "진짜 마지막 beat" 기준으로 수신 완료 확정
        if (rd_data_valid && rd_data_ready && rd_data_last)
          st_n = S_RD_WGT_REQ;
      end

      S_RD_WGT_REQ: begin
        rd_req_valid = 1'b1;
        rd_req_addr  = wgt_base_r;
        if (rd_req_ready) st_n = S_RD_WGT_RECV;
      end

      S_RD_WGT_RECV: begin
        if (rd_data_valid && rd_data_ready) wgt_we = 1'b1;
        if (rd_data_valid && rd_data_ready && rd_data_last)
          st_n = S_ACC_CLR;
      end

      S_ACC_CLR: begin
        mac_acc_clr = 1'b1;
        if (Kt_r == 0) st_n = S_POST_MAC;
        else           st_n = S_COMPUTE;
      end

      S_COMPUTE: begin
        // ✅ issue_fire: exactly one pulse per k (bounded)
        issue_fire = (issue_k_addr < Kt_r);
        act_issue  = issue_fire;
        wgt_issue  = issue_fire;

        // ✅ drive SRAM address directly from issue_k_addr in the issue cycle (sync read!)
        // Use issue_k_addr_trunc (continuous assign outside always) to avoid Icarus part-select warning
        rd_k_mux = issue_fire ? issue_k_addr_trunc : rd_k_r;

        // ✅ one-cycle SRAM read enable per issue is enough (rdata holds X-safe in SRAM model)
        act_re = issue_fire;
        wgt_re = issue_fire;

        // ✅ exit after MAC count has reached Kt_r (robust: >=)
        if (issue_k_mac >= Kt_r)
          st_n = S_POST_MAC;
      end

      S_POST_MAC: begin
        // ✅ fixed 1-cycle drain: always move forward
        st_n = S_LATCH;
      end

      S_LATCH: begin
        // 쓰기 FSM 이 비어야 acc_out_lat 을 덮어쓸 수 있다.
        if (wst == W_IDLE) begin
          wr_start = 1'b1;
          st_n     = S_IDLE;   // 래치 즉시 다음 디스크립터를 받는다
        end
      end

      default: st_n = S_IDLE;
    endcase
  end

endmodule

`default_nettype wire
