`timescale 1ns/1ps
`default_nettype none

// mxu_cluster.sv — BF16 텐서 병렬 클러스터
//
// ── 왜 ───────────────────────────────────────────────────────────
//
// `gemm_cluster` 가 INT8 쪽에서 증명한 것을 BF16 쪽에 옮긴다. 한
// 레이어의 가중치를 열 방향으로 쪼개 코어마다 다른 조각을 상주시키고,
// **같은 활성값**을 모든 코어에 브로드캐스트한다.
//
//   코어 0:  A · W[:, 0:DIM]        → OUT[:, 0:DIM]
//   코어 1:  A · W[:, DIM:2·DIM]    → OUT[:, DIM:2·DIM]
//
// 연산은 CORES 배가 되는데 활성값 트래픽은 그대로다. 산술 강도가
// √CORES 배로 좋아지는 것과 같은 이야기다 (9단계).
//
// ── INT8 클러스터와 다른 점 ──────────────────────────────────────
//
// 1. **디스크립터 FSM 이 없다.** `gemm_cluster` 는 `ctrl_fsm` 을 품고
//    64바이트 디스크립터를 받는다. BF16 쪽에는 아직 디스크립터 형식이
//    없으므로 `cmd_*` 를 그대로 밖으로 낸다. 스트라이드도 포트로
//    받는다. 디스크립터 형식이 생기면 그때 `ctrl_fsm` 을 얹는다.
// 2. 플래그가 `wgt_keep` 하나다 (INT4·K분할은 BF16 경로에 없다).
// 3. 원소가 2바이트라 K-step 이 `DIM*16` 비트다. 중재에는 영향이
//    없다 — 코어가 알아서 요청을 쪼갠다.
//
// 중재 구조와 그 근거(왜 valid 를 게이트하는지, 왜 완료를 코어마다
// 따로 세는지)는 `gemm_cluster.sv` 의 주석과 같다. 같은 함정을 같은
// 방식으로 피한다. 두 군데에 같은 설명을 베껴 두지 않고 요점만 적는다.
module mxu_cluster #(
  parameter int MAX_KT    = 256,
  parameter int DIM       = 16,
  parameter int RD_DATA_W = 256,
  parameter int WR_DATA_W = 128,
  parameter int BANKS     = 2,
  parameter int STREAM_RD = 0,
  // 코어 수. 연산이 CORES 배, 활성값 트래픽은 그대로.
  parameter int CORES     = 2
)(
  input  logic        clk,
  input  logic        rst_n,

  // 명령 (코어 전체가 락스텝으로 받는다)
  input  logic        cmd_valid,
  output logic        cmd_ready,
  input  logic [63:0] act_addr,
  input  logic [63:0] wgt_addr,   // 코어 0 의 가중치 조각
  input  logic [63:0] out_addr,   // 코어 0 의 출력 조각
  input  logic [31:0] Kt,
  input  logic        wgt_keep,
  // 코어 사이의 간격. 코어 k 는 wgt_addr + k·wgt_stride 를 읽고
  // out_addr + k·out_stride 에 쓴다.
  input  logic [31:0] wgt_stride,
  input  logic [31:0] out_stride,

  // 디스크립터 하나가 **모든 코어에서** 끝났을 때 한 사이클 선다.
  output logic        done,

  // memory read (1 포트)
  output logic        rd_req_valid,
  input  logic        rd_req_ready,
  output logic [63:0] rd_req_addr,
  output logic [15:0] rd_req_len_bytes,
  input  logic        rd_done,

  input  logic        rd_data_valid,
  input  logic [RD_DATA_W-1:0] rd_data,
  output logic        rd_data_ready,
  input  logic        rd_data_last,

  // memory write (1 포트)
  output logic        wr_req_valid,
  input  logic        wr_req_ready,
  output logic [63:0] wr_req_addr,
  output logic [15:0] wr_req_len_bytes,
  input  logic        wr_done,

  output logic        wr_data_valid,
  input  logic        wr_data_ready,
  output logic [WR_DATA_W-1:0] wr_data,
  output logic        wr_data_last,

  output logic        busy,
  output logic [31:0] perf_cycles,
  output logic [31:0] perf_bytes,
  output logic [31:0] perf_wgt_reuse,
  output logic [31:0] perf_wgt_load
);

  localparam int CSEL_W = (CORES > 1) ? $clog2(CORES) : 1;

  initial begin
    if (CORES < 1)
      $fatal(1, "mxu_cluster: CORES must be >= 1 (got %0d)", CORES);
  end

  // ------------------------------------------------------------------
  // 코어 배열
  // ------------------------------------------------------------------
  logic                  c_cmd_ready [0:CORES-1];
  logic                  c_done      [0:CORES-1];
  logic                  c_busy      [0:CORES-1];

  logic                  c_rd_req_valid [0:CORES-1];
  logic [63:0]           c_rd_req_addr  [0:CORES-1];
  logic [15:0]           c_rd_req_len   [0:CORES-1];
  logic                  c_rd_req_ready [0:CORES-1];
  logic                  c_rd_data_valid[0:CORES-1];
  logic                  c_rd_data_last [0:CORES-1];
  logic                  c_rd_data_ready[0:CORES-1];
  logic                  c_rd_done      [0:CORES-1];

  logic                  c_wr_req_valid [0:CORES-1];
  logic [63:0]           c_wr_req_addr  [0:CORES-1];
  logic [15:0]           c_wr_req_len   [0:CORES-1];
  logic                  c_wr_req_ready [0:CORES-1];
  logic                  c_wr_data_valid[0:CORES-1];
  logic [WR_DATA_W-1:0]  c_wr_data      [0:CORES-1];
  logic                  c_wr_data_last [0:CORES-1];
  logic                  c_wr_data_ready[0:CORES-1];
  logic                  c_wr_done      [0:CORES-1];

  logic [31:0]           c_perf_cycles  [0:CORES-1];
  logic [31:0]           c_perf_bytes   [0:CORES-1];
  logic [31:0]           c_wgt_reuse    [0:CORES-1];
  logic [31:0]           c_wgt_load     [0:CORES-1];

  // 모든 코어가 받을 준비가 되어야 명령을 발행한다 (락스텝 유지).
  //
  // valid 쪽을 게이트해야 한다. ready 를 AND 하는 것만으로는 각 코어가
  // **자기** cmd_ready 로 수락하므로, 먼저 준비된 코어가 같은
  // 디스크립터를 두 번 받는다. INT8 쪽에서 실제로 그렇게 깨졌다
  // (쓰기 주소가 16개 대신 4개). 코어의 cmd_ready 는 cmd_valid 를
  // 보지 않으므로 조합 루프가 없다.
  always_comb begin
    cmd_ready = 1'b1;
    for (int k = 0; k < CORES; k++)
      cmd_ready &= c_cmd_ready[k];
  end

  wire cmd_valid_gated = cmd_valid & cmd_ready;

  genvar ci;
  generate
    for (ci = 0; ci < CORES; ci++) begin : CORE
      wire [63:0] my_wgt = wgt_addr + 64'(wgt_stride) * ci;
      wire [63:0] my_out = out_addr + 64'(out_stride) * ci;

      mxu_core #(
        .MAX_KT(MAX_KT), .DIM(DIM),
        .RD_DATA_W(RD_DATA_W), .WR_DATA_W(WR_DATA_W),
        .BANKS(BANKS), .STREAM_RD(STREAM_RD)
      ) u_core (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(cmd_valid_gated), .cmd_ready(c_cmd_ready[ci]),
        .act_addr(act_addr), .wgt_addr(my_wgt), .out_addr(my_out),
        .Kt(Kt), .wgt_keep(wgt_keep),
        .done(c_done[ci]),

        .rd_req_valid(c_rd_req_valid[ci]), .rd_req_ready(c_rd_req_ready[ci]),
        .rd_req_addr(c_rd_req_addr[ci]), .rd_req_len_bytes(c_rd_req_len[ci]),
        .rd_done(c_rd_done[ci]),
        .rd_data_valid(c_rd_data_valid[ci]), .rd_data(rd_data),
        .rd_data_ready(c_rd_data_ready[ci]), .rd_data_last(c_rd_data_last[ci]),

        .wr_req_valid(c_wr_req_valid[ci]), .wr_req_ready(c_wr_req_ready[ci]),
        .wr_req_addr(c_wr_req_addr[ci]), .wr_req_len_bytes(c_wr_req_len[ci]),
        .wr_done(c_wr_done[ci]),
        .wr_data_valid(c_wr_data_valid[ci]), .wr_data_ready(c_wr_data_ready[ci]),
        .wr_data(c_wr_data[ci]), .wr_data_last(c_wr_data_last[ci]),

        .perf_cycles(c_perf_cycles[ci]), .perf_bytes(c_perf_bytes[ci]),
        .perf_wgt_reuse(c_wgt_reuse[ci]), .perf_wgt_load(c_wgt_load[ci]),
        .busy(c_busy[ci])
      );
    end
  endgenerate

  // ------------------------------------------------------------------
  // 읽기 중재 — ACT 는 브로드캐스트, WGT 는 라운드로빈
  //
  // 코어들이 락스텝이므로 ACT 요청은 같은 사이클에 같은 주소로 나온다.
  // 그때 한 번만 메모리에 묻고 응답을 전원에게 넘긴다. 그래서 활성값
  // 트래픽이 CORES 배가 되지 않는다. WGT 는 주소가 다르므로 하나씩.
  // ------------------------------------------------------------------
  logic              rd_all_valid;
  logic              rd_same_addr;
  logic              rd_any_valid;
  logic [CSEL_W-1:0] rd_grant;
  logic              rd_have_grant;

  always_comb begin
    rd_all_valid = 1'b1;
    rd_any_valid = 1'b0;
    rd_same_addr = 1'b1;
    for (int k = 0; k < CORES; k++) begin
      rd_all_valid &= c_rd_req_valid[k];
      rd_any_valid |= c_rd_req_valid[k];
      if (c_rd_req_addr[k] != c_rd_req_addr[0]) rd_same_addr = 1'b0;
    end

    // 가장 낮은 번호의 요청 코어 (고정 우선순위). 코어들이 같은 일을
    // 같은 순서로 하므로 WGT 구간에서 순서대로 한 번씩 돌아간다.
    rd_have_grant = 1'b0;
    rd_grant      = '0;
    for (int k = CORES - 1; k >= 0; k--)
      if (c_rd_req_valid[k]) begin
        rd_grant      = CSEL_W'(k);
        rd_have_grant = 1'b1;
      end
  end

  wire rd_bcast = rd_all_valid & rd_same_addr & (CORES > 1);

  logic              rd_busy_r, rd_bcast_r;
  logic [CSEL_W-1:0] rd_owner_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_busy_r  <= 1'b0;
      rd_bcast_r <= 1'b0;
      rd_owner_r <= '0;
    end else begin
      if (!rd_busy_r && rd_req_valid && rd_req_ready) begin
        rd_busy_r  <= 1'b1;
        rd_bcast_r <= rd_bcast;
        rd_owner_r <= rd_grant;
      end else if (rd_busy_r && rd_data_valid && rd_data_ready && rd_data_last) begin
        rd_busy_r <= 1'b0;
      end
    end
  end

  assign rd_req_valid     = !rd_busy_r && rd_any_valid;
  assign rd_req_addr      = c_rd_req_addr[rd_grant];
  assign rd_req_len_bytes = c_rd_req_len[rd_grant];

  always_comb begin
    for (int k = 0; k < CORES; k++)
      c_rd_req_ready[k] = 1'b0;
    if (!rd_busy_r && rd_req_valid && rd_req_ready) begin
      if (rd_bcast) begin
        for (int k = 0; k < CORES; k++) c_rd_req_ready[k] = 1'b1;
      end else begin
        c_rd_req_ready[rd_grant] = 1'b1;
      end
    end
  end

  always_comb begin
    for (int k = 0; k < CORES; k++) begin
      c_rd_data_valid[k] = 1'b0;
      c_rd_data_last[k]  = 1'b0;
      c_rd_done[k]       = 1'b0;
    end
    if (rd_busy_r) begin
      if (rd_bcast_r) begin
        for (int k = 0; k < CORES; k++) begin
          c_rd_data_valid[k] = rd_data_valid;
          c_rd_data_last[k]  = rd_data_last;
          c_rd_done[k]       = rd_done;
        end
      end else begin
        c_rd_data_valid[rd_owner_r] = rd_data_valid;
        c_rd_data_last[rd_owner_r]  = rd_data_last;
        c_rd_done[rd_owner_r]       = rd_done;
      end
    end
  end

  // 코어들은 rd_data_ready 를 상시 1 로 둔다 (mxu_core 가 그렇다).
  assign rd_data_ready = 1'b1;

  // ------------------------------------------------------------------
  // 쓰기 중재 — 라운드로빈 (주소가 코어마다 다르므로 공유 불가)
  // ------------------------------------------------------------------
  logic [CSEL_W-1:0] wr_grant;
  logic              wr_have_grant;
  logic              wr_busy_r;
  logic [CSEL_W-1:0] wr_owner_r;

  always_comb begin
    wr_have_grant = 1'b0;
    wr_grant      = '0;
    for (int k = CORES - 1; k >= 0; k--)
      if (c_wr_req_valid[k]) begin
        wr_grant      = CSEL_W'(k);
        wr_have_grant = 1'b1;
      end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_busy_r  <= 1'b0;
      wr_owner_r <= '0;
    end else begin
      if (!wr_busy_r && wr_req_valid && wr_req_ready) begin
        wr_busy_r  <= 1'b1;
        wr_owner_r <= wr_grant;
      end else if (wr_busy_r && wr_data_valid && wr_data_ready && wr_data_last) begin
        wr_busy_r <= 1'b0;
      end
    end
  end

  assign wr_req_valid     = !wr_busy_r && wr_have_grant;
  assign wr_req_addr      = c_wr_req_addr[wr_grant];
  assign wr_req_len_bytes = c_wr_req_len[wr_grant];

  always_comb begin
    for (int k = 0; k < CORES; k++) begin
      c_wr_req_ready[k]  = 1'b0;
      c_wr_data_ready[k] = 1'b0;
      c_wr_done[k]       = 1'b0;
    end
    if (!wr_busy_r && wr_req_valid && wr_req_ready)
      c_wr_req_ready[wr_grant] = 1'b1;
    if (wr_busy_r) begin
      c_wr_data_ready[wr_owner_r] = wr_data_ready;
      c_wr_done[wr_owner_r]       = wr_done;
    end
  end

  assign wr_data_valid = wr_busy_r && c_wr_data_valid[wr_owner_r];
  assign wr_data       = c_wr_data[wr_owner_r];
  assign wr_data_last  = wr_busy_r && c_wr_data_last[wr_owner_r];

  // ------------------------------------------------------------------
  // 완료 집계 — 코어마다 따로 센다
  // ------------------------------------------------------------------
  // 하나의 카운터로 c_done 펄스를 CORES 개 모으는 방식은 틀렸다. 쓰기가
  // 직렬화되므로 코어 0 이 먼저 끝나고, 다음 디스크립터의 쓰기까지
  // 끝내 **자기 완료 두 개**로 카운터를 채울 수 있다. 그러면 코어 1 이
  // 첫 디스크립터조차 안 끝낸 상태에서 호스트가 완료를 받는다.
  // INT8 쪽에서 CORES=2 는 8번째, CORES=4 는 첫 디스크립터에서 깨졌다.
  //
  // 코어들은 cmd_ready 게이트로 같은 사이클에 같은 디스크립터를 받으므로
  // 수락 수가 항상 같다. "모든 코어가 k번째를 끝냈다" 를 보면 된다.
  localparam int DCNT_W = 8;

  logic [DCNT_W-1:0] c_dcnt [0:CORES-1];
  logic [DCNT_W-1:0] emitted;
  logic              all_ahead;
  logic              done_all;

  always_comb begin
    all_ahead = 1'b1;
    for (int k = 0; k < CORES; k++)
      if (c_dcnt[k] == emitted) all_ahead = 1'b0;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int k = 0; k < CORES; k++) c_dcnt[k] <= '0;
      emitted  <= '0;
      done_all <= 1'b0;
    end else begin
      for (int k = 0; k < CORES; k++)
        if (c_done[k]) c_dcnt[k] <= c_dcnt[k] + 1'b1;
      if (all_ahead) begin
        emitted  <= emitted + 1'b1;
        done_all <= 1'b1;
      end else begin
        done_all <= 1'b0;
      end
    end
  end

  assign done = done_all;

  always_comb begin
    busy = 1'b0;
    for (int k = 0; k < CORES; k++)
      busy |= c_busy[k];
  end

  // 계측은 코어 0 의 사이클과 전체 바이트 합을 쓴다.
  assign perf_cycles = c_perf_cycles[0];
  always_comb begin
    perf_bytes     = '0;
    perf_wgt_reuse = '0;
    perf_wgt_load  = '0;
    for (int k = 0; k < CORES; k++) begin
      perf_bytes     = perf_bytes     + c_perf_bytes[k];
      perf_wgt_reuse = perf_wgt_reuse + c_wgt_reuse[k];
      perf_wgt_load  = perf_wgt_load  + c_wgt_load[k];
    end
  end

`ifdef COCOTB_SIM
  // 이 설계가 의존하는 가정. 어긋나면 조용히 틀리므로 터뜨린다.
  always_ff @(posedge clk) begin
    if (rst_n) begin
      // (1) 코어들은 rd_data_ready 를 상시 1 로 둔다. 그 덕에 데이터
      //     구간에 역압을 중재할 필요가 없다.
      for (int k = 0; k < CORES; k++)
        if (c_rd_data_ready[k] !== 1'b1)
          $fatal(1, "mxu_cluster: core %0d dropped rd_data_ready", k);

      // (2) 진행 중인 브로드캐스트에 요청자가 없으면 안 된다.
      if (rd_busy_r && rd_bcast_r && rd_data_valid && !rd_all_valid
          && !rd_any_valid)
        $fatal(1, "mxu_cluster: broadcast in flight with no requester");

      // (3) 코어들은 같은 사이클에 같은 수의 디스크립터를 받아야 한다.
      //     어긋나면 out_addr 가 어긋난 채 조용히 돌아간다.
      for (int k = 0; k < CORES; k++)
        if (CORE[k].u_core.wgt_load_r + CORE[k].u_core.wgt_reuse_r
            != CORE[0].u_core.wgt_load_r + CORE[0].u_core.wgt_reuse_r)
          $fatal(1, "mxu_cluster: core %0d accepted a different number of descriptors", k);

      // (4) 완료 스큐가 카운터 폭을 넘으면 비교가 뒤집힌다.
      for (int k = 0; k < CORES; k++)
        if ((c_dcnt[k] - emitted) > DCNT_W'(DCNT_W/2))
          $fatal(1, "mxu_cluster: core %0d completion skew %0d exceeds counter",
                 k, c_dcnt[k] - emitted);
    end
  end
`endif

endmodule

`default_nettype wire
