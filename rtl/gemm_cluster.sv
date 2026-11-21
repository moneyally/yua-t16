`timescale 1ns/1ps
`default_nettype none

// gemm_cluster.sv — 텐서 병렬 GEMM 클러스터
//
// ── 왜 ───────────────────────────────────────────────────────────
//
// 단일 코어는 연산이 묶는 쪽이 됐다 (N=32 에서 가동률 95.5%). 코어를
// 여러 개 두면 연산은 늘지만, 각자 활성값을 따로 읽으면 읽기 대역폭도
// 같이 늘어 아무것도 못 번다.
//
// 텐서 병렬은 그것을 피한다. 한 레이어의 가중치를 열 방향으로 쪼개
// 코어마다 다른 조각을 상주시키고, **같은 활성값**을 모든 코어에
// 흘린다. 각 코어는 출력의 다른 열 묶음을 만든다.
//
//   코어 0:  A · W[:, 0:N]      → OUT[:, 0:N]
//   코어 1:  A · W[:, N:2N]     → OUT[:, N:2N]
//   ...
//
// 활성값은 **한 번만 읽어 브로드캐스트**한다. 그래서 연산은 CORES 배가
// 되는데 활성값 트래픽은 그대로다. Groq / Cerebras 가 모델을 여러 칩에
// 쪼개는 방식과 같은 원리다 (spec/memory-bandwidth-analysis.md).
//
// 가중치는 코어마다 다르므로 브로드캐스트할 수 없다. 읽기 채널을
// 라운드로빈으로 나눠 쓴다. weight-stationary 와 함께 쓰면 가중치 적재는
// N 개 디스크립터마다 한 번이라 그 직렬화 비용이 분산된다.
//
// ── 메모리 포트 ──────────────────────────────────────────────────
//
// 바깥으로는 읽기 1개, 쓰기 1개다. 안에서 이렇게 나눈다.
//
//   읽기  ACT 구간: 한 번 요청해 모든 코어에 동시에 넘긴다 (브로드캐스트)
//         WGT 구간: 코어 순서대로 하나씩 (라운드로빈)
//   쓰기  코어 순서대로 하나씩 (라운드로빈)
//
// 코어들은 같은 디스크립터·같은 Kt 로 돌므로 ACT 구간에서 자연히
// 락스텝이 된다. 그 성질에 의존하므로, 어긋나면 조용히 틀리는 대신
// 터지도록 시뮬레이션 단언을 뒀다.
module gemm_cluster #(
  parameter int MAX_KT    = 256,
  parameter int RD_DATA_W = 256,
  parameter int WR_DATA_W = 256,
  parameter int BANKS     = 2,
  parameter int TILE_N    = 32,
  // 코어 수. 연산이 CORES 배, 활성값 트래픽은 그대로.
  parameter int CORES     = 2
)(
  input  logic        clk,
  input  logic        rst_n,

  input  logic        desc_valid,
  input  logic [7:0]  desc_bytes [0:63],
  output logic        desc_ready,

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
  output logic        done_pulse,
  output logic [31:0] perf_cycles,
  output logic [31:0] perf_bytes,
  output logic [31:0] perf_wgt_reuse,
  output logic [31:0] perf_wgt_load
);

  localparam int CSEL_W = (CORES > 1) ? $clog2(CORES) : 1;

  initial begin
    if (CORES < 1)
      $fatal(1, "gemm_cluster: CORES must be >= 1 (got %0d)", CORES);
  end

  // ------------------------------------------------------------------
  // 디스크립터 디코드 (ctrl_fsm 하나로 전체 클러스터를 몬다)
  // ------------------------------------------------------------------
  logic        cmd_valid_ctrl, cmd_ready_all;
  logic [63:0] act_addr_c, wgt_addr_c, out_addr_c;
  logic [31:0] Kt_c;
  logic        wgt_keep_c, wgt_int4_c;
  logic        ctrl_busy, ctrl_done_pulse;
  logic        core_done_any;

  // 코어별 스트라이드. 디스크립터 바이트 44..51 / 52..59 에서 온다.
  // 0 이면 모든 코어가 같은 가중치/출력을 쓴다 (디버그용으로만 의미 있다).
  logic [31:0] wgt_stride_r, out_stride_r;

  ctrl_fsm u_ctrl (
    .clk(clk), .rst_n(rst_n),
    .desc_valid(desc_valid), .desc_bytes(desc_bytes), .desc_ready(desc_ready),
    .cmd_valid(cmd_valid_ctrl), .cmd_ready(cmd_ready_all),
    .act_addr(act_addr_c), .wgt_addr(wgt_addr_c), .out_addr(out_addr_c),
    .Kt(Kt_c), .wgt_keep(wgt_keep_c), .wgt_int4(wgt_int4_c),
    .core_done(core_done_any),
    .busy(ctrl_busy), .done_pulse(ctrl_done_pulse)
  );

  // 스트라이드는 ctrl_fsm 이 내보내지 않으므로 여기서 직접 뽑는다.
  // ctrl_fsm 과 같은 사이클에 맞추기 위해 핸드셰이크에서 캡처한다.
  wire desc_accept_cl = desc_valid & desc_ready;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wgt_stride_r <= 32'd0;
      out_stride_r <= 32'd0;
    end else if (desc_accept_cl) begin
      wgt_stride_r <= {desc_bytes[47], desc_bytes[46],
                       desc_bytes[45], desc_bytes[44]};
      out_stride_r <= {desc_bytes[55], desc_bytes[54],
                       desc_bytes[53], desc_bytes[52]};
    end
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

  genvar ci;
  generate
    for (ci = 0; ci < CORES; ci++) begin : CORE
      // 코어 ci 는 가중치 조각 ci 와 출력 조각 ci 를 담당한다.
      wire [63:0] my_wgt = wgt_addr_c + 64'(wgt_stride_r) * ci;
      wire [63:0] my_out = out_addr_c + 64'(out_stride_r) * ci;

      gemm_core #(
        .MAX_KT(MAX_KT), .DMA_BEAT_BYTES(RD_DATA_W/8),
        .RD_DATA_W(RD_DATA_W), .BANKS(BANKS),
        .WR_DATA_W(WR_DATA_W), .TILE_N(TILE_N)
      ) u_core (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(cmd_valid_gated), .cmd_ready(c_cmd_ready[ci]),
        .act_addr(act_addr_c), .wgt_addr(my_wgt), .out_addr(my_out),
        .Kt(Kt_c), .wgt_keep(wgt_keep_c), .wgt_int4(wgt_int4_c),
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

  // 모든 코어가 받을 준비가 되어야 명령을 발행한다 (락스텝 유지).
  //
  // ready 를 AND 하는 것만으로는 부족하다. 각 코어는 **자기** cmd_ready 로
  // 수락하므로, cmd_valid 를 그대로 넘기면 먼저 준비된 코어가 바로 받는다.
  // ctrl_fsm 은 ST_DISPATCH 에서 cmd_ready_all 을 기다리며 cmd_valid 를
  // 계속 세워 두는데, 그 사이 먼저 받은 코어가 일을 끝내고 L_IDLE 로
  // 돌아오면 **같은 디스크립터를 또 받는다.**
  //
  // 실제로 그렇게 깨졌다: out_addr 가 두 값에서 멈추고, 16개 디스크립터를
  // 흘렸는데 쓰기 주소가 4개뿐이었다 (코어 2개 × 디스크립터 2개). 완료
  // 펄스는 16개가 와서 겉보기로는 정상이었다.
  //
  // 그래서 valid 쪽을 게이트한다. 코어의 cmd_ready 는 cmd_valid 에
  // 의존하지 않으므로(L_IDLE 상태와 뱅크 여유만 본다) 조합 루프가 없다.
  always_comb begin
    cmd_ready_all = 1'b1;
    for (int k = 0; k < CORES; k++)
      cmd_ready_all &= c_cmd_ready[k];
  end

  wire cmd_valid_gated = cmd_valid_ctrl & cmd_ready_all;

  // ------------------------------------------------------------------
  // 읽기 중재 — ACT 는 브로드캐스트, WGT 는 라운드로빈
  //
  // 코어들은 같은 디스크립터로 돌기 때문에 ACT 요청을 같은 사이클에
  // 같은 주소로 낸다. 그때는 한 번만 메모리에 묻고 응답을 모두에게
  // 동시에 넘긴다. 그래서 활성값 트래픽이 CORES 배가 되지 않는다.
  //
  // WGT 요청은 주소가 다르므로 하나씩 처리한다.
  // ------------------------------------------------------------------
  logic              rd_all_valid;    // 전원이 요청 중
  logic              rd_same_addr;    // 전원이 같은 주소 (= ACT 구간)
  logic              rd_any_valid;
  logic [CSEL_W-1:0] rd_grant;        // WGT 구간에서 고른 코어
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

    // 가장 낮은 번호의 요청 코어를 고른다 (고정 우선순위).
    // 공정성은 필요 없다 — 코어들이 같은 일을 같은 순서로 하므로
    // WGT 구간에서 순서대로 한 번씩 돌아간다.
    rd_have_grant = 1'b0;
    rd_grant      = '0;
    for (int k = CORES - 1; k >= 0; k--)
      if (c_rd_req_valid[k]) begin
        rd_grant      = CSEL_W'(k);
        rd_have_grant = 1'b1;
      end
  end

  // 브로드캐스트 모드: 전원이 같은 주소를 요청하는 중
  wire rd_bcast = rd_all_valid & rd_same_addr & (CORES > 1);

  // 진행 중인 읽기가 브로드캐스트였는지 기억한다 (데이터 구간에 필요).
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

  // 바깥 읽기 요청: 진행 중이 아닐 때만 낸다.
  assign rd_req_valid     = !rd_busy_r && rd_any_valid;
  assign rd_req_addr      = c_rd_req_addr[rd_grant];
  assign rd_req_len_bytes = c_rd_req_len[rd_grant];

  // 요청 수락을 코어에 돌려준다.
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

  // 데이터는 소유자(들)에게만 넘긴다.
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

  // 코어들은 rd_data_ready 를 상시 1 로 둔다 (gemm_core 가 그렇다).
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
  // 완료 집계 — 모든 코어가 끝나야 디스크립터 하나가 끝난 것이다
  // ------------------------------------------------------------------
  // 완료는 **코어마다 따로** 센다.
  //
  // 처음에는 c_done 펄스를 하나의 카운터로 모아 CORES 개가 되면 완료로
  // 봤다. 그것은 틀렸다. 쓰기가 직렬화되므로 코어 0 이 먼저 끝나고,
  // 그 다음 디스크립터의 쓰기까지 끝내 **자기 완료 두 개**로 카운터를
  // 채울 수 있다 — 코어 1 은 첫 디스크립터조차 안 끝낸 상태로 호스트가
  // 완료를 받고 미완성 결과를 읽는다. 디스크립터 하나만 보내면 드러나지
  // 않고, 연속으로 밀어 넣을 때 코어 수가 많을수록 빨리 틀어진다
  // (CORES=2 는 #8, CORES=4 는 #0 에서 깨졌다).
  //
  // 코어들은 cmd_ready_all 로 같은 사이클에 같은 디스크립터를 받으므로
  // 수락 수는 항상 같다. 따라서 "모든 코어가 k번째를 끝냈다" 를 보면
  // 된다. 코어별 완료 수와 이미 내보낸 수를 비교한다.
  localparam int DCNT_W = 8;   // 스큐 한계는 ctrl_fsm 의 outstanding (<=3)

  logic [DCNT_W-1:0] c_dcnt [0:CORES-1];   // 코어별 누적 완료 수
  logic [DCNT_W-1:0] emitted;              // 이미 내보낸 완료 수
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

  assign core_done_any = done_all;
  assign done_pulse    = ctrl_done_pulse;

  always_comb begin
    busy = ctrl_busy;
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
  // 이 설계는 두 가정에 의존한다. 어긋나면 조용히 틀리므로 터뜨린다.
  always_ff @(posedge clk) begin
    if (rst_n) begin
      // (1) 코어들은 rd_data_ready 를 상시 1 로 둔다. gemm_core 가 그렇게
      //     만들어져 있고, 그 덕에 데이터 구간에 역압을 중재할 필요가 없다.
      for (int k = 0; k < CORES; k++)
        if (c_rd_data_ready[k] !== 1'b1)
          $fatal(1, "gemm_cluster: core %0d dropped rd_data_ready", k);

      // (2) 브로드캐스트는 전원이 같은 주소를 요청할 때만 한다.
      //     진행 중인 브로드캐스트 동안 소유권이 바뀌면 안 된다.
      if (rd_busy_r && rd_bcast_r && rd_data_valid && !rd_all_valid
          && !rd_any_valid)
        $fatal(1, "gemm_cluster: broadcast in flight with no requester");

      // (3) 코어들은 같은 사이클에 같은 수의 디스크립터를 받아야 한다.
      //     어긋나면 out_addr 가 어긋난 채 조용히 돌아간다.
      for (int k = 0; k < CORES; k++)
        if (CORE[k].u_core.wgt_load_r + CORE[k].u_core.wgt_reuse_r
            != CORE[0].u_core.wgt_load_r + CORE[0].u_core.wgt_reuse_r)
          $fatal(1, "gemm_cluster: core %0d accepted a different number of descriptors", k);

      // (4) 코어 간 완료 스큐가 카운터 폭을 넘으면 비교가 뒤집힌다.
      //     ctrl_fsm 의 outstanding 이 3 이하로 묶으므로 8 로 충분하지만,
      //     가정이 깨지면 조용히 틀리는 대신 터뜨린다.
      for (int k = 0; k < CORES; k++)
        if ((c_dcnt[k] - emitted) > DCNT_W'(DCNT_W/2))
          $fatal(1, "gemm_cluster: core %0d completion skew %0d exceeds counter",
                 k, c_dcnt[k] - emitted);
    end
  end
`endif

endmodule

`default_nettype wire
