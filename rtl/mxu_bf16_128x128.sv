// mxu_bf16_128x128.sv — BF16 128×128 MXU (병렬 타일 그리드)
// SSOT: ORBIT_G3_RTL_PLAN.md (G3-RTL-004), ORBIT_G3_ARCHITECTURE.md
//
// 외부 인터페이스는 이전 반복(iterative) 버전과 동일하다. 드롭인 교체된다.
//
// ── 이전 구현의 문제 두 가지 ──────────────────────────────
//  1) 정확성: S_STORE_TILE 이 FP32 비트패턴을 정수 덧셈으로 누적했다.
//     (`tile_acc <= tile_out + tile_acc`). K=1 에서는 우연히 맞지만
//     K>1 이면 결과가 틀린다. 원본 주석도 "NOT correct FP32 add!" 라고
//     적어 두었고, 기존 테스트 5개가 전부 k_steps=1 이라 잡히지 않았다.
//  2) 성능: 16×16 타일 1개를 64개 논리 타일 위치에 재사용하면서
//     위치마다 CLR→LOAD→COMPUTE→WAIT→STORE 5상태를 돌았다.
//     K-step 당 약 320 사이클.
//
// ── 이번 구현 ─────────────────────────────────────────────
//  PAR×PAR 물리 타일 그리드를 둔다. PAR 은 TILES 를 나누어 떨어져야 한다.
//
//  PAR == TILES (기본 8): 논리 타일마다 물리 타일이 하나씩 있으므로
//    누산을 타일 내부 FP32 누산기가 그대로 수행한다. 외부 누산기도,
//    외부 덧셈도 필요 없다 — 정확성이 구조적으로 보장된다.
//    처리량 K-step 당 1 사이클.
//
//  PAR < TILES: 물리 타일이 여러 논리 위치를 나눠 맡는다. 이때만
//    외부 누산기를 쓰고, 덧셈은 bf16_pkg::fp32_add 로 올바르게 한다.
//    K-step 당 PASSES×4 사이클 (PASSES = (TILES/PAR)²).
//
//  면적/성능 트레이드오프는 PAR 하나로 조절한다:
//    PAR=8 → 64 타일 = 16384 MAC,  K-step 1 사이클   (최대 성능)
//    PAR=4 → 16 타일 =  4096 MAC,  K-step 16 사이클
//    PAR=2 →  4 타일 =  1024 MAC,  K-step 64 사이클
//    PAR=1 →  1 타일 =   256 MAC,  K-step 256 사이클 (최소 면적)
`timescale 1ns/1ps
`default_nettype none

/* verilator lint_off UNUSEDSIGNAL */

module mxu_bf16_128x128 #(
  parameter int DIM      = 128,
  parameter int TILE_DIM = 16,
  parameter int TILES    = 8,   // DIM / TILE_DIM
  parameter int PAR      = 8,   // 물리 타일 그리드 한 변. TILES 를 나누어야 함

  // 출력 방식 (mxu_bf16_16x16 과 같은 의미).
  // FP32_OUT=1 (기본): acc_flat 에 128×128 FP32 를 조합으로 내보낸다.
  //   타일마다 PE 256개분 변환기가 붙는다.
  // FP32_OUT=0 + STREAM_RD=1: acc_flat 을 0 으로 묶고, 타일당 변환기
  //   하나를 공유해 rd_idx/rd_fp32 로 순차로 읽는다.
  //   PAR=8 기준 변환기가 16,384개에서 64개로 줄어든다.
  parameter int FP32_OUT  = 1,
  parameter int STREAM_RD = 0
)(
  input  logic        clk,
  input  logic        rst_n,

  // 제어
  input  logic        start,
  input  logic        acc_clr,
  input  logic [15:0] k_steps,

  // 데이터: K-step 하나당 A 의 열 128개, B 의 행 128개 (BF16)
  input  logic [15:0] a_col [0:DIM-1],
  input  logic [15:0] b_row [0:DIM-1],
  input  logic        data_valid,

  // 상태
  output logic        busy,
  output logic        done_pulse,
  output logic        ready,

  // 누산기 출력: 128×128 FP32 (FP32_OUT=0 이면 전부 0)
  output logic [31:0] acc_flat [0:DIM*DIM-1],

  // 공유 변환기 순차 읽기 (STREAM_RD=1 일 때만 유효)
  // rd_idx 는 행 우선 평탄 인덱스다: idx = row * DIM + col
  input  logic [$clog2(DIM*DIM)-1:0] rd_idx,
  output logic [31:0]                rd_fp32
);

  import bf16_pkg::fp32_add;

  localparam int GROUPS = TILES / PAR;          // 한 변당 패스 수
  localparam int PASSES = GROUPS * GROUPS;      // K-step 당 패스 수

  // 파라미터 정합성 — 합성/시뮬 양쪽에서 조기에 잡는다
  initial begin
    if (TILES * TILE_DIM != DIM)
      $fatal(1, "mxu_bf16_128x128: TILES*TILE_DIM != DIM");
    if (PAR < 1 || TILES % PAR != 0)
      $fatal(1, "mxu_bf16_128x128: PAR must divide TILES");
    if (STREAM_RD != 0 && PAR != TILES)
      $fatal(1, "mxu_bf16_128x128: STREAM_RD requires PAR == TILES (누산값이 타일에 있어야 한다)");
  end

  // ═══════════════════════════════════════════════════════════
  // 물리 타일 그리드 PAR×PAR
  //
  // 포트 연결은 원소 단위 continuous assign 으로 한다.
  // 2D unpacked 배열 슬라이스를 포트에 직접 묶는 문법
  // (`.a_row(tile_a[gr])`) 은 Icarus 가 지원하지 않는다.
  // ═══════════════════════════════════════════════════════════
  logic [31:0] tile_out [0:PAR-1][0:PAR-1][0:TILE_DIM-1][0:TILE_DIM-1];
  logic        tile_en, tile_clr;

  // 현재 패스가 맡은 논리 타일 그룹의 시작 좌표
  localparam int GRP_W = (GROUPS > 1) ? $clog2(GROUPS) : 1;
  logic [GRP_W-1:0] grp_r, grp_c;

  // ─── 공유 변환기 읽기 라우팅 ──────────────────────────────────
  //
  // rd_idx 는 행 우선 평탄 인덱스다: idx = row * DIM + col
  //   row = idx[2*TD_W+TIL_W-1 : TD_W+TIL_W]   (상위 절반)
  //   col = idx[TD_W+TIL_W-1   : 0]            (하위 절반)
  // 각 절반을 다시 (논리 타일 번호, 타일 내 위치) 로 쪼갠다.
  //
  // STREAM_RD 은 PAR == TILES 일 때만 쓸 수 있다. PAR 이 작으면 누산값이
  // 타일이 아니라 외부 acc_mem 에 있어서 타일을 읽어봐야 의미가 없다.
  localparam int TD_W  = $clog2(TILE_DIM);   // 타일 내 좌표 폭 (4)
  localparam int TIL_W = $clog2(TILES);      // 타일 번호 폭 (3)
  localparam int HALF  = TD_W + TIL_W;       // 행/열 한 쪽 폭 (7)

  logic [TD_W*2-1:0] tile_rd_idx;                      // 모든 타일에 같은 값
  logic [31:0]       tile_rd_fp32 [0:PAR-1][0:PAR-1];

  logic [TIL_W-1:0] rd_tile_r, rd_tile_c;
  logic [TD_W-1:0]  rd_in_r,   rd_in_c;

  always_comb begin
    rd_tile_r = rd_idx[2*HALF-1    -: TIL_W];   // row 상위 TIL_W
    rd_in_r   = rd_idx[2*HALF-TIL_W-1 -: TD_W]; // row 하위 TD_W
    rd_tile_c = rd_idx[HALF-1      -: TIL_W];   // col 상위 TIL_W
    rd_in_c   = rd_idx[TD_W-1      -: TD_W];    // col 하위 TD_W
    tile_rd_idx = {rd_in_r, rd_in_c};
  end

  genvar gr, gc, gi, gj;
  generate
    for (gr = 0; gr < PAR; gr++) begin : G_TROW
      for (gc = 0; gc < PAR; gc++) begin : G_TCOL

        logic [15:0] a_s [0:TILE_DIM-1];
        logic [15:0] b_s [0:TILE_DIM-1];
        logic [31:0] o   [0:TILE_DIM-1][0:TILE_DIM-1];

        // 물리 타일 (gr,gc) ← 논리 타일 (grp_r*PAR+gr, grp_c*PAR+gc)
        for (gi = 0; gi < TILE_DIM; gi++) begin : G_SLICE
          assign a_s[gi] = a_col[(grp_r * PAR + gr) * TILE_DIM + gi];
          assign b_s[gi] = b_row[(grp_c * PAR + gc) * TILE_DIM + gi];
        end

        mxu_bf16_16x16 #(
          .ROWS      (TILE_DIM),
          .COLS      (TILE_DIM),
          .FP32_OUT  (FP32_OUT),
          .STREAM_RD (STREAM_RD)
        ) u_tile (
          .clk     (clk),
          .rst_n   (rst_n),
          .en      (tile_en),
          .acc_clr (tile_clr),
          .a_row   (a_s),
          .b_col   (b_s),
          .acc_out (o),
          .rd_idx  (tile_rd_idx),
          .rd_fp32 (tile_rd_fp32[gr][gc]),
          .busy    ()
        );

        // 절차적 코드에서 인덱싱할 수 있도록 모듈 레벨 배열로 끌어낸다
        for (gi = 0; gi < TILE_DIM; gi++) begin : G_OR
          for (gj = 0; gj < TILE_DIM; gj++) begin : G_OC
            assign tile_out[gr][gc][gi][gj] = o[gi][gj];
          end
        end

      end
    end
  endgenerate

  // 상태 타입은 모듈 스코프에 둔다 (Icarus 는 generate 안의 typedef 를 못 묶는다)
  typedef enum logic [1:0] { P_IDLE, P_RUN, P_DONE } pstate_t;
  typedef enum logic [2:0] {
    Q_IDLE, Q_CLR, Q_COMPUTE, Q_WAIT, Q_STORE, Q_NEXT_PASS, Q_DONE
  } qstate_t;

  logic [15:0] k_cnt, k_total;

  generate
    if (PASSES == 1) begin : G_FULL_PARALLEL
      // ═══════════════════════════════════════════════════════
      // 완전 병렬: 누산은 타일 내부 FP32 누산기가 한다.
      // 외부 누산기 없음 → 정수 덧셈 버그가 구조적으로 불가능.
      // ═══════════════════════════════════════════════════════
      assign grp_r = '0;   // 패스가 하나뿐이라 그룹 오프셋은 항상 0
      assign grp_c = '0;

      for (gr = 0; gr < TILES; gr++) begin : F_ROW
        for (gc = 0; gc < TILES; gc++) begin : F_COL
          for (genvar i = 0; i < TILE_DIM; i++) begin : F_I
            for (genvar j = 0; j < TILE_DIM; j++) begin : F_J
              assign acc_flat[(gr * TILE_DIM + i) * DIM + (gc * TILE_DIM + j)]
                     = tile_out[gr][gc][i][j];
            end
          end
        end
      end

      pstate_t pst;

      assign busy  = (pst == P_RUN);
      assign ready = (pst == P_IDLE) || (pst == P_RUN);

      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          pst        <= P_IDLE;
          k_cnt      <= 16'd0;
          k_total    <= 16'd0;
          tile_en    <= 1'b0;
          tile_clr   <= 1'b0;
          done_pulse <= 1'b0;
        end else begin
          done_pulse <= 1'b0;
          tile_en    <= 1'b0;
          tile_clr   <= 1'b0;

          case (pst)
            P_IDLE: begin
              if (acc_clr) begin
                tile_clr <= 1'b1;
              end else if (start) begin
                k_total <= k_steps;
                k_cnt   <= 16'd0;
                pst     <= P_RUN;
              end
            end

            P_RUN: begin
              if (k_cnt >= k_total) begin
                pst <= P_DONE;
              end else if (data_valid) begin
                tile_en <= 1'b1;        // 64개 타일이 동시에 한 K-step 누산
                k_cnt   <= k_cnt + 16'd1;
              end
            end

            P_DONE: begin
              done_pulse <= 1'b1;
              pst        <= P_IDLE;
            end

            default: pst <= P_IDLE;
          endcase
        end
      end

    end else begin : G_PARTIAL
      // ═══════════════════════════════════════════════════════
      // 부분 병렬: 물리 타일이 여러 논리 위치를 돌려 맡으므로
      // 외부 누산기가 필요하다. 덧셈은 fp32_add 로 올바르게.
      // ═══════════════════════════════════════════════════════
      logic [31:0] acc_mem [0:DIM*DIM-1];
      // 평탄 16384 루프는 Verilator 의 기본 언롤 한도(3075)를 넘는다.
      // 중첩 128×128 로 쓰면 어느 툴에서도 문제없다.
      for (gr = 0; gr < DIM; gr++) begin : F_OROW
        for (gc = 0; gc < DIM; gc++) begin : F_OCOL
          assign acc_flat[gr * DIM + gc] = acc_mem[gr * DIM + gc];
        end
      end

      qstate_t qst;

      assign busy  = (qst != Q_IDLE) && (qst != Q_DONE);
      // ready 는 '새 K-step 데이터가 실제로 필요할 때'만 올라가야 한다.
      // Q_CLR 은 패스마다 다시 들르므로 조건에 !data_latched 가 필요하다.
      // 빠지면 호스트가 다음 K-step 데이터를 패스 도중에 밀어 넣는다.
      assign ready = (qst == Q_IDLE)
                  || (qst == Q_CLR && !data_latched && k_cnt < k_total);

      logic data_latched;
      localparam int PAR_W = (PAR > 1) ? $clog2(PAR) : 1;
      logic [PAR_W-1:0] st_r, st_c;   // Q_STORE 순회 좌표

      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          qst          <= Q_IDLE;
          grp_r        <= '0;
          grp_c        <= '0;
          k_cnt        <= 16'd0;
          k_total      <= 16'd0;
          tile_en      <= 1'b0;
          tile_clr     <= 1'b0;
          done_pulse   <= 1'b0;
          data_latched <= 1'b0;
          st_r         <= '0;
          st_c         <= '0;
          for (int i = 0; i < DIM*DIM; i++) acc_mem[i] <= 32'd0;
        end else begin
          done_pulse <= 1'b0;
          tile_en    <= 1'b0;
          tile_clr   <= 1'b0;

          case (qst)
            Q_IDLE: begin
              if (acc_clr) begin
                for (int i = 0; i < DIM*DIM; i++) acc_mem[i] <= 32'd0;
                tile_clr <= 1'b1;
              end else if (start) begin
                k_total <= k_steps;
                k_cnt   <= 16'd0;
                grp_r   <= '0;
                grp_c   <= '0;
                qst     <= Q_CLR;
              end
            end

            Q_CLR: begin
              if (k_cnt >= k_total) begin
                qst <= Q_DONE;
              end else if (data_valid || data_latched) begin
                data_latched <= 1'b1;
                tile_clr     <= 1'b1;   // 패스마다 타일 누산기 비움
                qst          <= Q_COMPUTE;
              end
            end

            Q_COMPUTE: begin
              tile_en <= 1'b1;          // 이 패스의 외적 1회
              qst     <= Q_WAIT;
            end

            Q_WAIT: begin
              qst <= Q_STORE;           // tile_out 은 레지스터 출력
            end

            Q_STORE: begin
              // 물리 타일 한 개(16×16 = 256 덧셈)씩만 외부 누산기에 반영한다.
              // PAR×PAR 전체를 한 사이클에 더하면 PAR=2 에서도 1024 덧셈이라
              // 합성이 되지 않는다 (16×16 타일 자체가 이미 256 가산기다).
              for (int i = 0; i < TILE_DIM; i++)
                for (int j = 0; j < TILE_DIM; j++)
                  acc_mem[((grp_r * PAR + st_r) * TILE_DIM + i) * DIM
                          + (grp_c * PAR + st_c) * TILE_DIM + j]
                    <= fp32_add(
                         acc_mem[((grp_r * PAR + st_r) * TILE_DIM + i) * DIM
                                 + (grp_c * PAR + st_c) * TILE_DIM + j],
                         tile_out[st_r][st_c][i][j]);

              if (st_c < PAR - 1) begin
                st_c <= st_c + 1'b1;
              end else if (st_r < PAR - 1) begin
                st_c <= '0;
                st_r <= st_r + 1'b1;
              end else begin
                st_c <= '0;
                st_r <= '0;
                qst  <= Q_NEXT_PASS;
              end
            end

            Q_NEXT_PASS: begin
              if (grp_c < GROUPS - 1) begin
                grp_c <= grp_c + 1'b1;
                qst   <= Q_CLR;
              end else if (grp_r < GROUPS - 1) begin
                grp_c <= '0;
                grp_r <= grp_r + 1'b1;
                qst   <= Q_CLR;
              end else begin
                grp_r        <= '0;
                grp_c        <= '0;
                k_cnt        <= k_cnt + 16'd1;
                data_latched <= 1'b0;   // 다음 K-step 데이터 요청
                qst          <= Q_CLR;
              end
            end

            Q_DONE: begin
              done_pulse <= 1'b1;
              qst        <= Q_IDLE;
            end

            default: qst <= Q_IDLE;
          endcase
        end
      end
    end
  endgenerate

  // ─── 공유 변환기 최종 선택 ────────────────────────────────────
  // STREAM_RD 일 때 PAR == TILES 이므로 물리 타일 번호 = 논리 타일 번호다.
  generate
    if (STREAM_RD != 0) begin : G_RD_MUX
      assign rd_fp32 = tile_rd_fp32[rd_tile_r][rd_tile_c];
    end else begin : G_RD_TIE
      assign rd_fp32 = 32'd0;
    end
  endgenerate

endmodule

`default_nettype wire
