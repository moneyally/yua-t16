// ============================================================================
// wdog_timer.sv — 워치독 타이머
//
// SSOT: spec/watchdog.md (비트 배치·동작·타임아웃 수식이 전부 거기서 온다)
//
// ----------------------------------------------------------------------------
// 왜 있는가
// ----------------------------------------------------------------------------
// `rtl/reg_top.sv` 에 이렇게 적혀 있었다:
//     // Watchdog control (Proto-A stub: register only, no actual timer)
// 값을 받아 두고 **아무 일도 하지 않았다.** 유일하게 동작하던 것은 bit[31]
// 테스트 주입뿐이다.
//
// `reset_seq` 는 이미 `wdog_reset` 입력과 `BOOT_CAUSE[1]=WDOG` 래치를 갖고 있다.
// **없던 것은 타이머 하나뿐이었다.** 이 파일이 그것이다.
//
// 보드에서 이게 왜 필요한가: `DELTA_STEP` 이 멈추면 호스트가 할 수 있는 일이 없다.
// `SW_RESET` 은 호스트가 레지스터를 쓸 수 있을 때만 듣는다. 칩이 스스로 빠져나올
// 길이 하나는 있어야 한다.
//
// ----------------------------------------------------------------------------
// 계약 (spec/watchdog.md 2절)
// ----------------------------------------------------------------------------
//   타임아웃 = (period + 1) × 2^PRESCALE_LOG2 사이클
//   en=0      → 카운터를 리로드값으로 **유지**한다 (세지 않는다)
//   kick=1    → 리로드 (en 과 무관)
//   period 쓰기 → **다음 리로드부터** 적용. 진행 중인 창은 안 건드린다
//                 (즉시 반영이면 period 를 계속 다시 쓰는 것만으로 영원히
//                  안 터진다 — 그건 워치독이 아니다)
//   timeout   → 1사이클 펄스. 펄스와 함께 스스로 리로드한다
// ============================================================================
`timescale 1ns/1ps
`default_nettype none

module wdog_timer #(
  // 2^10 = 1024 사이클이 창 하나의 눈금이다. **테스트에서 바꾸지 않는다** —
  // 바꾸면 보드에서 도는 것과 다른 회로를 검증하게 된다 (spec/watchdog.md 1절).
  parameter int PRESCALE_LOG2 = 10,
  parameter int PERIOD_W      = 16
)(
  input  wire                 clk,
  input  wire                 rst_n,

  input  wire                 en,        // WDOG_CTRL[0]
  input  wire                 kick,      // WDOG_CTRL[1] 쓰기 펄스
  input  wire [PERIOD_W-1:0]  period,    // WDOG_CTRL[23:8]

  output logic                timeout    // 1사이클 펄스 → reset_seq.wdog_reset
);

  localparam int PRE_W   = PRESCALE_LOG2;
  localparam logic [PRE_W-1:0] PRE_MAX = {PRE_W{1'b1}};   // 2^N - 1

  logic [PRE_W-1:0]    pre_cnt;
  logic [PERIOD_W-1:0] win_cnt;

  // 리로드 조건을 한 곳에 모은다. en=0 이 리로드를 **매 사이클** 걸어 두는 것이
  // 곧 "en 을 올리면 처음부터 센다" 를 만든다 — 따로 처리하지 않는다.
  wire reload = (!en) || kick;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      pre_cnt <= '0;
      win_cnt <= '0;
      timeout <= 1'b0;
    end else begin
      timeout <= 1'b0;                       // 기본은 0 — 펄스는 1사이클이다

      if (reload) begin
        pre_cnt <= '0;
        win_cnt <= period;                   // period 는 **리로드 순간에만** 샘플된다
      end else if (pre_cnt == PRE_MAX) begin
        pre_cnt <= '0;
        if (win_cnt == '0) begin
          timeout <= 1'b1;
          win_cnt <= period;                 // 터진 뒤 새 창 (연속 발화 방지)
        end else begin
          win_cnt <= win_cnt - 1'b1;
        end
      end else begin
        pre_cnt <= pre_cnt + 1'b1;
      end
    end
  end

endmodule

`default_nettype wire
