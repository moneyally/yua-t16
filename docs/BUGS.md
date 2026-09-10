# BUGS.md — 파형 근거가 있는 미해결 버그

각 항목은 **재현 명령 + 실행 출력 + 파형 파일**을 반드시 갖는다.
근거 없이 "버그인 것 같다"는 여기 적지 않는다 (그건 `docs/AUDIT.md` 의 "미검증" 항목).

---

## BUG-001 — fault 난 디스크립터가 완료 IRQ(`DESC_DONE`)를 올린다

**상태**: 재현 완료 · 파형 확보 · **미수정**
**발견**: `docs/AUDIT.md` §4 (코드 읽기 가설) → 2026-09-10 테스트로 확정
**심각도**: 높음. 호스트가 실패를 성공으로 처리한다.

### 증상

CRC 불일치나 illegal opcode 로 **실패한** 디스크립터 하나가
`TC0_FAULT` 와 `DESC_DONE` **양쪽 IRQ 를 동시에** 올린다.
호스트 소프트웨어(`tools/orbit_device.py` 의 poll 경로)는 `DESC_DONE` 을 보고
"디스크립터가 정상 완료됐다"고 판단할 수 있다.

### 재현

```bash
$ python3 tb/run_tb.py g2_ctrl_top tb_g2_ctrl_top_fault_irq $(ls rtl/*.sv)
```

출력 (해당 부분):

```
  2460.00ns INFO  cocotb.g2_ctrl_top   NOP        IRQ_PENDING = 0x00000001 ['DESC_DONE']
  4920.00ns INFO  cocotb.g2_ctrl_top   ILLEGAL    IRQ_PENDING = 0x00000021 ['DESC_DONE', 'TC0_FAULT']
  4930.00ns INFO  cocotb.g2_ctrl_top   ILLEGAL    TC0_FAULT_STATUS = 0x00000001
  7390.00ns INFO  cocotb.g2_ctrl_top   CRCFAIL    IRQ_PENDING = 0x00000021 ['DESC_DONE', 'TC0_FAULT']
** TESTS=3 PASS=1 FAIL=2 SKIP=0
```

- 대조군(정상 NOP): `0x01` = `DESC_DONE` 만. ✔ 기대대로.
- illegal opcode: `0x21` = `DESC_DONE | TC0_FAULT`. ✘ **`DESC_DONE` 이 있으면 안 된다.**
- CRC 불일치: `0x21`. ✘ 같은 문제.

파형: `build/tb/g2_ctrl_top/g2_ctrl_top.fst` (GTKWave 로 열 것)

### 어느 사이클에 어떤 신호가 기대와 다른가

`desc_fsm_v2` 단독, illegal opcode(0xFF) 주입. 사이클 0 은 디스크립터 수락 직후.

```bash
$ python3 tb/run_tb.py desc_fsm_v2 tb_desc_fsm_v2_done_pulse   # test_diag_fault_cycle_table
```

```
cyc | state | busy | fault_valid | fault_code | done_pulse
----+-------+------+-------------+------------+-----------
  0 |     2 |    1 |           0 |        0x0 |          0     ST_CRC_CHECK
  1 |     3 |    1 |           0 |        0x0 |          0     ST_DECODE
  2 |     7 |    1 |           0 |        0x1 |          0     ST_FAULT   ← fault_code 확정
  3 |     6 |    0 |           1 |        0x1 |          1     ST_DONE    ← ★ fault_valid 와 done_pulse 가 같은 사이클
  4 |     0 |    0 |           1 |        0x1 |          0     ST_IDLE
  5 |     0 |    0 |           0 |        0x1 |          0
```

(state 인코딩: 0=ST_IDLE 1=ST_LATCH 2=ST_CRC_CHECK 3=ST_DECODE 4=ST_DISPATCH 5=ST_WAIT 6=ST_DONE 7=ST_FAULT — `rtl/desc_fsm_v2.sv:75-82`)

**사이클 3 이 문제다.** FSM 이 `ST_FAULT` 에서 곧바로 `ST_DONE` 으로 가고,
`ST_DONE` 은 무조건 `done_pulse` 를 낸다. 그래서 실패 트랜잭션인데 완료 펄스가 나온다.

### 원인 (코드 3줄)

```systemverilog
// rtl/desc_fsm_v2.sv:331-334
ST_FAULT: begin
  busy    = 1'b1;
  state_n = ST_DONE;  // transition to DONE to re-enter IDLE   ← 여기
end

// rtl/desc_fsm_v2.sv:327-329
ST_DONE: begin
  done_pulse = 1'b1;  // ← 진입 경로를 구분하지 않는다
  state_n    = ST_IDLE;
end
```

```systemverilog
// rtl/g2_ctrl_top.sv:456-460
assign irq_sources = {20'd0, trace_wrap_irq, 1'b0, 1'b0, 1'b0, 1'b0, 1'b0,
                       fsm_fault_valid, oom_emerg_edge, oom_press_edge,
                       1'b0, 1'b0, fsm_done_pulse};
//                                  ^^^^^^^^^^^^^^ = irq_sources[0] = IrqBit.DESC_DONE
//     fsm_fault_valid = irq_sources[5] = IrqBit.TC0_FAULT
```

`ST_FAULT` 가 `ST_IDLE` 로 직접 가지 않고 `ST_DONE` 을 경유하는 것이 근본 원인이다.
주석 `// transition to DONE to re-enter IDLE` 이 의도를 그대로 보여준다 — IDLE 로 돌아가는
경로를 재사용하려다가 완료 펄스까지 같이 딸려 나왔다.

### 부수 영향 (같은 원인, 아직 별도 테스트 없음 — 미검증)

`fsm_done_pulse` 는 `rtl/g2_ctrl_top.sv` 에서 IRQ 외에 세 곳을 더 움직인다.
fault 트랜잭션마다 이들도 함께 틀어질 것으로 **추정**한다:

| 라인 | 코드 | 추정 영향 |
|---|---|---|
| 375 | `if (fsm_done_pulse) done_count_r <= done_count_r + 1'b1;` | 실패한 디스크립터가 완료 카운터를 올린다 |
| 388 | `assign oom_alloc_dec = fsm_done_pulse;` | OOM 사용량 감소가 fault 경로에서도 일어난다 |
| 390 | `assign oom_dma_dec = gemm_done_pulse \| (fsm_fault_valid & opcode==0x02)` | fault 시 `oom_alloc_dec` 와 중복 감소 가능 |

### 수정 방향 (아직 적용하지 않음 — 사용자 확인 후)

세 가지 중 하나. **어느 쪽이든 `docs/DESIGN.md` 9절의 완료 신호 정의를 먼저 고쳐야 한다.**

1. `ST_FAULT: state_n = ST_IDLE;` 로 직행. 가장 작다. `ST_DONE` 을 거치며 하던
   다른 일이 있는지 확인 필요.
2. `ST_DONE` 진입 경로를 구분하는 플래그를 두고 `done_pulse = ST_DONE && !came_from_fault`.
3. 완료 신호를 `done_pulse` 하나에서 `done_ok` / `done_err` 둘로 나눈다.
   DR1 의 `DELTA_STEP` 완료 신호를 설계할 때 어차피 필요한 구분이라 이쪽이 미래에 맞다.

**이 버그를 고치기 전에 `docs/DESIGN.md` 9절을 먼저 고친다** (DESIGN.md 는 SSOT).

---

## BUG-002 — `cdc_fifo.sv` 는 컴파일된 적이 없었다 (2026-09-10 수정됨)

**상태**: **수정 완료**. 기록용으로 남긴다.

`initial` 블록이 **포트 리스트 안**에 들어 있었다 (`)(` 와 포트 선언 사이).
합법 SystemVerilog 가 아니라서 iverilog·sv2v 둘 다 거부한다.

```bash
$ iverilog -g2012 -o /dev/null rtl/cdc_fifo.sv     # 수정 전
rtl/cdc_fifo.sv:19: syntax error
rtl/cdc_fifo.sv:19: Errors in port declarations.
rtl/cdc_fifo.sv:20: syntax error
I give up.
```

즉 `tb/tb_cdc_fifo_async.py`, `tb/tb_cdc_fifo_reset.py` 는 **한 번도 실행된 적이 없다.**
그런데 `CLAUDE.md` 2절은 `cdc_fifo` 를 "구조 양호"로 분류하고 있었다.

수정: 파라미터 검사 `initial` 을 모듈 본문으로 옮기고 `` `ifdef COCOTB_SIM `` 으로 감쌌다
(CLAUDE.md 규칙 1 이 허용하는 유일한 `initial` 형태). 동작은 동일하다.

```bash
$ iverilog -g2012 -DCOCOTB_SIM=1 -o /dev/null rtl/cdc_fifo.sv && echo OK
COMPILE OK
```

**남은 일**: Vivado 쪽 elaboration-time 파라미터 검사가 없어졌다. `rtl/cdc_fifo.sv` 의
TODO 주석 참조. 그리고 위 두 테스트벤치는 아직 **실행해 본 적이 없다** — 통과할지 미지수.

---

## BUG-003 — `mxu_bf16_16x16.sv` 의 함수 안 무한 `while` 루프 (2026-09-10 수정됨)

**상태**: **수정 완료**. 기록용.

```systemverilog
// rtl/mxu_bf16_16x16.sv:176 (수정 전) — fp32_add 정규화 시프트
while (!sum[24] && r_e > 0) begin
  sum = sum << 1;
  r_e = r_e - 1;
end
```

경계 없는 `while` 은 하드웨어가 아니다. yosys 는 이걸
`ERROR: Function \fp32_add can only be called with constant arguments.` 로 거부하고
(0.33·0.69 동일) `scripts/synth_gate.sh` 가 여기서 막혔다.

수정: 경계 25 의 `for` 루프로 교체. `sum` 은 26비트이고 이 분기는 `sum!=0 && !sum[25]`
일 때만 오므로 선두 1 은 `[24:0]` 안에 있다 → 최대 24회 시프트. 조건이 거짓이 되면
이후 반복은 아무 일도 하지 않으므로 동작은 동일하다.

**남은 일**: `CLAUDE.md` 규칙 1 의 금지 토큰 목록에 `while` 이 없다.
`real`·`$exp` 와 같은 이유로 금지 대상이다. 규칙 1 에 추가할 것을 제안한다.
(`scripts/check_banned_tokens.py` 에는 아직 넣지 않았다 — 규칙 변경은 사용자 결정.)

## BUG-006 — 시뮬레이션 전용 X 가드가 합성에서 로직을 통째로 삭제한다

**상태**: **수정 완료** (2026-09-10). `scripts/synth_gate.sh` STAGE 2 의 `cells=0` 이 단서였다.
**심각도**: 매우 높음. 고치기 전 상태로 FPGA/ASIC 에 올렸다면 **가중치 SRAM 이 없고
GEMM 결과가 0 으로 기록되는 칩**이 나온다.

### 증상

`scripts/synth_gate.sh` STAGE 2 출력에서:

```
  ok      act_sram                 64s      cells=99294
  ok      wgt_sram                 41s      cells=0        ← ★
  ok      gemm_core               162s      cells=1523     ← ★ 너무 작다
```

`act_sram` 과 `wgt_sram` 은 사실상 같은 SRAM 인데 셀 수가 99,294 대 **0** 이다.
0 셀은 "합성 성공"이 아니라 **"전부 최적화로 사라졌다"** 는 뜻이다.

### 원인

```systemverilog
// rtl/wgt_sram.sv:26 (수정 전)
function automatic logic has_x_addr(input logic [AW-1:0] a);
  has_x_addr = (^a === 1'bx);
endfunction
...
if (we && !has_x_addr(waddr)) mem[waddr] <= wdata;
if (re && !has_x_addr(raddr)) rdata <= mem[raddr];
```

`===` 는 **시뮬레이션 전용 연산자**다 (IEEE 합성 서브셋에 없다).
yosys 는 `has_x_addr` 를 상수로 접고, 그 결과 `we && !has_x_addr(...)` 가 **항상 거짓**이 된다.
쓰기도 읽기도 일어나지 않으니 `rdata` 는 리셋값 0 에 고정되고, `opt_clean` 이 메모리 전체를 지운다.

`act_sram.sv` 에는 이 가드가 없다. 그래서 둘의 셀 수가 갈렸다.

`gemm_core.sv` 도 같은 패턴을 결과 쓰기 경로에 쓰고 있었다 (`x32()`, `pack4()`).

### 측정 (가드만 상수 0 으로 바꾸고 재합성)

| 모듈 | 수정 전 | 수정 후 | 배수 |
|---|---|---|---|
| `wgt_sram` | **0** | **99,294** | ∞ (act_sram 과 정확히 동일) |
| `gemm_core` | **1,523** | **34,735** | 22.8× |

```bash
$ yosys -p "read_verilog -defer build/synth_gate/flat.v; hierarchy -check -top wgt_sram; synth -top wgt_sram; stat"
   Number of cells:              99294        # 수정 후
$ yosys -p "... -top gemm_core; synth -top gemm_core; stat"
   Number of cells:              34735        # 수정 후
```

즉 **`gemm_core` 로직의 약 95% 가 합성에서 사라지고 있었다.** 결과 쓰기 데이터가
상수 0 으로 접히면서 그 앞단이 전부 죽은 로직이 된 것이다.

### 수정

두 파일 모두 X 가드를 `` `ifdef COCOTB_SIM `` 안으로 넣고, 합성 경로에서는 상수 0 /
가드 없는 경로를 쓰게 했다. **시뮬레이션 동작은 완전히 동일하다** (CLAUDE.md 규칙 1 이
허용하는 유일한 시뮬 전용 코드 형태).

```systemverilog
function automatic logic has_x_addr(input logic [AW-1:0] a);
`ifdef COCOTB_SIM
  has_x_addr = (^a === 1'bx);
`else
  has_x_addr = 1'b0;
`endif
endfunction
```

### 재발 방지

1. `scripts/synth_gate.sh` STAGE 2 가 이제 `cells=0` 을 **ZERO 로 따로 표시**한다.
   0 셀은 통과로 세지 않는다.
2. **`CLAUDE.md` 규칙 1 의 금지 토큰에 `===` / `!==` (X 비교) 를 추가할 것을 제안한다.**
   `real`·`$exp` 와 정확히 같은 이유다 — 시뮬레이션에서만 의미가 있고 합성에서는
   조용히 다른 회로가 된다. **사용자 결정 필요** (`docs/LOG.md`).
   `` `ifdef COCOTB_SIM `` 안의 사용은 예외로 둬야 하므로 체커에 아직 넣지 않았다.

### 이 버그가 시사하는 것

`tests/` 261개와 `tb/` 147개가 통과하는 동안, **가중치 SRAM 이 없는 설계**였다.
시뮬레이션은 `` `ifdef COCOTB_SIM `` 이 켜진 채로 돌기 때문에 X 가드가 살아 있고,
아무 문제도 보이지 않는다. **시뮬레이션과 합성이 서로 다른 회로를 보고 있었다.**
`docs/AUDIT.md` §3 의 "골든 모델을 import 하는 테스트 0개"와 같은 뿌리다 —
정답을 바깥에서 가져오지 않으면 이런 것은 잡히지 않는다.

---

## BUG-004 / BUG-005 — `fault_code_r` 다중 드라이버 (2026-09-10 수정됨)

**상태**: **수정 완료**. `scripts/synth_gate.sh` STAGE 1 이 처음 잡아냈다.

`rtl/desc_fsm_v2.sv` 와 `rtl/g3_desc_fsm.sv` 둘 다, `fault_code_r` 를
**두 개의 서로 다른 `always_ff` 블록에서 구동**하고 있었다.

```systemverilog
// 블록 A (fault_valid 래치)          // 블록 B (fault code 래치)
always_ff @(posedge clk ...) begin    always_ff @(posedge clk ...) begin
  if (!rst_n) begin                     if (!rst_n) begin
    fault_code_r  <= 8'h00;   ← A         fault_code_r <= 8'h00;   ← B
    fault_valid_r <= 1'b0;              end else begin
  end else begin ...                      ... fault_code_r <= 8'h01; ...
```

SystemVerilog 에서 하나의 변수는 하나의 `always_ff` 에서만 구동해야 한다.
Vivado 와 verilator 는 MULTIDRIVEN 으로 거부한다. 시뮬레이션에서는 이벤트 스케줄링상
우연히 동작해서 **기존 테스트 12개가 전부 통과하고 있었다.**

```
$ bash scripts/synth_gate.sh --stage1
  FAIL    desc_fsm_v2              ERROR: Found 8 problems in 'check -assert'.
$ grep "multiple conflicting" build/synth_gate/elab_desc_fsm_v2.log
Warning: multiple conflicting drivers for desc_fsm_v2.\fault_code_r [7]:
... [0] 까지 8비트 전부
```

수정: 블록 A 에서 `fault_code_r` 리셋 문장을 제거했다. 블록 B 가 이미 리셋한다.
두 파일 모두 같은 수정. 수정 후 STAGE 1 통과.

**이것이 W1-2 게이트를 만든 이유다.** 테스트 261개가 통과하는 동안 두 개의
제어 평면 FSM 이 합성 불가 상태였고, 아무도 몰랐다.

---

---

*다음 후보 (아직 재현 안 함)*: `docs/AUDIT.md` §4 의 `core_done_seen` 클리어 조건
(`rtl/desc_fsm_v2.sv:171` — `ST_IDLE`/`ST_DISPATCH` 에서만 클리어). BUG-001 을 고친 뒤에 본다.
