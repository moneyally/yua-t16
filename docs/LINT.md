# LINT.md — verilator 린트 경고 기록 (PLAN W1-3)

작성 2026-09-11. **기록만 한다. 고치지 않는다** (PLAN W1-3 지시).

재현:

```bash
for f in $(ls rtl/*.sv rtl/*.v | sort); do
  echo "##### $f"; verilator --lint-only -Wall -Irtl "$f"; echo "##### rc=$?"
done
```

도구: `verilator --version` → `Verilator 5.020 2024-01-01 rev (Debian 5.020-1)`
대상: `rtl/*.sv`, `rtl/*.v` 31개 파일 (`rtl/behavioral/` 제외 — 합성 대상 아님)

> `-Wall` 은 경고를 치명으로 만든다. 그래서 아래 파일들의 종료 코드는 1 이지만,
> `%Error: Exiting due to N warning(s)` 는 **경고 때문에 멈춘 것**이고 진짜 에러는 하나뿐이다
> (`BLKLOOPINIT`, `vpu_core_synth.sv:554`).
> `scripts/synth_gate.sh` 는 린트를 게이트에 넣지 않는다 — W1-3 은 기록 단계다.

## 요약

| | 개수 |
|---|---|
| 검사한 파일 | 31 |
| 경고 0건 파일 | **13** |
| 경고/에러 있는 파일 | 18 |
| 총 경고·에러 | **134** |

코드별:

| 코드 | 건수 | 무엇인가 |
|---|---|---|
| `UNUSEDSIGNAL` | 59 | 선언·연결됐지만 읽히지 않는 신호. 대부분 스켈레톤에서 포트만 뚫어둔 것 — 의도된 것이 많다. |
| `WIDTHTRUNC` | 34 | 대입에서 상위 비트가 **잘린다**. 의도라면 명시적 캐스트/슬라이스로 적어야 한다. 실제 버그가 숨을 수 있는 등급. |
| `WIDTHEXPAND` | 27 | 대입에서 폭이 **확장된다**. 부호 확장 의도와 어긋나면 버그가 된다. |
| `UNUSEDPARAM` | 8 | 쓰이지 않는 파라미터. 대개 무해하나 인터페이스가 실체와 다르다는 신호. |
| `PINCONNECTEMPTY` | 5 | 인스턴스 포트를 `()` 로 비워 연결. 의도라면 그렇게 적혀야 한다. |
| `BLKLOOPINIT` | 1 | **verilator 가 지원하지 않는 구문** — for 루프 안에서 배열에 논블로킹 대입. 경고가 아니라 에러다. |

## 경고 0건 파일 (13개)

```
desc_fsm_v2.sv  dma_bridge.sv  gemm_stub.sv  irq_ctrl.sv  mac_array.sv  mac_pe.sv  mxu_bf16_16x16.sv  oom_guard.sv  pcie_ep_versal.sv  reg_top.sv  reset_seq.sv  trace_ring.sv  vpu_lut.sv
```
`mac_pe`, `mac_array`, `desc_fsm_v2`, `reg_top`, `irq_ctrl`, `trace_ring`, `oom_guard`, `reset_seq` —
**북극성의 핵심 부품과 제어 평면 주요 모듈이 전부 여기 있다.** 이건 좋은 신호다.

## 파일별 (경고 많은 순)

| 건수 | 파일 | 내역 |
|---|---|---|
| 21 | `rtl/vpu_core_synth.sv` | WIDTHTRUNC 19, UNUSEDSIGNAL 1, BLKLOOPINIT 1 |
| 19 | `rtl/g2_ctrl_top.sv` | UNUSEDSIGNAL 11, WIDTHEXPAND 6, WIDTHTRUNC 2 |
| 19 | `rtl/g2_protob_top.sv` | UNUSEDSIGNAL 11, WIDTHEXPAND 6, WIDTHTRUNC 2 |
| 17 | `rtl/gemm_top.sv` | UNUSEDSIGNAL 11, WIDTHEXPAND 4, WIDTHTRUNC 2 |
| 16 | `rtl/gemm_core.sv` | UNUSEDSIGNAL 10, WIDTHEXPAND 4, WIDTHTRUNC 2 |
| 13 | `rtl/gemm_wb_wrapper.v` | WIDTHTRUNC 7, UNUSEDSIGNAL 4, UNUSEDPARAM 2 |
| 6 | `rtl/g3_int_top.sv` | PINCONNECTEMPTY 4, UNUSEDPARAM 2 |
| 3 | `rtl/act_sram.sv` | UNUSEDSIGNAL 3 |
| 3 | `rtl/gemm_int4_sky130.v` | UNUSEDPARAM 2, UNUSEDSIGNAL 1 |
| 3 | `rtl/gemm_int4_synth.sv` | WIDTHEXPAND 2, UNUSEDSIGNAL 1 |
| 2 | `rtl/backward_engine.sv` | PINCONNECTEMPTY 1, WIDTHEXPAND 1 |
| 2 | `rtl/cdc_fifo.sv` | WIDTHEXPAND 2 |
| 2 | `rtl/desc_queue.sv` | WIDTHEXPAND 2 |
| 2 | `rtl/g3_desc_fsm.sv` | UNUSEDPARAM 2 |
| 2 | `rtl/scale_fabric_ctrl.sv` | UNUSEDSIGNAL 2 |
| 2 | `rtl/wgt_sram.sv` | UNUSEDSIGNAL 2 |
| 1 | `rtl/ctrl_fsm.sv` | UNUSEDSIGNAL 1 |
| 1 | `rtl/gemm_int4_fpga.sv` | UNUSEDSIGNAL 1 |

## 코드별 대표 샘플

### `UNUSEDSIGNAL` — 59건

선언·연결됐지만 읽히지 않는 신호. 대부분 스켈레톤에서 포트만 뚫어둔 것 — 의도된 것이 많다.

```
rtl/act_sram.sv:23:11  Signal is not driven, nor used: 'i'
rtl/act_sram.sv:32:63  Signal is not used: 'a'
rtl/act_sram.sv:33:13  Signal is not driven, nor used: 'k'
... (총 59건. 전체는 build/lint/raw.txt)
```

### `WIDTHTRUNC` — 34건

대입에서 상위 비트가 **잘린다**. 의도라면 명시적 캐스트/슬라이스로 적어야 한다. 실제 버그가 숨을 수 있는 등급.

```
rtl/gemm_core.sv:102:18  Operator ASSIGNDLY expects 16 bits on the Assign RHS, but Assign RHS's MUL generates 32 bits.
rtl/gemm_core.sv:410:22  Operator ASSIGN expects 16 bits on the Assign RHS, but Assign RHS's VARREF 'OUT_BYTES' generate
rtl/gemm_core.sv:102:18  Operator ASSIGNDLY expects 16 bits on the Assign RHS, but Assign RHS's MUL generates 32 bits.
... (총 34건. 전체는 build/lint/raw.txt)
```

### `WIDTHEXPAND` — 27건

대입에서 폭이 **확장된다**. 부호 확장 의도와 어긋나면 버그가 된다.

```
rtl/backward_engine.sv:213:21  Operator EQ expects 32 bits on the LHS, but LHS's VARREF 'k_cnt' generates 5 bits.
rtl/cdc_fifo.sv:92:50  Operator AND expects 5 bits on the LHS, but LHS's VARREF 'wr_valid' generates 1 bits.
rtl/cdc_fifo.sv:92:50  Operator AND expects 5 bits on the RHS, but RHS's VARREF 'wr_ready' generates 1 bits.
... (총 27건. 전체는 build/lint/raw.txt)
```

### `UNUSEDPARAM` — 8건

쓰이지 않는 파라미터. 대개 무해하나 인터페이스가 실체와 다르다는 신호.

```
rtl/g3_desc_fsm.sv:82:26  Parameter is not used: 'OP_KVC'
rtl/g3_desc_fsm.sv:83:26  Parameter is not used: 'OP_VPU'
rtl/g3_desc_fsm.sv:82:26  Parameter is not used: 'OP_KVC'
... (총 8건. 전체는 build/lint/raw.txt)
```

### `PINCONNECTEMPTY` — 5건

인스턴스 포트를 `()` 로 비워 연결. 의도라면 그렇게 적혀야 한다.

```
rtl/backward_engine.sv:99:6  Cell pin connected by name with empty reference: 'busy'
rtl/g3_int_top.sv:81:6  Cell pin connected by name with empty reference: 'bkwd_cmd_valid'
rtl/g3_int_top.sv:81:25  Cell pin connected by name with empty reference: 'opt_cmd_valid'
... (총 5건. 전체는 build/lint/raw.txt)
```

### `BLKLOOPINIT` — 1건

**verilator 가 지원하지 않는 구문** — for 루프 안에서 배열에 논블로킹 대입. 경고가 아니라 에러다.

```
rtl/vpu_core_synth.sv:554:19  Unsupported: Delayed assignment to array inside for loops (non-delayed is ok - see docs)
```

## 읽은 소감 — 고칠 우선순위 (고치지는 않았다)

1. **`BLKLOOPINIT` 1건 (`vpu_core_synth.sv:554`)** — 유일한 진짜 에러. verilator 가 아예 지원하지 않는 구문이라 이 모듈은 **verilator 로 시뮬레이션할 수 없다.** 그런데 `sim/cocotb/test_vpu_synth.py` 가 이 모듈을 DUT 로 쓴다. `scripts/synth_gate.sh` STAGE 2 에서는 117,158 셀로 합성은 된다.
2. **`WIDTHTRUNC` 34건** — 상위 비트가 잘리는 대입. 이 등급에 실제 버그가 숨는다. `vpu_core_synth.sv` 19건, `gemm_wb_wrapper.v` 7건이 대부분.
3. **`WIDTHEXPAND` 27건** — 부호 확장 의도와 어긋나면 버그. `g2_ctrl_top`/`g2_protob_top`/`gemm_top`/`gemm_core` 에 몰려 있다.
4. **`UNUSEDSIGNAL` 59건** — 가장 많지만 가장 무해하다. 대부분 스켈레톤에서 포트만 뚫어둔 것이다. `pcie_ep_versal` 이 0건인 것이 오히려 특이하다 (그쪽 undriven 문제는 yosys `check -assert` 가 잡는다 — `scripts/synth_gate.sh` known-incomplete).

**주의**: 이 숫자들은 `verilator 5.020` 기준이다. 다른 버전은 경고 집합이 다르다.
PLAN W1-3 의 완료 기준은 "경고 목록을 기록"이고, 경고 0 은 8주차(W8) 기준이다.
