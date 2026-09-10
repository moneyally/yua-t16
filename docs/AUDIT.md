# AUDIT.md — 레포 현황 실측 감사 (2026-09-10)

목적: `CLAUDE.md` 2절 "레포 현재 상태"와 `README.md`의 주장을, 추측 없이 **실행한 명령의 출력으로만** 검증한다.
모든 항목에 실행 명령과 출력을 붙였다. 실행하지 못한 것은 "실행 못 함"으로 표시했다.

환경: Linux 6.18 / Python 3.11.15 / yosys 0.33 (설치함) / yosys 0.69 (yowasp, 설치함) / sv2v v0.0.13 (설치함) / iverilog 12 (설치함).
**verilator 없음 — 린트 관련 항목은 전부 "실행 못 함".**

---

## 1. 합성 금지 토큰 (`real`, `$itor`, `$rtoi`, `$exp`, `$sqrt`, `#delay`) 사용 파일

### 방법

CLAUDE.md 4절의 `grep | grep -v '^\s*//'` 는 **블록 주석(`/* */`)과 줄 끝 주석을 걸러내지 못한다.**
그래서 주석·문자열을 공백으로 치환한 뒤 검사하는 스크립트를 썼다 (`scan_tokens.py`, 스크래치패드).

```bash
python3 scan_tokens.py rtl/*.sv
```

### 출력 (요약)

```
=== FILES WITH HITS (7) ===
  collective_engine.sv  (7 hits: $itor, $rtoi, real)
  gemm_int4.sv          (10 hits: $itor, real)
  loss_scaler.sv        (9 hits: $itor, $rtoi, real)
  moe_router.sv         (18 hits: $exp, $itor, $rtoi, real)
  optimizer_unit.sv     (11 hits: $itor, $rtoi, $sqrt, real)
  vpu_core.sv           (19 hits: $exp, $itor, $rtoi, $sqrt, real)
  vpu_fp16_utils.sv     (22 hits: $exp, $itor, $rtoi, $sqrt, real)
```

대표 라인:

```
rtl/optimizer_unit.sv:195: [$sqrt] update = lr_r * m_new / ($sqrt(v_new > 0.0 ? v_new : 0.0) + eps_r);
rtl/vpu_core.sv:104:       [$exp]  else sigmoid_r = 1.0 / (1.0 + $exp(-x));
rtl/vpu_fp16_utils.sv:141: [$exp]  return $exp(x);
```

`rtl/*.v` (2개) 도 검사: **hit 0** (`clean`).
`#delay` 패턴(`#[0-9]`)은 `rtl/` 전체에서 **0건**.

### CLAUDE.md 2절과의 차이

**차이 없음. 7개 파일이 정확히 일치한다.**
`collective_engine, gemm_int4, loss_scaler, moe_router, optimizer_unit, vpu_core, vpu_fp16_utils`

### 다만 추가로 확인된 사실 (2절에 없는 것)

1. **`vpu_fp16_utils.sv` 는 모듈이 아니라 `package` 다.** 그리고 **레포 어디에서도 `import` 되지 않는다.**
   ```bash
   $ grep -rn "vpu_fp16_utils" rtl/ tb/ sim/ scripts/
   rtl/vpu_fp16_utils.sv:1:// vpu_fp16_utils.sv
   rtl/vpu_fp16_utils.sv:5:package vpu_fp16_utils;
   ```
   → 완전한 죽은 코드. 이동 대상이지만 "의존 테스트"는 없다.

2. **`gemm_int4.sv` 와 `moe_router.sv` 도 다른 RTL에서 인스턴스화되지 않는다.**
   ```bash
   $ grep -ln "\bgemm_int4\b *[uU]_" rtl/*.sv   → rtl/gemm_int4.sv (자기 자신뿐)
   $ grep -ln "\bmoe_router\b *[uU]_" rtl/*.sv  → rtl/moe_router.sv (자기 자신뿐)
   ```
   OpenLane 플로우는 `gemm_int4.sv`가 아니라 별개 파일 `gemm_int4_sky130.v`를 쓴다 (`openlane/gemm_int4/config.json`의 `DESIGN_NAME: gemm_int4_sky130`). 그 파일은 금지 토큰 clean.

3. **`real` 을 쓰는 모듈을 (전이적으로) 인스턴스화하는 상위 모듈 5개** — W1-1에서 같이 이동해야 하는 것들:

   | 상위 모듈 | 파일 | 직접 의존 |
   |---|---|---|
   | `vpu_top` | `rtl/vpu_top.sv` | `vpu_core` |
   | `g3_train_int_top` | `rtl/g3_train_int_top.sv` | `optimizer_unit` |
   | `g3_multistep_int_top` | `rtl/g3_multistep_int_top.sv` | `loss_scaler`, `optimizer_unit` |
   | `g3_2chip_int_top` | `rtl/g3_2chip_int_top.sv` | `collective_engine`, `optimizer_unit` |
   | `g3_2chip_fabric_int_top` | `rtl/g3_2chip_fabric_int_top.sv` | `collective_engine`, `optimizer_unit` |

   `g3_asic_top`, `g3_int_top`, `g3_ctrl_top`, `backward_engine` 는 **의존 없음** (clean 쪽에 남는다).

---

## 2. yosys

### 설치 여부

세션 시작 시 **없었다.**

```bash
$ which yosys verilator iverilog
/bin/bash: line 1: yosys: command not found
/bin/bash: line 1: verilator: command not found
```

### 설치 가능 여부: 가능

```bash
$ apt-get update -qq && apt-get install -y yosys
$ yosys -V
Yosys 0.33 (git sha1 2584903a060)
```

`iverilog` 도 apt로 설치됨. **`verilator` 는 설치 시도하지 않음 — 이번 세션 범위 밖(W1-3).**

### mac_array + mac_pe 합성 — 1차 시도: **실패**

```bash
$ yosys -p "read_verilog -sv rtl/mac_array.sv rtl/mac_pe.sv; hierarchy -check -top mac_array; synth -top mac_array; stat"
1. Executing Verilog-2005 frontend: rtl/mac_array.sv
rtl/mac_array.sv:9: ERROR: syntax error, unexpected '[', expecting ',' or '=' or ')'
```

원인은 `real` 이 아니다. **yosys 내장 Verilog 프론트엔드가 unpacked array 포트를 파싱하지 못한다.**

```systemverilog
rtl/mac_array.sv:9:  input  logic signed [7:0] a_row [0:15],   // ← unpacked array port
```

yosys **0.69**(yowasp, 최신)에서도 동일하게 실패한다:

```bash
$ yowasp-yosys -p "read_verilog -sv rtl/mac_array.sv rtl/mac_pe.sv; ..."
Yosys 0.69 (git sha1 9f75ca1f9, ...)
rtl/mac_array.sv:9: ERROR: syntax error, unexpected '[', expecting ')' or ',' or '='
```

`mac_pe.sv` 단독은 통과한다 (포트가 전부 packed):

```bash
$ yosys -p "read_verilog -sv rtl/mac_pe.sv; hierarchy -check -top mac_pe; synth -top mac_pe; stat"
=== mac_pe ===
   Number of cells:                730
     $_ANDNOT_  246   $_AND_  45   $_DFFE_PN0P_  32   $_NAND_  35
     $_NOR_      23   $_NOT_  29   $_ORNOT_      36   $_OR_     80
     $_XNOR_     43   $_XOR_ 161
```

### mac_array 합성 — 2차 시도: **sv2v 경유로 성공**

`sv2v`(SystemVerilog→Verilog-2005 변환기, MIT, 무료, GitHub 릴리스 바이너리)로 변환 후 합성했다.

```bash
$ sv2v rtl/mac_array.sv rtl/mac_pe.sv > mac_array_sv2v.v      # exit 0, 경고 없음
$ yosys -p "read_verilog mac_array_sv2v.v; hierarchy -check -top mac_array; synth -top mac_array; stat"
```

출력 (마지막 20줄):

```
=== design hierarchy ===

   mac_array                         1
     mac_pe                        256

   Number of wires:             182279
   Number of wire bits:         210180
   Number of public wires:        2055
   Number of public wire bits:   22020
   Number of memories:               0
   Number of memory bits:            0
   Number of processes:              0
   Number of cells:             196352
     $_ANDNOT_                   64768
     $_AND_                       6400
     $_DFFE_PN0P_                 8192
     $_MUX_                        256
     $_NAND_                     11520
     $_NOR_                       5376
     $_NOT_                       8960
     $_ORNOT_                    11264
     $_OR_                       24064
     $_XNOR_                      6656
     $_XOR_                      48896
```

**셀 수 리포트**

| 항목 | 값 |
|---|---|
| top | `mac_array` |
| 하위 인스턴스 | `mac_pe` × 256 |
| 총 셀 수 | **196,352** |
| 플립플롭 (`$_DFFE_PN0P_`) | **8,192** = 256 PE × 32비트 누산기 ✔ |
| `mac_pe` 단독 셀 수 | 730 (× 256 = 186,880, 나머지 9,472는 배선/최적화 잔여) |
| 라이브러리 | 없음 (generic gate, `abc` 기본 매핑). **면적·타이밍은 미검증.** |

### CLAUDE.md 2절 "`mac_pe`, `mac_array`: 합성 가능"에 대한 정정

**부분적으로 틀렸다.** `mac_array.sv`는 **현재 상태로는 yosys가 읽지 못한다.** sv2v 전처리를 거쳐야만 합성된다.
논리 자체는 합성 가능(하드웨어로 말이 됨)하지만, CLAUDE.md 1절이 정의한 게이트 명령
`yosys -p "read_verilog -sv <files>; synth -top <module>"` 는 **통과하지 못한다.**

### 전체 게이트 (CLAUDE.md 4절 명령) 현재 결과

```bash
$ yosys -q -p "read_verilog -sv rtl/*.sv; hierarchy -check; synth" 2>&1 | tail -8
Warning: Replacing memory \mem with list of registers. See rtl/act_sram.sv:57
rtl/backward_engine.sv:33: ERROR: syntax error, unexpected '[', expecting ',' or '=' or ')'
exit=1
```

**`real` 모듈에 도달하기도 전에, 두 번째 파일에서 죽는다.**

### rtl/ 파일별 yosys 파싱 단독 결과 (`read_verilog -sv <file>` 만)

```bash
$ for f in rtl/*.sv rtl/*.v; do yosys -q -p "read_verilog -sv $f" ...; done
→ ok=18, FAIL=29
```

실패 사유별 분류 (**yosys가 보고한 첫 번째 에러 기준**. `optimizer_unit` 처럼 unpacked 포트와 `real` 을 둘 다 가진 파일은 먼저 걸린 쪽으로 분류됨):

| 사유 | 개수 | 파일 |
|---|---|---|
| unpacked array 포트 | 18 | `backward_engine, collective_engine, ctrl_fsm, desc_fsm_v2, desc_queue, g3_2chip_fabric_int_top, g3_2chip_int_top, g3_asic_top, g3_desc_fsm, g3_int_top, g3_multistep_int_top, g3_train_int_top, gemm_top, mac_array, mxu_bf16_128x128, mxu_bf16_16x16, reg_top, optimizer_unit` |
| `real` 타입 | 5 | `gemm_int4, loss_scaler, moe_router, vpu_core, vpu_fp16_utils` |
| `initial` (guard 없음) | 1 | `cdc_fifo.sv:19` |
| 기타 언어 기능 | 5 | `g2_ctrl_top:157 (Invalid array access)`, `g3_ctrl_top:93 (unexpanded memory)`, `gemm_core:309 (Non-constant function call in constant expression)`, `gemm_int4_fpga:63 / gemm_int4_synth:84 (TOK_INT)` |
| 통과 | 18 | `act_sram, dma_bridge, g2_protob_top, gemm_stub, irq_ctrl, kvc_core, mac_pe, oom_guard, pcie_ep_versal, reset_seq, scale_fabric_ctrl, trace_ring, vpu_core_synth, vpu_lut, vpu_top, wgt_sram, gemm_int4_sky130.v, gemm_wb_wrapper.v` |

**결론: W1-2 `synth_gate.sh` 는 `real` 7개를 격리해도 통과하지 않는다.**
가장 큰 장애물은 `real` 이 아니라 **unpacked array 포트 18개 파일**이다. 선택지는 두 가지 — (a) 게이트에 sv2v 전처리 단계를 넣는다, (b) 포트를 packed flat 벡터로 고친다(`mac_array.sv` 가 이미 출력에 대해 그렇게 하고 있다: `acc_out_flat`). **결정은 사용자 몫 → LOG.md "결정 필요"에 적었다.**

### 추가 발견 — 존재하지 않는 모듈 `g3_reg_top` 을 인스턴스화하는 파일 2개

```bash
$ grep -rn "g3_reg_top" rtl/
rtl/g3_asic_top.sv:114:  g3_reg_top u_reg (
rtl/g3_ctrl_top.sv:85:   g3_reg_top u_g3_reg (
$ grep -rln "module g3_reg_top" rtl/
(출력 없음)
```

**`module g3_reg_top` 은 레포 어디에도 정의되어 있지 않다.**
따라서 `g3_asic_top` 과 `g3_ctrl_top` 은 **어떤 툴로도 elaborate 될 수 없다.**
(위 파일별 파싱 스캔에서 두 파일이 다른 에러로 먼저 죽는 바람에 드러나지 않았다.)

git log 최상단 커밋이 `feat(g3): G3-RTL-031 — ASIC top skeleton (261 tests PASS)` 인데,
그 "261 tests" 는 `g3_asic_top` 을 시뮬레이션한 결과일 수 없다. `tests/test_g3_asic_top_contract.py` 는
파이썬으로 **소스 텍스트를 검사하는 계약 테스트**이지 RTL 실행이 아니다.

### 추가 발견 — 아무도 인스턴스화하지 않는 모듈 (죽은 코드)

```bash
$ for m in vpu_lut gemm_stub kvc_core mxu_bf16_128x128 gemm_int4 moe_router; do
    grep -ln "\b$m\b *u_" rtl/*.sv | grep -v "rtl/$m.sv"; done
(전부 출력 없음)
```

`vpu_lut`, `gemm_stub`, `kvc_core`, `mxu_bf16_128x128`, `gemm_int4`, `moe_router`, `vpu_fp16_utils`(package) —
**7개가 어떤 상위 모듈에도 연결되어 있지 않다.** `mxu_bf16_128x128` 은 전용 테스트벤치(`tb_mxu_bf16_128x128.py`, 5 테스트)까지 있지만
설계 계층에는 들어가 있지 않다. CLAUDE.md 2절의 "`mxu_bf16_128x128` 은 이름이 실체보다 크다"는 지적에 **"게다가 아무 데도 안 쓰인다"** 를 덧붙여야 한다.

---

---

## 3. tb/ 테스트의 기대값 출처

### 기계적 사실 먼저

```bash
$ grep -rn "golden\|from sim" tb/*.py
(출력 없음)
$ grep -rn "golden" sim/cocotb/*.py
(출력 없음)
```

**`tb/` 36개 파일 147개 cocotb 테스트, `sim/cocotb/` 29개 테스트 — 그 어느 것도 `sim/golden/` 을 import 하지 않는다.**

`sim/golden/` 은 존재하지만 총 177줄이고 GEMM INT8 + DMA만 있다:
```bash
$ wc -l sim/golden/*.py
   0 sim/golden/__init__.py
  64 sim/golden/executor.py
  14 sim/golden/memory.py
  15 sim/golden/ops.py
  84 sim/golden/pack.py
```

### 분류표 (`tb/` 36개 파일)

분류 기준:
- **[골든]** = `sim/golden/` 의 numpy 모델에서 기대값이 나옴
- **[내장모델]** = 테스트 파일 안에 파이썬 참조식을 직접 구현 (`python_adam`, `crc8` 등). 골든은 아니지만 RTL 바깥이다
- **[하드코딩]** = 기대값이 리터럴 상수 (`== 0x47320001`, `~2.0`, `== 5`)
- **[프로토콜]** = 데이터값이 아니라 핸드셰이크/상태 전이만 검사 (기대값 개념이 약함)

| 테스트 파일 | 기대값 출처 | 근거 |
|---|---|---|
| tb_backward_engine.py | **내장모델 + 하드코딩** | `float_to_bf16`/`fp32_bits_to_float` 자체 구현, 기대값은 `dW[0][0] ≈ 0.5`, `≈2.0` 리터럴 |
| tb_cdc_fifo_async.py | **프로토콜** | 쓴 값 == 읽은 값 (self-consistent) |
| tb_cdc_fifo_reset.py | **하드코딩** | `== 0xCAFE_0001` |
| tb_collective_engine.py | **하드코딩** | `abs(val - 3.0) < 0.01` (1.0+2.0) |
| tb_desc_fsm_v2.py | **내장모델(crc8) + 하드코딩** | `fault_code == 0x01/0x02` |
| tb_desc_fsm_v2_timeout.py | **하드코딩** | `fault_code == 0x03` |
| tb_desc_queue_backpressure.py | **하드코딩** | `pushed == 64` |
| tb_desc_queue_basic.py | **프로토콜 + 하드코딩** | `q0_data == 0x1111` |
| tb_g2_ctrl_top_bootcause.py | **하드코딩** | `bc&1 == 1`, `v == 0x0001_0001` |
| tb_g2_ctrl_top_dma_shim.py | **하드코딩** | `st&0xF == 0` |
| tb_g2_ctrl_top_host_e2e.py | **하드코딩 + 프로토콜** | `gid == 0x4732_0001`, `reads_served == 2`. **연산 결과값 비교는 없다** |
| tb_g2_ctrl_top_multiqueue.py | **프로토콜** | "드레인 됐는가" |
| tb_g2_ctrl_top_oom.py | **하드코딩** | `usage0 == 0` |
| tb_g2_ctrl_top_perf.py | **하드코딩** | `v == 0` |
| tb_g2_ctrl_top_smoke.py | **하드코딩** | `val == 0x4732_0001` |
| tb_g2_ctrl_top_tc_status.py | **하드코딩** | `rs&0x7 == 0` |
| tb_g2_ctrl_top_trace_dump.py | **하드코딩** | `opcode_field == 0x01` |
| tb_g3_desc_fsm.py | **내장모델(crc8) + 하드코딩** | `fault_code == 0x01` |
| tb_g3_int_2chip_allreduce.py | **내장모델** | `python_adam()` 를 파일 안에서 구현, tol 2% |
| tb_g3_int_2chip_fabric.py | **내장모델** | `py_adam()`, tol 2% |
| tb_g3_int_gemm_e2e.py | **하드코딩** | `abs(result - 2.0) < 0.1` |
| tb_g3_int_multistep_loop.py | **내장모델** | `py_adam()` 2스텝, tol 2% |
| tb_g3_int_single_layer.py | **내장모델 + 하드코딩** | `python_adam()`, `dw00 ≈ 0.5` |
| tb_irq_ctrl_basic.py | **하드코딩** | 비트마스크 리터럴 |
| tb_irq_ctrl_w1c.py | **하드코딩** | 비트마스크 리터럴 |
| tb_loss_scaler.py | **하드코딩** | `abs(s - 32768.0) < 1.0`, `- 1.0`, `- 32.0` |
| tb_mxu_bf16_128x128.py | **하드코딩** | `≈2.0`, `≈3.0` |
| tb_mxu_bf16_16x16.py | **하드코딩** | `≈2.0`, `≈4.0`, `≈-6.0` |
| tb_oom_guard_race.py | **하드코딩** | `== 1500`, `== 1600` |
| tb_oom_guard_thresholds.py | **하드코딩** | 상태 상수 |
| tb_optimizer_unit.py | **내장모델** | `python_adam_step()`, tol 1% |
| tb_reg_top_rw.py | **하드코딩** | `== 0x47320001` 등 |
| tb_reset_seq_order.py | **프로토콜** | 릴리스 순서 |
| tb_scale_fabric_ctrl.py | **프로토콜** | 보낸 데이터 == 받은 데이터 |
| tb_trace_ring_freeze.py | **하드코딩** | `tail == 5`, `drop_count == 3` |
| tb_trace_ring_wrap.py | **하드코딩** | `tail == 10` |

집계: **골든 0 / 내장모델 8 / 하드코딩 22 / 프로토콜 6** (중복 분류 있음)

### RTL 내부 함수를 참조 모델로 쓰는 테스트

```bash
$ grep -noE "dut\.[a-zA-Z_0-9]+\.[a-zA-Z_0-9]+\.value" tb/*.py
(출력 없음 — 36개 파일 전부)
```

**"DUT 내부 계층을 뒤져 기대값을 만드는" 형태의 직접 위반은 없다.** 테스트들은 top 포트만 본다.

다만 **의미상 등가인 문제**가 하나 있다:
`tb_optimizer_unit.py` 의 `python_adam_step()` 은 파이썬 float Adam 구현이고,
`rtl/optimizer_unit.sv` 도 **`real` 타입 float Adam 구현**이다 (`$sqrt`, `real` 산술).
즉 **파이썬 float 모델을 SystemVerilog float 모델과 비교하고 있다.** 두 float 구현이 2% 안에서 일치한다는 것은
하드웨어에 대해 아무것도 증명하지 않는다. `tb_g3_int_*` 4개도 같은 구조다.

### CLAUDE.md 규칙 2 위반 테스트 목록

규칙 2를 문자 그대로 적용하면 **`tb/` 36개 파일 147개 테스트 전부가 위반**이다 (`sim/golden/` 미사용 + docstring 첫 줄에 "기대값의 출처" 없음).
`sim/cocotb/` 7개 파일 29개 테스트도 동일하게 위반.

심각도 순으로 좁히면:

**A급 — "증명이 없다" (float 모델 vs float 모델, 최우선)**
1. `tb_optimizer_unit.py` (6 테스트)
2. `tb_g3_int_single_layer.py` (4)
3. `tb_g3_int_multistep_loop.py` (6)
4. `tb_g3_int_2chip_allreduce.py` (5)
5. `tb_g3_int_2chip_fabric.py` (4)
6. `tb_collective_engine.py` (7)
7. `tb_loss_scaler.py` (7)
8. `sim/cocotb/test_vpu.py` (8), `test_moe.py` (3), `test_gemm_int4.py` (3)

→ 이 10개 파일 53개 테스트는 전부 **W1-1 이동 대상 모듈**을 DUT로 쓴다. 즉 A급 위반과 behavioral 격리 대상이 정확히 겹친다.

**B급 — 골든이 있어야 하는데 하드코딩으로 때운 것**
- `tb_mxu_bf16_16x16.py` (6), `tb_mxu_bf16_128x128.py` (5), `tb_g3_int_gemm_e2e.py` (5), `tb_backward_engine.py` (7)
- 단위 행렬·상수 입력으로만 테스트해서 **곱셈기 배선 오류를 잡아낼 수 없다.** `sim/golden/ops.py:gemm_int8` 이 이미 있는데 안 쓴다.
- `tb_g2_ctrl_top_host_e2e.py` (5): README가 "GEMM E2E 검증"이라 부르지만 **GEMM 결과값을 검사하지 않는다** (DMA 횟수·IRQ·trace tail만 본다).

**C급 — 제어 평면 (하드코딩이 타당함)**
- 레지스터맵·IRQ·큐·리셋·트레이스 테스트 17개 파일. 기대값이 `spec/`·`tools/orbit_mmio_map.py` 의 상수이므로 하드코딩이 오히려 맞다. **다만 SSOT인 `orbit_mmio_map.py` 에서 import 하지 않고 파일마다 주소를 재타이핑한다** (`A_G2_ID = 0x0_0000` 이 8개 파일에 중복). 이건 규칙 2보다 SSOT 규율 문제.

---

## 4. `done_pulse`

### 코드 위치

```bash
$ grep -c "done_pulse" rtl/*.sv | grep -v ":0"
rtl/backward_engine.sv:6          rtl/g3_asic_top.sv:1
rtl/collective_engine.sv:5        rtl/g3_ctrl_top.sv:1
rtl/ctrl_fsm.sv:3                 rtl/g3_desc_fsm.sv:3
rtl/desc_fsm_v2.sv:3              rtl/g3_int_top.sv:4
rtl/g2_ctrl_top.sv:12             rtl/g3_multistep_int_top.sv:10
rtl/g3_2chip_fabric_int_top.sv:13 rtl/g3_train_int_top.sv:7
rtl/g3_2chip_int_top.sv:10        rtl/gemm_top.sv:2
rtl/loss_scaler.sv:5              rtl/mxu_bf16_128x128.sv:5
rtl/optimizer_unit.sv:5           rtl/scale_fabric_ctrl.sv:5
```

**생산 지점 (GEMM 경로, 안쪽 → 바깥쪽)**

| # | 위치 | 코드 | 폭 |
|---|---|---|---|
| 1 | `rtl/gemm_core.sv:357` | `assign done = (st == S_DONE);` — `S_DONE → S_IDLE` (line 480-481) | 1 사이클 |
| 2 | `rtl/gemm_top.sv:167-173` | `core_done_r <= core_done;` (레지스터 1단 지연) | 1 사이클, +1 지연 |
| 3 | `rtl/ctrl_fsm.sv:104,125` | `ST_DONE` 에서 `done_pulse = 1'b1`, `ST_DONE → ST_IDLE` (line 148) | 1 사이클 |
| 4 | `rtl/gemm_top.sv:85` | 3번을 그대로 top 포트로 냄 (`gemm_done_pulse`) | 1 사이클 |
| 5 | `rtl/desc_fsm_v2.sv:270,328` | `ST_DONE` 에서 `done_pulse = 1'b1`, `ST_DONE → ST_IDLE` (line 327-329) | 1 사이클 |

**소비 지점 (`rtl/g2_ctrl_top.sv`)**

| 라인 | 코드 | 의미 |
|---|---|---|
| 247 | `.core_done(gemm_done_pulse)` | gemm 완료 → desc_fsm_v2 피드백 |
| 337 | `if (gemm_done_pulse) dma_done_latch <= 1'b1;` | DMA 상태 래치 |
| 374 | `if (gemm_done_pulse) tile_count_r <= tile_count_r + 1'b1;` | 성능 카운터 |
| 375 | `if (fsm_done_pulse) done_count_r <= done_count_r + 1'b1;` | 디스크립터 완료 카운터 |
| 388 | `assign oom_alloc_dec = fsm_done_pulse;` | OOM 사용량 감소 |
| 390 | `assign oom_dma_dec = gemm_done_pulse \| (fsm_fault_valid & opcode==0x02);` | OOM DMA 감소 |
| 460 | `... , fsm_done_pulse}` (IRQ 소스 벡터) | `DESC_DONE` IRQ |

**핸드셰이크 방어 로직 2곳** (버그가 있었음을 시사하는 흔적):
- `rtl/ctrl_fsm.sv:88-97` `core_done_seen` 래치 — 1사이클 펄스를 놓치지 않으려는 캡처
- `rtl/desc_fsm_v2.sv:165-176` 같은 패턴. **클리어 조건이 `state == ST_IDLE || state == ST_DISPATCH` 뿐**이다 (`ST_LATCH`, `ST_CRC_CHECK`, `ST_DECODE`, `ST_WAIT`, `ST_FAULT` 에서는 클리어되지 않음)
- `rtl/gemm_top.sv:95-119` "command bridge (robust)" + 주석 `// ✅ IMPORTANT: ctrl sees "accept-ready", not raw core ready` — 과거에 커맨드를 놓친 적이 있다는 뜻

**코드만 읽고 보이는 의심 지점 (미검증, 파형 필요 — W2-2 대상)**

`rtl/desc_fsm_v2.sv:331-334`:
```systemverilog
ST_FAULT: begin
  busy    = 1'b1;
  state_n = ST_DONE;  // transition to DONE to re-enter IDLE
end
```
→ **fault 가 나도 `ST_DONE` 을 거치므로 `done_pulse` 가 1회 발생한다.**
그러면 `g2_ctrl_top.sv:460` 을 통해 **실패한 디스크립터에 대해서도 `DESC_DONE` IRQ 가 뜨고**,
`done_count_r`(375) 이 증가하며, `oom_alloc_dec`(388) 도 fault 경로와 done 경로 양쪽에서 한 번씩 관여한다.
`fault_valid` 와 `done_pulse` 가 같은 트랜잭션에서 둘 다 나오는 것이 의도인지 아닌지 **문서에 정의가 없다.**
`docs/DESIGN.md` 9절이 "완료 신호는 '상태 쓰기 완료 후 1사이클'로 명시" 하라고 한 것과 정확히 같은 종류의 문제다.
**이것이 버그인지는 파형으로 확인해야 한다. 지금은 추측이며, W2-2에서 재현한다.**

### 현재 테스트가 `done_pulse` 를 어떻게 검사하는가 — **폭을 검사하지 않는다**

전형적인 패턴 (`tb/tb_desc_fsm_v2.py:104-110`):
```python
    # Wait for done_pulse
    for _ in range(10):
        await RisingEdge(dut.clk)
        if dut.done_pulse.value == 1:
            break

    assert dut.done_pulse.value == 1, "done_pulse not asserted"
```

즉 **"N 사이클 안에 1이 된 적이 있는가"만 본다.** 검사하지 않는 것:

1. **펄스 폭이 정확히 1사이클인가** — 다음 사이클에 0으로 떨어지는지 아무도 안 본다
2. **한 트랜잭션에 정확히 1회만 발생하는가** — 중복 펄스를 잡을 수 없다
3. **fault 트랜잭션에서 `done_pulse` 가 나오는가** — 위 의심 지점을 아무도 검사하지 않는다
4. **상태 쓰기 완료와의 타이밍 관계** — DESIGN.md 가 요구하는 "쓰기 완료 후 1사이클"

`done_pulse` 관련 전체 검색:
```bash
$ grep -rn "done_pulse" tb/*.py | grep -iE "== 0|!= 1|count|width|twice|second|edge"
tb/tb_backward_engine.py:257:    assert dut.done_pulse.value == 0, "No spurious done after reset"
tb/tb_collective_engine.py:212:  assert dut.done_pulse.value == 0
```

**펄스 폭 / 펄스 개수를 검사하는 테스트는 레포 전체에 0개.**
위 2건도 "리셋 후에 0인가"일 뿐 폭 검사가 아니다.
`tb/`·`sim/cocotb/` 어디에도 `pulse_count`, `n_pulse` 류의 변수가 없다.

### 덧붙임 — 애초에 cocotb 테스트를 돌릴 수단이 레포에 없다

```bash
$ find . -iname "Makefile*" -o -iname "conftest.py" -o -iname "runner*.py" | grep -v .git/
./driver/Makefile
```

`tb/` 에 Makefile 이 없다. CLAUDE.md 4절의 `cd tb && make SIM=verilator ...` 는 **실행 불가**.
`sim/cocotb/run_*.py` 7개는 러너가 있지만 `tb/` 36개 파일은 커버하지 않는다.

레포에 커밋된 단 하나의 cocotb 결과 파일 `tb/results.xml` (git 추적됨) 이 기록하는 것:
```xml
<testsuite name="all" package="all">
  <testcase name="test_read_only_defaults"  classname="tb_reg_top_rw" file="/mnt/c/Users/um020810/yua-t16/tb/tb_reg_top_rw.py" .../>
  ... (총 6건, 전부 tb_reg_top_rw)
</testsuite>
```
**6건, 단일 모듈, Windows 경로.** "24 tests all pass"의 근거가 아니다.

→ **이번 세션에서 cocotb 테스트는 하나도 실행하지 못했다** (verilator 없음 + 러너 없음). `done_pulse` 파형 확보는 W2-2.

---

## 5. README 주장 중 현재 코드로 뒷받침되지 않는 것

### 검증 명령

```bash
$ pip install -q pytest numpy && python3 -m pytest tests/ -q | tail -12
```
```
FAILED tests/test_vck190_build_contract.py::TestBuildFilesExist::test_build_doc
FAILED tests/test_vck190_build_contract.py::TestBuildFilesExist::test_bringup_doc
FAILED tests/test_vck190_build_contract.py::TestBuildFilesExist::test_failure_matrix_doc
FAILED tests/test_vck190_build_contract.py::TestTclContent::test_cpm_tcl_bar_sizes
FAILED tests/test_vck190_build_contract.py::TestTclContent::test_cpm_tcl_vendor_device_id
FAILED tests/test_vck190_build_contract.py::TestTclContent::test_cpm_tcl_msix
FAILED tests/test_vck190_build_contract.py::TestBarConsistency::test_bar0_1mib_everywhere
FAILED tests/test_vck190_build_contract.py::TestBarConsistency::test_bar4_64k_everywhere
8 failed, 247 passed in 0.27s
```

```bash
$ ls rtl/*.sv rtl/*.v | wc -l            → 47
$ grep -h "^module " rtl/*.sv rtl/*.v | wc -l → 46  (+ package 1)
$ ls tools/*.py | wc -l                  → 15
$ ls tests/test_*.py | wc -l             → 18
$ ls tb/tb_*.py | wc -l                  → 36
$ grep -c "@cocotb.test" tb/tb_*.py | awk -F: '{s+=$2} END{print s}' → 147
$ grep -c "@cocotb.test" sim/cocotb/test_*.py | awk -F: '{s+=$2} END{print s}' → 29
$ git ls-files docs/ | wc -l             → 0
$ find . -iname "*.pdi" -o -iname "*.bit" -o -iname "*.xsa" → (출력 없음)
$ find fpga -maxdepth 2
fpga/vck190/create_project.tcl  fpga/vck190/vck190_pcie.xdc
fpga/vck190/install_vivado.sh   fpga/vck190/create_cpm_ip.tcl
```

### 뒷받침되지 않는 주장 목록

| # | README 문장 (위치) | 실측 | 판정 |
|---|---|---|---|
| 1 | "**237 tests** across ... All pass" (헤더 + Test Results 표) | `pytest tests/` = **8 failed, 247 passed** (255 수집). cocotb는 실행 불가 | **거짓.** 숫자도 틀리고 "All pass"도 틀림 |
| 2 | "Python unit/integration \| 208 \| All pass" | 255 수집 / 247 통과 / 8 실패 | **거짓** |
| 3 | "RTL cocotb (module-level) \| 24 \| All pass" | `tb/` 147개 + `sim/cocotb/` 29개. 실행 수단 없음. 커밋된 결과는 6건뿐 | **미검증 + 숫자 불일치** |
| 4 | "Host-driven DUT (GEMM E2E) \| 5 \| All pass" | 파일은 존재. 실행 못 함. **그리고 GEMM 결과값을 비교하지 않는다** (§3 B급) | **미검증 + 과장** |
| 5 | "**23 RTL modules** in SystemVerilog" | 47 파일 / 46 모듈 / 1 패키지 | **거짓 (과소)**. 표기 자체가 관리되지 않음 |
| 6 | "├── rtl/  # 23 SystemVerilog modules" (Project Structure) | 위와 동일 | **거짓** |
| 7 | "├── tb/   # 9 cocotb testbenches, 29 tests" | 36 파일 / 147 테스트 | **거짓** |
| 8 | "├── tests/ # 16 test files, 208 tests" | 18 파일 / 255 테스트 | **거짓** |
| 9 | "├── tools/ # Python host stack (14 modules)" | 15 파일 (`__init__.py` 포함) | **경미한 불일치** |
| 10 | "└── docs/  # 15 design documents" | **`.gitignore:9` 에 `docs/` 가 통째로 들어 있어 git이 추적한 파일이 0개다.** 아래 §5.1 참조 | **거짓 + 근본 원인 발견** |
| 11 | "**Bitstream generated**: Vivado 2025.2, synthesis + implementation + PDI complete" | `.pdi`/`.bit`/`.xsa` 파일 0개. `fpga/vck190/` 에 tcl 4개뿐 | **미검증.** 레포 안에 증거 없음 |
| 12 | "**Bitstream (PDI) generated** \| **Done** — 0 errors" (Status 표) | 위와 동일 | **미검증** |
| 12b | "VCK190 Vivado project + CPM config \| Done" (Status 표) | `fpga/vck190/create_cpm_ip.tcl` 본문: *"versal_cips requires IPI (Block Design). Cannot use create_ip ... use Vivado GUI to configure CPM manually."* — **CPM 설정이 스크립트화되지 않았다.** pytest 실패 5건이 이 파일에서 BAR 크기(`{20}`,`{16}`)·vendor/device ID·MSI-X 문자열을 찾다가 나는 것 | **거짓** |
| 13 | "Custom LLM Inference Accelerator" (제목) / "open-source hardware accelerator for LLM inference" | 구현된 연산기는 INT8 16×16 MAC 배열과 BF16 16×16 타일. 어텐션·소프트맥스·정규화·KV캐시 실동작 없음. LLM 추론을 돌린 적 없음 | **과장.** CLAUDE.md 0절이 금지한 표현 |
| 14 | "Full software-hardware closed loop verified" | 폐루프의 하드웨어 끝단이 시뮬레이션이고, 그 시뮬레이션조차 현재 레포에서 실행할 수 없다 | **과장** |
| 15 | "No mocks, no shortcuts — the Python host stack talks to the SystemVerilog DUT" | E2E 테스트는 `tb/dma_responder.py`(파이썬 메모리 모델)를 DMA로 쓴다. 그것이 mock이다 | **거짓** |
| 16 | "**FPGA target**: VCK190 ... PCIe Gen4 x8" + "pcie_ep_versal ── CPM PCIe Gen4 x8 adapter" | CLAUDE.md 2절이 이미 "CPM AXI-Stream 포트가 스텁, 호스트와 통신한 적 없음"이라 적음 | **과장** |
| 17 | "dma_bridge ── DMA submit/status state machine" 을 아키텍처 그림에 실 경로처럼 그림 | CLAUDE.md 2절: "상태머신이고 실제 메모리 인터페이스가 아님". DDR/HBM 경로 없음 | **과장** |
| 18 | "gemm_core (INT8 16×16 MAC array)" 가 `mac_array` 를 쓰는 것처럼 읽힘 | `gemm_core.sv` 는 `mac_array` 를 인스턴스화하지 않는다 (별도 구현) | **오해 유발** |
| 19 | "`pip install cocotb verilator`" (Quick Start) | verilator 는 pip 패키지가 아니다. 그리고 `tb/` 에 Makefile 이 없어 이 지시를 따라도 아무것도 못 돌린다 | **실행 불가능한 지시** |
| 20 | "48 MMIO registers" / "48 registers across 11 blocks" | 미검증 (`tools/orbit_mmio_map.py` 대조 안 함) | **미검증** |
| 21 | "*Simulation-verified, bitstream-generated, awaiting silicon.*" (푸터) | 위 1~4, 11~12에 따라 세 주장 모두 근거 부족 | **과장** |

### 5.1 근본 원인 — `.gitignore` 가 `docs/` 전체를 무시한다

```bash
$ cat .gitignore | head -9
__pycache__/
*.pyc
sim_build/
*.vvp
*.vcd
*.fst
.venv/
build/
docs/          ← 이것

$ git check-ignore -v docs/DESIGN.md
.gitignore:9:docs/	docs/DESIGN.md
```

파급:

1. **`CLAUDE.md` 가 SSOT로 지정한 `docs/DESIGN.md`, `docs/PLAN.md` 는 커밋될 수 없었다.** `docs/AUDIT.md`, `docs/LOG.md`, `docs/BUGS.md`, `docs/LINT.md` 도 마찬가지. **PLAN 0~4단계 전체가 이 한 줄에 막혀 있었다.**
2. README 의 "docs/ 15 design documents" 는 **작성자 로컬에는 있었지만 원격에는 한 번도 올라간 적이 없는** 파일들을 가리킨다.
3. `pytest` 실패 8건 중 **3건**(`test_build_doc`, `test_bringup_doc`, `test_failure_matrix_doc`)이 정확히 이 때문이다 — `docs/ORBIT_G2_VCK190_BUILD.md` 등을 찾는데 클론에는 없다.
4. 나머지 **5건**은 `fpga/vck190/create_cpm_ip.tcl` 내용 불일치 (위 #12b).

**조치: `.gitignore` 에서 `docs/` 한 줄을 삭제했다** (W1-1 커밋에 포함 예정). 이걸 안 고치면 W1-3 `docs/LINT.md`, W2-2 `docs/BUGS.md`, W2-3 `docs/LOG.md` 가 전부 커밋되지 않는다.

### CLAUDE.md 2절의 README 관련 서술 정정

> "README가 "MPW ready", "training" 등 현재 상태를 과장함."

```bash
$ grep -niE "MPW|training|tapeout|silicon|GPU|frontier" README.md
213:*Built by YUA AI. Simulation-verified, bitstream-generated, awaiting silicon.*
```

**현재 README 에 "MPW ready" 와 "training" 이라는 문구는 없다.** (과거 버전이었거나 외부 검토자가 다른 문서를 본 듯)
대신 실제로 있는 과장은 위 표의 **"LLM inference accelerator"(#13), "Full closed loop verified"(#14), "No mocks"(#15), "awaiting silicon"(#21)** 이다.
W2-1에서 제거할 문구 목록은 CLAUDE.md 2절이 아니라 **이 표를 기준으로 삼아야 한다.**

---

## 6. 감사 결과 한 줄 요약

| CLAUDE.md 2절 주장 | 감사 결과 |
|---|---|
| `real` 사용 7개 모듈 | ✅ **정확히 일치** |
| `mac_pe`, `mac_array` 합성 가능 | ⚠️ **`mac_pe`만 사실.** `mac_array`는 sv2v 없이는 yosys가 읽지 못함 (196,352 셀 — sv2v 경유 확인) |
| 제어 평면 "구조 양호, cocotb E2E 연결됨" | ⚠️ 구조는 양호. **E2E는 이 레포에서 실행할 수 없음** (Makefile 없음, verilator 없음) |
| `done_pulse` 미해결 버그 존재 | ⚠️ **버그 확정 못 함.** 다만 방어 로직 3곳과 fault→done_pulse 경로라는 의심 지점 확보. **현재 테스트는 펄스 폭·개수를 전혀 검사하지 않음** |
| README 과장 | ✅ **사실.** 단, 지목된 "MPW ready"/"training" 문구는 현재 없음. 실제 과장은 21건 (§5) |
| `g3_asic_top` / `g3_ctrl_top` | ❌ **존재하지 않는 모듈 `g3_reg_top` 을 인스턴스화** — elaborate 불가 |
| `spec/` SSOT 규율 | ⚠️ `spec/` 자체는 있으나 **tb/ 테스트들이 `orbit_mmio_map.py` 를 import 하지 않고 주소를 재타이핑** |

| `.gitignore` | ❌ **`docs/` 전체를 무시** — PLAN이 요구하는 모든 문서가 커밋 불가였다. 이 세션에서 제거 |

**W1-2(synth_gate.sh)의 진짜 장애물은 `real` 7개가 아니라 unpacked array 포트 18개 파일이다.** 사용자 결정 필요 → `docs/LOG.md`.

---

*감사 수행: 2026-09-10. 실행하지 못한 항목: verilator 린트, 모든 cocotb 시뮬레이션, FPGA 비트스트림 검증.*
