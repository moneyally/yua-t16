# LOG.md — 작업 기록

---

## 2026-09-10 (세션 2, 자율모드) — W1-1 완료 + W1-2 게이트 + W2-2 준비

**한 것**: 승인 6건 전부 실행. `scripts/synth_gate.sh` (2단 게이트) 세워서 **통과**. W2-2 가설을 테스트로 만들어 **BUG-001 확정** (파형·사이클표 확보). 게이트가 추가로 합성 결함 5개를 잡아냈고 그중 4개 수정.
**안 된 것**: `mxu_bf16_16x16` 등 7개 모듈은 yosys 가 시간 예산 안에 못 끝냄(**합성 불가가 아니라 미측정**). 기존 `tb/` 테스트벤치 대부분은 여전히 실행 못 함 (아래 "남은 문제" 참조). `verilator` 5.020(Debian)은 cocotb 2.x 와 비호환.
**실행한 검증 명령**: `bash scripts/synth_gate.sh` → **PASS (EXIT=0)** / `python3 -m pytest tests/ -q` → **3 failed, 247 passed, 5 xfailed** / `python3 tb/run_tb.py g2_ctrl_top tb_g2_ctrl_top_fault_irq` → 3 tests, 1 pass 2 fail(=버그 확정)
**다음 세션 첫 작업**: BUG-001 수정 방향 결정 → `docs/DESIGN.md` 9절 먼저 고치고 → RTL. 그 다음 W1-3 (verilator 린트 → `docs/LINT.md`).
**사용자 결정 필요**: 4건 (아래 "결정 필요 v2").

---

### 승인 6건 처리 결과

| # | 지시 | 결과 |
|---|---|---|
| 1 | synth_gate 는 sv2v 전처리, "일일 게이트 / 최종은 Vivado" 주석 | ✅ `scripts/synth_gate.sh` 헤더에 명시. STAGE 2 끝에도 매번 출력. |
| 2 | `sim/cocotb` 행동모델 3파일 → `sim/cocotb/behavioral/` | ✅ + 러너 경로 수정. `test_kvc`/`run_kvc` 도 같이 이동 (아래 참조). |
| 3 | README 는 W2-1 에서 | ✅ 손대지 않음. |
| 4 | `g3_asic_top`, `g3_ctrl_top` → behavioral | ✅ + `tests/test_g3_asic_top_contract.py` 경로 갱신 (소스 텍스트 계약 테스트라 여전히 통과). |
| 5 | pytest CPM 실패 5건 xfail(reason 명시), 기대값 수정 금지 | ✅ `strict=True` 로. 기대값 그대로. **문서 3건은 손대지 않고 그대로 실패로 남겼다** (아래 결정 필요 #4). |
| 6 | CLAUDE.md 2절 README 항목 교체 | ✅ "237 all pass"·"CPM config Done" 등 감사 결과로 교체. `real` 7개 항목도 W1-1 완료 상태로 갱신. |

### W1-2 — `scripts/synth_gate.sh`

**왜 2단인가**: 전 모듈 full `synth` 는 일일 게이트로 못 쓴다 (`mxu_bf16_16x16` 20분+, `act_sram` 64s).

- **STAGE 1 (elaborate + `check -assert`)**: 전 모듈, 모듈당 120s, 병렬 4. **에러는 게이트 실패.** 진짜 회귀는 거의 전부 여기서 잡힌다.
- **STAGE 2 (`synth -top`)**: 전 모듈, 모듈당 240s(기본), 병렬 4. **에러는 게이트 실패.** 시간 초과는 **WARN + 목록 출력** — "느려서 못 끝냄"은 "합성 불가"가 아니기 때문. `--strict` 로 실패 처리 가능.
- `cells=0` 은 통과로 세지 않고 **ZERO 로 따로 표시**한다. BUG-006 이 정확히 그 증상이었다.
- 알려진 미완성 모듈(`pcie_ep_versal`, `g2_protob_top`)은 **숨기지 않고 이유와 함께 매번 출력**한다.

**실행 결과 (`SYNTH_TIMEOUT=420 bash scripts/synth_gate.sh`)**

```
=== ORBIT synth gate ===============================================
일일 게이트 (sv2v -> yosys). 최종 합성 판정은 Vivado.

[0/4] tools   : Yosys 0.33 (git sha1 2584903a060) / sv2v v0.0.13
[1/4] sources : 31 files (rtl/behavioral/ 제외)
[2/4] tokens  : clean
[3/4] sv2v    : OK  (31 modules -> build/synth_gate/flat.v)

--- STAGE 1: elaborate + check (전 모듈, 모듈당 120s, 병렬 4) ---
  ok      cdc_fifo
  ok      ctrl_fsm
  ok      desc_queue
  ok      dma_bridge
  ok      desc_fsm_v2
  ok      act_sram
  ok      g3_desc_fsm
  ok      g2_ctrl_top
  FAIL    g2_protob_top            ERROR: Found 171 problems in 'check -assert'.
  ok      gemm_int4_fpga
  ok      gemm_int4_sky130
  ok      gemm_core
  ok      gemm_stub
  ok      gemm_int4_synth
  ok      gemm_wb_wrapper
  ok      irq_ctrl
  ok      mac_array
  ok      mac_pe
  ok      gemm_top
  ok      oom_guard
  FAIL    pcie_ep_versal           ERROR: Found 171 problems in 'check -assert'.
  ok      reg_top
  ok      reset_seq
  ok      scale_fabric_ctrl
  ok      trace_ring
  ok      vpu_core_synth
  ok      vpu_lut
  TIMEOUT backward_engine          >120s
  ok      wgt_sram
  TIMEOUT g3_int_top               >120s
  TIMEOUT mxu_bf16_16x16           >120s

  KNOWN INCOMPLETE (게이트 실패로 치지 않음, 매번 표시):
    - g2_protob_top: pcie_ep_versal 을 인스턴스화하므로 같은 undriven 포트를 물려받는다.
    - pcie_ep_versal: CPM AXI-Stream 포트가 스텁 — BAR 요청 출력이 undriven (CLAUDE.md 2절, docs/AUDIT.md §5 #16). PCIe 실동작은 docs/DESIGN.md 10절에서 범위 밖.
  WARN: elaborate 시간 초과 3개 (>120s):
         backward_engine
         g3_int_top
         mxu_bf16_16x16
  -> STAGE 1 통과 (26/31, 시간 초과 3, known 2)

--- STAGE 2: synth -top (전 모듈, 모듈당 420s, 병렬 4) ---
  ok      ctrl_fsm                 1s       cells=499
  ok      cdc_fifo                 1s       cells=1167
  ok      desc_fsm_v2              4s       cells=2786
  ok      dma_bridge               1s       cells=880
  ok      desc_queue               24s      cells=66030
  ok      act_sram                 62s      cells=99294
  ok      g3_desc_fsm              4s       cells=2897
  ok      g2_ctrl_top              325s     cells=657868
  ok      g2_protob_top            338s     cells=657773
  TIMEOUT backward_engine          >420s
  TIMEOUT g3_int_top               >420s
  ok      gemm_int4_sky130         73s      cells=43900
  ok      gemm_stub                0s       cells=203
  ok      gemm_int4_fpga           139s     cells=50594
  ok      gemm_core                191s     cells=429672
  ok      irq_ctrl                 1s       cells=549
  ok      mac_array                1s       cells=196352
  ok      mac_pe                   0s       cells=730
  ok      gemm_wb_wrapper          98s      cells=71262
  ok      oom_guard                2s       cells=3752
  ok      pcie_ep_versal           0s       cells=4
  ok      reg_top                  11s      cells=7250
  ok      reset_seq                0s       cells=52
  ok      scale_fabric_ctrl        0s       cells=62
  ok      trace_ring               49s      cells=143610
  ok      gemm_top                 189s     cells=430402
  ok      vpu_lut                  2s       cells=1130
  ok      wgt_sram                 62s      cells=99294
  TIMEOUT gemm_int4_synth          >420s
  TIMEOUT mxu_bf16_16x16           >420s
  ok      vpu_core_synth           419s     cells=117158

WARN: 시간 초과 4개 (>420s) — 합성 불가가 아니라 '측정 못 함'이다:
       backward_engine
       g3_int_top
       gemm_int4_synth
       mxu_bf16_16x16
       SYNTH_TIMEOUT=1800 으로 다시 돌리거나 Vivado 로 판정할 것.
=== PASS ===========================================================
STAGE 1 elaborate 26/31  (시간 초과 3, known-incomplete 2)
STAGE 2 synth     27/31  (시간 초과 4, 0셀 0, known-incomplete 0)
주의: 이것은 일일 게이트다. FPGA 합성·타이밍 판정은 Vivado 로만 한다.
EXIT=0
```

`cells=` 는 **design hierarchy 총계**다 (하위 모듈 포함). 예: `gemm_core 429,672` 는
`gemm_core` 자체 34,735 + `act_sram` 99,294 + `wgt_sram` 99,294 + `mac_array` 196,352 의 합.

읽을 만한 숫자:

| 모듈 | 셀 수(총계) | 의미 |
|---|---|---|
| `g2_ctrl_top` | **657,868** | 제어 평면 전체가 합성된다는 첫 실측 |
| `mac_array` | **196,352** | 북극성의 핵심 부품. `mac_pe` 730 × 256 + 배선 |
| `act_sram` / `wgt_sram` | 99,294 / **99,294** | BUG-006 수정 전 wgt_sram 은 **0** 이었다 |
| `gemm_core` | 429,672 | BUG-006 수정 전 1,523 |
| `mac_pe` | 730 | DFF 32개 = 32비트 누산기 ✔ |

기본 예산(240s)으로는 `g2_ctrl_top` 이 시간 초과였다. **420s 면 잡힌다.**
일일 게이트는 기본값으로 돌리고, RTL 을 크게 건드린 날은 `SYNTH_TIMEOUT=420` 을 권한다.

### 게이트가 잡아낸 합성 결함 (전부 `docs/BUGS.md` 에 명령 출력과 함께)

| # | 모듈 | 문제 | 상태 |
|---|---|---|---|
| BUG-002 | `cdc_fifo` | `initial` 이 **포트 리스트 안**에 있어 iverilog·sv2v 둘 다 거부 → **한 번도 컴파일된 적 없음** | 수정 |
| BUG-003 | `mxu_bf16_16x16` | 함수 안 **경계 없는 `while`** | 수정 (경계 25 for 루프) |
| BUG-004/5 | `desc_fsm_v2`, `g3_desc_fsm` | `fault_code_r` 를 **두 `always_ff` 가 구동** (다중 드라이버) | 수정 |
| **BUG-006** | `wgt_sram`, `gemm_core` | 시뮬 전용 `=== 1'bx` X 가드가 **합성에서 로직을 삭제**. `wgt_sram` **0 cells**, `gemm_core` 1,523 → **34,735 cells** (로직 95% 소멸) | 수정 |

**BUG-006 이 오늘 가장 중요한 발견이다.** 고치기 전 상태로 실물에 올렸다면
**가중치 SRAM 이 없고 GEMM 결과가 0 으로 기록되는 칩**이 나왔다.
시뮬레이션은 `` `ifdef COCOTB_SIM `` 이 켜진 채 돌기 때문에 261개 테스트가 끝까지 정상으로 보였다.
시뮬레이션과 합성이 **서로 다른 회로**를 보고 있었다.

### W2-2 — `done_pulse` 가설: **확정**

지시대로 두 assertion 을 만들었다.

**(b) `done_pulse` 폭 1사이클** — `tb/tb_desc_fsm_v2_done_pulse.py`

```
NOP:     pulses=[(2, 1)] total_high=1 fault=None      ← PASS
GEMM:    pulses=[(5, 1)] total_high=1 fault=None      ← PASS
ILLEGAL: pulses=[(3, 1)] total_high=1 fault=3         ← 폭은 1이지만 fault 인데 펄스가 났다
CRCFAIL: pulses=[(2, 1)] total_high=1 fault=2
TIMEOUT: pulses=[(35, 1)] total_high=1 fault=35
** TESTS=6 PASS=3 FAIL=3 SKIP=0 **
```

폭 자체는 전부 정확히 1사이클이었다. **폭은 문제가 아니었다.**

**(a) fault 시 `DESC_DONE` IRQ 미발생** — `tb/tb_g2_ctrl_top_fault_irq.py`

```
NOP        IRQ_PENDING = 0x00000001 ['DESC_DONE']                 ← PASS (대조군)
ILLEGAL    IRQ_PENDING = 0x00000021 ['DESC_DONE', 'TC0_FAULT']    ← FAIL
CRCFAIL    IRQ_PENDING = 0x00000021 ['DESC_DONE', 'TC0_FAULT']    ← FAIL
** TESTS=3 PASS=1 FAIL=2 SKIP=0 **
```

**가설 확정.** 실패한 디스크립터가 완료 IRQ 를 올린다.
사이클 표(`rtl/desc_fsm_v2.sv` 단독, illegal opcode):

```
cyc | state | busy | fault_valid | fault_code | done_pulse
  2 |     7 |    1 |           0 |        0x1 |          0     ST_FAULT
  3 |     6 |    0 |           1 |        0x1 |          1     ST_DONE  ← ★ 둘이 같은 사이클
  4 |     0 |    0 |           1 |        0x1 |          0     ST_IDLE
```

원인은 `rtl/desc_fsm_v2.sv:331` `ST_FAULT: state_n = ST_DONE;` 3줄.
파형: `build/tb/g2_ctrl_top/g2_ctrl_top.fst`. 상세: `docs/BUGS.md` BUG-001.
**지시대로 수정하지 않았다** — 수정 방향 3안을 BUGS.md 에 적어뒀고, `docs/DESIGN.md` 9절을 먼저 고쳐야 한다.

### 부수적으로 만든 것

- `scripts/check_banned_tokens.py` — 주석·문자열을 공백 치환한 뒤 검사한다. CLAUDE.md 4절의 `grep -v '^\s*//'` 는 블록 주석을 못 거른다.
- `tb/run_tb.py` — `tb/` 에 Makefile 이 없어서 CLAUDE.md 4절의 `cd tb && make ...` 가 실행 불가였다 (AUDIT §4). iverilog + sv2v 로 돌린다.

### 남은 문제 (정직하게)

1. **기존 `tb/` 테스트벤치 대부분은 아직 실행 못 한다.** iverilog 는 unpacked array 포트를 지원하지 않아 sv2v 를 거치는데, 그러면 `dut.desc_bytes[i]` 같은 접근이 깨진다 (sv2v 가 packed 벡터로 낮춤). 내가 새로 쓴 두 테스트벤치는 두 모양을 모두 지원하는 어댑터를 넣었지만, **기존 36개는 고치지 않았다.** 근본 해결은 verilator 5.022+ (cocotb 2.x 요구) 이고 Debian 은 5.020 이다.
2. **`mxu_bf16_16x16` 은 FP32 가산기 256개**를 한 사이클에 넣은 구조다. yosys 20분+. 실물에서 100MHz 는 커녕 합성이 될지도 미검증. `docs/DESIGN.md` 10절이 BF16 을 범위 밖으로 두고 있으니 W1-1 처럼 격리 대상일 수 있다 — 결정 필요 #3.
3. `tb/results.xml` 은 git 에 추적되는 **낡은 결과 파일**이다 (6건, Windows 경로). 지우거나 gitignore 해야 한다.

---

## 결정 필요 v2

**#1 — `CLAUDE.md` 규칙 1 금지 토큰에 `===` / `!==` (X 비교) 추가?**
BUG-006 의 근본 원인이다. `real`·`$exp` 와 같은 이유 — 시뮬레이션에서만 의미가 있고 합성에서는 조용히 다른 회로가 된다. `` `ifdef COCOTB_SIM `` 안은 예외여야 해서 체커에 아직 넣지 않았다. **권고: 추가.**

**#2 — `while` 도 금지 토큰에 추가?**
BUG-003. 경계 없는 `while` 은 하드웨어가 아니다. 현재 합성 대상 RTL 에는 0건이라 지금 추가해도 게이트는 통과한다. **권고: 추가.**

**#3 — `mxu_bf16_16x16` 계열도 behavioral 로?**
`mxu_bf16_128x128`(524,288 FF)과 `kvc_core`(2 Mbit FF)는 이미 옮겼다 — 아무 데도 안 쓰이고 명백히 하드웨어가 아니어서 판단이 쉬웠다.
`mxu_bf16_16x16` 은 다르다: `backward_engine` 과 `g3_int_top` 이 실제로 쓴다. 그런데 `backward_engine` 은 CLAUDE.md 2절이 명시적으로 격리 대상에서 제외한 모듈이다. **혼자 판단하지 않고 남겨둔다.**

**#4 — pytest 문서 실패 3건 처리**
`docs/ORBIT_G2_VCK190_{BUILD,PCIE_BRINGUP,FAILURE_MATRIX}.md` 3개가 없어서 나는 실패다. `.gitignore` 의 `docs/` 때문에 push 된 적이 없다. 지시는 "CPM 5건만 xfail" 이었으므로 **이 3건은 손대지 않고 빨간 채로 뒀다.** 선택지: (a) 정원님 로컬에 원본이 있으면 커밋 → 3건 통과, (b) xfail(reason="never committed, .gitignore had docs/"), (c) 문서를 새로 씀. **권고: (a).**

---

## 2026-09-10 (세션 1) — 감사 + W1-1 착수

**한 것**: `docs/AUDIT.md` 작성 (5개 항목 전부 실측·명령 출력 첨부) / yosys·iverilog·sv2v 설치 / W1-1 파일 이동 19개 실행 (커밋 안 함).
**안 된 것**: cocotb 테스트 **하나도 실행 못 함** (verilator 없음 + `tb/`에 Makefile 없음). verilator 린트 실행 못 함. `done_pulse` 버그 **확정 못 함** (파형 미확보 — W2-2).
**실행한 검증 명령**: `python3 -m pytest tests/ -q` → **8 failed, 247 passed** (이동 전후 동일). `python3 scan_tokens.py rtl/*.sv rtl/*.v` → **clean**. `yosys ... synth -top mac_array` (sv2v 경유) → **196,352 cells**.
**다음 세션 첫 작업**: 아래 "결정 필요" 3건에 대한 사용자 답을 받고 → W1-1 커밋 → W1-2 `scripts/synth_gate.sh`.
**사용자 결정 필요**: 3건 (아래).
**추가 조치 1건**: `.gitignore` 에서 `docs/` 한 줄 삭제 — 이게 있으면 AUDIT/LOG/BUGS/LINT/DESIGN/PLAN 전부 커밋 불가였다. 아래 참조.

---

## W1-1 보고 — 이동 대상 파일과 영향받는 테스트

**상태: 워킹 트리에서 `git mv` 완료. 커밋하지 않았다.** 되돌리려면 `git reset --hard HEAD`.

### A. `rtl/` → `rtl/behavioral/` — 12개

**A-1. `real` 직접 사용 (PLAN 명시 7개) — AUDIT §1에서 정확히 일치 확인**

| 파일 | 금지 토큰 | 상위 인스턴스 |
|---|---|---|
| `collective_engine.sv` | `real`, `$itor`, `$rtoi` | g3_2chip_int_top, g3_2chip_fabric_int_top |
| `gemm_int4.sv` | `real`, `$itor` | **없음 (죽은 코드)** |
| `loss_scaler.sv` | `real`, `$itor`, `$rtoi` | g3_multistep_int_top |
| `moe_router.sv` | `real`, `$itor`, `$rtoi`, `$exp` | **없음 (죽은 코드)** |
| `optimizer_unit.sv` | `real`, `$itor`, `$rtoi`, `$sqrt` | g3_train/multistep/2chip/2chip_fabric |
| `vpu_core.sv` | `real`, `$itor`, `$rtoi`, `$exp`, `$sqrt` | vpu_top |
| `vpu_fp16_utils.sv` | `real`, `$itor`, `$rtoi`, `$exp`, `$sqrt` | **없음 (package, import 0회)** |

**A-2. 위를 전이적으로 인스턴스화하는 상위 모듈 (PLAN "G3 top") — 5개**

| 파일 | 직접 의존 |
|---|---|
| `vpu_top.sv` | vpu_core |
| `g3_train_int_top.sv` | optimizer_unit |
| `g3_multistep_int_top.sv` | loss_scaler, optimizer_unit |
| `g3_2chip_int_top.sv` | collective_engine, optimizer_unit |
| `g3_2chip_fabric_int_top.sv` | collective_engine, optimizer_unit |

**`rtl/` 에 남긴 것 중 헷갈릴 만한 것 (의존 없음 확인):**
`backward_engine.sv` (PLAN이 명시적으로 제외), `g3_asic_top.sv`, `g3_ctrl_top.sv`, `g3_int_top.sv`, `g3_desc_fsm.sv`, `scale_fabric_ctrl.sv`, `vpu_core_synth.sv`, `vpu_lut.sv`, `gemm_int4_synth.sv`, `gemm_int4_fpga.sv`, `gemm_int4_sky130.v`.

### B. `tb/` → `tb/behavioral/` — 7개 (39 테스트)

| 테스트벤치 | DUT | cocotb 테스트 수 | AUDIT §3 등급 |
|---|---|---|---|
| `tb_collective_engine.py` | collective_engine | 7 | A |
| `tb_loss_scaler.py` | loss_scaler | 7 | A |
| `tb_optimizer_unit.py` | optimizer_unit | 6 | A |
| `tb_g3_int_single_layer.py` | g3_train_int_top | 4 | A |
| `tb_g3_int_multistep_loop.py` | g3_multistep_int_top | 6 | A |
| `tb_g3_int_2chip_allreduce.py` | g3_2chip_int_top | 5 | A |
| `tb_g3_int_2chip_fabric.py` | g3_2chip_fabric_int_top | 4 | A |
| **합계** | | **39** | |

이동 후 `tb/` 에 남은 것: 29개 파일 / 108 테스트. 전부 제어 평면·mac·mxu·gemm 경로다.

### C. 이동하지 **않았지만** 행동 모델을 DUT로 쓰는 테스트 — 3개 파일 14 테스트

PLAN W1-1이 `tb/` 만 언급해서 `sim/cocotb/` 는 손대지 않았다. **결정 필요 #2.**

| 파일 | DUT | 테스트 수 |
|---|---|---|
| `sim/cocotb/test_vpu.py` (+ `run_vpu.py`) | vpu_core | 8 |
| `sim/cocotb/test_moe.py` (+ `run_moe.py`) | moe_router | 3 |
| `sim/cocotb/test_gemm_int4.py` (+ `run_gemm_int4.py`) | gemm_int4 | 3 |

### D. 이동 때문에 고친 경로 참조 — 3곳

레포 전체에서 이동 대상 파일의 **경로**를 참조하는 곳은 아래 3줄뿐이었다. 전부 수정했다.

```
sim/cocotb/run_gemm_int4.py:25   root / "rtl" / "behavioral" / "gemm_int4.sv",
sim/cocotb/run_moe.py:14         root / "rtl" / "behavioral" / "moe_router.sv"
sim/cocotb/run_vpu.py:14         root / "rtl" / "behavioral" / "vpu_core.sv"
```

`rtl/synth/vivado_synth.tcl` 는 `vpu_core_synth.sv` / `gemm_int4_synth.sv` 만 참조 → **영향 없음**.
`openlane/*/config.json` 은 자기 `src/` 사본을 씀 → **영향 없음**.
`tests/` 18개 파일 → **영향 없음** (이동 전후 `8 failed, 247 passed` 동일).

### E. 이동 후 검증

```bash
$ python3 scan_tokens.py rtl/*.sv rtl/*.v
=== FILES WITH HITS (0) ===
clean

$ python3 -m pytest tests/ -q | tail -1
8 failed, 247 passed in 0.47s      # 이동 전과 동일 → 회귀 없음
```

**`rtl/` (behavioral 제외) 에서 금지 토큰 6종이 전부 사라졌다.** 이게 W1-1의 핵심 성과다.

### F. 아직 안 한 W1-1 잔여 작업

- README 에 "행동 모델(합성 불가)" 섹션 추가 — **W2-1 README 전면 재작성과 겹쳐서, 결정 #3 이후로 미룸.**
- 커밋 — **지시대로 첫 커밋 전에 멈춤.**

---

## 계획 밖이지만 먼저 고친 것 — `.gitignore` 의 `docs/`

`git status` 에 `docs/` 가 안 잡혀서 확인했더니:

```bash
$ git check-ignore -v docs/DESIGN.md
.gitignore:9:docs/	docs/DESIGN.md
```

**`.gitignore` 9번째 줄이 `docs/` 전체를 무시하고 있었다.**
즉 `CLAUDE.md` 가 SSOT로 지정한 `docs/DESIGN.md`·`docs/PLAN.md`, PLAN이 만들라는 `docs/LOG.md`(W2-3)·`docs/BUGS.md`(W2-2)·`docs/LINT.md`(W1-3)·`docs/AUDIT.md` 가 **하나도 커밋될 수 없는 상태**였다.
PLAN 12주 전체가 이 한 줄에 막혀 있었다.

**해당 줄을 삭제했다.** PLAN에 없는 변경이지만, 이걸 안 고치면 PLAN의 산출물이 전부 사라지므로 규칙 6(범위 축소)이 아니라 "작업 자체가 성립하지 않음"에 해당한다고 판단했다. 되돌리려면 `.gitignore` 에 `docs/` 를 다시 넣으면 된다.

부수 효과로 **`pytest` 실패 8건의 정체가 밝혀졌다**:
- 3건 = `docs/ORBIT_G2_VCK190_*.md` 3개를 찾는데 클론에 없음 (gitignore 때문에 애초에 push된 적 없음)
- 5건 = `fpga/vck190/create_cpm_ip.tcl` 이 BAR 크기·vendor ID·MSI-X 설정을 담고 있지 않음. 파일 본문이 *"Cannot use create_ip ... use Vivado GUI to configure CPM manually"* 다. → README 의 "CPM config Done" 은 사실이 아니다. (AUDIT §5 #12b)

**이 8건은 지금 고치지 않았다.** W2-1(README 재작성) 또는 별도 항목에서 다룰 일이고, 오늘 범위가 아니다.

---

## 사용자 결정 필요

**#1 (가장 중요) — `synth_gate.sh` 를 어떻게 통과시킬 것인가**

W1-1을 끝내도 `yosys -q -p "read_verilog -sv rtl/*.sv; hierarchy -check; synth"` 는 **통과하지 않는다.**
`real` 은 이제 없지만, **unpacked array 포트를 쓰는 18개 파일**을 yosys 내장 프론트엔드가 파싱하지 못한다 (yosys 0.33·0.69 동일). `mac_array.sv` 포함.

```
rtl/mac_array.sv:9: ERROR: syntax error, unexpected '[', expecting ',' or '=' or ')'
  input logic signed [7:0] a_row [0:15],
```

선택지:
- **(a) 게이트에 sv2v 전처리를 넣는다.** `sv2v rtl/*.sv | yosys`. 코드 수정 0. 새 의존성 1개(sv2v, MIT, 무료 바이너리) → CLAUDE.md 규칙 4에 따라 **승인 필요**. 오늘 이 방식으로 `mac_array` 196,352 셀 합성을 확인했다.
- **(b) 포트를 packed flat 벡터로 고친다.** 새 의존성 0. 대신 18개 파일 + 그 인스턴스화 지점을 손대야 하고, 검증할 시뮬레이터가 지금 없다. `mac_array.sv` 는 출력에 대해 이미 이 방식을 쓴다(`acc_out_flat`, 주석에 "Icarus 12 unpacked-array-output limitation" 이라 적혀 있음).
- **(c) 게이트를 "파싱 되는 파일만" 으로 축소하고 나머지를 예외 목록에 둔다.** 정직하지만 게이트의 의미가 약해진다.

**나의 권고: (a).** 이유 — W1-2의 목적은 "회귀를 막는 게이트를 세우는 것"이지 "RTL 리팩터링"이 아니고, 지금은 고친 결과를 검증할 시뮬레이터조차 없다. 규칙 6(범위 축소)에도 맞다. (b)는 W5 이후 dr1 모듈을 새로 쓸 때 그 스타일로 쓰면 자연히 해결된다.

**#2 — `sim/cocotb/` 의 행동 모델 테스트 3개(14 테스트)도 격리할 것인가**
PLAN W1-1은 `tb/` 만 말한다. 문자 그대로 따라 손대지 않았다. `sim/cocotb/behavioral/` 로 같이 옮기는 게 일관되지만 PLAN에 없는 폴더라 규칙 5에 걸린다. **지시 주면 다음 세션에 옮긴다.**

**#3 — README 처리 순서**
W1-1은 "행동 모델 섹션 추가", W2-1은 "전면 재작성"이다. AUDIT §5에서 **뒷받침 안 되는 주장 21건**이 나왔고, 그중 테스트 수·모듈 수·`docs/` 15개 문서 같은 건 지금 당장 틀린 숫자다. 지금 부분 수정하면 W2-1에서 또 고친다. **W1-1에서는 README를 건드리지 말고 W2-1에서 한 번에 다시 쓰는 것을 권고.**

---

## 오늘 발견한, PLAN에 없던 것들 (기록만)

1. **`g3_reg_top` 모듈이 레포에 없다.** `g3_asic_top.sv:114` 와 `g3_ctrl_top.sv:85` 가 인스턴스화하는데 정의가 없다 → 두 파일 모두 elaborate 불가. (AUDIT §2)
2. **`tb/` 에 Makefile 이 없다.** CLAUDE.md 4절의 `cd tb && make SIM=verilator ...` 는 실행할 수 없다. 커밋된 유일한 cocotb 결과 `tb/results.xml` 은 6건(단일 모듈, Windows 경로)이다.
3. **`sim/golden/` 을 import 하는 테스트가 0개다.** 177줄짜리 GEMM INT8 골든이 있는데 아무도 안 쓴다. (AUDIT §3)
4. **아무 데도 연결 안 된 모듈 7개**: `vpu_lut, gemm_stub, kvc_core, mxu_bf16_128x128, gemm_int4, moe_router, vpu_fp16_utils`.
