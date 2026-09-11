# yua-t16 / ORBIT — 상태 고정 델타룰 헤드 (d=16) + 검증된 제어 평면

Q1.15 델타룰 헤드 1개가 **골든 모델과 비트 단위로 일치한다** (1,000토큰 × 시드 3개).
INT8 16×16 외적 누산 타일과 제어 평면은 그대로 쓰인다. 설계는 [docs/DESIGN.md](docs/DESIGN.md).

SystemVerilog RTL 과 Python 호스트 스택. **시뮬레이션 단계이며 실물 보드에서 동작한 적이 없다.**

---

## 현재 상태

| 블록 | 상태 | 근거 |
|---|---|---|
| **ORBIT-DR1 델타룰 헤드 (d=16)** — `dr1_top` + `matvec_unit`/`update_unit`/`err_unit`/`requant_q15`/`state_sram`/`vec_regs`/`dr1_scratch` | **동작 + 비트 정확** | `DELTA_INIT`/`STEP`/`DUMP` 전부. 골든 `step()` 과 **1,000토큰 × 시드 3개 비트 일치** (`bash scripts/run_dr1_tb.sh` → **17/17, 101 테스트**). 126사이클/토큰 (d=16, `dr1_top` 내부), 462 (d=64). 호스트가 읽는 `DR1_CYCLES` 는 디스패치 1사이클을 포함해 **127** 이다 |
| **호스트 경로 E2E** — `dev.delta_step()` → 디스크립터 → RTL → 결과 | **동작** | `tb/tb_dr1_host_e2e.py` 6/6. 스크래치 MMIO 창 왕복 포함 |
| **보드 최상위 전체** — `dr1_soc_top` = `axil_reg_bridge` + `g2_ctrl_top` + `axi4_master_adapter` | **시뮬레이션 검증** | `tb/tb_dr1_soc_top.py` 6/6 — 호스트 스택이 **AXI4-Lite 만으로** `G2_ID` 읽기 → `DELTA_INIT` → `DELTA_STEP` 10토큰 골든 비트 일치. `tb/tb_axil_reg_bridge.py` 7/7. **보드에서는 안 돌려봤다** ([docs/FPGA.md](docs/FPGA.md)) |
| `mac_pe`, `mac_array` — INT8 16×16 출력 고정 외적 누산 | **합성됨** | `mac_array` 196,352 cells (`scripts/synth_gate.sh`) |
| 제어 평면 — `reg_top`, `desc_queue`, `desc_fsm_v2`, `irq_ctrl`, `trace_ring`, `oom_guard`, `reset_seq`, `wdog_timer`, `cdc_fifo` | **합성됨** | `g2_ctrl_top` 885,790 cells (DR1 포함) |
| **워치독** — `wdog_timer` | **동작** | `tb/tb_wdog_timer.py` 7/7 (사이클 단위로 파이썬 모델과 대조) + `tb/tb_g2_ctrl_top_wdog.py` 6/6 (레지스터→리셋→`BOOT_CAUSE[1]` 전체 경로). 전에는 레지스터만 있고 타이머가 없었다 |
| `gemm_core` / `gemm_top` — DMA + MAC 오케스트레이션 | **합성됨** | 429,672 / 430,404 cells |
| Python 호스트 스택 (`tools/`) — HAL, 디스크립터 패커, 레지스터맵 SSOT, 트레이스 디코더, CLI | **동작** | `python3 -m pytest tests/ -q` → **344 passed, 8 xfailed** |
| PCIe (`pcie_ep_versal`, `g2_protob_top`) | **범위 밖** | Versal CPM 용이다. 타깃 보드가 KV260(Zynq MPSoC)이라 이 경로는 안 쓴다. 아래 "범위 밖" 참조 |
| **AXI4 마스터 (외부 메모리)** — `axi4_master_adapter` | **시뮬레이션 검증** | `tb/tb_axi4_master_adapter.py` 8/8. 256 beat 상한·4KB 경계에서 버스트를 쪼갠다 (슬레이브 모델이 위반을 assert 한다). `dr1_soc_top` 에 결선됨. **실제 DDR 에 붙여 본 적 없다** |
| 학습 경로 (`optimizer_unit`, `loss_scaler`, `collective_engine`, G3 top) | **행동 모델 (합성 불가)** | `rtl/behavioral/` 로 격리. 아래 참조 |
| BF16 (`mxu_bf16_16x16`) | **미측정** | 손으로 만든 FP32 가산기 256개. yosys 가 시간 예산 안에 못 끝낸다 |
| **긴 시퀀스 안정성** — Q1.15 상태가 실수 델타룰을 계속 따라가는가 | **유계 확인** | α 를 표현 가능한 최대값(0.999969, 사실상 망각 없음)으로 고정해도 400→3200토큰에서 드리프트가 **~11 LSB 로 평탄**하다 (풀스케일의 0.034%). `tests/test_golden_drift.py` 가 **기울기**를 지킨다. 표는 [docs/DESIGN.md](docs/DESIGN.md) 6.5절 |
| 실물 보드 | **없음** | 보드를 산 적이 없다 |

### 범위 밖 — "스텁" 이 아니라 **안 쓰는 길**이다

아래 모듈들은 미완성인 채로 레포에 있다. 그런데 **끝낼 계획이 없다.** 이전
설계(Versal 보드 + 학습 지원 + 멀티칩)의 유물이고, 지금 목표(KV260 위의 델타룰
헤드 1개)와 관계가 없다. "스텁" 이라고만 적어 두면 갚아야 할 빚처럼 보이므로
여기에 이유를 적어 둔다.

| 모듈 | 왜 안 쓰나 |
|---|---|
| `pcie_ep_versal`, `g2_protob_top` | Versal CPM 전용. 타깃이 KV260(Zynq MPSoC)이라 호스트는 **PS 의 AXI4-Lite** 로 들어온다 (`dr1_soc_top`). CQ→BAR TLP 디코드를 지금 쓰면 PG347 도 보드도 없어 **검증할 방법이 없는 코드**가 된다 |
| `g3_desc_fsm` 의 BACKWARD / OPTIMIZER / COLLECTIVE 디스패치 | 학습 경로다. 이 레포의 범위가 아니다 (`CLAUDE.md` 0절) |
| `scale_fabric_ctrl` 의 PHY | 멀티칩 패브릭. 칩이 하나도 없다 |
| `reg_top` 의 `BOOT_VECTOR_LO/HI` | 부팅할 CPU 코어가 없다. 레지스터만 존재 |

**끝낼 계획이 있는 것**은 이 표에 없다. 위 상태표에서 "시뮬레이션 검증"인 항목들이
그것이고, 남은 일은 전부 **보드가 있어야** 한다.

### 알려진 한계

- **`DELTA_STEP` 은 토큰당 126 사이클**이다. `docs/DESIGN.md` 6절 계약 상한은 64 —
  **2.0배 초과**다 (최초 205에서 줄였다. 내역은 `docs/DESIGN.md` 6.2·6.3절).
  상한을 고쳐 쓰지 않았다.
- **d=64 는 시뮬레이션까지만 확인했다.** 같은 RTL 을 `PARAM_D=64 PARAM_SCRATCH=8192`
  로 돌려 골든과 20토큰 비트 일치, 462사이클/토큰
  (`tb/tb_dr1_d64.py`). **통합 빌드(`g2_ctrl_top`)는 면적 때문에 d=16 그대로**다 —
  d=64 가 보드에 들어갈지는 Vivado 를 돌려야 안다.
- **실물 보드 없음.** Vivado 합성·타이밍은 한 번도 돌린 적이 없다 ([docs/FPGA.md](docs/FPGA.md)).
- **PCIe 없음.** 위 표 참조.
- **외부 메모리는 시뮬레이션까지만.** `axi4_master_adapter` 가 AXI4 로 버스트를
  내지만, 그것을 받은 것은 파이썬 슬레이브 모델뿐이다. 실제 DDR 컨트롤러의
  타이밍·역압·재정렬은 보지 않았다.

미해결 버그 목록과 수정된 버그의 근거는 전부 [docs/BUGS.md](docs/BUGS.md) 에 있다
(BUG-001 `done_pulse` 는 2026-09-11 수정됨 — 완료 신호 3분할).

---

## 행동 모델 (합성 불가) — `rtl/behavioral/`

`rtl/behavioral/` 에 있는 16개 모듈은 **하드웨어가 아니다.** SystemVerilog 로 쓰인
동작 기술이며, 합성 대상이 아니고 `scripts/synth_gate.sh` 검사에서 제외된다.

| 모듈 | 왜 하드웨어가 아닌가 |
|---|---|
| `optimizer_unit`, `loss_scaler`, `collective_engine`, `vpu_core`, `vpu_fp16_utils`, `gemm_int4`, `moe_router` | `real` 타입 + `$itor`/`$rtoi`/`$exp`/`$sqrt` 부동소수 연산 |
| `g3_train_int_top`, `g3_multistep_int_top`, `g3_2chip_int_top`, `g3_2chip_fabric_int_top`, `vpu_top` | 위 모듈들을 인스턴스화한다 |
| `g3_asic_top`, `g3_ctrl_top` | 존재하지 않는 모듈 `g3_reg_top` 을 인스턴스화한다 — elaborate 불가 |
| `kvc_core` | `kv_store` 6차원 배열 = 2 Mbit 플립플롭. yosys `std::bad_alloc` |
| `mxu_bf16_128x128` | `tile_acc` = 524,288 플립플롭 |

이들의 테스트벤치는 `tb/behavioral/`, `sim/cocotb/behavioral/` 에 있다.

---

## 실행 가능한 명령

아래는 전부 실제로 돌아가는 것만 적었다. 툴체인 설치는 `bash scripts/setup_tools.sh`.

### 합성 가능성 게이트

```bash
bash scripts/synth_gate.sh; echo $?     # 0 이어야 한다
```

sv2v → yosys 로 `rtl/` 전체를 3단 검사한다:
1. **elaborate + `check -assert`** — undriven 포트, 다중 드라이버 같은 구조 오류
2. **`synth -top`** — 합성이 끝까지 도는지, 0셀로 사라지지 않는지
3. **셀 수 회귀** — `scripts/cell_baseline.txt` 대비 **+10% 초과면 FAIL**

3번이 있는 이유: **기능 테스트는 면적 회귀를 못 잡는다.** 실제로 논리적으로 같은
식으로 바꿨는데 `mac_array` 가 +46% 커진 적이 있고, 그때 cocotb 는 전부 통과했다
([docs/BUGS.md](docs/BUGS.md) BUG-011). 의도한 증가면 기준선을 갱신하고 이유를
커밋 메시지에 적는다.

**일일 게이트다. 최종 합성·타이밍 판정은 Vivado 로만 한다.**
기본 예산(모듈당 240s)으로는 `g2_ctrl_top` 이 시간 초과로 뜬다 — 정상이다.
전부 잡으려면 `SYNTH_TIMEOUT=420`. 빠른 확인만 하려면 `--stage1`.

### 호스트 스택 테스트

```bash
python3 -m pytest tests/ -q
# 344 passed, 8 xfailed
```

xfail 8건: `fpga/vck190/create_cpm_ip.tcl` 에 CPM 설정이 없어서 5건,
`docs/ORBIT_G2_VCK190_*.md` 3개가 저장소에 없어서 3건. 둘 다 **단언이 옳고
대상이 미완성**인 경우다 — 지우거나 기대값을 바꿔 통과시키지 않았다.

### RTL 시뮬레이션 (cocotb)

```bash
python3 tb/run_tb.py <toplevel> <module> [소스.sv ...]

# 예: BUG-001 재현
python3 tb/run_tb.py g2_ctrl_top tb_g2_ctrl_top_fault_irq $(ls rtl/*.sv)
```

`tb/` 에 Makefile 은 없다. 러너를 쓴다. 파형은 `build/tb/<toplevel>/` 에 FST 로 남는다.

### ORBIT-DR1 테스트벤치 (전부 골든 비트 비교)

```bash
bash scripts/run_dr1_tb.sh; echo $?     # 0 이어야 한다
```

각 RTL 모듈이 `sim/golden/deltarule.py` 의 **같은 이름 함수**와 비트 단위로
일치하는지 본다. 대응은 `requant_q15↔requantize_q15`, `matvec_unit↔matvec`,
`update_unit↔update_row`, `err_unit↔compute_err`, `dr1_top↔step`.

### 뮤테이션 테스트 (불변 코드)

```bash
python3 scripts/mutation_test.py        # 18/18 killed 여야 한다
```

소스를 한 군데씩 고의로 망가뜨리고 그 오라클 테스트가 **반드시 실패하는지** 본다.
살아남는 뮤턴트가 있으면 테스트에 구멍이 있는 것이다. 실제로 이 방법으로
구멍 하나(α/β 정의역 미검사)를 찾아서 메웠다.

대상은 골든 모델(`sim/golden/deltarule.py`)과 `tools/orbit_mmio_map.py` 의 레지스터
비트 배치다 — 둘 다 **한 번 정하면 안 바뀌고, 틀려도 조용히 도는** 코드다.

### 금지 토큰 검사

```bash
python3 scripts/check_banned_tokens.py rtl/*.sv rtl/*.v rtl/dr1/*.sv; echo $?
```

### 디버그 CLI (RTL 없이 동작 — SimBackend)

```bash
python3 -m tools.orbit_debug_protoa info
python3 -m tools.orbit_debug_protoa queue-status
python3 -m tools.orbit_debug_protoa trace-dump --count 16
```

---

## 구조

```
호스트 (Python)
  OrbitDevice HAL ── SimBackend / CocotbBackend / MmapBackend
        │  MMIO + 64B 디스크립터
        ▼
g2_ctrl_top  (제어 평면, 합성됨)
  ├── reg_top ────── MMIO 레지스터 뱅크 (SSOT: tools/orbit_mmio_map.py)
  ├── desc_queue ─── 4큐 링버퍼 + 우선순위 arbiter
  ├── desc_fsm_v2 ── CRC-8 / opcode / 타임아웃 검증
  ├── gemm_top ───── ctrl_fsm + gemm_core
  │      └── gemm_core ── act_sram/wgt_sram + mac_array (INT8 16×16)
  ├── dr1_top ────── **델타룰 헤드 (d=16, Q1.15)**
  │      ├── state_sram ── d×d 상태 (온칩 상주, BRAM)
  │      ├── vec_regs ──── q/k/v
  │      ├── matvec_unit ─ p = S·k, o = S_next·q  (requant_q15)
  │      ├── err_unit ──── err = v − α·p
  │      └── update_unit ─ S ← α·S + β·err·kᵀ  (mac_pe × d)
  └── dr1_scratch ── q/k/v/o 벡터 + 덤프 (호스트 MMIO 창 0x8033_1000)
  ├── oom_guard ──── 4상태 메모리 압력 제어
  ├── trace_ring ─── 디버그 이벤트 링 (SAT_EVENT / CLAMP_EVENT 포함)
  ├── irq_ctrl ───── 인터럽트 컨트롤러 (W1C)
  ├── reset_seq ──── 리셋 시퀀서 (POR/SW/WDOG)
  └── wdog_timer ─── 워치독 타이머 (WDOG_CTRL → reset_seq)

보드용 최상위는 `dr1_soc_top` =
`axil_reg_bridge` (AXI4-Lite 슬레이브, 호스트→레지스터)
+ `g2_ctrl_top`
+ `axi4_master_adapter` (AXI4 마스터, 코어→외부 메모리).
```

```
rtl/              합성 대상 35개 파일 (.sv 33 + .v 2)
rtl/dr1/          DR1 델타룰 헤드 10개 — 역시 합성 대상
rtl/behavioral/   행동 모델 16개 — 합성 대상 아님
tb/               cocotb 테스트벤치 51개 + run_tb.py
tb/behavioral/    행동 모델용 테스트벤치 8개
tools/            Python 호스트 스택 17개 모듈
tests/            호스트 스택 pytest 23개 파일
sim/golden/       numpy 골든 모델 6개 — **deltarule.py 가 DR1 의 정답지다**
spec/             SSOT 설계 문서 11개 (deltarule.md, watchdog.md 등)
scripts/          synth_gate.sh, setup_tools.sh, check_banned_tokens.py
fpga/vck190/      Vivado Tcl (CPM 설정 미완성)
fpga/kv260/       Vivado Tcl — **실행해 본 적 없음** (docs/FPGA.md)
openlane/         gemm_int4_sky130 OpenLane 설정
docs/             DESIGN / PLAN / AUDIT / BUGS / LINT / LOG
```

---

## 문서

| 문서 | 내용 |
|---|---|
| [docs/DESIGN.md](docs/DESIGN.md) | ORBIT-DR1 델타룰 헤드 설계 (SSOT) |
| [docs/PLAN.md](docs/PLAN.md) | 12주 실행 계획 |
| [docs/AUDIT.md](docs/AUDIT.md) | 레포 현황 실측 감사 — 모든 항목에 명령 출력 첨부 |
| [docs/BUGS.md](docs/BUGS.md) | 파형·명령 출력 근거가 있는 버그만 |
| [docs/LINT.md](docs/LINT.md) | verilator 린트 경고 기록 |
| [docs/LOG.md](docs/LOG.md) | 세션 기록 |
| [docs/FPGA.md](docs/FPGA.md) | 보드 빌드 상태 — **측정값 없음** |
| [docs/PRIOR_ART.md](docs/PRIOR_ART.md) | 선행 연구. DR1 은 최초가 아니다 |
| [docs/RESEARCH.md](docs/RESEARCH.md) | 빈 질문 조사 (계획만) |
| [docs/GPU_FEASIBILITY.md](docs/GPU_FEASIBILITY.md) | "그래픽카드로 틀면?" 조사와 기각 사유 |
| [spec/deltarule.md](spec/deltarule.md) | DR1 디스크립터·레지스터 계약 |
| [CLAUDE.md](CLAUDE.md) | 작업 규칙 |

**계획·목표는 `docs/PLAN.md` 에만 쓴다. 이 README 는 현재 상태만 말한다.**

---

## 툴체인

전부 무료·오픈소스. `bash scripts/setup_tools.sh` / 상태 확인은 `--check`.

| 도구 | 용도 | 비고 |
|---|---|---|
| yosys | 합성 게이트 | apt |
| sv2v | SystemVerilog → Verilog-2005 | yosys·iverilog 가 unpacked array 포트를 못 읽는다 |
| iverilog | cocotb 시뮬레이션 | apt |
| verilator | 린트 (`--lint-only`) | apt 5.020 으로 충분 |
| verilator 5.022+ | cocotb 2.x 시뮬레이션 | **소스 빌드 필요** — apt 판에는 `VerilatedVpi` API 가 없다 |
| cocotb 2.x, pytest, numpy | 테스트 | pip |

---

## License

MIT
