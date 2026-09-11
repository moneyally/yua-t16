# yua-t16 / ORBIT — 상태 고정 델타룰 헤드 (d=16) + 검증된 제어 평면

Q1.15 델타룰 헤드 1개가 **골든 모델과 비트 단위로 일치한다** (1,000토큰 × 시드 3개).
INT8 16×16 외적 누산 타일과 제어 평면은 그대로 쓰인다. 설계는 [docs/DESIGN.md](docs/DESIGN.md).

SystemVerilog RTL 과 Python 호스트 스택. **시뮬레이션 단계이며 실물 보드에서 동작한 적이 없다.**

---

## 현재 상태

| 블록 | 상태 | 근거 |
|---|---|---|
| **ORBIT-DR1 델타룰 헤드 (d=16)** — `dr1_top` + `matvec_unit`/`update_unit`/`err_unit`/`requant_q15`/`state_sram`/`vec_regs`/`dr1_scratch` | **동작 + 비트 정확** | `DELTA_INIT`/`STEP`/`DUMP` 전부. 골든 `step()` 과 **1,000토큰 × 시드 3개 비트 일치** (`bash scripts/run_dr1_tb.sh`). `dr1_top` 124,461 cells |
| **호스트 경로 E2E** — `dev.delta_step()` → 디스크립터 → RTL → 결과 | **동작** | `tb/tb_dr1_host_e2e.py` 6/6. 스크래치 MMIO 창 왕복 포함 |
| **AXI4-Lite 브리지** — `axil_reg_bridge`, `dr1_soc_top` | **시뮬레이션 검증** | `tb/tb_axil_reg_bridge.py` 7/7. **보드에서는 안 돌려봤다** ([docs/FPGA.md](docs/FPGA.md)) |
| `mac_pe`, `mac_array` — INT8 16×16 출력 고정 외적 누산 | **합성됨** | `mac_array` 196,352 cells (`scripts/synth_gate.sh`) |
| 제어 평면 — `reg_top`, `desc_queue`, `desc_fsm_v2`, `irq_ctrl`, `trace_ring`, `oom_guard`, `reset_seq`, `cdc_fifo` | **합성됨** | `g2_ctrl_top` 666,029 cells (DR1 포함) |
| `gemm_core` / `gemm_top` — DMA + MAC 오케스트레이션 | **합성됨** | 429,672 / 430,402 cells |
| Python 호스트 스택 (`tools/`) — HAL, 디스크립터 패커, 레지스터맵 SSOT, 트레이스 디코더, CLI | **동작** | `python3 -m pytest tests/ -q` |
| PCIe (`pcie_ep_versal`) | **스텁** | CPM AXI-Stream 포트가 연결되지 않았다. 호스트와 통신한 적 없음 |
| 외부 메모리 (DDR/HBM) | **없음** | `dma_bridge` 는 상태머신이고 실제 메모리 인터페이스가 아니다 |
| 학습 경로 (`optimizer_unit`, `loss_scaler`, `collective_engine`, G3 top) | **행동 모델 (합성 불가)** | `rtl/behavioral/` 로 격리. 아래 참조 |
| BF16 (`mxu_bf16_16x16`) | **미측정** | 손으로 만든 FP32 가산기 256개. yosys 가 시간 예산 안에 못 끝낸다 |
| 실물 보드 | **없음** | 보드를 산 적이 없다 |

### 알려진 한계

- **`DELTA_STEP` 은 토큰당 205 사이클**이다. `docs/DESIGN.md` 6절 계약 상한은 64 —
  **3.2배 초과**다. 내역과 줄일 방법은 `docs/DESIGN.md` 6.2절에 적어 뒀다.
  상한을 고쳐 쓰지 않았다.
- **d=64 는 아직 안 된다.** 스크래치(1024 원소)에 64×64 덤프가 안 들어간다
  (`spec/deltarule.md` 3.6절). 4608 원소로 키워야 한다.
- **실물 보드 없음.** Vivado 합성·타이밍은 한 번도 돌린 적이 없다 ([docs/FPGA.md](docs/FPGA.md)).
- **PCIe·외부 메모리 없음.** 아래 표 참조.

미해결 버그 목록과 수정된 버그의 근거는 전부 [docs/BUGS.md](docs/BUGS.md) 에 있다
(BUG-001 `done_pulse` 는 2026-09-11 수정됨 — 완료 신호 3분할).

---

## 행동 모델 (합성 불가) — `rtl/behavioral/`

`rtl/behavioral/` 에 있는 14개 모듈은 **하드웨어가 아니다.** SystemVerilog 로 쓰인
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

sv2v → yosys 로 `rtl/` 전체를 2단 검사한다 (elaborate + synth).
**일일 게이트다. 최종 합성·타이밍 판정은 Vivado 로만 한다.**
기본 예산(모듈당 240s)으로는 `g2_ctrl_top` 이 시간 초과로 뜬다 — 정상이다.
전부 잡으려면 `SYNTH_TIMEOUT=420`. 빠른 확인만 하려면 `--stage1`.

### 호스트 스택 테스트

```bash
python3 -m pytest tests/ -q
# 322 passed, 8 xfailed
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

### 골든 모델 뮤테이션 테스트

```bash
python3 scripts/mutation_test.py        # 13/13 killed 여야 한다
```

골든을 한 군데씩 고의로 망가뜨리고 `tests/test_golden_deltarule.py` 가
**반드시 실패하는지** 본다. 살아남는 뮤턴트가 있으면 테스트에 구멍이 있는 것이다.
실제로 이 방법으로 구멍 하나(α/β 정의역 미검사)를 찾아서 메웠다.

### 금지 토큰 검사

```bash
python3 scripts/check_banned_tokens.py rtl/*.sv rtl/*.v; echo $?
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
  └── reset_seq ──── 리셋 시퀀서 (POR/SW/WDOG)

보드용 최상위는 `dr1_soc_top` = `axil_reg_bridge` (AXI4-Lite) + `g2_ctrl_top`.
```

```
rtl/              합성 대상 31개 파일
rtl/behavioral/   행동 모델 14개 — 합성 대상 아님
tb/               cocotb 테스트벤치 + run_tb.py
tb/behavioral/    행동 모델용 테스트벤치
tools/            Python 호스트 스택 15개 모듈
tests/            호스트 스택 pytest 18개 파일
sim/golden/       numpy 골든 모델 — **deltarule.py 가 DR1 의 정답지다**
spec/             SSOT 설계 문서 9개
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
