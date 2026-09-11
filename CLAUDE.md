# CLAUDE.md — yua-t16 / ORBIT 작업 규칙

이 파일은 이 레포에서 작업하는 모든 Claude Code 세션이 먼저 읽는다.
사용자(정원)는 낮에 폰으로 짧은 지시를 보내고, 저녁에 결과를 검증한다.
따라서 이 문서의 규칙은 "지시가 짧아도 방향이 흔들리지 않게" 하기 위한 것이다.

## 0. 북극성 (바뀌지 않는 방향)

이 레포의 장기 목표는 **상태 고정(state-stationary) AI 추론 유닛**이다.
- 가중치가 아니라 **세션별 상태 행렬**이 연산기 옆 SRAM에 상주하고, 토큰이 지나가며 그 상태를 제자리에서 갱신한다.
- 첫 목표물은 **델타룰(Gated DeltaNet 계열) 헤드 1개**를 FPGA에서 돌리는 것이다.
- 기존 `mac_array.sv`(외적 누산)가 그 갱신 유닛의 핵심이다. 새로 만들지 말고 진화시킨다.
- 상세는 `docs/DESIGN.md`, 일정은 `docs/PLAN.md`.

"GPU급", "프론티어 모델", "학습 지원" 같은 목표는 이 레포의 범위가 아니다. README나 문서에 그런 표현을 쓰지 않는다.

## 1. 절대 규칙 (어기면 그 커밋은 되돌린다)

1. **합성 가능성이 정답이다.** 합성 대상 RTL(`rtl/*.sv`, `rtl/*.v` — `rtl/behavioral/` 제외)은 `bash scripts/synth_gate.sh` 가 0으로 끝나야 한다. 금지 토큰은 두 등급이다:

   | 등급 | 토큰 | 예외 |
   |---|---|---|
   | **HARD** — 어디에 있든 금지 | `real`, `$itor`, `$rtoi`, `$exp`, `$sqrt`, `#delay` | 없음. 해당 모듈은 `rtl/behavioral/` 로 |
   | **SIM-ONLY** — 합성 경로에서 금지 | `===`, `!==`, `while`, `initial` 로직 | `` `ifdef COCOTB_SIM `` 안은 허용. `tb/` 는 검사 대상 아님 |

   SIM-ONLY 가 왜 금지인가: 시뮬레이션에서는 의미가 있지만 **합성에서는 조용히 다른 회로가 된다.** `===`/`!==` 는 합성기가 조건을 상수로 접고, 경계 없는 `while` 은 하드웨어가 아니다. 실제로 이것 때문에 `wgt_sram` 이 통째로 사라졌고(0 cells vs 99,294) `gemm_core` 로직 95%가 죽었다 — `docs/BUGS.md` BUG-003, BUG-006, BUG-007.

   검사: `python3 scripts/check_banned_tokens.py rtl/*.sv rtl/*.v` (주석·문자열·`ifdef` 인식. `grep` 은 블록 주석을 못 거른다)
2. **테스트는 RTL 바깥의 정답과 비교한다.** cocotb 테스트의 기대값은 `sim/golden/`의 numpy 모델에서 나와야 한다. RTL 내부 함수를 참조 모델로 쓰는 테스트는 테스트가 아니다. 새 테스트마다 "기대값의 출처"를 docstring 첫 줄에 적는다.
3. **완료 주장에는 명령 출력이 붙는다.** "테스트 통과"라고 쓰지 않는다. 실행한 명령과 마지막 20줄 출력을 그대로 보고에 붙인다. 실행하지 못했으면 "실행 못 함"이라고 쓴다.
4. **비용 0.** 클라우드, 유료 API, 유료 툴을 추가하지 않는다. 툴체인은 verilator, iverilog, yosys, cocotb, Vivado(무료 에디션 범위)로 한정한다. 새 의존성은 이유를 적고 사용자 승인 후에 추가한다.
5. **한 번에 하나.** `docs/PLAN.md`의 현재 주차 작업만 한다. 다음 주차 작업이 눈에 보여도 시작하지 않는다. 새 최상위 모듈, 새 폴더, 새 서브프로젝트는 PLAN에 있는 것만 만든다.
6. **범위 축소가 기본 동작이다.** 막히면 기능을 추가해서 우회하지 않고, 범위를 줄여서 끝낸다. 예: 64×64가 안 되면 16×16으로 줄여 끝내고 보고한다.
7. **모르면 모른다고 쓴다.** 타이밍, 면적, 보드 동작처럼 실제로 돌려보지 않은 것은 "추정" 또는 "미검증"이라고 표시한다.
8. **README는 현재 상태만 말한다.** 계획·희망·목표는 `docs/PLAN.md`에만 쓴다.

## 2. 레포 현재 상태 (2026-09 기준, 외부 검토 결과)

잘 된 것:
- 제어 평면: `reg_top`, `desc_queue`, `desc_fsm_v2`, `irq_ctrl`, `trace_ring`, `reset_seq`, `cdc_fifo` — 구조 양호, 호스트 스택(`tools/`)과 cocotb E2E 연결됨.
- `mac_pe`, `mac_array`: 16×16 INT8 출력 고정 외적 누산. 합성 가능. **이것이 북극성의 핵심 부품이다.**
- `spec/`의 SSOT 규율, `tools/orbit_mmio_map.py` 레지스터맵 단일 소스.

고쳐야 하는 것 (우선순위 순):
- ~~**`real` 타입 사용 모듈 7개 — 합성 불가.**~~ **2026-09-10 W1-1 완료:** 7개(`collective_engine`, `gemm_int4`, `loss_scaler`, `moe_router`, `optimizer_unit`, `vpu_core`, `vpu_fp16_utils`) + 이들에 의존하는 상위 모듈을 `rtl/behavioral/` 로 격리했다. 테스트벤치는 `tb/behavioral/`, `sim/cocotb/behavioral/`. `rtl/*.sv` 는 이제 금지 토큰 0건.
- **합성 게이트를 막는 진짜 원인은 `real` 이 아니었다 — unpacked array 포트다.** yosys 내장 프론트엔드(0.33·0.69)가 파싱하지 못한다. 합법 SystemVerilog 이고 Vivado 는 합성하므로 **RTL 을 고치지 않고 `sv2v` 전처리를 게이트에 넣는다** (`scripts/synth_gate.sh`). 일일 게이트는 sv2v→yosys, **최종 합성 판정은 Vivado.**
- ~~`pcie_ep_versal.sv`: CPM AXI-Stream 포트가 스텁.~~ **2026-09-11 판정: 범위 밖.** Versal CPM 전용인데 타깃 보드가 KV260(Zynq MPSoC)이다. 호스트는 PS 의 AXI4-Lite 로 들어온다 (`dr1_soc_top`, `tb/tb_dr1_soc_top.py` 6/6). CQ→BAR TLP 디코드를 지금 쓰면 PG347 도 보드도 없어 **검증할 방법이 없는 코드**가 된다 — 규칙 6·7 에 어긋난다. README "범위 밖" 표에 이유를 적었다. **정원이 다르게 판단하면 되돌린다** (LOG 결정 필요 (1)).
- ~~외부 메모리(DDR/HBM) 경로 없음. `dma_bridge`는 상태머신이고 실제 메모리 인터페이스가 아님.~~ **2026-09-11 완료(시뮬레이션 한정):** `rtl/axi4_master_adapter.sv` 가 `rd_req_*`/`wr_req_*` 를 AXI4 마스터로 바꾼다 — 256 beat 상한과 4KB 경계에서 버스트를 쪼갠다. `rtl/dr1_soc_top.sv` 에 결선됨(전에는 tie-off). 검증: `python3 tb/run_tb.py axi4_master_adapter tb_axi4_master_adapter rtl/axi4_master_adapter.sv` = 8/8, 슬레이브 모델이 프로토콜 위반을 assert 한다. **보드의 실제 DDR 에 붙여 본 적은 없다 — 미검증.**
- `mxu_bf16_128x128`은 16×16 타일 1개를 64회 반복 — 연산기 수는 256개. 이름이 실체보다 크다.
- **README 과장 — `docs/AUDIT.md` §5 에 21건 목록.** 실제 문구는 다음이다 (2026-09-10 감사, 명령 출력 첨부됨):
  - "**237 tests** ... All pass" → 실제 `python3 -m pytest tests/ -q` = **8 failed, 247 passed**.
  - "VCK190 Vivado project + **CPM config Done**" → `fpga/vck190/create_cpm_ip.tcl` 본문이 *"Cannot use create_ip ... configure CPM manually"*. **CPM 설정이 없다.** pytest 실패 5건의 원인.
  - "23 RTL modules"(실제 47 파일), "tb/ 9 testbenches 29 tests"(실제 36 파일 147 테스트), "docs/ 15 design documents"(`.gitignore` 가 `docs/` 를 무시해 추적 파일 0개였음 — 2026-09-10 수정).
  - "Custom LLM Inference Accelerator", "Full closed loop verified", "No mocks", "awaiting silicon" — 0절이 금지한 종류의 표현.
  - 이전 판의 "MPW ready", "training" 문구는 **현재 README 에 없다.** (`grep -niE "MPW|training" README.md` → 해당 없음). 위 목록으로 대체한다.
- **`done_pulse` 버그 확정 — `docs/BUGS.md` BUG-001.** fault 난 디스크립터가 `DESC_DONE` 완료 IRQ 를 올린다 (`IRQ_PENDING=0x21`). 재현 테스트·사이클표·파형 있음. **미수정** — `docs/DESIGN.md` 9절(완료 신호 정의)을 먼저 고쳐야 한다.

## 3. 작업 방식

- 세션 시작 시: `git status`, `git log --oneline -5`, `docs/PLAN.md`의 "현재 주차" 섹션을 읽고, 오늘 할 일 1~3개를 먼저 적는다.
- 커밋은 작게, 메시지는 `<영역>: <무엇을> — <검증 명령>` 형식. 예: `rtl: move real-typed modules to rtl/behavioral — yosys synth of rtl/*.sv passes`
- 세션 종료 시 `docs/LOG.md`에 5줄 이내로 추가: 한 것 / 안 된 것 / 실행한 검증 명령 / 다음 세션 첫 작업 / 사용자 결정이 필요한 것.
- 사용자에게 질문이 필요하면 작업을 멈추지 말고, 가장 보수적인 선택으로 진행한 뒤 LOG의 "결정 필요"에 적는다.

## 4. 검증 명령 모음

```bash
# 합성 가능성 게이트 — 이것 하나가 정답이다. 0 이어야 한다.
bash scripts/synth_gate.sh; echo $?
#   기본 예산(모듈당 240s)으로는 g2_ctrl_top 이 시간 초과로 뜬다. 정상이다.
#   전부 잡으려면: SYNTH_TIMEOUT=420 bash scripts/synth_gate.sh
#   빠른 확인만:   bash scripts/synth_gate.sh --stage1
#   일일 게이트는 sv2v -> yosys. **최종 합성 판정은 Vivado.**

# 금지 토큰만 따로 (주석·문자열·ifdef 인식)
python3 scripts/check_banned_tokens.py rtl/*.sv rtl/*.v; echo $?

# 린트
verilator --lint-only -Wall -Irtl rtl/<module>.sv

# 호스트 스택 테스트
python -m pytest tests/ -q

# cocotb — tb/ 에 Makefile 이 없다. 러너를 쓴다.
python3 tb/run_tb.py <toplevel> <module> [소스.sv ...]
#   예: python3 tb/run_tb.py g2_ctrl_top tb_g2_ctrl_top_fault_irq $(ls rtl/*.sv)
#   iverilog + sv2v 로 돌린다. Debian verilator 5.020 은 cocotb 2.x 와 비호환
#   (VerilatedVpi API 없음). verilator 소스 빌드가 다음 세션 1순위.
```

## 5. 저녁 검증 체크리스트 (사용자용)

Claude Code의 보고를 읽을 때 이 순서로 확인한다. 하나라도 "아니오"면 그 작업은 미완료다.
1. 보고에 실제 명령 출력이 붙어 있는가? (없으면 미완료)
2. yosys 합성 게이트가 통과했는가? (RTL을 건드린 날은 필수)
3. 새 테스트의 기대값이 `sim/golden/`에서 왔는가? (테스트 파일 첫 줄 확인)
4. 오늘 만든 것이 PLAN의 현재 주차 항목인가? (아니면 되돌린다)
5. LOG.md에 "안 된 것"이 정직하게 적혀 있는가? (전부 됐다는 보고는 의심한다)
6. 이해 안 되는 코드 한 덩어리를 골라 "이 줄이 왜 필요한지 설명해"라고 물어본다. (하루 1회, 학습용)
