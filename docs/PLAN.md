# PLAN.md — 12주 실행 계획 (ORBIT-DR1)

작성 2026-09-11. 매주 일요일 저녁 사용자가 "현재 주차"를 갱신한다. Claude Code는 **현재 주차 항목만** 작업한다.
투입 전제: 평일 저녁 4시간 중 RTL 1~2시간(주말은 칩 5~6시간), 낮에는 에이전트가 자율 진행.
예산: 1년 하드웨어 상한 100만원. 비상금 800만원은 건드리지 않는다.

## 현재 주차: 1

---

## 0단계 · 1~2주 · 청소와 정직화 (보드 구매 금지)

목표: 레포가 "지금 무엇인지"를 정확히 말하게 만든다. 새 기능 0개.

- [ ] W1-1 `rtl/behavioral/` 생성, `real` 사용 7개 모듈 이동 (`collective_engine, gemm_int4, loss_scaler, moe_router, optimizer_unit, vpu_core, vpu_fp16_utils`). 이들을 참조하는 G3 top·테스트벤치도 `rtl/behavioral/`, `tb/behavioral/`로 이동. README에 "행동 모델(합성 불가)" 섹션으로 분리 표기.
- [ ] W1-2 `scripts/synth_gate.sh` 작성: `rtl/*.sv`(behavioral 제외) 전체를 yosys로 합성, 실패 시 비-0 종료. `grep` 금지 토큰 검사 포함. 이 스크립트가 통과하는 상태를 첫 커밋으로.
- [ ] W1-3 `verilator --lint-only -Wall` 을 합성 대상 전 모듈에 돌리고, 경고 목록을 `docs/LINT.md`에 기록. 고치지는 않는다(기록만).
- [ ] W2-1 README 전면 재작성: 현재 상태표(제어 평면 ✔ / mac_array ✔ / PCIe 스텁 / DDR 없음 / 학습 경로 = 행동 모델), 실행 가능한 명령만 기재. "MPW ready", "LLM inference accelerator", "training" 문구 제거. 새 한 줄: "INT8 16×16 외적 누산 타일 + 검증된 제어 평면. 다음 목표: 델타룰 헤드 (docs/DESIGN.md)".
- [ ] W2-2 `done_pulse` 버그 재현: 실패하는 최소 테스트벤치 하나를 `tb/tb_gemm_core_done.py`로 만들고, VCD를 덤프해 `docs/BUGS.md`에 "어느 사이클에 어떤 신호가 기대와 다른지" 한 문단 기록. 이번 주에 고치지 않아도 된다. **재현 + 파형 기록이 완료 기준.**
- [ ] W2-3 `docs/LOG.md` 시작.

**2주 종료 기준**: `scripts/synth_gate.sh` 통과 / README에 과장 문구 0 / BUGS.md에 파형 근거가 있는 항목 1개 이상.
**사용자 검증**: 저녁에 `bash scripts/synth_gate.sh; echo $?` 직접 실행해 0 확인.

---

## 1단계 · 3~4주 · 골든 모델과 계약 (RTL 금지)

목표: 정답지를 먼저 만든다. 이 2주 동안 RTL 파일을 만들거나 수정하지 않는다.

- [ ] W3-1 `sim/golden/deltarule.py`: DESIGN.md 2절 정확식, Q1.15 정수 연산, round-half-to-even, 포화. 함수 `step(S, q, k, v, alpha, beta) -> (S_next, o, sat_count)`. float 참조 구현을 별도 함수로 두고, 정수 구현과 float 구현의 오차가 예상 범위(1 LSB 이내)인지 검사하는 자체 테스트.
- [ ] W3-2 `tests/test_golden_deltarule.py`: 불변조건 I1~I5를 골든 모델 수준에서 검사. d=16, 64 모두.
- [ ] W4-1 `spec/deltarule.md`: opcode 3개(DELTA_INIT/STEP/DUMP) 디스크립터 필드, 레지스터 추가분, 완료 신호 정의("상태 쓰기 완료 후 1사이클"). `tools/orbit_mmio_map.py`, `tools/orbit_desc.py` 갱신 및 `tests/test_desc_pack.py` 통과.
- [ ] W4-2 `tb/tb_dr1_top.py` 골격: RTL이 아직 없으므로 골든 모델을 DUT 자리에 두고 테스트 하네스 자체를 검증(하네스가 골든을 골든과 비교해 통과하는지). "RTL이 오면 DUT 한 줄만 바꾼다"가 완료 기준.

**4주 종료 기준**: `pytest tests/test_golden_deltarule.py` 통과 / spec 문서와 mmio_map이 일치 / 하네스가 골든-대-골든으로 통과.

---

## 2단계 · 5~8주 · RTL (d=16 먼저)

- [ ] W5 `rtl/dr1/state_sram.sv`(16×16 Q1.15, 1R1W), `rtl/dr1/vec_regs.sv`, `rtl/dr1/matvec_unit.sv`. 각각 단위 cocotb + yosys 게이트.
- [ ] W6 `rtl/dr1/dr1_top.sv` FSM: INIT / LOAD_QKV / MATVEC_K / ERR / UPDATE(mac_array 재사용) / MATVEC_Q / WRITE_O / DONE. 첫 목표: `DELTA_INIT` + `DELTA_DUMP`만으로 I1 통과.
- [ ] W7 `DELTA_STEP` 전체 경로. 골든 대비 **1토큰 비트 일치** → 10토큰 → 100토큰. 불일치 시 첫 불일치 지점 출력이 동작하는지 먼저 확인.
- [ ] W8 1,000토큰 무작위 시드 3개 비트 일치. 트레이스 링 `SAT_EVENT` 연동(I4). Verilator 린트 경고 0. **여기서 d=16 완료 선언.** d=64 확장은 W8 완료 후에만, 남는 시간에.

**8주 종료 기준**: `make -C tb dr1` 이 1,000토큰 × 3시드 비트 일치 출력 / synth_gate 통과 / DESIGN.md 6절 사이클 예산과 실측 사이클 비교표 기록.
**사용자 검증**: 파형(GTKWave)으로 `DELTA_STEP` 한 토큰을 직접 열어 FSM 상태 전이를 눈으로 따라간다. 이 주가 파형 읽기를 배우는 주다.

---

## 3단계 · 9~10주 · 호스트 연결

- [ ] W9 `tools/orbit_device.py`에 `delta_init/delta_step/delta_dump` 추가. `CocotbBackend`로 Python HAL → RTL E2E (기존 `tb_g2_ctrl_top_host_e2e.py` 패턴). 기대값은 골든.
- [ ] W10 `tests/test_dr1_host_e2e.py`: 호스트가 100토큰을 보내고 결과 o_t를 받아 골든과 비교. `driver/`의 uapi 헤더에 새 opcode 상수 추가(커널 드라이버 코드 수정은 보드 뒤로).

**10주 종료 기준**: Python에서 `dev.delta_step(...)` 호출 → RTL → 결과가 골든과 일치.

---

## 4단계 · 11~12주 · 실물로 가는 준비

- [ ] W11 보드 결정 게이트: 0~3단계 종료 기준이 전부 충족됐을 때만 KV260(또는 동급, 50만원 이내) 구매. 미충족이면 구매를 다음 달로 미루고 미충족 항목을 먼저 끝낸다.
- [ ] W11 `fpga/kv260/` Vivado 프로젝트: `dr1_top` + AXI-Lite 래퍼(기존 reg_top 재사용). 합성·구현·타이밍 리포트를 `docs/FPGA.md`에 기록. 목표 100MHz. 안 나오면 50MHz로 낮추고 기록.
- [ ] W12 보드 도착 시: LED → AXI 레지스터 읽기(G2_ID `0x47320001` 확인) → `DELTA_INIT` → `DELTA_STEP` 1토큰 → 골든과 비교. **여기까지가 12주의 최종 목표.** 보드가 안 왔으면 W12는 Tiny Tapeout 서브셋(4×4 mac_array + SPI) 준비로 대체.

**12주 종료 기준(최종)**: 실보드에서 `DELTA_STEP` 1토큰이 골든과 일치하는 영상 1개 + README에 "실보드 검증" 한 줄. 이 영상이 12개월 계획의 첫 외부 증명이다.

---

## 손절 규칙

- 8주 종료 기준을 **10주째에도** 못 채우면 d=16으로 범위를 영구 고정하고 3단계로 넘어간다.
- 12주 종료 기준을 **6개월(26주)째에도** 못 채우면 하드웨어 트랙을 보류하고 소프트웨어 트랙(파형 도구 A3)으로 전환한다. 감정이 아니라 규칙으로.

---

## 매일 루프

**아침 (폰, 10분)** — 보낼 메시지 형식:
```
CLAUDE.md 읽고 docs/PLAN.md 현재 주차 진행. 오늘은 W?-? 항목.
끝나면 docs/LOG.md에 5줄 보고. 명령 출력 첨부. 안 되면 범위 줄여서 끝내.
```
**저녁 (4시간)**:
1. (60분) LOG.md 읽기 → CLAUDE.md 5절 체크리스트 6개 통과 확인. 미통과면 되돌리기 지시.
2. (90분) 직접 작업: 파형 열기, 골든 모델 한 줄 손으로 계산해서 맞는지 확인, 스펙 문서 검토.
3. (60분) 학습: 오늘 생긴 코드 중 이해 안 되는 부분 하나를 골라 Claude Code에 "왜"를 묻고 답을 자기 말로 LOG에 한 줄 적기.
4. (10분) 내일 아침 메시지 미리 작성.

---

## 첫 세션 킥오프 프롬프트 (그대로 붙여넣기)

```
CLAUDE.md, docs/DESIGN.md, docs/PLAN.md를 읽어라.

그다음 레포를 직접 뒤져서 다음을 검증하고 docs/AUDIT.md로 정리해라. 추측 금지, 각 항목에 실행한 명령과 출력 첨부:
1. rtl/*.sv 중 `real`, $itor, $rtoi, $exp, $sqrt, #delay를 실제 코드(주석 제외)에서 쓰는 파일 목록. CLAUDE.md 2절의 7개와 다르면 차이를 적어라.
2. yosys가 설치돼 있는지, 없으면 설치 가능한지. 설치 후 rtl/mac_array.sv + mac_pe.sv를 synth -top mac_array로 합성한 셀 수 리포트.
3. tb/ 의 각 테스트가 기대값을 어디서 가져오는지 표: 골든 모델 / RTL 내부 함수 / 하드코딩. CLAUDE.md 규칙 2 위반 테스트 목록.
4. done_pulse 관련 코드 위치와 현재 테스트가 그것을 어떻게(또는 안) 검사하는지.
5. README의 문장 중 현재 코드로 뒷받침되지 않는 주장 목록.

AUDIT.md를 쓴 뒤에는 PLAN W1-1 (behavioral 격리)을 시작하되, 첫 커밋 전에 멈추고 이동 대상 파일과 영향받는 테스트 목록을 LOG.md에 보고해라. 오늘은 거기까지.
```
