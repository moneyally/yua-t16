# LOG.md — 작업 기록

---

## 2026-09-11 (세션 11, 자율 위임) — W7~W11: DELTA_STEP 완성 · 호스트 E2E · 보드 前 준비

사용자가 자율 진행을 위임했다 (보드 구매 단계 전까지, 허락 없이 결정·진행).

- **한 것**: **W7** `DELTA_STEP` 전체 경로 — `err_unit.sv`(골든 `compute_err` 비트 일치) + `dr1_scratch.sv`(q/k/v/o 온칩 스크래치, 호스트 MMIO 창 `0x8033_1000`) 신설, `dr1_top` 에 LOAD→MV_K→ERR→UPD×d→MV_Q→WR_O FSM. **W8** 1,000토큰 × 시드 3개 비트 일치, `SAT_EVENT`/`CLAMP_EVENT` 트레이스 연동. **W9·W10** `OrbitDevice.delta_init/step/dump` + 스크래치 접근 → 호스트 스택으로 RTL E2E (`tb_dr1_host_e2e.py`). **W11(보드 前)** `axil_reg_bridge.sv`(AXI4-Lite, tb 7/7) + `dr1_soc_top.sv` + `fpga/kv260/` + `docs/FPGA.md`. **스텁 정리**: `pcie_ep_versal` 의 BAR 출력을 정의된 값으로 구동해 `check -assert` 171건 → 0건, **게이트 known-incomplete 목록이 비었다**. **뮤테이션 테스트** `scripts/mutation_test.py` 신설 — 13/13 killed.
- **안 된 것**: `DELTA_STEP` 이 **205사이클**이다 (계약 상한 64의 3.2배). 내역과 줄일 방법을 DESIGN 6.2 에 적고 **상한은 안 고쳤다**. **d=64 불가** — 스크래치 1024원소에 64² 덤프가 안 들어간다 (4608 필요, spec 3.6절에 한계로 명시). Vivado 는 이 컨테이너에 없어서 **자원·타이밍 전부 미측정** (docs/FPGA.md 에 "측정값 없음"으로 비워 뒀다). CQ→BAR TLP 디코드는 PG347 확인 전이라 구현 안 함 — **PCIe 는 여전히 동작하지 않는다**. `backward_engine`/`g3_int_top`/`gemm_int4_synth`/`mxu_bf16_16x16` 시간 초과 여전.
- **검증 명령**: `bash scripts/synth_gate.sh --check-mem dr1_scratch` → **STAGE1 43/43 (known 0)** · `dr1_scratch $mem_v2 x1` · `bash scripts/run_dr1_tb.sh` → **13/13 ok (72 테스트, d=64 포함)** · `python3 -m pytest tests/ -q` → **322 passed, 8 xfailed** (실패 0) · `python3 scripts/mutation_test.py` → **13/13 killed** · `python3 scripts/check_banned_tokens.py rtl/*.sv rtl/*.v rtl/dr1/*.sv` → clean · verilator 경고 신규 0건.
- **다음 세션 첫 작업**: (a) 사이클 최적화 — 스크래치 포트 128비트화(LOAD 50→8) + `update_unit` 행 파이프라인(96→~34) 로 205 → 약 85 목표, **먼저 골든 비트 일치를 깨지 않는지부터** 확인. (b) d=64 로 가려면 스크래치를 4608원소로 키우고 `dr1_scratch_layout(64)` 를 열어야 한다. (c) `docs/RESEARCH.md` Q-F 실험은 **승인 대기** 상태 그대로 둠.
- **결정 필요**: (1) 205사이클을 최적화할지, 아니면 d=16 을 이대로 굳히고 보드로 갈지. (2) **BUG-009 가 중요한 교훈을 줬다** — "1토큰 비트 일치"는 초기 상태가 0 이라 상태 경로 버그를 숨긴다. 앞으로 어떤 경로든 **최소 10토큰**을 통과 기준으로 삼자는 제안. (3) 보드 구매는 손대지 않았다 (위임 범위 밖). PLAN W11 결정 게이트는 정원 몫.

### 사이클 최적화 (같은 세션, 커밋 분리) — 205 → 126

계약 초과를 기록만 하고 넘기지 않고, **비트 일치를 유지한 채** 줄였다.
두 단계 다 1,000토큰 × 시드 3개 비트 일치를 먼저 확인하고 진행했다.

| 고친 것 | 전 | 후 | 방법 |
|---|---|---|---|
| 행 갱신 | 96 | 49 | `mac_pe` 에 **DUAL 파라미터**(한 사이클에 곱 2개) → `update_unit` 4→2사이클. `dr1_top` 이 **다음 행을 미리 읽어** 행당 6→3 |
| 피연산자 적재 | 50 | 18 | q,k,v 를 차례로 다 읽던 것을, **k 만 기다리고 v·q 는 계산과 겹쳐서** 적재 (스크래치 포트가 MV_K/UPD 동안 놀고 있었다) |
| **합계** | **205** | **126** | 계약 상한 64 의 3.2배 → **2.0배** |

`mac_pe` 는 `mac_array` 도 쓰는 모듈이라 회귀가 가장 무서웠다. `DUAL=0`(기본)이면
두 번째 곱셈기를 **generate 로 아예 안 만들도록** 했고, **셀 수로 확인**했다
(`mac_array` 196,352 / `mac_pe` 730 — 변경 전후 동일). GEMM E2E 도 5/5 통과.

남은 거리와 성격은 DESIGN 6.3 에 적었다: `matvec` 40사이클은 1R1W SRAM 의
**구조적 하한**이고, 나머지(갱신 3→2, 스크래치 128비트)를 다 해도 약 80 이다.
계약 64 는 **피연산자 이동을 세지 않은 숫자**라, 그때 계약 쪽을 고칠지 정한다.

### d=64 확장 확인 (PLAN W8 "d=64 확장은 W8 완료 후에만")

**같은 RTL 을 파라미터만 바꿔** 돌렸다. 파일을 복제하지 않았다 — 복제하면
한쪽만 고치는 날이 온다. `tb/run_tb.py` 에 `PARAM_*` 환경변수 지원을 넣었다.

```
PARAM_D=64 PARAM_SCRATCH=8192 python3 tb/run_tb.py dr1_tb_wrap tb_dr1_d64 <소스>
  X1 d=64 INIT→DUMP 골든 일치 / X2 1토큰 / X3 20토큰 + 포화 수 / X4 사이클 실측
  TESTS=4 PASS=4
```

| | d=16 | d=64 |
|---|---|---|
| `DELTA_STEP` 사이클 | 126 | **462** (3.7배) |
| 계약 상한 4·d | 64 | 256 |
| 초과 배수 | 2.0× | **1.8×** ← d 가 커질수록 가까워진다 |

구조가 d 에 대해 선형이라는 뜻이다 (고정 오버헤드가 희석된다).
**통합 빌드는 d=16 그대로** 뒀다 — d=64 는 면적이 약 4배라 KV260 에 들어갈지
Vivado 없이는 모른다.

### 게이트에 **셀 수 회귀 검사**를 넣었다 (BUG-011 의 결과)

`mac_pe` 를 고치면서 논리적으로 같은 식으로 바꿨는데 `mac_array` 가
**196,352 → 287,488 (+46%)** 로 커졌다. cocotb 12/12, GEMM E2E 5/5, pytest 전부
통과했다. **기능 테스트는 면적 회귀를 못 잡는다.** 눈으로 셀 수를 비교하다 걸렸다.

운에 맡길 일이 아니라서 `scripts/synth_gate.sh` 에 3단계를 추가했다:
`scripts/cell_baseline.txt` 대비 **+10% 초과면 FAIL**, 감소/신규는 WARN.
의도한 증가면 기준선을 갱신하고 **이유를 커밋 메시지에 적는다**.

`DUAL=0` 경로를 글자 그대로 예전 코드로 되돌려 `mac_array` 는 196,352 로 복귀했다.

### 이번 세션에 찾아서 고친 버그

- **BUG-009**: `dr1_top` 이 갱신에 **이전 행**의 상태를 썼다. 1토큰은 통과하고 10토큰에서 잡혔다 (상태가 0 이면 안 보인다).
- **BUG-010**: 포트 연결식의 `$signed()` 가 sv2v→yosys 에서 내부 assert 로 죽인다. 린트·시뮬레이션은 둘 다 통과했다 — **게이트가 정답이다**.
- 테스트벤치 쪽 실수 3건(명령 씹힘 / 이중 실행 / AXI VALID 안 내림)도 `docs/BUGS.md` 에 표로 남겼다. 셋 다 원인이 같다: **등록된 신호를 rising edge 직후에 읽었다.**
- **뮤테이션 테스트가 골든 테스트의 구멍을 찾았다**: α/β 정의역(>0x8000) 검사가 `tests/test_golden_deltarule.py` 에 아예 없었다. 3개 테스트를 추가해 메웠다.

---

## 2026-09-11 (세션 10, 자율모드) — 참고문헌 출처 · 기준선 사실 보강 · W6

- **한 것**: [0] 참고문헌 출처를 아래에 기록(SANE 은 전용 검색을 한 적이 없었다는 것까지) · [1] 기준선 2603.05931 사실 보강 — **토큰당 읽기 1 + 쓰기 1**, 5단계 파이프라인, GVA, FP32는 [추정] (DESIGN 0.8, PRIOR_ART 0.2) · [2] **W6**: `update_unit.sv`(골든 `update_row` 와 300세트 비트 일치, 행당 **4사이클**), `dr1_top.sv` FSM 골격(INIT/DUMP 만, STEP→`0x07 DR1_UNIMPL`), `mac_pe` 폭 파라미터화 재사용(DESIGN 8.1), `desc_fsm_v2` 에 **`core_err` 포트 신설** + opcode 0x50/0x52, `reg_top` DR1 레지스터 4개, `CocotbDut.init/dump` 구현, **하네스 I1 을 실 RTL 로 통과**, RTL 경로 오류 주입, BUG-001 회귀를 DR1 fault 로 확장 · [3] VCK190 문서 3건 xfail · [4] `docs/RESEARCH.md` 신설(계획만).
- **안 된 것**: `DELTA_STEP` 계산 경로는 **W7** (이번 주 금지 항목이라 손대지 않음). **사이클 예산 문제를 발견했다** — STEP 추정 ≈104 사이클 vs 계약 상한 4·d=64 (DESIGN 6.1 에 적었다. 상한을 고쳐 쓰지 않았다). DUMP 는 아직 **스트림 포트**일 뿐 메모리에 안 쓴다. 덤프 **내용**은 0 상태만 검증했다(쓰기 경로가 W7이라 비영 상태를 만들 수단이 없다). STAGE 1 에서 `backward_engine` elaborate 가 처음으로 시간 초과(>120s) — 설계가 커져서지 회귀는 아니다. arxiv 본문은 여전히 못 읽음.
- **검증 명령**: `bash scripts/synth_gate.sh --check-mem state_sram` → **EXIT=0** (STAGE1 35/38, STAGE2 34/38) · **`mac_pe` 730 / `mac_array` 196,352 — 파라미터화 전후 셀 수 동일** · `dr1_top` 8,808, `update_unit` 42,198 · `bash scripts/run_dr1_tb.sh` → **8/8 ok (37 테스트)** · g2_ctrl_top 기존 tb 11개 → **35/35 PASS** · `python3 -m pytest tests/ -q` → **308 passed, 8 xfailed** (실패 0) · verilator 경고 수 HEAD 와 동일(신규 0).
- **다음 세션 첫 작업**: W7 — `dr1_top` 에 `LOAD_QKV`/`MATVEC_K`/`ERR`/`UPDATE`/`MATVEC_Q`/`WRITE_O` 를 붙여 `DELTA_STEP` 완성. 1토큰 비트 일치 → 10 → 100. 동시에 **행 파이프라인으로 사이클을 64에 맞출 수 있는지 실측**. `desc_fsm_v2` 유효 opcode 에 0x51 추가는 그 다음.
- **결정 필요**: (1) `docs/RESEARCH.md` 4절 — **Q-C(세션 상주)와 Q-D(헤드 상태 압축)는 닫자**는 제안. 검색에서 UNISON 2609.09643(하드웨어 co-design!), 2602.04852 등이 이미 답하고 있다. (2) 남은 Q-E/Q-F 중 **Q-F(하드클립 vs tanh, SAT_EVENT 트리거)를 1주 실험으로** 제안 — 실행은 승인 후. (3) STEP 사이클 예산 104 vs 64 — 파이프라인으로 줄일지, 상한을 다시 정할지. (4) 하네스에 `run_harness_async` 를 추가했다. "CocotbDut 한 클래스만 채운다"는 W4-2 주장은 **반만 맞았다** — 비교 로직은 한 곳(`_harness_core`)으로 유지했지만 **await 때문에 호출 껍데기 하나가 더 필요했다.** 정직하게 적는다.

### 참고문헌 출처 (규칙 7 — 어디서 얻었는가)

**2608.15533 DeltaLog** — 두 경로로 얻었다.
1. 세션 9 에서 `WebSearch` 질의 **"arXiv 2608.15533 DeltaLog Deferred Materialization of Recurrent States"** 를
   직접 실행했고, 결과 링크 목록 첫 줄이 이것이었다:
   `{"title":"[2608.15533] DeltaLog: Deferred Materialization of Recurrent States for Linear Attention Decoding",`
   `"url":"https://arxiv.org/abs/2608.15533"}`.
   같은 결과의 요약에 저자 **Junqing Lin, Jingwei Sun, Guangzhong Sun**, 제출일 **2026-08-16**,
   "represents the recurrent state as a dense base state together with a bounded log of recent compact updates",
   "periodic merge steps fold the accumulated updates back into the dense base state" 가 있었다.
2. 정원이 **arXiv 페이지를 직접 열어 확인**(2026-09-11): cs.DC, 2026-08-16, Lin/Sun/Sun,
   초록에 커널 **1.86×**, 상태 쓰기 트래픽 **7.83×**, end-to-end **1.05~1.20×**, GDN/KDA/RWKV6.
   → **GPU 서빙 스택용 소프트웨어**이지 하드웨어 가속기가 아니라는 점을 `PRIOR_ART.md` 3.1 에 명시했다.

**2608.22354 SANE** — 출처가 더 약했다. **전용 검색을 한 적이 없다.**
위 DeltaLog 질의의 결과 링크 목록에 섞여 나온 한 줄이 전부였다:
`{"title":"SANE: State Anomaly Neutralization for Stable Extreme-Context Delta-Rule Models",`
`"url":"https://arxiv.org/html/2608.22354"}`.
즉 세션 9 시점에 내가 가진 것은 **제목과 URL 뿐**이었고, 저자·날짜·초록은 몰랐다.
그래서 PRIOR_ART 초판에 "제목·초록 수준만 확인" 이라고 적었다.
2026-09-11 정원이 **직접 확인**: cs.LG, **2026-08-23 (v2 08-25)**, **Lin/Xu/Liu/Hao/Cai**,
초록 요지 — RWKV-7 1억 토큰에서 **국소 채널 노름 폭발**(전체 포화 아님), **청크 경계 tanh 적응 압축
(3 ≤ α ≤ 5)**, **α ≥ 8 은 안정하나 추론 상실**. → 실재 확인, "미검증" 표시 해소.

**교훈**: 검색 결과 *목록에 섞여 나온 제목*은 "검색으로 확인" 이 아니다.
`PRIOR_ART.md` 0절에 출처 등급 **[검색] / [사용자] / [추정]** 을 도입했다.
논문 **본문은 여전히 한 편도 못 읽었다** — arxiv.org 가 이 컨테이너에서 egress 프록시에 막힌다.

---

## 2026-09-11 (세션 9, 자율모드) — W5 RTL 4개 · 신규성 주장 철회

- **한 것**: 골든을 RTL 단위로 분해(`requantize_q15`/`matvec`/`update_row`/`compute_err`, `step()` 은 조합만) → `rtl/dr1/` 4개 모듈(`requant_q15` 172셀, `state_sram` 8,516셀 `$mem_v2 x1`, `vec_regs` 3,202셀, `matvec_unit` 34,968셀) + tb 4개 전부 골든 비트 일치, `tools/orbit_pack.py`(평탄화 규칙 SSOT), 게이트에 `--check-mem`, DESIGN 6.1 실측표(`matvec_unit` = **19사이클 = D+3**, 계약 상한 20). 그리고 **DESIGN 1절 신규성 주장 철회** — 선행 연구 arXiv 2603.05931 인용 + `docs/PRIOR_ART.md` 신설(5편).
- **안 된 것**: 논문 **본문을 못 읽었다** (arxiv.org 가 egress 프록시 차단) — 기준선의 수치 형식·패스 수는 "미확인" 으로 남겼다. STAGE 2 시간 초과 4개(`backward_engine`, `g3_int_top`, `gemm_int4_synth`, `mxu_bf16_16x16`) 여전히 측정 못 함. pytest 3건 실패(VCK190 문서, 사용자 과제). `update_unit`/`dr1_top` 은 W6.
- **검증 명령**: `bash scripts/synth_gate.sh --check-mem state_sram` → **EXIT=0** (STAGE1 34/36, STAGE2 32/36) · `bash scripts/run_dr1_tb.sh` → 4/4 ok (TESTS=16 PASS=16 FAIL=0) · `python3 -m pytest tests/ -q` → **3 failed, 308 passed, 5 xfailed** · `python3 sim/golden/deltarule.py` → 전부 통과 (S 0.634/0.661 LSB, o 고립 0.499/0.500) · `grep -nE "세상에 없는|최초|아무도" README.md CLAUDE.md` → **0건**.
- **다음 세션 첫 작업**: W6 — `rtl/dr1/update_unit.sv`(행 단위 스트리밍, `mac_pe` D개, 골든 `update_row()` 와 비트 일치) → `dr1_top.sv` FSM 골격 INIT/DUMP 만(STEP 은 `DONE_ERR` + `fault_code=UNIMPL`) → `CocotbDut.init/dump` 구현(`done_ok` 폴링, `done_pulse` 사용 금지).
- **결정 필요**: (1) `docs/PRIOR_ART.md` 3절 경고 — **2608.15533 DeltaLog 가 Q-B(게으른 감쇠)를, 2608.22354 SANE 이 Q-A(장기 상태 안정화)를 이미 다룬다.** 주말 연구 세션의 4개 질문 중 2개가 이미 절반 답이 나와 있다는 뜻이다. 그래도 원래 4개로 갈지, 질문을 다시 뽑을지. (2) 기준선 논문 본문 수치가 필요하면 정원님이 PDF 를 받아 붙여줘야 한다 (프록시 차단).

---

## 2026-09-11 (세션 8, 자율모드) — W4-2 하네스 골격 · 1단계(3~4주) 종료

**한 것**: **W4-2 `tb/tb_dr1_top.py` 완성** — DUT 추상 클래스(`init`/`step`/`dump`), `GoldenDut`, `CocotbDut`(자리만+안내), `OffByOneDut`(오류 주입), 하네스, 불일치 출력 형식 확정. `tests/test_dr1_harness.py` **27개 전부 통과** — 골든-대-골든 **N=1000, d=16·64, 시드 3개** 포함. `pytest.ini` 로 `slow` 마커 등록 (빠른 확인용).
**안 된 것**: BUG-008a-3 세그폴트·008c~f 그대로. VCK190 문서 3건 그대로. 하네스에 **스크래치 주소 개념이 없다** — `CocotbDut` 이 그걸 직접 다뤄야 한다 (아래 자기 점검).
**실행한 검증 명령**: `python3 -m pytest tests/ -q` → **3 failed, 303 passed, 5 xfailed** (276 → **+27**), 43s / `-m "not slow"` → 5s / `python3 tb/tb_dr1_top.py` → 골든-대-골든 18/18 PASS, 오류 주입 검출 확인. **RTL 무수정**.
**다음 세션 첫 작업**: 2단계 W5 — `rtl/dr1/state_sram.sv`, `vec_regs.sv`, `matvec_unit.sv`. **여기서부터 RTL 금지가 풀린다.**
**사용자 결정 필요**: 1건 — 현재 주차를 5로 올릴지 (1단계 종료 기준 3/3 충족).

---

### 오류 주입 — 하네스가 살아 있다

골든-대-골든은 **항상 통과한다.** 그래서 그것만으로는 하네스가 비교를 하는지,
아무것도 안 하고 통과하는지 구분할 수 없다. `OffByOneDut` 으로 1 LSB 를 고의로 넣었다:

```
DR1 harness  d=16  N=50  seed=7  → FAIL (2 mismatches)
  [MISMATCH] token=   13  o[5]           expected=0xFFF6 (    -10)  actual=0xFFF7 (     -9)  diff=+1
  [MISMATCH] token=  end  S[2][9]        expected=0xFEE3 (   -285)  actual=0xFEE4 (   -284)  diff=+1
```

**토큰·행·열·기대·실제·차이를 정확히 찍는다.** 테스트는 위치가 주입한 곳과 같은지까지 확인하고,
−1 LSB(부호 반대)도 잡는지 별도로 본다. 출력 형식 자체도 테스트로 고정했다 (DESIGN.md 7절 3항).

### 완료 기준 자기 점검 — "RTL 이 오면 CocotbDut 한 클래스만 채운다"

**기계적으로 확인했다.** DUT 를 **엄격한 프록시**로 감싸서 `init`/`step`/`dump`/`d` 외의
속성 접근에 `AttributeError` 를 내게 하고, 그 프록시로 하네스를 끝까지 돌렸다. 통과했다.

처음에는 `__getattribute__` 로 접근을 세는 방식이었는데 **그건 약했다** — DUT 자신의
메서드가 내부적으로 `self.foo` 를 건드리는 것까지 같이 세어서, 하네스가 쓴 것과
구분되지 않았다. 프록시가 그 구분을 강제한다.

**그래서 답은 "참이지만 조건이 있다"** 다:

| | 상태 |
|---|---|
| 하네스가 DUT 에 대해 쓰는 것 | `init` / `step` / `dump` / `d` **넷뿐** (프록시로 증명) |
| 불일치 비교·보고·시퀀스 생성 | 전부 하네스 쪽. DUT 교체와 무관 |
| 완료 신호 계약 | `Dr1Fault` 예외 하나로 표현 — 시그니처 안 바뀜 |
| **하네스에 없는 것** | **스크래치 주소 개념.** `step(slot, q, k, v, α, β)` 는 벡터를 직접 받는다. 실제 RTL 은 q/k/v 를 스크래치 SRAM 에서 읽으므로, `CocotbDut.step` 이 **MMIO 로 스크래치를 채우고 주소를 디스크립터에 넣는 일**을 해야 한다 |

즉 **"한 클래스만 채운다"는 참**이다 — 하네스·테스트·비교 로직은 손댈 필요가 없다.
다만 그 한 클래스가 하는 일이 작지는 않다 (디스크립터 패킹 + 스크래치 적재 + 완료 폴링).
`CocotbDut` docstring 에 그 4단계를 순서대로 적어뒀고, 테스트가 그 안내 문구의 존재를 검사한다.

### `CocotbDut` 자리에 남긴 것 — cdc_fifo 에서 배운 것

docstring 에 박아뒀다 (`test_cocotb_dut_is_a_stub_with_guidance` 가 존재를 검사):

1. **레지스터 출력은 falling edge 에서 샘플링한다.** `RisingEdge` 직후에 읽으면
   논블로킹 대입 전이라 이전 값을 본다 (BUG-008a/b — RTL 은 정상인데 테스트가 틀렸다).
2. **핸드셰이크 결정 신호도 레지스터면 한 사이클 어긋난다.** edge N 의 핸드셰이크는
   *N 직전* 값으로 정해지는데 edge 직후 보이는 값은 *N+1* 용이다 → 마지막 1개를 놓친다.
   참고 구현: `tb/tb_cdc_fifo_async.py` 의 `read_n()`.
3. **`done_pulse` 는 리타이어(성공+실패)다.** 완료 판정에 쓰면 fault 를 성공으로 읽는다 —
   `done_ok` 를 봐야 한다 (BUG-001).
4. 스크래치 주소는 **16바이트 정렬** (spec/deltarule.md 4절, `act_sram` 데이터 폭 128비트).

### 테스트 27개 구성

| 묶음 | 내용 |
|---|---|
| 골든-대-골든 | d=16 N=1000 ×3시드, d=64 N=1000 ×3시드(`slow`), 같은 시드 = 같은 시퀀스 |
| 오류 주입 | o 위치 정확, S 셀 정확, −1 LSB, 출력 형식 고정 |
| I1~I6 (하네스 수준) | I1 init→0, I2 α=1·β=0 불변, I3 β=0 → o=S·q, I4 포화 수 일치(강제 유발), I5 결정성, **I6 done_ok/done_err 하나만 + 실패 시 상태 불변** |
| 스펙 경계 | α/β > 1.0 은 fault 가 아니라 **클램프**(DUT) / 골든은 `ValueError` |
| 구조 | `CocotbDut` 스텁·안내 문구, 엄격 프록시 인터페이스 검사, dump 모양 오류 |

실행 시간: 전체 **43s**, `-m "not slow"` **5s**. `pytest.ini` 에 마커를 등록했고
**기본 실행은 전부 돌린다** — W4-2 완료 기준이 d=64 N=1000 이므로 기본에서 빼지 않았다.

---

## 1단계 (3~4주) 종료 기준 — **3/3 충족**

`docs/PLAN.md` "4주 종료 기준" 대조:

| 기준 | 상태 | 근거 |
|---|---|---|
| `pytest tests/test_golden_deltarule.py` 통과 | ✅ | **16 passed** |
| spec 문서와 mmio_map 이 일치 | ✅ | `tests/test_dr1_spec_consistency.py` **13 passed** — spec 마크다운 표를 파싱해서 대조. 필드가 64바이트를 빈틈없이 덮는지까지 |
| 하네스가 골든-대-골든으로 통과 | ✅ | **N=1000, d=16·64, 시드 3개** 전부. 오류 주입으로 하네스가 살아 있음도 확인 |

W3-1 ✅ W3-2 ✅ W4-1 ✅ W4-2 ✅ — **1단계 전 항목 완료. 이 2주 동안 RTL 파일 0개 수정.**

### 누적 현황

| 단계 | 상태 |
|---|---|
| 0단계 (1~2주) 청소와 정직화 | ✅ 완료 (세션 4) |
| **1단계 (3~4주) 골든 모델과 계약** | ✅ **완료 (세션 8)** |
| 2단계 (5~8주) RTL — d=16 먼저 | ⬜ 다음. **W5 부터 RTL 금지 해제** |
| 3단계 (9~10주) 호스트 연결 | ⬜ |
| 4단계 (11~12주) 실물 준비 | ⬜ |

pytest 누적: 247 → 263 → 276 → **303 passed** (실패 3건은 VCK190 문서 미커밋, 변화 없음)

---

## 2026-09-11 (세션 7, 자율모드) — UQ1.15 도입 · W4-1 스펙과 SSOT

**한 것**: α/β 를 **부호 없는 UQ1.15** 로 바꿨다 (`0x8000` = 1.0 **정확**). DESIGN.md **0.4**, 골든 모델, 손계산 테스트 전부 갱신 — **손계산 오차 0 확인**. **W4-1 `spec/deltarule.md` 작성** (opcode 3개, 64B 레이아웃, 레지스터 4개, 실패 조건). `tools/orbit_mmio_map.py`·`orbit_desc.py` 갱신 + **spec 문서를 파싱해 상수와 대조하는 테스트 13개** 신규.
**안 된 것**: BUG-008a-3 세그폴트·008c~f 그대로. VCK190 문서 3건 그대로.
**실행한 검증 명령**: `python3 -m pytest tests/ -q` → **3 failed, 276 passed, 5 xfailed** (263 → +13) / `python3 -m pytest tests/test_dr1_spec_consistency.py -q` → **13 passed** / `python3 -m pytest tests/test_golden_deltarule.py -q` → **16 passed** / `python3 sim/golden/deltarule.py` → 전부 통과. **RTL 무수정**.
**다음 세션 첫 작업**: W4-2 `tb/tb_dr1_top.py` 골격 (골든을 DUT 자리에 두고 하네스 자체 검증).
**사용자 결정 필요**: 없음.

---

### UQ1.15 — 손계산 오차 0

`0x8000` 을 특별 해석하는 대신 **형식 자체를 부호 없는 UQ1.15 로** 바꾼 결정이 맞았다.
같은 16비트인데 1.0 이 그냥 들어간다.

| 무엇 | 형식 | 1.0 |
|---|---|---|
| `S`, `q`, `k`, `v`, `p`, `err`, `o` | 부호 있는 Q1.15 | **없음** (최대 0.999969) |
| `α`, `β` | **부호 없는 UQ1.15** | **`0x8000` = 1.0 정확** |

1.0 초과는 미정의 — 골든은 `ValueError`, RTL 은 `0x8000` 클램프 + `CLAMP_EVENT` + `DR1_CLAMP_COUNT`.
골든이 예외를 내는 이유: 골든은 검증 기준이라 미정의 입력을 조용히 처리하면 안 된다.

손계산 대조 결과가 바뀌었다:

```
                          S1오차  o1오차  S2오차
α/β 근사(Q15_MAX) 였을 때    2      3      1     ← 이전 세션
α/β UQ1.15 로 바꾼 뒤        1      2      0     ← 지금
k=q 도 1.0 로 두면           0      0      0     ← 수식·양자화 순서 검증
```

**남은 오차는 전부 k, q 탓이다** — 이들은 상태와 같은 부호 있는 Q1.15 라 1.0 이 없다.
실제 입력에서 k 가 정확히 1.0 일 일은 없으므로 문제가 되지 않는다.
허용치는 여전히 임의값이 아니라 **연쇄의 ≈1.0 곱셈 횟수**로 정했다 (S1 1회, o1 2회).

### W4-1 — `spec/deltarule.md`

**opcode**: 기존 `0x01~0x04`(G2), `0x10/0x20/0x30/0x40`(G3)과 충돌하지 않게 `0x5x` 블록.

| 값 | 이름 |
|---|---|
| 0x50 | `DELTA_INIT` |
| 0x51 | `DELTA_STEP` |
| 0x52 | `DELTA_DUMP` |

**디스크립터 64B — 기존 필드 위치를 그대로 재사용했다.** `rtl/desc_fsm_v2.sv` 의 추출 로직을
안 건드려도 된다는 뜻이다.

| off | size | 필드 | 기존 필드 |
|---|---|---|---|
| 0 | 1 | `opcode` | `opcode` |
| 1 | 1 | `slot` | — |
| 2 | 2 | `reserved0` | — |
| 4 | 4 | `reserved1` | — |
| 8 | 8 | `reserved2` | — |
| **16** | 8 | `q_addr` | `act_addr` |
| **24** | 8 | `k_addr` | `wgt_addr` |
| **32** | 8 | `o_addr` | `out_addr` |
| 40 | 4 | `reserved3` | `Kt` (DR1 미사용) |
| **44** | 8 | `v_addr` | — |
| **52** | 2 | `alpha_uq15` | — |
| **54** | 2 | `beta_uq15` | — |
| 56 | 7 | `reserved4` | — |
| 63 | 1 | `crc8` | `crc8` (동일) |

타임아웃은 디스크립터 필드가 아니다 — `desc_fsm_v2` 의 `timeout_cycles` **포트**로 들어온다
(레지스터 설정값). 기존과 동일하다.

**레지스터**: 새 블록 `DR1 = 0x8033_0000` (OOM 0x8032 와 TC0 0x8034 사이 빈 공간).

| 주소 | 이름 | 접근 |
|---|---|---|
| 0x8033_0000 | `DR1_STATUS` (`[0] busy`, `[11:4] last_slot`) | RO |
| 0x8033_0004 | `DR1_SAT_COUNT` | W1C |
| 0x8033_0008 | `DR1_CLAMP_COUNT` | W1C |
| 0x8033_000C | `DR1_CYCLES` (마지막 STEP 사이클 수) | RO |

W1C 는 기존과 동일 — 읽은 값을 그대로 쓰면 0 이 되고, 증가와 겹치면 **증가가 이긴다**(set-wins).

**실패 조건 → `done_err`** (DESIGN 5.1 계약 그대로 참조):

| fault_code | 조건 |
|---|---|
| 0x05 `DR1_BAD_SLOT` | `slot >= DR1_NUM_SLOTS` (v1 은 slot≠0) |
| 0x06 `DR1_UNALIGNED` | q/k/v/o 주소가 **16바이트 정렬** 아님 (`act_sram` 데이터 폭 128비트) |

α/β 가 `0x8000` 초과인 것은 **fault 가 아니다** — 클램프하고 센다.

### SSOT 방향 — 문서를 파싱해서 코드를 검사한다

`tests/test_dr1_spec_consistency.py` 13개는 **`spec/deltarule.md` 의 마크다운 표를 직접 파싱**해
`tools/orbit_mmio_map.py` 상수와 대조한다. 방향을 명시했다:

```
docs/DESIGN.md → spec/deltarule.md → tools/orbit_mmio_map.py → orbit_desc.py / RTL / 테스트
                      ↑ 이게 옳다
```

**상수에서 문서를 생성하지 않은 이유**: 그러면 문서가 상수의 그림자가 되고
"문서가 SSOT" 라는 말이 거짓이 된다. 사람이 읽고 고치는 쪽이 문서이므로,
문서를 읽어서 코드를 검사하는 방향이 맞다.

검사 항목: opcode 값·중복, fault code, 필드 오프셋·폭, **필드가 겹치거나 64바이트를 넘지 않고
빈틈 없이 덮는지**, 기존 주소 슬롯 재사용, 레지스터 주소·접근·리셋, 블록 주소 충돌,
트레이스 이벤트 값, `UQ15_ONE` 이 골든과 일치, 패커가 스펙 오프셋에 넣는지,
패커·골든이 스펙 위반을 거부하는지.

---

## 2026-09-11 (세션 6, 자율모드) — DESIGN 0.3 · W3-2 불변조건 테스트 · d=2 손계산 대조

**한 것**: 결정 2건 반영 (DESIGN.md 2절을 `err = v − α·p` 한 줄로 정리하고 "주의" 삭제, 3절에 Q2.30 1회 재양자화 명시, **버전 0.3**; PLAN W3-1 문구를 고립/전체 오차 분리로 조정). **W3-2 `tests/test_golden_deltarule.py` 완성 — I1~I5 전부, d=16·64, 16개 테스트 전부 통과.** 지시하신 **d=2 손계산 대조 테스트** 추가 — 손계산과 **완전 일치**.
**안 된 것**: BUG-008a-3 세그폴트·008c~f 그대로. VCK190 문서 3건 그대로 (정원님 로컬 대기).
**실행한 검증 명령**: `python3 -m pytest tests/test_golden_deltarule.py -q` → **16 passed** / `python3 -m pytest tests/ -q` → **3 failed, 263 passed, 5 xfailed** (이전 247 → +16) / `python3 sim/golden/deltarule.py` → 전부 통과. **RTL 무수정** (`git status --short rtl/` 비어 있음).
**다음 세션 첫 작업**: W4-1 `spec/deltarule.md` — opcode 3개(DELTA_INIT/STEP/DUMP) 디스크립터 필드, 레지스터 추가분, 완료 신호 정의. **α/β 필드 폭을 여기서 결론내야 한다** (아래).
**사용자 결정 필요**: 1건 — Q1.15 에 1.0 이 없다.

---

### 손계산 대조 — **완전 일치**

주신 손계산을 두 방식으로 검사했다.

**(1) 1.0 이 정확할 때 (`ONE_Q15`=32768)** — `test_hand_worked_d2_exact_one`

```
1단계  S=0, k=q=[1,0], v=[3,5]·u, α=β=1
       → S = [[3,0],[5,0]]·u,  o = [3,5]·u      (u = 1/8 = 4096)
2단계  k=q=[1,0], v=[7,1]·u, α=1, β=1/2
       → S = [[5,0],[3,0]]·u
```

정수 결과가 **한 비트도 안 틀리고** 손계산과 같다. `S = [[12288,0],[20480,0]]`,
`o = [12288,20480]`, 2단계 `S = [[20480,0],[12288,0]]`. 1/8 을 되돌리면 `[[3,0],[5,0]]`,
`[3,5]`, `[[5,0],[3,0]]` 그대로다. float 참조 구현으로도 같은 답이 나오는 것을 따로 검사했다.

**(2) 실제로 표현 가능한 값만 쓸 때 (`Q15_MAX`=32767)** — `test_hand_worked_d2_representable_q15`

```
1단계 S 오차 2 LSB   [[12288,0],[20478,0]]
1단계 o 오차 3 LSB   [12288, 20477]
2단계 S 오차 1 LSB   [[20479,0],[12288,0]]
```

허용치를 임의로 정하지 않고 **연쇄에 들어간 ≈1.0 곱셈 횟수**로 정했다:
`S1` 은 β·v 와 ·k 로 2회 → ≤2, `o1` 은 거기에 ·q 가 붙어 3회 → ≤3.
**한 번 곱할 때마다 정확히 1 LSB 를 잃는다.** 이 숫자가 커지면 골든 모델이나 Q1.15 정의가 바뀐 것이다.

### 결정 필요 v4 — **Q1.15 에 1.0 이 없다**

부호 있는 Q1.15 의 최대값은 32767 = 0.999969... 다. **1.0(=32768)은 표현되지 않는다.**
그런데 DESIGN.md 2절은 α, β ∈ (0,1] 로 **1 을 포함**한다. 손계산도 α=β=k=q=1 을 쓴다.

지금 골든 모델은 파이썬 정수라 32768 을 그냥 받아서 정확히 계산한다. **RTL 은 그럴 수 없다.**
16비트 필드에 32768 이 안 들어간다. 위 (2) 의 3 LSB 오차가 그 대가다.

선택지:
- **(a) α/β 를 17비트 unsigned 로** — `0x10000`=1.0. 필드 2바이트를 넘어 디스크립터 레이아웃이 바뀐다.
- **(b) α/β 를 unsigned Q0.16 으로** (0~65535 = 0~0.99998) — 여전히 1.0 이 없다. 지금과 같은 문제.
- **(c) 16비트 필드에서 `0x8000` 을 1.0 으로 특별 해석** — 필드 폭 유지. α,β>0 이라 부호 비트가 남는다. **권고.**
- **(d) 1.0 을 쓰지 않기로 하고 상한을 0.999969 로 고정** — DESIGN.md 2절의 `(0,1]` 을 `(0,1)` 로 고친다. 가장 정직하지만 α=1(감쇠 없음)·β=1(완전 갱신)이라는 자연스러운 경계를 못 쓴다.

**k, q 는 다르다** — 이들은 상태와 같은 Q1.15 벡터라 (c) 를 쓸 수 없다. k=1.0 은 그냥 표현 불가다.
다만 실제 입력에서 k 가 정확히 1.0 일 일은 없으므로 문제가 되지 않는다. 손계산 테스트에서만 드러난다.

**W4-1 에서 디스크립터 필드(`alpha_q15`, `beta_q15`)를 정의할 때 결론이 필요하다.** 권고는 (c).

### W3-2 — `tests/test_golden_deltarule.py`

16개 테스트, 전부 docstring 첫 줄에 기대값 출처를 적었다.

| 불변조건 | 어떻게 검사했나 |
|---|---|
| I1 | 0 상태에서 시작 확인 + "S=0 이면 p=0 이므로 S_1 = β·v·kᵀ" 를 원소마다 대조 |
| I2 | α=1(정확) → **완전 동일**, α≈1(Q15_MAX) → ≤1 LSB 두 가지로 |
| I3 | β=0 일 때 o 가 `S_{t-1}·q` 와 같은지, 재양자화 순서까지 같은 기대값으로 |
| I4 | **포화 강제**: S=전부 +Q15_MAX, v=전부 Q15_MIN, k=[1,0…] → `err = −32768−32767 = −65535` 가 행마다 1회 잘려서 **정확히 d 회**. 손계산을 docstring 에 적고 결과값(`S_next[:,0]=−1`, `o=−1`)까지 대조 |
| I5 | 한 스텝·8토큰 시퀀스 양쪽 + `run()` 이 입력 상태를 망가뜨리지 않는지 |

I4 는 반대편(작은 입력 10토큰 → 포화 0회)도 같이 검사한다.

---

## 2026-09-11 (세션 5, 자율모드) — 현재 주차 3 · cdc_fifo 판정 · OOM 테스트 · W3-1 골든 모델

**한 것**: PLAN 현재 주차 → 3. **BUG-008a/b 는 RTL 이 아니라 테스트벤치 버그로 판정** (2/3 + 1/1 해결). `oom_alloc_dec` 가 done_err 경로에서 정확히 1회 감소하는 테스트가 **없었고, 추가해서 3/3 통과**. **W3-1 `sim/golden/deltarule.py` 완성** — 자체 테스트 전부 통과.
**안 된 것**: `tb_cdc_fifo_async.test_continuous_streaming` 이 verilator 에서 **SIGSEGV(rc=-11)** — 시간 박스 종료, `docs/BUGS.md` 008a-3 에 남기고 넘어감. BUG-008c~f(oom_guard 4건, trace_ring 1건, g3_desc_fsm 1건) 미착수.
**실행한 검증 명령**: `python3 sim/golden/deltarule.py` → 전부 통과 / `python3 tb/run_tb.py g2_ctrl_top tb_g2_ctrl_top_oom_fault $(ls rtl/*.sv)` → 3/3 PASS / `python3 -m pytest tests/ -q` → 3 failed, 247 passed, 5 xfailed. **RTL 파일은 하나도 건드리지 않았다** (`git status --short rtl/` 비어 있음) — 지시대로.
**다음 세션 첫 작업**: W3-2 `tests/test_golden_deltarule.py` (불변조건 I1~I5 를 골든 모델 수준에서, d=16·64).
**사용자 결정 필요**: 2건 — DESIGN.md 2절 문구 보완, `o` 오차 기준 (아래).

---

### GitHub 확인

- **PR 0건** (열림·닫힘 모두 없음)
- **CI 워크플로 0개** (`.github/` 없음)
- 브랜치: `main`, `claude/hyo-r7u3an`(+9), **`spec-v1`** — G1 시절 옛 분기로 main 에 병합된 적 없다. 정리 여부는 결정 사항이 아니라 그냥 기록만.

### BUG-008a/b — **`cdc_fifo` RTL 은 정상이었다**

프로브를 붙여 `rd_valid` 를 계속 1 로 두고 매 사이클 관측:

```
cyc | rd_ready | rd_data
  0 |        1 | 0xdead0000
  ...
  6 |        1 | 0xdead0006
  7 |        0 | 0xdead0007     <- 8개가 순서대로 전부 나온다
```

문제는 테스트벤치 샘플링이었다. 두 가지를 놓치고 있었다:

1. `rd_data` 는 **레지스터 출력**이다. `await RisingEdge` 직후에 읽으면 논블로킹 대입 전이라 **이전 값**을 본다 → 첫 읽기에서 리셋값 `0x0`.
2. `rd_ready`(=`~empty`) **도** 레지스터 출력이다. rising edge N 의 핸드셰이크는 *N 직전* `rd_ready` 로 정해지는데 edge 직후에 보이는 값은 *N+1* 용이다 → 마지막 1개를 놓친다 (8개 중 7개).

falling edge 에서 샘플링하는 `read_n()` 헬퍼로 고쳤다. `tb_cdc_fifo_reset` **1/1 PASS**, `tb_cdc_fifo_async` **2/3 PASS**.

남은 1건(`test_continuous_streaming`)은 **테스트 도중 세그폴트**다 (pass/fail 도 못 찍는다). 코루틴 정리를 넣어봤지만 효과 없었다. 같은 DUT 로 다른 두 테스트가 통과하므로 **RTL 문제라는 근거는 없다.** 시간 박스대로 여기서 멈췄다.

### BUG-001 보완 — `oom_alloc_dec` 테스트

**그런 테스트는 없었다.** `tb/tb_g2_ctrl_top_oom_fault.py` 를 새로 썼다.
`rtl/oom_guard.sv:73` 의 카운터는 **레벨 감지**라 `alloc_dec` 가 1인 사이클마다 감소한다 —
"정확히 1사이클, 정확히 1회"가 안 지켜지면 사용량이 어긋난다.

```
ILLEGAL: usage 0 -> 0, alloc_dec high cycles = 1     <- fault(done_err) 경로
NOP:     usage 0 -> 0, alloc_dec high cycles = 1     <- 성공(done_ok) 경로 (대조군)
fault #0..#3: usage = 0                              <- 4회 반복해도 누수 없음
** TESTS=3 PASS=3 FAIL=0 SKIP=0 **
```

`ST_FAULT → ST_DONE` 을 남긴 판단이 옳았다는 근거가 이제 테스트로 고정됐다.

### W3-1 — `sim/golden/deltarule.py`

`step(S, q, k, v, alpha, beta) -> (S_next, o, sat_count)`. 전부 numpy 정수 연산이다.
float 로 계산한 뒤 변환하지 않는다 (DESIGN.md 3절).

**DESIGN.md 2절 "정확한 식" 을 하드웨어 한 줄로 접었다.** 문서가 경고한 대로
`err = v − p` 는 α=1 일 때만 맞다. 정확한 전개 `S_t = α·S − α·β·p·kᵀ + β·v·kᵀ` 는

    S_t = α·S + β·(v − α·p)·kᵀ

로 **정확히** 접히므로 **`err = v − α·p`** 를 쓴다. 모든 α 에서 정확하고, 여전히
"스케일 1회 + 외적 누산 1회"라 하드웨어 구조가 바뀌지 않는다.

**연산 순서를 한 번 고쳤다.** 처음엔 `α·S` 와 `berr·kᵀ` 를 각각 Q1.15 로 내린 뒤 더했는데
재양자화가 2번 일어나 오차가 **1.06~1.08 LSB** 로 기준을 넘었다. Q2.30 누산기 안에서
더하고 **마지막에 한 번만** 내리도록 바꾸니 **0.63~0.66 LSB** 로 떨어졌다.
이 순서는 DESIGN.md 8절의 `"acc 초기값을 α·S 행으로 로드하는 경로 추가"` 와 일치한다.
**기준을 늘리지 않고 구현을 고쳤다.**

자체 테스트 (`python3 sim/golden/deltarule.py`):

```
  round-half-to-even        OK  (10 케이스)
  포화(saturate)             OK  (경계 3케이스, SAT 카운트 일치)
  불변조건 I2/I3/I5 (d=16)   OK
  불변조건 I2/I3/I5 (d=64)   OK
  float 대비 (d=16)  OK   S 0.634 / o(고립) 0.499 LSB   [기준 1 LSB]
                     참고: o(전체) 1.064 LSB — S 양자화가 16회 누산으로 전파된 것. 게이트 아님
  float 대비 (d=64)  OK   S 0.661 / o(고립) 0.500 LSB   [기준 1 LSB]
                     참고: o(전체) 1.254 LSB — S 양자화가 64회 누산으로 전파된 것. 게이트 아님
   100토큰 드리프트 (d=16)       2.1 LSB, 포화 0회  — 관측용
    20토큰 드리프트 (d=64)       2.4 LSB, 포화 0회  — 관측용
=== 전부 통과 ===
```

---

## 결정 필요 v3

**#1 — DESIGN.md 2절 문구 보완?**
2절 본문의 하드웨어 전개가 `err = v_t − p` 로 적혀 있고, 바로 아래 "주의"가 그게 α=1 에서만
맞다고 경고한다. 골든 모델은 정확한 쪽(`err = v − α·p`)을 구현했다.
**본문을 `err = v − α·p` 로 고치면 주의 문단과 본문이 한 식으로 합쳐진다.**
SSOT 라서 임의로 안 고쳤다. 승인하면 2절을 그렇게 정리한다.

**#2 — `o` 의 float 오차 기준**
PLAN W3-1 은 "1 LSB 이내"라고 했는데, `o` 를 **순수 float 경로와 끝까지** 비교하면
d=16 에서 1.06, d=64 에서 1.25 LSB 다. 이건 `o` 계산의 오차가 아니라 **S 의 양자화 오차가
d 번 누산으로 전파된 것**이고, d 가 커지면 반드시 커진다. 1 LSB 로 묶을 수 없다.

그래서 자체 테스트는 이렇게 나눴다:
- **게이트**: S 오차 ≤1 LSB, o 오차(고립: 정수 S_next 기준) ≤1 LSB → 둘 다 통과
- **관측**: o 오차(전체), N토큰 드리프트 → 숫자만 보고

근거: DESIGN.md 7절의 RTL 검증은 **골든 모델과 비트 일치**다. 둘 다 정수라 float 드리프트는
판정에 안 쓰인다. float 비교는 부호·시프트 같은 굵직한 실수를 잡는 용도다.
**이 분리에 동의하시면 PLAN W3-1 문구를 그렇게 조정하겠다.**

---

## 2026-09-11 (세션 4, 자율모드) — W1-3 · verilator 복구 · W2-1 README · BUG-001 수정

**한 것**: 4건 순서대로 전부. `docs/LINT.md`(W1-3) / verilator 5.034 소스 빌드 + `scripts/setup_tools.sh` → **기존 tb 38개 중 32개 실행 복구** / README 전면 재작성(W2-1) / **BUG-001 수정** (DESIGN.md 5.1절 완료 신호 계약 신설 후 RTL).
**안 된 것**: verilator 복구로 **기존 테스트 실패 6건(BUG-008)이 드러났다 — 미수정.** `cdc_fifo` 는 데이터가 통과하지 않는다. `cocotb-test` 는 이 환경에서 wheel 빌드 실패(선택 사항). 4개 모듈은 여전히 yosys 시간 예산 초과(미측정).
**실행한 검증 명령**: `SYNTH_TIMEOUT=420 JOBS=2 bash scripts/synth_gate.sh` → **PASS (EXIT=0)** — STAGE 1 29/31, STAGE 2 27/31 / tb 스윕 → **38개 중 PASS 32, 개별 153개 중 146 PASS** / `python3 -m pytest tests/ -q` → 3 failed, 247 passed, 5 xfailed / `verilator --version` → 5.034
**다음 세션 첫 작업**: BUG-008a/b (`cdc_fifo` 데이터 미통과) — 테스트벤치가 틀렸나 RTL 이 틀렸나부터 가른다. 0단계 종료 기준은 이제 전부 충족.
**사용자 결정 필요**: 1건 — PLAN "현재 주차"를 3으로 올릴지 (0단계 종료 기준 3/3 충족).

---

### 1. W1-3 — `docs/LINT.md`

```bash
for f in $(ls rtl/*.sv rtl/*.v | sort); do
  echo "##### $f"; verilator --lint-only -Wall -Irtl "$f"; echo "##### rc=$?"
done
```

31개 파일 / 총 **134건** / **경고 0건 파일 13개**.

| 코드 | 건수 |
|---|---|
| `UNUSEDSIGNAL` | 59 |
| `WIDTHTRUNC` | 34 |
| `WIDTHEXPAND` | 27 |
| `UNUSEDPARAM` | 8 |
| `PINCONNECTEMPTY` | 5 |
| `BLKLOOPINIT` | **1 (유일한 진짜 에러)** |

`mac_pe`, `mac_array`, `desc_fsm_v2`, `reg_top`, `irq_ctrl`, `trace_ring`, `oom_guard`, `reset_seq` —
**북극성 핵심 부품과 제어 평면 주요 모듈이 전부 경고 0건이다.**

유일한 에러는 `vpu_core_synth.sv:554` `BLKLOOPINIT` (for 루프 안 배열 논블로킹 대입).
verilator 가 지원하지 않는 구문이라 **이 모듈은 verilator 로 시뮬레이션할 수 없다.**
그런데 `sim/cocotb/test_vpu_synth.py` 가 이걸 DUT 로 쓴다. 기록만 했다 (W1-3 지시).

**지적한 대로 이건 verilator 소스 빌드를 기다릴 필요가 없었다** — `--lint-only` 는 5.020 으로 된다.

### 2. verilator 소스 빌드 + `scripts/setup_tools.sh`

```
$ verilator --version
Verilator 5.034 2025-02-24 rev v5.034
```

`scripts/setup_tools.sh` 에 전 과정을 기록했다 (`--check` 로 상태만 확인 가능).
그 과정에서 함정 두 개를 만났고 스크립트에 주석으로 남겼다:

- **numpy 가 깨져 있었다.** apt 의 `python3-numpy`(yosys → xdot → graphviz 의존으로 딸려온다)가 `/usr/lib/python3/dist-packages` 에 들어가는데 `/usr/local/bin/python3` 에서 import 되지 않는다 (`No module named 'numpy.core._multiarray_umath'`). pip 로 덮으려 해도 `RECORD file not found (installed by debian)` 로 막힌다. `--ignore-installed` 로 우회. **1단계(골든 모델) 전체가 numpy 에 의존하므로 이건 선택이 아니었다.**
- **`cocotb-test` 는 wheel 빌드가 실패한다.** 다행히 필수가 아니다 — `tb/run_tb.py` 는 cocotb 2.x 내장 `cocotb_tools.runner` 를 쓴다. 레거시 `sim/cocotb/run_*.py` 만 `cocotb_test` 를 참조하므로 그 러너들을 옮기는 것이 남은 일이다.

### 3. 기존 tb 복구 — **38개 중 32개**

`tb/run_tb.py` 를 고쳐 verilator 5.022+ 가 있으면 자동으로 쓰고(네이티브 SV, sv2v 불필요),
없으면 icarus+sv2v 로 내려가게 했다. `tb/` 와 `tb/behavioral/` 양쪽에서 모듈을 찾는다.

```
PASS      32
TESTFAIL   5
TIMEOUT    0
BUILDERR   1
개별 테스트: 153개 중 PASS 146 / FAIL 7
```

전체: `build/tb/sweep.txt` · 모듈별 로그: `build/tb/sweep_<module>.log`

**이전 상태: 0개 실행 가능** (`tb/` 에 Makefile 없음 + verilator 비호환).

실패 6건은 `docs/BUGS.md` **BUG-008** 에 각각의 assertion 출력과 함께 기록했다. **미수정.**
가장 심각한 것:

| 테스트벤치 | 증상 |
|---|---|
| `tb_cdc_fifo_async` | `Mismatch at 0: wrote 0xdead0000, read 0x0` — 데이터가 통과하지 않는다 |
| `tb_cdc_fifo_reset` | `Data corruption after reset: assert 0 == 3405643777` |
| `tb_oom_guard_thresholds` | `Expected PRESSURE, got 0` (2/2 실패) |

`cdc_fifo` 는 `CLAUDE.md` 2절이 **"구조 양호"** 로 분류한 모듈이다. BUG-002 때문에 컴파일이 안 됐고,
컴파일을 고치고 처음 돌려보니 동작하지 않는다. **"구조 양호"의 근거가 된 테스트는 한 번도 실행된 적이 없었다.**

### 4. W2-1 — README 전면 재작성

`grep -niE "MPW|training|LLM inference|GPU|frontier|awaiting silicon|closed loop|no mocks|all pass|237" README.md` → **없음**

- 첫 줄을 지시대로 바꿨다: "INT8 16×16 외적 누산 타일 + 검증된 제어 평면. 다음 목표: 델타룰 헤드."
- 현재 상태표에 **근거(셀 수·명령)를 붙였다.** 스텁·없음·미측정을 그대로 적었다.
- **"행동 모델(합성 불가)" 섹션**을 분리하고 14개 모듈마다 왜 하드웨어가 아닌지 적었다.
- 실행 가능한 명령만 남겼다. `pip install verilator` 같은 거짓 지시 제거.
- 알려진 미해결 버그를 README 에 명시 (BUG-008 링크).
- pytest 실패 3건·xfail 5건의 이유를 README 에 그대로 적었다.

### 5. BUG-001 수정 — SSOT 먼저

**`docs/DESIGN.md` 를 먼저 고쳤다** (버전 0.1 → 0.2). 5.1절 **완료 신호 계약** 신설:

| 신호 | 의미 | 걸려 있는 것 |
|---|---|---|
| `done_ok` | 성공 완료, 1사이클 | **완료 IRQ (`DESC_DONE`)** |
| `done_err` | 실패 종료, 1사이클 | (fault 는 `TC0_FAULT` 로도 보고) |
| `done_pulse` | 리타이어 = ok\|err | **자원 회수** (OOM 감소) |

7절 불변조건에 **I6** 추가: 한 트랜잭션에 ok/err 중 정확히 하나, 폭 1사이클.

RTL: `desc_fsm_v2` 에 `came_from_fault` 레지스터로 `ST_DONE` 진입 경로를 구분해
`done_ok`/`done_err` 를 파생. `g2_ctrl_top` 의 `irq_sources[0]` 을 `fsm_done_ok` 로 바꿨다.

**`ST_FAULT → ST_DONE` 전이는 그대로 뒀다.** 권고했던 "ST_IDLE 로 직행"(1안)을 쓰지 않은 이유:
`oom_alloc_dec` 가 `done_pulse` 에 걸려 있어서, 직행시키면 **fault 트랜잭션마다 OOM 사용량이 누수된다.**
3안(신호 분리)이 맞았던 것은 이 때문이다 — 계약 규칙 4 로 명시했다.

```
ILLEGAL    IRQ_PENDING = 0x00000021 ['DESC_DONE','TC0_FAULT']   (수정 전)
ILLEGAL    IRQ_PENDING = 0x00000020 ['TC0_FAULT']               (수정 후)
** TESTS=3 PASS=3 FAIL=0 SKIP=0 **
```

`tb_desc_fsm_v2_done_pulse.py` 도 새 계약(I6)으로 고쳤다 — 6/6 PASS.
예전 단언("fault 면 done_pulse 가 없어야 한다")은 **새 계약과 모순**이라 바꿨다.
`done_pulse` 는 리타이어이므로 fault 에서도 나는 게 맞다.

`g3_desc_fsm` 은 같은 `ST_FAULT → ST_DONE` 모양이지만 **IRQ 소비자가 없다**
(`g3_int_top` 이 출력으로 그냥 내보낸다). 사용자에게 보이는 버그가 아니라 손대지 않았다.

---

### 6. 게이트가 거짓 실패를 냈다 — OOM 등급 추가

BUG-001 수정 후 게이트를 돌렸더니 **EXIT=1** 이 나왔다.

```
FAIL    gemm_int4_synth   ERROR: ABC: execution of command ... failed: return code 137.
FAIL: STAGE 2 합성 실패 1개
EXIT=1
```

로그를 보면 `ABC: Killed`, rc=137 = SIGKILL = **OOM killer**.
병렬 4개에 무거운 모듈(`g2_ctrl_top` 657k cells, `mxu_bf16_16x16`)이 겹쳐 abc 가 메모리로 죽은 것이다.
**설계 실패가 아니라 자원 한계다.** 시간 초과와 같은 등급이어야 하는데 하드 실패로 처리하고 있었다.

`scripts/synth_gate.sh` 에 **OOMKILL 등급**을 추가했다 (rc 137/139, `bad_alloc`, `ABC: Killed` 감지).
시간 초과와 같이 WARN + 목록으로 표시하고, `JOBS` 를 줄여보라고 안내한다.
`--strict` 에서는 둘 다 실패로 친다.

`JOBS=2` 로 다시 돌린 결과:

```
WARN: 시간 초과 4개 (>420s) — 합성 불가가 아니라 '측정 못 함'이다:
       backward_engine  g3_int_top  gemm_int4_synth  mxu_bf16_16x16
=== PASS ===========================================================
STAGE 1 elaborate 29/31  (시간 초과 0, known-incomplete 2)
STAGE 2 synth     27/31  (시간 초과 4, OOM 0, 0셀 0, known-incomplete 0)
EXIT=0
```

`g2_ctrl_top` 657,872 cells · `desc_fsm_v2` 2,789 (BUG-001 수정으로 2,786 → 2,789, `came_from_fault` 1비트 + 파생 2개).

**교훈**: 게이트의 판정 등급이 부족하면 게이트 자체가 거짓 신호를 낸다.
"실패"와 "측정 못 함"을 구분하지 않으면, 통과시키려고 범위를 줄이는 압력이 생긴다.

---

### 0단계 (1~2주) 종료 기준 — **3/3 충족**

| 기준 | 상태 |
|---|---|
| `scripts/synth_gate.sh` 통과 | ✅ EXIT=0 |
| README 에 과장 문구 0 | ✅ grep 결과 없음 |
| BUGS.md 에 파형 근거 항목 1개 이상 | ✅ BUG-001 (+7건) |

W1-1 ✅ W1-2 ✅ W1-3 ✅ W2-1 ✅ W2-2 ✅ W2-3 ✅ — **0단계 전 항목 완료.**

**1단계(3~4주)는 RTL 금지 · 골든 모델 전용이다.** 이번 세션이 그 이유를 한 번 더 보여줬다:
BUG-008 의 6건은 전부 "테스트벤치가 틀렸나 RTL 이 틀렸나"를 가릴 수 없는 상태다.
`sim/golden/` 을 import 하는 테스트는 여전히 **0개**다.

---

## 2026-09-11 (세션 3, 자율모드) — 금지 토큰 2등급 도입 + BUG-007

**한 것**: 승인 (1)(2)(3) 반영. `===`/`!==`/`while` 을 SIM-ONLY 등급으로 금지 토큰에 추가하고 체커가 `` `ifdef COCOTB_SIM `` 중첩을 인식하게 했다. 추가하자마자 **남아 있던 X 가드 2건(BUG-007)** 이 드러나서 선제 수정. `mxu_bf16_16x16`·`backward_engine` 에 "범위 밖 · DR1 v2 검토" 주석. CLAUDE.md 규칙 1·4절 갱신.
**안 된 것**: BUG-001 미수정(지시대로). 기존 `tb/` 36개 여전히 실행 못 함. `backward_engine`·`g3_int_top`·`gemm_int4_synth`·`mxu_bf16_16x16` 은 420s 예산 안에 yosys 가 못 끝냄(**미측정**).
**실행한 검증 명령**: `bash scripts/synth_gate.sh` → **PASS (EXIT=0)** / `python3 scripts/check_banned_tokens.py rtl/*.sv rtl/*.v` → **clean (EXIT=0)** / `python3 -m pytest tests/ -q` → **3 failed, 247 passed, 5 xfailed**
**다음 세션 첫 작업**: verilator 소스 빌드 + `scripts/setup_tools.sh` + 기존 tb 36개 중 복구 수 보고. **단, W1-3(린트)은 이것을 기다릴 필요가 없다** — 아래 참조.
**사용자 결정 필요**: BUG-001 수정 3안 중 선택 (DESIGN.md 9절을 먼저 고쳐야 한다).

---

### 승인 4건 처리

| # | 지시 | 결과 |
|---|---|---|
| 1·2 | `===`, `!==`, `while` 금지 토큰 추가. `ifdef COCOTB_SIM` 내부와 `tb/` 는 허용 | ✅ HARD / SIM-ONLY 2등급으로 나눠 `scripts/check_banned_tokens.py` 구현. `` `ifdef/`ifndef/`elsif/`else/`endif `` 중첩 추적. `tb/` 는 애초에 게이트 대상이 아님 |
| 3 | `mxu_bf16_16x16`, `backward_engine` 이동 없이 느린 게이트 + "범위 밖·DR1 v2 검토" 주석 | ✅ 두 파일 module 선언 앞에 근거(DESIGN.md 10절·8절, CLAUDE.md 2절 제외 조항)까지 적어 넣음 |
| 4 | 문서 3건은 정원님이 로컬 확인 후 처리 | ✅ 손대지 않음. pytest 에 빨간 채로 남아 있다 |

### BUG-007 — 규칙을 추가하자마자 2건이 더 나왔다

```
$ python3 scripts/check_banned_tokens.py rtl/*.sv rtl/*.v
rtl/act_sram.sv:33: [!==] if (a[k] !== 1'b0 && a[k] !== 1'b1)
rtl/ctrl_fsm.sv:94: [===] end else if (core_done === 1'b1) begin
EXIT=1
```

`ctrl_fsm.sv:94` 는 `core_done_seen` 캡처다. 조건이 상수 0 으로 접히면 **GEMM FSM 이 `ST_WAIT` 에 영영 갇힌다.**

측정해 보니 **이번엔 피해가 없었다**:

| 모듈 | 가드 있음 | 가드 제거 | 판정 |
|---|---|---|---|
| `ctrl_fsm` | 499 | 499 | 동일 |
| `act_sram` | 99,294 | 99,294 | 동일 |

`wgt_sram` 은 사라졌는데 `act_sram` 은 멀쩡했던 이유가 여기 있다. **같은 X 가드라도 접히는 방향이 코드 모양에 따라 달랐다** — `wgt_sram` 은 "X면 쓰지 마라"(→ 항상 쓰지 마라, 메모리 소멸), `act_sram` 은 "X면 invalid"(→ 항상 valid, 무해).

**도구가 어느 쪽으로 접을지에 기대는 코드다.** yosys 가 무해했다고 Vivado 도 그러리라는 보장은 없다(**미검증**). 그래서 피해가 없어도 `` `ifdef COCOTB_SIM `` 으로 굳혔다. 수정 후 셀 수 동일(499 / 99,294) 확인.

### 게이트 재실행

```
--- STAGE 2: synth -top (전 모듈, 모듈당 420s, 병렬 4) ---
  ok      ctrl_fsm                 0s       cells(design total)=499
  ok      cdc_fifo                 0s       cells(design total)=1167
  ok      desc_fsm_v2              4s       cells(design total)=2786
  ok      dma_bridge               1s       cells(design total)=880
  ok      desc_queue               19s      cells(design total)=66030
  ok      act_sram                 46s      cells(design total)=99294
  ok      g3_desc_fsm              3s       cells(design total)=2897
  ok      g2_ctrl_top              249s     cells(design total)=657868
  ok      g2_protob_top            250s     cells(design total)=657773
  ok      gemm_int4_fpga           101s     cells(design total)=50576
  ok      gemm_core                146s     cells(design total)=429672
  TIMEOUT backward_engine          >420s
  ok      gemm_stub                0s       cells(design total)=203
  ok      gemm_int4_sky130         54s      cells(design total)=43882
  TIMEOUT g3_int_top               >420s
  ok      irq_ctrl                 1s       cells(design total)=549
  ok      mac_array                1s       cells(design total)=196352
  ok      mac_pe                   0s       cells(design total)=730
  ok      gemm_wb_wrapper          76s      cells(design total)=71263
  ok      oom_guard                1s       cells(design total)=3752
  ok      pcie_ep_versal           0s       cells(design total)=4
  ok      reg_top                  9s       cells(design total)=7250
  ok      reset_seq                0s       cells(design total)=52
  ok      scale_fabric_ctrl        0s       cells(design total)=62
  ok      trace_ring               41s      cells(design total)=143610
  ok      gemm_top                 149s     cells(design total)=430402
  ok      vpu_lut                  1s       cells(design total)=1130
  ok      wgt_sram                 49s      cells(design total)=99294
  TIMEOUT gemm_int4_synth          >420s
  ok      vpu_core_synth           337s     cells(design total)=117221
  TIMEOUT mxu_bf16_16x16           >420s

WARN: 시간 초과 4개 (>420s) — 합성 불가가 아니라 '측정 못 함'이다:
       backward_engine
       g3_int_top
       gemm_int4_synth
       mxu_bf16_16x16
       SYNTH_TIMEOUT=1800 으로 다시 돌리거나 Vivado 로 판정할 것.
=== PASS ===========================================================
```

STAGE 1 이 26/31 → **27/31** 로 개선됐다 (시간 초과 3 → 2).

---

### 짚을 것 — W1-3 은 verilator 소스 빌드를 기다릴 필요가 없다

다음 세션 1순위로 verilator 소스 빌드를 주셨는데, **W1-3(린트 기록)은 지금 있는 Debian 5.020 으로 바로 된다.**

```bash
$ verilator --lint-only -Wall -Irtl rtl/mac_pe.sv; echo $?
0
```

verilator 의 cocotb 2.x 비호환은 **시뮬레이션 실행** 경로에만 해당한다 (`VerilatedVpi::clearEvalNeeded` 등 VPI API). `--lint-only` 는 영향이 없다.

즉 두 작업은 독립이다:
- **W1-3 (린트 → `docs/LINT.md`)**: 지금 가능. 몇 분.
- **verilator 소스 빌드**: 기존 `tb/` 36개를 되살리기 위한 것. 이쪽이 더 크고 더 중요하지만 PLAN 항목은 아니다.

### 2주 종료 기준 대비 현황

PLAN 0단계 종료 기준 3개 중:

| 기준 | 상태 |
|---|---|
| `scripts/synth_gate.sh` 통과 | ✅ EXIT=0 |
| `docs/BUGS.md` 에 파형 근거 항목 1개 이상 | ✅ BUG-001 (+ 5건 더) |
| README 에 과장 문구 0 | ⬜ **W2-1. 유일하게 남은 것** |

W1-1 ✅ / W1-2 ✅ / W1-3 ⬜ / W2-1 ⬜ / W2-2 ✅(조기 완료) / W2-3 ✅

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
