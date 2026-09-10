# LOG.md — 작업 기록

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
