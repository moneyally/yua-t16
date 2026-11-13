# 검증 인프라 — 테스트벤치 36개 중 2개만 실행 가능했던 문제

## 발견

`tb/` 에 cocotb 테스트벤치가 36개 있었지만 러너는 2개뿐이었다.
**나머지 34개는 실행할 수단이 없었다.** `sim/cocotb/` 의 테스트 9개를
합치면 45개 중 2개만 돌릴 수 있는 상태였다.

README 는 "24 RTL cocotb (module-level) — All pass" 와 "Total 237 — All pass"
를 명시하고 있었으나, 그 숫자를 재현할 방법이 저장소에 없었다.

실제로 돌려보니 RTL 버그 3건과 테스트벤치 결함 2건이 나왔다.

## 인프라

| 파일 | 역할 |
|---|---|
| `sim/cocotb/rtl_deps.py` | top 모듈 하나에서 RTL 의존 파일 닫힘(closure)을 자동 수집 |
| `sim/cocotb/run_one.py` | 테스트벤치 하나를 격리된 프로세스에서 실행 |
| `sim/cocotb/run_all.py` | 전체 발견 → top 추론 → 실행 → JSON 리포트 |
| `sim/cocotb/summarize.py` | `results.xml` 들을 테스트 단위로 집계 |
| `sim/cocotb/verify_matrix.sh` | MXU 구성별(PAR=1/2/4/8) 검증 매트릭스 |

```bash
export PATH=/opt/verilator/bin:$PATH
python3.13 sim/cocotb/run_all.py            # 전체
python3.13 sim/cocotb/run_all.py desc_queue # 이름 필터
python3.13 sim/cocotb/summarize.py          # 집계
```

### top 모듈 추론

이름 접두사로 맞추되(`tb_desc_queue_basic` → `desc_queue`), **SSOT 이슈 ID를
먼저 본다.** 모듈과 테스트벤치 양쪽 헤더에 `G3-INT-00N` 이 적혀 있다.

이름만으로 짐작하면 `tb_g3_int_single_layer`(G3-INT-002)를
`g3_int_top`(G3-INT-001)에 잘못 붙이게 된다. 실제로 그렇게 붙였다가
`contains no child object named start` 로 4건이 실패했다. 올바른 대상은
`g3_train_int_top` 이다.

## 고친 것

### RTL 버그 3건

| 모듈 | 문제 | 수정 |
|---|---|---|
| `cdc_fifo.sv` | `initial` 블록이 **포트 리스트 안**에 있어 모듈이 파싱조차 되지 않음 | 본문으로 이동 |
| `cdc_fifo.sv` | 포트 주석은 `rd_ready // data available`(FWFT)인데 구현은 핸드셰이크 **다음** 사이클에 데이터를 냄 → 첫 읽기가 0 | FWFT 읽기로 변경 |
| `trace_ring.sv` | 덮어쓰기 시 head 전진 조건이 `tail_next == head_r` 라 한 바퀴 돈 뒤 영영 참이 되지 않아 head 가 0 에 고정 | `tail_r == head_r` 로 수정 |
| `g3_desc_fsm.sv` | `ST_DECODE` 가 "모르는 opcode" 와 "알지만 미지원" 두 경우로 분기하는데, fault_code 할당부가 state 만 보고 무조건 `0x01` | `opcode_known()` 으로 `0x01`/`0x04` 구분 |

`cdc_fifo` 는 어디에도 인스턴스화되지 않은 데다 파싱조차 안 돼서
**한 번도 검증된 적이 없었다.**

### 테스트벤치 결함 2건

- `tb_cdc_fifo_async` — 소비자 루프 안의 "safety timeout" `Timer(500ns)` 가
  `read_data` 가 빈 동안 매 반복 걸려서 전체 예산 5000ns 를 태웠다.
  제거하니 32건을 순서대로 모두 읽는다 (RTL 은 정상이었다).
- `tb_oom_guard_*` — `await RisingEdge(clk)` 직후 바로 레지스터 출력을 읽어
  엣지 반영 **전** 값을 보고 있었다. `settle()` 헬퍼로 한 델타 넘긴다.
  (ReadOnly 는 이후 신호 쓰기가 막혀 쓸 수 없다.)

### 도구 문제 2건

- **Icarus 는 쓸 수 없다.** unpacked 배열 포트를 바인딩하지 못해
  MXU 계열 누산기가 전부 X 로 읽힌다. 레포 자체 테스트벤치도 Icarus 에선
  5/5 실패한다. RTL 문제가 아니다.
- **cocotb 2.1.0 은 Verilator 5.028+ 를 요구한다.** Debian 의 5.020 으로는
  `clearEvalNeeded` 심볼 오류가 난다.
- Verilator 는 작은 설계에서 "fast build" 경로를 타며 `verilator_includer`
  를 `#!/usr/bin/env python3` 로 실행한다. cocotb 가 심은 PYTHONPATH 를
  다른 버전의 python3 가 읽으면 `SRE module mismatch` 로 죽는다.
  `run_one.py` 가 PATH 앞에 python3 shim 을 둔다.

## 결과

```
실행된 테스트벤치  46 / 46
총 테스트         188개 — 통과 188, 실패 0
```

이전: 실행 가능한 테스트벤치 2개.

### 빌드는 top 모듈별로 공유한다

테스트벤치마다 따로 빌드하면 `g2_ctrl_top` 을 10번, PAR=8 의
`mxu_bf16_128x128` 을 2번 컴파일한다. 전체 회귀가 30분을 넘겼다.
`sim_build/build/<top>` 으로 묶으니 46개 테스트벤치가 15개 빌드로 줄고
전체 회귀가 수 분 안에 끝난다. 결과는 `sim_build/results/<tb>.xml` 로
분리해 충돌을 막는다.

## 남은 공백

`pytest tests/` 는 247 통과 / 8 실패다. 8건 모두
`tests/test_vck190_build_contract.py` 이고, 내가 건드린 영역이 아니다.

- **3건** — `docs/ORBIT_G2_VCK190_*.md` 존재 검사. `docs/` 가 `.gitignore`
  에 있어 **한 번도 커밋된 적이 없다.** 클론 환경에서는 구조적으로 통과
  불가능했다. `.gitignore` 에서 `docs/` 를 빼 두었으니, 문서 자체는
  작성자가 커밋해야 한다.
- **5건** — `fpga/vck190/create_cpm_ip.tcl` 이 BAR 크기/벤더 ID/MSI-X 를
  설정하지 않는다. tcl 이 직접 "If errors occur, use Vivado GUI to configure
  CPM manually" 라고 적고 있다. 즉 **PCIe 엔드포인트 자동 구성이 없다.**
  이는 `pcie_ep_versal.sv` 의 CQ/CC/RQ/RC 스텁과 함께 보드 브링업의
  실제 블로커다. Vivado 없이 IP 속성명을 추측해 채우면 합성 단계에서
  깨지므로 여기서는 손대지 않았다.
