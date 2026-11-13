# ORBIT-G2 VCK190 실패 매트릭스

> **재구성 문서.** 원본은 커밋된 적이 없다. RTL 의 상태 인코딩,
> 스모크 스크립트, 실제로 겪은 툴 문제에서 재구성했다.

## 1. 보드가 `lspci` 에 뜨지 않는다

| 증상 | 가능한 원인 | 확인 |
|---|---|---|
| 호스트가 리셋/리부팅 반복 | PERST# 해제 후 20 ms 안에 link training 을 시작하지 못함 | `reset_seq` 의 `RELEASE_DELAY` 총합을 계산 |
| 디바이스 없음, 호스트는 정상 | 비트스트림 미적재, CPM 미설정 | Vivado 하드웨어 매니저에서 적재 확인 |
| 링크는 뜨는데 BAR 가 0 | `create_cpm_ip.tcl` 의 BAR 설정 실패 | 속성 오류 로그 확인 (파일 상단 절차) |

링크가 뜬 뒤라면 `BOOT_CAUSE` 를 읽어라.

| `boot_cause` 비트 | 뜻 |
|---|---|
| [0] | POR (정상 전원 투입) |
| [1] | WDOG (워치독 타임아웃) |
| [2] | SW (소프트웨어 리셋) |
| [3] | PCIE_FLR (Function Level Reset) |

## 2. 레지스터 읽기가 이상하다

| 증상 | 원인 | 조치 |
|---|---|---|
| 전부 0xFFFFFFFF | 완결자 경로가 완료를 못 냄 | `pcie_ep_versal` 의 `CQ_*`/`CC_*` 를 PG343 과 대조 |
| 전부 0 | BAR 라우팅이 어긋남 | BAR ID 매핑 확인 (`tb_pcie_ep_completer.py` 의 `test_bar_routing`) |
| `G2_ID != 0x4732_0001` | 다른 디바이스이거나 BAR0 오프셋 오류 | `scripts/vck190_first_smoke.py` 로 재확인 |
| 32비트 외 접근이 실패 | 설계상 1 DW 만 지원 | 호스트는 32비트로만 접근해야 한다 |

## 3. 디스크립터가 실패로 돌아온다

`g3_desc_fsm` / `desc_fsm_v2` 의 `fault_code`:

| 코드 | 뜻 | 흔한 원인 |
|---|---|---|
| 0x01 | Illegal opcode | 모르는 opcode |
| 0x02 | CRC 실패 | 디스크립터 패킹 불일치 (`tools/orbit_desc.py`) |
| 0x03 | 타임아웃 | 타깃이 `core_done` 을 올리지 않음 |
| 0x04 | Unsupported target | 아는 opcode지만 아직 타깃이 없음 (BACKWARD/OPTIMIZER/COLLECTIVE) |

0x01 과 0x04 를 구분하지 못하던 버그가 있었다 — 지금은 구분한다.

## 4. 메모리 압력

`oom_guard` 의 `pressure_state`:

| 값 | 상태 | 부수 효과 |
|---|---|---|
| 0 | NORMAL | — |
| 1 | PRESSURE | — |
| 2 | CRITICAL | `prefetch_clamp` |
| 3 | EMERG | `prefetch_clamp` + `admission_stop` |

## 5. 성능이 기대보다 낮다

정상이다. 측정값이 문서화돼 있다.

| Kt | op당 사이클 | MAC 가동률 |
|---:|---:|---:|
| 1 | 68.4 | 7.3% |
| 16 | 74.0 | 27.0% |
| 64 | 210.1 | 32.4% |

읽기 대역폭이 1.9 배 모자라 **완벽히 겹쳐도 상한이 53%** 다.
자세한 내용은 `spec/memory-bandwidth-analysis.md`.

`gemm_top` 경유로는 이보다도 낮다 — `ctrl_fsm` 이 `core_done` 을
기다린 뒤에야 다음 디스크립터를 받기 때문이다.

## 6. 시뮬레이션이 깨진다 (RTL 문제가 아님)

| 증상 | 원인 | 조치 |
|---|---|---|
| 누산기가 전부 X | Icarus 가 unpacked 배열 포트를 바인딩 못 함 | Verilator 를 쓴다 |
| `clearEvalNeeded` 심볼 오류 | cocotb 2.1.0 이 Verilator 5.028+ 요구 | Verilator 를 올린다 (Debian 5.020 불가) |
| `SRE module mismatch` | Verilator 의 `verilator_includer` 가 다른 python3 로 cocotb 의 PYTHONPATH 를 읽음 | `run_one.py` 의 python3 shim |
| 신호가 한 사이클 전 값으로 보임 | `RisingEdge` 직후 읽음 | `Timer(1ns)` 로 한 델타 넘긴 뒤 읽는다 |

마지막 항목은 실제로 테스트 두 개에서 버그를 만들었다
(`tb_oom_guard_*` 의 `settle()`, 처리량 테스트의 핸드셰이크 중복 집계).
