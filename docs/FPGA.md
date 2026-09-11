# FPGA.md — 보드 빌드 상태 (KV260)

버전 0.1 (2026-09-11). PLAN W11.

## 0. 한 줄 요약

**아직 아무것도 돌리지 않았다.** 이 문서는 *무엇을 준비했는지*와 *무엇이
미측정인지*를 적어 둔 것이다. 자원·타이밍·전력 수치는 **하나도 없다.**

## 1. 준비된 것

| 파일 | 내용 | 상태 |
|---|---|---|
| `rtl/axil_reg_bridge.sv` | AXI4-Lite 슬레이브 → `reg_top` 버스 어댑터 | **시뮬레이션 검증됨** (`tb/tb_axil_reg_bridge.py` 7/7) |
| `rtl/dr1_soc_top.sv` | 보드 최상위 = 브리지 + `g2_ctrl_top` | 린트·합성 게이트 통과 |
| `fpga/kv260/create_project.tcl` | Vivado 프로젝트 생성 (xck26, OOC 합성) | **실행 안 해봄** |
| `fpga/kv260/kv260_dr1.xdc` | 100MHz 클럭 + AXI 지연 제약 | **실행 안 해봄** |

## 2. 미측정 (이 컨테이너에 Vivado 가 없다)

- LUT / FF / BRAM / DSP 사용량 — **모름**
- 타이밍 클로징 여부, 최대 주파수 — **모름** (목표 100MHz 는 목표일 뿐이다)
- `state_sram` / `dr1_scratch` / `act_sram` / `wgt_sram` 이 Vivado 에서
  **BRAM 으로 추론되는지** — 모름. yosys 쪽에서는 `$mem_v2` 로 확인했지만
  (`scripts/synth_gate.sh --check-mem`) 그것이 Vivado 를 보장하지 않는다.
- 전력 — 모름

**가장 큰 미지수: 크기.** yosys 게이트 수로는 `g2_ctrl_top` 이 66만 셀이다.
이 숫자는 일반 게이트 환산이라 LUT 수와 직접 비교할 수 없지만, KV260(K26 SOM)의
LUT 예산 안에 들어갈지는 **합성해 봐야 안다**. 안 들어가면 선택지는:

1. `g2_ctrl_top` 대신 **DR1 만** 담은 축소 top 을 만든다 (GEMM 경로 제거).
   `act_sram`/`wgt_sram`/`gemm_core` 가 빠지면 대부분이 사라진다.
2. `MAX_KT`/`QUEUE_DEPTH` 를 줄인다.

## 3. 보드가 오면 하는 순서 (PLAN W12)

```bash
# 1) 프로젝트 생성 + OOC 합성 (PS 없이 PL 만)
vivado -mode batch -source fpga/kv260/create_project.tcl
#    Vivado Tcl 에서:
#      launch_runs synth_1 -jobs 4; wait_on_run synth_1
#      open_run synth_1
#      report_utilization -file build/vivado/util_synth.rpt
#      report_timing_summary -file build/vivado/timing_synth.rpt
```

리포트가 나오면 **출력 그대로** 이 문서 4절에 붙인다 (CLAUDE.md 규칙 3).
목표를 못 맞추면 맞췄다고 쓰지 않는다 — 주파수를 낮추고 낮췄다고 적는다.

그 다음:
1. Zynq MPSoC PS 블록 디자인 생성, `dr1_soc_top` 을 AXI4-Lite 슬레이브로 연결
2. 비트스트림 → SD 부팅 → `devmem` 으로 `G2_ID` 읽기 (`0x47320001` 나와야 함)
3. `DELTA_INIT` → `DELTA_STEP` 1토큰 → 골든과 대조

3번은 시뮬레이션에서 이미 통과한 경로다 (`tb/tb_dr1_host_e2e.py`).
보드에서는 **백엔드만** 바뀐다 — `tools/orbit_backend.py` 에 `/dev/mem` 백엔드를
하나 추가하면 `OrbitDevice` 위쪽 코드는 그대로 돌아간다.

## 4. 측정 결과

_(비어 있음 — 아직 합성하지 않았다.)_
