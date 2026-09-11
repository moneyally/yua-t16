# FPGA.md — 보드 빌드 상태 (KV260)

버전 0.1 (2026-09-11). PLAN W11.

## 0. 한 줄 요약

**아직 아무것도 돌리지 않았다.** 이 문서는 *무엇을 준비했는지*와 *무엇이
미측정인지*를 적어 둔 것이다. 자원·타이밍·전력 수치는 **하나도 없다.**

## 1. 준비된 것

| 파일 | 내용 | 상태 |
|---|---|---|
| `rtl/axil_reg_bridge.sv` | AXI4-Lite 슬레이브 → `reg_top` 버스 어댑터 | **시뮬레이션 검증됨** (`tb/tb_axil_reg_bridge.py` 7/7) |
| `rtl/axi4_master_adapter.sv` | 코어의 `rd_req_*`/`wr_req_*` → AXI4 마스터 (PS DDR) | **시뮬레이션 검증됨** (`tb/tb_axi4_master_adapter.py` 8/8) |
| `rtl/dr1_soc_top.sv` | 보드 최상위 = AXI-Lite 슬레이브 + `g2_ctrl_top` + AXI4 마스터 | **시뮬레이션 검증됨** (`tb/tb_dr1_soc_top.py` 6/6) — 아래 3절 2·4번을 시뮬레이션에서 먼저 밟았다 |
| `tools/orbit_axil_backend.py` | 호스트 스택용 AXI4-Lite 백엔드 | 보드에서는 이 자리에 `/dev/mem` 백엔드가 들어간다 |
| `rtl/wdog_timer.sv` | 워치독 타이머 (`WDOG_CTRL` → 리셋) | **시뮬레이션 검증됨** (`tb/tb_wdog_timer.py` 7/7, `tb/tb_g2_ctrl_top_wdog.py` 6/6) |
| `fpga/kv260/create_project.tcl` | Vivado 프로젝트 생성 (xck26, OOC 합성) | **실행 안 해봄** |
| `fpga/kv260/kv260_dr1.xdc` | 100MHz 클럭 + AXI 지연 제약 | **실행 안 해봄** |

## 2. 미측정 (이 컨테이너에 Vivado 가 없다)

- LUT / FF / BRAM / DSP 사용량 — **모름**
- 타이밍 클로징 여부, 최대 주파수 — **모름** (목표 100MHz 는 목표일 뿐이다)
- `state_sram` / `dr1_scratch` / `act_sram` / `wgt_sram` 이 Vivado 에서
  **BRAM 으로 추론되는지** — 모름. yosys 쪽에서는 `$mem_v2` 로 확인했지만
  (`scripts/synth_gate.sh --check-mem`) 그것이 Vivado 를 보장하지 않는다.
- 전력 — 모름
- **AXI4 마스터가 실제 DDR 에서 도는지** — 모름. `axi4_master_adapter` 를 받아 본 것은
  파이썬 슬레이브 모델뿐이다. 모델은 프로토콜 위반(4KB 경계, 256 beat 상한, WLAST
  위치)을 assert 하지만, 실제 DDR 컨트롤러의 **지연·역압·응답 순서**는 흉내 내지
  않는다. 특히 이 어댑터는 **outstanding 1** 이라 지연이 길면 대역폭이 그대로 죽는다 —
  보드에서 처음 볼 숫자가 그것이다.
- PS 쪽 결선 — 미정. `m_axi_*` 를 S_AXI_HP0 에 붙이려면 Vivado 블록 디자인에서
  폭(128비트)·클럭 도메인을 맞춰야 한다. `create_project.tcl` 에는 아직 없다.

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
1. Zynq MPSoC PS 블록 디자인 생성, `dr1_soc_top` 을 AXI4-Lite 슬레이브로 연결.
   AXI4 마스터(`m_axi_*`, 128비트)는 S_AXI_HP0 에 붙인다 — **DR1 경로는 안 쓰므로
   처음 bring-up 에서는 안 붙이고 넘어가도 된다.** 붙일 때 폭과 클럭 도메인을 맞춘다.
2. 비트스트림 → SD 부팅 → `devmem` 으로 `G2_ID` 읽기 (`0x47320001` 나와야 함)
3. **워치독을 먼저 켜지 않는다.** 2번이 성공할 때까지는 꺼 둔다 — 안 그러면
   칩이 1초마다 리셋되는데 그게 워치독 때문인지 다른 문제인지 구분이 안 된다.
   2번이 되면 `dev.watchdog_enable(period)` 로 켜고, 그 뒤부터 멈춤을 관측한다.
4. `DELTA_INIT` → `DELTA_STEP` 1토큰 → 골든과 대조

2번과 4번은 **시뮬레이션에서 이미 통과한 경로다** — `tb/tb_dr1_soc_top.py` 가
`dr1_soc_top` 을 AXI4-Lite 로만 두드려서 `G2_ID` 읽기(S1), `DELTA_INIT`(S2),
`DELTA_STEP` 1토큰(S3)·10토큰(S4) 골든 비트 일치를 확인한다.

보드에서는 **백엔드만** 바뀐다 — `tools/orbit_axil_backend.py` 자리에 `/dev/mem`
백엔드를 하나 넣으면 `OrbitDevice` 위쪽 코드는 그대로 돌아간다. 그 교체가
**이 테스트가 증명하는 것**이다: 백엔드를 바꿔도 호스트 코드가 안 바뀐다는 것.

그래도 보드가 돈다는 뜻은 아니다 (2절). 시뮬레이션이 안 보는 것: 실제 PS 의
AXI 지연·클럭 도메인, 타이밍 클로징, 전원, DDR 컨트롤러의 실제 거동.

## 4. 측정 결과

_(비어 있음 — 아직 합성하지 않았다.)_
