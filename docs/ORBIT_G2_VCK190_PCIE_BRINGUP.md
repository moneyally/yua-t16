# ORBIT-G2 VCK190 PCIe 브링업

> **재구성 문서.** 원본은 저장소에 커밋된 적이 없다 (`.gitignore` 에
> `docs/` 가 들어 있었다). 이 문서는 RTL, 스모크 스크립트, 레지스터 맵,
> PCIe CEM 요구사항에서 **거꾸로 재구성**한 것이다. 설계 당시의 원본이
> 아니므로, 원본이 나오면 그쪽이 SSOT 다.

## 1. 전원 투입부터 enumeration 까지

PCIe 카드는 전원 투입 후 **밀리초 단위 데드라인**이 있다. 거의 모든
1차 보드가 여기서 실패한다.

```
① 전원 레일 안정화 (순서대로)
         ↓  최소 지연
② PERST# 디어서션
         ↓  ★ 20 ms 안에
③ Link training 시작          ← PCIe CEM 스펙
         ↓  PERST# 해제 후 최대 100 ms
④ L0 진입 (링크 업)
         ↓  전원 안정 기준 총 120 ms
⑤ BIOS enumeration → lspci 에 보임
```

Intel 은 브링업을 **System reset → Link training → BIOS enumeration**
3단계로 정의한다.

### 가장 흔한 실패

SoC 부팅이 20 ms 를 넘기면 PERST# 가 풀렸는데 아직 부팅 중이라
link training 이 실패하고, 호스트가 enumerate 될 때까지 리셋을
반복한다. 호스트는 카드가 파워업 중 어떤 상태인지 알 수 없다.

ORBIT-G2 는 이에 대해 두 가지를 갖고 있다.

- **`rtl/reset_seq.sv`** — 리셋 해제 순서가
  `always_on → io → mem → core`, 도메인마다 `RELEASE_DELAY` 사이클 지연.
  리셋 원인을 `boot_cause[3:0]` 에 래치한다 (POR / WDOG / SW / PCIE_FLR).
  보드가 안 뜰 때 `BOOT_CAUSE` 만 읽어도 왜 죽었는지 나온다.
  **`RELEASE_DELAY` 가 ②→③ 의 20 ms 예산 안에 들어가는지 숫자로
  확인할 것.**
- **`rtl/trace_ring.sv`** — 1K 엔트리. 링크가 뜬 뒤 카드 안에서 무슨
  일이 있었는지 볼 수 있는 유일한 창이다.

## 2. BAR 맵

| BAR | 크기 | 용도 | 캐시 |
|---|---|---|---|
| BAR0 | 1 MiB (2^20) | 레지스터 | UC |
| BAR2 | 2 MiB (2^21) | CMEM 윈도우 | WC |
| BAR4 | 64 KiB (2^16) | DMA 엔진 | UC |

세 군데가 같은 값을 들고 있고, `tests/test_vck190_build_contract.py` 가
어긋나는지 감시한다.

- `rtl/pcie_ep_versal.sv` 의 `BAR0_SIZE` / `BAR2_SIZE` / `BAR4_SIZE`
- `tools/orbit_mmap_backend.py` 의 `EXPECTED_BAR_SIZES`
- `fpga/vck190/create_cpm_ip.tcl` 의 구경 상수

호스트 측 오프셋 라우팅은 `tools/orbit_mmap_backend.py` 가 정의한다:

```
[0x0_0000, 0x1_0000)  → BAR0 (레지스터)
[0x1_0000, 0x2_0000)  → BAR4 (DMA 엔진)
[0x2_0000, 0x10_0000) → BAR0 확장 (OOM / TC0 / Perf / IRQ / Trace)
```

## 3. 완결자 경로 (`pcie_ep_versal.sv`)

호스트의 BAR 접근은 CQ 로 들어와 BAR 포트로 라우팅되고, 읽기는 CC 로
완료된다.

```
P_IDLE → P_REQ → P_RD_WAIT → P_CC → P_IDLE      (읽기)
P_IDLE → P_REQ → P_IDLE                          (쓰기, 완료 없음)
```

- **지원: 1 DW (32비트) 메모리 읽기/쓰기만.** 호스트 스택이 BAR 를
  mmap 해 32비트 단위로만 접근하므로 레지스터 경로에 필요한 전부다.
- 2 DW 이상, 모르는 BAR ID → **Unsupported Request** 로 완료한다.
- 처리 중에는 `m_axis_cq_tready` 를 내려 CQ 를 더 받지 않는다.
- `s_axis_cc_tready` 가 올 때까지 완료를 유지한다.
- Requester ID 와 Tag 는 그대로 되돌려준다.

RQ/RC (요청자 DMA) 는 아직 구현하지 않았다. `dma_bridge` 가 담당한다.

### ⚠ 합성 전 반드시 확인할 것

CQ/CC 디스크립터의 **비트 위치**는 PG343 과 대조되지 않았다. 이 저장소에
PG343 이 없고 Vivado 도 없다. `rtl/pcie_ep_versal.sv` 의 `CQ_*` / `CC_*`
localparam 이 그 형식을 한 곳에 모아둔 것이고, 틀렸다면 **그 상수만**
고치면 된다. FSM 과 라우팅은 상수를 통해서만 디스크립터를 보므로
바꿀 필요가 없다.

`tb/tb_pcie_ep_completer.py` 도 같은 상수로 자극을 만든다. 즉 그 7개
테스트가 검증하는 것은 **형식이 아니라 로직**이다 (BAR 라우팅,
읽기/쓰기 처리, 완료 필드, UR 처리, 백프레셔, CC 유지).

## 4. 첫 보드 통과 기준

`scripts/vck190_first_smoke.py` 가 자동 판정한다.

```
lspci 에 디바이스 보임
BAR0 ≥ 1 MiB,  BAR4 ≥ 64 KiB
G2_ID == 0x4732_0001
G2_VERSION / CAP0 / TC0 / IRQ 가 리셋 기본값으로 읽힘
```

그다음 `tools/orbit_bringup.py` 의
`bringup()` → `run_gemm_smoke()` → `collect_diagnostics()` 로 넘어간다.

## 5. 순서

1. `fpga/vck190/create_cpm_ip.tcl` 의 CPM 속성 **이름**을 이 Vivado
   버전에 맞춘다 (값은 고치지 말 것 — 파일 상단 절차 참조).
2. `rtl/pcie_ep_versal.sv` 의 `CQ_*`/`CC_*` 를 PG343 과 대조한다.
3. 비트스트림을 올리고 `lspci` 에 뜨는지 본다.
4. `scripts/vck190_first_smoke.py` 를 돌린다.

**기성 평가보드에서 enumeration 이 안 되면 커스텀 보드에서는 100% 안 된다.**
