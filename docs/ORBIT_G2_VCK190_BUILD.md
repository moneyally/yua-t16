# ORBIT-G2 VCK190 빌드

> **재구성 문서.** 원본은 커밋된 적이 없다 (`docs/` 가 `.gitignore` 에
> 있었다). 저장소의 tcl, xdc, 계약 테스트에서 재구성했다.

## 대상

| | |
|---|---|
| 보드 | VCK190 |
| 디바이스 | xcvc1902 (Versal VC1902) |
| PCIe | CPM, Gen4 (16.0 GT/s) x8, Endpoint |
| 최상위 | `g2_protob_top` |
| Vivado | 2025.2 기준 작성 |

**PL PCIE 가 아니라 CPM 이 맞다.** VCK190 은 하드닝된 CPM 블록을 쓴다
(`rtl/pcie_ep_versal.sv` 헤더 참조).

## 절차

```bash
cd fpga/vck190
vivado -mode batch -source create_project.tcl
vivado -mode batch -source create_cpm_ip.tcl
```

`create_project.tcl` 이 프로젝트와 최상위를 잡고,
`create_cpm_ip.tcl` 이 CPM PCIe 엔드포인트를 Block Design 으로 만든다.

### CPM 속성 이름 문제

`versal_cips` 의 `CONFIG.CPM_CONFIG` 속성 **이름**은 IP 버전마다 다르다.
`create_cpm_ip.tcl` 은 설계에서 나온 **값**과 버전 의존적인 **이름**을
분리해 두었다.

속성 오류가 나면:

1. GUI (IP Integrator) 에서 CPM 을 같은 값으로 설정한다.
2. `write_bd_tcl -force exported_cpm_bd.tcl` 로 뽑는다.
3. 뽑힌 tcl 의 **이름만** `create_cpm_ip.tcl` 에 반영한다. 값은 그대로.

스크립트는 BAR 바이트 크기와 log2 구경 리터럴이 어긋나면 시작 전에
멈춘다.

## 설정값의 출처

| 항목 | 값 | SSOT |
|---|---|---|
| BAR0 | 1 MiB (구경 20) | `rtl/pcie_ep_versal.sv`, `tools/orbit_mmap_backend.py` |
| BAR2 | 2 MiB (구경 21) | 〃 |
| BAR4 | 64 KiB (구경 16) | 〃 |
| 벤더 ID | 0x10EE (Xilinx) | `tools/orbit_mmap_backend.py` |
| 디바이스 ID | 0x9038 | 〃 (현재 placeholder) |
| MSI-X | 16 벡터 (테이블 0x00F) | `rtl/irq_ctrl.sv` 12 소스 + 여유 |

한쪽만 고치면 `tests/test_vck190_build_contract.py` 가 잡는다.

## 제약

`fpga/vck190/vck190_pcie.xdc` 가 GTY 쿼드와 레퍼런스 클럭 핀을 잡는다.

## 시뮬레이션 (하드웨어 불필요)

```bash
export PATH=/opt/verilator/bin:$PATH
python3.13 sim/cocotb/run_all.py        # 전체 회귀
python3.13 sim/cocotb/summarize.py      # 집계
python3.13 -m pytest tests/ -q          # 호스트 스택
```

**검증은 Verilator 로 한다.** Icarus 는 unpacked 배열 포트를 바인딩하지
못해 MXU 계열 누산기가 전부 X 로 읽힌다 (RTL 문제가 아니다).
cocotb 2.1.0 은 Verilator 5.028 이상을 요구한다.
