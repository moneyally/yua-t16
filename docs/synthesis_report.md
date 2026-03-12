# ORBIT-G1 Synthesis Report

**Tool**: Yosys 0.52 (`synth_xilinx -family xc7`)
**Target**: Arty A7-100T (`xc7a100tcsg324-1`)
**Date**: 2026-03-13
**Constraint**: 150 MHz (`constraints.xdc`)

---

## 1. 개별 모듈 합성 결과

### 1-A. VPU core synth (`vpu_core_synth.sv`)
256-wide SIMD, Q8.8 fixed-point, 10 ops (ADD/MUL/SCALE/RESIDUAL/RMSNORM/SILU/ROPE/SOFTMAX/CLAMP/GELU)

| 리소스 | 사용량 | A7-100T 가용량 | 점유율 |
|--------|--------|----------------|--------|
| LUT (전체) | 25,036 | 82,800 | **30.2%** |
| FF (FDCE) | 4,262 | 126,800 | 3.4% |
| DSP48E1 | 11 | 90 | 12.2% |
| CARRY4 | 1,200 | — | — |
| Estimated LC | ~22,418 | — | — |

LUT 상세:
```
LUT1: 720   LUT2: 3,136  LUT3: 1,761
LUT4: 339   LUT5: 1,894  LUT6: 17,906
MUXF7: 2,494  MUXF8: 947
```

---

### 1-B. GEMM INT4 synth (원본, `gemm_int4_synth.sv`)
16×16 INT4 GEMM, 병렬 곱셈 — **FPGA 초과**

| 리소스 | 사용량 | A7-100T 가용량 | 점유율 |
|--------|--------|----------------|--------|
| LUT | 8,744 | 82,800 | 10.6% |
| FF | 16,142 | 126,800 | 12.7% |
| **DSP48E1** | **1,040** | 90 | **1,156% ← 초과** |
| CARRY4 | 1,986 | — | — |

> 원인: S_COMPUTE에서 16개 INT8×INT8 병렬 곱셈 + S_SCALE에서 256개 32×16 병렬 곱셈 → 대부분 DSP 매핑

---

### 1-C. GEMM INT4 FPGA (`gemm_int4_fpga.sv`) ✅ NEW
INT8×INT4 shift-and-add (무 DSP) + 직렬화 SCALE (2 DSP)

**개선 전략:**
1. `mul_i8_i4()`: INT4 weight를 shift-and-add로 분해 (`-8*b[3] + 4*b[2] + 2*b[1] + b[0]`) → DSP 0
2. S_SCALE 직렬화: 256 병렬 → 1개/clock → DSP 2개만 사용
3. 레이턴시: COMPUTE 256 + SCALE 256 = 512 cycles @ 150 MHz ≈ **3.4 µs/tile**

| 리소스 | 원본 synth | FPGA 버전 | 감소율 |
|--------|-----------|-----------|--------|
| LUT | 8,744 | **8,848** | +1% (FF 감소 보상) |
| FF | 16,142 | **15,118** | -6% |
| **DSP48E1** | **1,040** | **4** | **-99.6%** |
| CARRY4 | 1,986 | 479 | -76% |
| Estimated LC | — | **7,083** | — |

---

## 2. 전체 ORBIT-G1 리소스 예상치 (Arty A7-100T)

> VPU + GEMM FPGA 기준. KVC Controller / MoE Router는 RTL 구현됨, 합성 미측정 → 추정치 포함.

| 컴포넌트 | LUT | FF | DSP48E1 | BRAM36 |
|----------|-----|-----|---------|--------|
| VPU core (×1) | 25,036 | 4,262 | 11 | 0 |
| GEMM INT4 FPGA (×1) | 8,848 | 15,118 | 4 | 0 |
| KVC Controller (추정) | ~2,000 | ~1,500 | 0 | ~4 |
| MoE Router (추정) | ~1,500 | ~800 | 4 | 0 |
| PCIe / 인터커넥트 (추정) | ~3,000 | ~2,000 | 0 | 2 |
| **합계** | **~40,384** | **~23,680** | **~19** | **~6** |
| **A7-100T 가용량** | 82,800 | 126,800 | 90 | 135 |
| **점유율** | **48.8%** | **18.7%** | **21.1%** | **4.4%** |

### 80% 목표 대비 여유

| 리소스 | 80% 목표 | 예상 사용 | 여유 |
|--------|---------|-----------|------|
| LUT | 66,240 | ~40,384 | **+25,856 (39%)** |
| FF | 101,440 | ~23,680 | **+77,760 (77%)** |
| DSP48E1 | 72 | ~19 | **+53 (74%)** |
| BRAM36 | 108 | ~6 | **+102 (94%)** |

→ **전체 ORBIT-G1이 Arty A7-100T 80% 이내에 충분히 들어감** ✅

---

## 3. 성능 추정

| 항목 | 값 |
|------|----|
| 목표 클럭 | 150 MHz |
| VPU throughput (256-wide, 1 op/cycle) | 256 × 150M = **38.4 GOPS** |
| GEMM INT4 (16×16 tile, 512 cycles) | 16×16×16 / 512 × 150M ≈ **1.2 GOPS** |
| 예상 최대 클럭 (Vivado 기준, 추정) | **120~160 MHz** |

> VPU는 단순 FSM + LUT, 크리티컬 패스 짧아서 150 MHz 달성 가능성 높음.
> GEMM은 INT8×INT4 shift-and-add 체인 깊이에 따라 100~130 MHz 예상.

---

## 4. 결론

| 항목 | 결과 |
|------|------|
| VPU FPGA 구현 가능 | ✅ LUT 30%, DSP 12% |
| GEMM INT4 FPGA 구현 가능 | ✅ DSP 1040→4개 (99.6% 감소) |
| 전체 ORBIT-G1 80% 이내 | ✅ LUT 49%, DSP 21% |
| 실물 검증 (FPGA 보드) | ❌ 미완료 (Vivado P&R 필요) |
| 타이밍 분석 (정확) | ❌ Yosys 미지원 → Vivado 필요 |

---

*Raw logs: `docs/yosys_vpu_synth.log`, `docs/yosys_gemm_synth.log`, `docs/yosys_gemm_fpga_synth.log`*
