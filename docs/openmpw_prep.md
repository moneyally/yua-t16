# ORBIT-G1 OpenMPW 테이프아웃 준비 가이드

> Google + Efabless 무료 ASIC 셔틀로 YUA-T16 칩 찍기

---

## 1. OpenMPW 개요

**OpenMPW(Open Multi-Project Wafer)** 는 Google, Efabless, SkyWater Technology가 공동으로 운영하는 무료 ASIC 테이프아웃 셔틀 프로그램이다.

| 항목 | 내용 |
|------|------|
| 공정 | SKY130B (SkyWater 130nm CMOS) |
| 비용 | **무료** (오픈소스 프로젝트 대상) |
| 주기 | 약 3~4개월마다 셔틀 오픈 |
| 신청 URL | https://efabless.com/open_shuttle_program |
| 셔틀 알림 | https://platform.efabless.com/shuttle_requests |
| EDA 플로우 | OpenLane (오픈소스 RTL-to-GDSII) |
| Harness | Caravel SoC (Wishbone + Logic Analyzer 포함) |

오픈소스 라이선스(Apache 2.0, MIT 등)로 공개된 프로젝트면 신청 가능하다. ORBIT-G1은 Apache 2.0이므로 조건 충족.

---

## 2. ORBIT-G1에서 OpenMPW에 적합한 컴포넌트

### SKY130 면적 한계

SKY130B 공정에서 OpenMPW에 허용되는 사용자 영역은 **약 10mm²** 이다 (Caravel harness 기준 `user_project_wrapper`).

ORBIT-G1 전체(GEMM + VPU 256-wide + KVC + MoE)를 ASIC으로 구현하면 면적 초과가 예상된다. 따라서 **컴포넌트를 분리해서 단계적으로 테이프아웃**하는 전략을 택한다.

### 후보 컴포넌트

| 후보 | 예상 면적 | 우선순위 | 비고 |
|------|-----------|----------|------|
| **GEMM INT4 타일 (16×16)** | ~1~2mm² | ★★★ 최우선 | DSP 최적화 완료, 면적 여유 |
| VPU 32-wide (축소) | ~2~3mm² | ★★☆ | 256-wide는 면적 초과 예상 |
| GEMM + VPU 32-wide 조합 | ~3~5mm² | ★★☆ | 10mm² 이내 가능성 높음 |
| KVC Controller | ~0.5mm² | ★☆☆ | 단독으론 의미 작음 |

**1차 목표**: GEMM INT4 타일 단독 테이프아웃
**2차 목표**: GEMM + VPU 32-wide 조합

### SKY130 성능 추정

130nm 공정은 FPGA보다 느리지만 커스텀 레이아웃으로 밀도를 높일 수 있다.

| 항목 | 추정값 |
|------|--------|
| 최대 클럭 | ~100 MHz (SKY130 표준 셀 기준) |
| GEMM 16×16 INT8 @ 100MHz | 16×16×16 MAC / 256 cycles × 100M ≈ **1.6 GOPS** |
| VPU 32-wide @ 100MHz | 32 × 100M = **3.2 GOPS** |
| GEMM 타일 면적 (INT4) | **~1~2mm²** 예상 |

> 참고: FPGA 대비 칩 전력이 10~100배 낮다. 같은 연산을 훨씬 적은 전력으로 수행.

---

## 3. 준비 체크리스트

### 3-1. RTL 정리

- [ ] Verilog 2005 호환성 검토 (`unique case`, `$clog2` 등 제거)
- [ ] `include` 경로 정리 (상대경로 → OpenLane 플로우 호환)
- [ ] 타이밍 크리티컬 패스 식별 (100 MHz 목표 기준)
- [ ] `gemm_int4_fpga.sv` → SKY130 표준 셀용 포팅 (`mul_i8_i4` LUT → 표준 셀 곱셈기)

### 3-2. OpenLane 디렉토리 구조 설정

```
yua-t16/
├── openlane/
│   └── gemm_int4/
│       ├── config.json          # OpenLane 2.x 설정
│       ├── src/
│       │   └── gemm_int4.v      # 단일 플랫 파일 또는 include 목록
│       └── pin_order.cfg        # 핀 배치 (선택)
```

### 3-3. `config.json` 작성 (OpenLane 2.x)

```json
{
    "DESIGN_NAME": "gemm_int4",
    "VERILOG_FILES": ["dir::src/gemm_int4.v"],
    "CLOCK_PORT": "clk",
    "CLOCK_PERIOD": 10.0,
    "DIE_AREA": "0 0 500 500",
    "FP_CORE_UTIL": 45,
    "PL_TARGET_DENSITY": 0.4,
    "SYNTH_STRATEGY": "DELAY 0",
    "scl::sky130_fd_sc_hd": {
        "CLOCK_PERIOD": 10.0,
        "FP_CORE_UTIL": 45
    }
}
```

> `CLOCK_PERIOD: 10.0` = 100 MHz. SKY130HD 라이브러리 기준 달성 가능한 타겟.

### 3-4. SKY130 표준 셀 매핑 확인

- [ ] `sky130_fd_sc_hd` (High Density) 라이브러리 사용
- [ ] DSP가 없으므로 모든 곱셈은 표준 셀 기반 → `mul_i8_i4` shift-and-add 방식 유리
- [ ] SRAM이 필요하면 `sky130_sram_1kbyte_1rw1r_32x256_8` 매크로 사용

### 3-5. Caravel SoC Harness 연결

Efabless의 Caravel은 RISC-V SoC로 사용자 IP를 감싸는 표준 harness다.

- [ ] `caravel_user_project` 포크
- [ ] `user_project_wrapper.v`에 `gemm_int4` 인스턴스 연결
- [ ] Wishbone 버스 인터페이스 구현 (레지스터 맵 설계)
- [ ] Logic Analyzer 38핀 연결 (디버그용)
- [ ] `io_oeb` (Output Enable) 설정

Wishbone 인터페이스 레지스터 맵 예시:

| 주소 (오프셋) | 레지스터 | 역할 |
|---------------|----------|------|
| 0x00 | CTRL | 시작/리셋 |
| 0x04 | STATUS | 완료/에러 플래그 |
| 0x08~0x87 | ACT_BUF | 16개 INT8 activation 입력 |
| 0x88~0xC7 | WGT_BUF | 16×16 INT4 weight 입력 |
| 0xC8~0x147 | OUT_BUF | 16개 INT32 결과 출력 |

### 3-6. LVS/DRC 통과 확인

- [ ] OpenLane 플로우 전체 실행 (`flow.tcl` 또는 `openlane` CLI)
- [ ] DRC 0 violations 확인 (`magic` DRC 체크)
- [ ] LVS 통과 확인 (`netgen`)
- [ ] Antenna 위반 없음 확인
- [ ] 최종 GDS 생성 확인 (`klayout` 뷰어로 검토)

### 3-7. 신청서 작성

- [ ] 프로젝트 설명 (영어, 500자 내외)
- [ ] 사용 목적: LLM 추론 가속기 연구, 오픈소스
- [ ] 라이선스: Apache 2.0
- [ ] GitHub URL: https://github.com/moneyally/yua-t16
- [ ] 연락처 및 소속

---

## 4. 빠른 시작 명령어

```bash
# 1. OpenLane 설치 (Docker 기반, 권장)
git clone https://github.com/The-OpenROAD-Project/OpenLane
cd OpenLane
make             # Docker 이미지 빌드 (~20분)
make test        # 기본 테스트 (spm 디자인으로 검증)

# 2. caravel_user_project 포크 및 설정
git clone https://github.com/efabless/caravel_user_project
cd caravel_user_project
make setup       # PDK + OpenLane 환경 설정

# 3. PDK 직접 설치 (Docker 없이)
git clone https://github.com/google/skywater-pdk
cd skywater-pdk
git submodule update --init libraries/sky130_fd_sc_hd/latest
python sky130/utils/build_rules.py  # 라이브러리 빌드

# 4. OpenLane 플로우 실행 (gemm_int4 디자인)
cd /home/dmsal020813/project/yua-t16
./OpenLane/flow.tcl -design openlane/gemm_int4 -tag first_run

# 5. 결과 확인
# logs/synthesis/1-synthesis.log  → 합성 결과
# logs/routing/               → 라우팅 결과
# results/final/gds/          → 최종 GDS
```

---

## 5. 다음 셔틀 대비 타임라인

| 단계 | 목표 | 예상 기간 |
|------|------|-----------|
| **지금** | RTL 정리 + Verilog 2005 호환 | 1~2일 |
| **단기** | OpenLane 설치 + config 작성 | 2~3일 |
| **중기** | Caravel 연결 + DRC/LVS 통과 | 1~2주 |
| **셔틀 오픈 시** | 신청서 제출 + GDS 업로드 | 1일 |

셔틀 오픈 알림을 받으려면:
- https://platform.efabless.com/shuttle_requests 에서 계정 생성 후 알림 신청
- Efabless Discord: https://discord.gg/efabless

---

## 6. 참고 자료

- OpenLane 문서: https://openlane.readthedocs.io
- Caravel 문서: https://caravel-harness.readthedocs.io
- SKY130 PDK: https://github.com/google/skywater-pdk
- OpenMPW 신청: https://efabless.com/open_shuttle_program
- 합성 결과: `docs/synthesis_report.md`
- 프로젝트 RTL: `rtl/`

---

*작성일: 2026-03-13*
