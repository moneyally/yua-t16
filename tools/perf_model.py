#!/usr/bin/env python3
"""perf_model.py — 이 가속기의 성능 모델. 측정값과 가정값을 구분한다.

사용법:
    python3 tools/perf_model.py              # 전체 보고
    python3 tools/perf_model.py --json       # 기계 판독용

이 파일의 모든 숫자에는 출처 태그가 붙는다.

    MEASURED  이 저장소의 시뮬레이션/합성으로 직접 측정한 값.
              재현 명령이 주석에 있다.
    SOURCED   외부 공개 자료. 출처를 적는다.
    ASSUMED   측정하지 않은 값. 가정이고, 바꾸면 결과가 바뀐다.
              **동작 주파수가 여기 있다** — 배치·배선을 돌리지 않았다.
    DERIVED   위 값들로 계산한 값.

가정값으로 결론을 내지 않기 위해, 보고서는 ASSUMED 에 의존하는 줄을
따로 표시한다.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict

# ─────────────────────────────────────────────────────────────────────
# MEASURED — 이 저장소의 시뮬레이션
#
# 재현:
#   TILE_N=16 RD_DATA_W=128 WR_DATA_W=128 WGT_KEEP=1 \
#     python3.13 sim/cocotb/run_pipe_profile.py
#   TILE_N=32 RD_DATA_W=256 WR_DATA_W=256 \
#     TEST_MODULE=test_tile32 python3.13 sim/cocotb/run_gemm_top_cfg.py
#
# 정상상태 = 완료 펄스 사이 간격, 24개 연속 중 앞뒤 2개 제외.
# 측정 구간에서 간격이 한 사이클도 흔들리지 않았다.
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    name: str
    tile_n: int           # MAC 타일 한 변
    rd_bits: int          # DMA 읽기 폭
    wr_bits: int          # DMA 쓰기 폭
    kt: int               # 측정에 쓴 K-step 수
    cycles_per_op: float  # MEASURED 정상상태 디스크립터 간격
    weight_stationary: bool
    cores: int = 1        # 텐서 병렬 코어 수

    @property
    def macs(self) -> int:
        return self.cores * self.tile_n ** 2

    @property
    def act_bytes_per_desc(self) -> int:
        """디스크립터당 활성값 바이트. 브로드캐스트라 코어 수와 무관하다."""
        return self.kt * self.tile_n

    @property
    def intensity(self) -> float:
        """산술 강도 (MAC / 활성값바이트). 코어 C 개의 N×N 이면 C·N 이다.

        같은 MAC 수의 단일 타일은 한 변이 N√C 이라 강도가 N√C 다.
        즉 클러스터가 √C 배 유리하다 — 브로드캐스트 덕이다.
        """
        return self.macs_per_desc / self.act_bytes_per_desc

    @property
    def macs_per_desc(self) -> int:
        return self.macs * self.kt

    @property
    def macs_per_cycle(self) -> float:          # DERIVED
        return self.macs_per_desc / self.cycles_per_op

    @property
    def utilisation(self) -> float:             # DERIVED
        return self.macs_per_cycle / self.macs

    @property
    def ops_per_cycle(self) -> float:           # DERIVED (MAC = 2 OPS)
        return self.macs_per_cycle * 2


# 측정된 구성들. cycles_per_op 는 전부 MEASURED.
CONFIGS = [
    Config("N=16 128b, WS off", 16, 128, 128, 64, 131, False),
    Config("N=16 128b, WS on",  16, 128, 128, 64,  70, True),
    Config("N=16 256b, WS on",  16, 256, 256, 32,  38, True),
    Config("N=32 256b, WS on",  32, 256, 256, 128, 134, True),
    Config("N=64 512b, WS on",  64, 512, 512, 256, 262, True),
    # 텐서 병렬 클러스터 (rtl/gemm_cluster.sv). 활성값을 브로드캐스트하므로
    # 읽기 대역폭이 코어 수와 무관하다.
    #   TILE_N=32 RD_DATA_W=256 WR_DATA_W=<폭> CORES=<n> \
    #     python3.13 sim/cocotb/run_cluster.py
    Config("2코어 N=32 wr256",  32, 256, 256, 256, 262,   True, cores=2),
    Config("4코어 N=32 wr256",  32, 256, 256, 256, 588.5, True, cores=4),
    Config("4코어 N=32 wr512",  32, 256, 512, 256, 262,   True, cores=4),
]

# 최고 측정 구성과 FPGA 설계점.
# VC1902 는 DSP58 1,968 개다. 4코어×1,024 = 4,096 과 단일 N=64 = 4,096 은
# 둘 다 2.1배 초과이고, 2코어×1,024 = 2,048 도 넘는다. 따라서 단일
# VC1902 에 들어가는 최대는 N=32 단일 코어(1,024)다.
BEST = next(c for c in CONFIGS if c.name == "4코어 N=32 wr512")
FPGA_BEST = next(c for c in CONFIGS if c.name == "N=32 256b, WS on")


# MEASURED — Yosys, `abc -g simple` 일반 셀 기준.
#   yosys -p "read_verilog -sv rtl/mac_pe.sv; hierarchy -top mac_pe;
#             proc; opt; techmap; opt; abc -g simple; opt_clean; stat; ltp -noff"
MAC_PE_CELLS = 817
MAC_PE_LOGIC_DEPTH = 67     # 2입력 게이트 단수. 한 사이클 곱셈+누산의 대가.

# MEASURED — 가중치 트래픽. 재현: TEST_MODULE=test_int4_weights
#   Kt=64 에서 가중치 비트 INT8 64 → INT4 32 (정확히 절반).
WGT_BITS_SUPPORTED = (8, 4)

# ─────────────────────────────────────────────────────────────────────
# SOURCED — 외부 자료
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Device:
    name: str
    luts: int | None
    dsp: int | None
    sram_mb: float          # 온칩 SRAM (가중치를 담을 수 있는 양)
    dram_gb: float
    dram_gbps: float        # GB/s
    source: str


DEVICES = {
    # AMD 제품 페이지: DSP Engines 1,968 / LUTs 899,840.
    # EPCC 문서: PL 4.2MB BRAM + 16.3MB URAM.
    # VCK190 보드: PS DDR4 + PL DDR4 DIMM. 채널당 x64@3200 = 25.6 GB/s.
    "vck190": Device("VCK190 (VC1902)", 899_840, 1_968, 20.5, 16.0, 25.6,
                     "AMD 제품 페이지 + EPCC VCK5000 문서 + UG1366"),
    # H200 SXM. 세션 앞부분 조사 수치.
    "h200": Device("NVIDIA H200 SXM", None, None, 0.0, 141.0, 4_800.0,
                   "세션 앞부분 조사 (NVIDIA 공개 스펙)"),
    # Groq LPU 1세대. 벤더 주장이며 독립 벤치마크로 확인하지 못했다.
    "groq": Device("Groq LPU (1세대)", None, None, 230.0, 0.0, 80_000.0,
                   "Groq 자사 백서 (벤더 주장, 미확인)"),
    # Cerebras WSE-3. 벤더 주장.
    "wse3": Device("Cerebras WSE-3", None, None, 44_000.0, 0.0, 21_000_000.0,
                   "Cerebras 데이터시트 인용 2차 자료 (벤더 주장, 미확인)"),
}

# SOURCED — H200 피크 연산 (dense, sparsity 제외)
H200_INT8_TOPS = 1_979.0
H200_BF16_TFLOPS = 989.0

# ─────────────────────────────────────────────────────────────────────
# ASSUMED — 측정하지 않은 값
# ─────────────────────────────────────────────────────────────────────

# 동작 주파수. **이 저장소에서 측정된 값이 없다.**
# 배치·배선을 돌리지 않았고 (이 환경에 Vivado 가 없다), mac_pe 의
# 논리 단수 67 이 임계 경로다. 실제 Vivado 는 곱셈·누산을 DSP58 하드
# 블록으로 뽑을 것이므로 일반 셀 단수에서 주파수를 추정할 수 없다.
# 아래는 DSP58 기반 설계에서 흔히 보는 범위의 양 끝이다.
CLOCK_MHZ = {"보수적": 300.0, "낙관적": 600.0}

# ─────────────────────────────────────────────────────────────────────
# DERIVED
# ─────────────────────────────────────────────────────────────────────

def peak_tops(cfg: Config, clock_mhz: float) -> float:
    """INT8 TOPS. ASSUMED 주파수에 의존한다."""
    return cfg.ops_per_cycle * clock_mhz * 1e6 / 1e12


def array_cells(cfg: Config) -> int:
    """MAC 배열 셀 수. 측정된 PE 를 배열 크기로 환산한다.

    Yosys 0.33 은 unpacked array 포트를 파싱하지 못해 배열 전체를
    합성할 수 없다. 이 저장소가 PE 단위로 재는 이유이고
    spec/pe-fixed-point-redesign.md 에 같은 제약이 적혀 있다.
    """
    return cfg.macs * MAC_PE_CELLS


def decode_tokens_per_s(model_params_b: float, wgt_bits: int,
                        dram_gbps: float, batch: int = 1,
                        compute_tops: float | None = None) -> dict:
    """단일/배치 디코드 토큰 속도. 메모리와 연산 중 묶는 쪽을 고른다.

    디코드는 토큰마다 가중치 전체를 한 번 읽는다. 배치 B 면 그 한 번으로
    B 토큰을 처리하므로 메모리 쪽 속도가 B 배가 된다. 연산은 토큰당
    2·P OPS 이므로 배치에 비례해 늘어난다 — 즉 배치는 메모리 한계를
    밀어올리고 연산 한계는 그대로 둔다.
    """
    wgt_bytes = model_params_b * 1e9 * wgt_bits / 8
    mem_limit = dram_gbps * 1e9 / wgt_bytes * batch
    out = {"weight_gb": wgt_bytes / 1e9, "mem_limit_tok_s": mem_limit}
    if compute_tops is not None:
        ops_per_token = 2 * model_params_b * 1e9
        comp_limit = compute_tops * 1e12 / ops_per_token
        out["compute_limit_tok_s"] = comp_limit
        out["tok_s"] = min(mem_limit, comp_limit)
        out["bound_by"] = "메모리" if mem_limit < comp_limit else "연산"
    else:
        out["tok_s"] = mem_limit
        out["bound_by"] = "메모리"
    return out


def sram_resident_devices(model_params_b: float, wgt_bits: int,
                          sram_mb: float) -> float:
    """가중치를 전부 온칩 SRAM 에 올리려면 칩이 몇 개 필요한가.

    Groq / Cerebras 가 HBM 대역폭 벽을 넘은 방식이 이것이다 — 대역폭을
    키운 게 아니라 DRAM 을 임계 경로에서 뺐다. 대가는 용량이다.
    """
    wgt_mb = model_params_b * 1e9 * wgt_bits / 8 / 1e6
    return wgt_mb / sram_mb


def report() -> dict:
    out: dict = {"measured": {}, "derived": {}, "assumed": {}}

    print("=" * 74)
    print(" 성능 모델 — MEASURED / SOURCED / ASSUMED / DERIVED")
    print("=" * 74)

    # ── 1. 측정된 구성 ────────────────────────────────────────────
    print("\n[MEASURED] 정상상태 처리량 (cocotb + Verilator)")
    print(f"  {'구성':<20}{'코어':>4}{'MAC':>7}{'Kt':>5}{'사이클/op':>10}"
          f"{'MAC/사이클':>11}{'가동률':>8}{'강도':>9}")
    print("  " + "-" * 76)
    for c in CONFIGS:
        print(f"  {c.name:<20}{c.cores:>4}{c.macs:>7,}{c.kt:>5}"
              f"{c.cycles_per_op:>10.1f}{c.macs_per_cycle:>11.0f}"
              f"{c.utilisation*100:>7.1f}%{c.intensity:>7.0f} M/B")
        out["measured"][c.name] = {
            "kt": c.kt, "cycles_per_op": c.cycles_per_op,
            "macs_per_cycle": round(c.macs_per_cycle, 1),
            "utilisation": round(c.utilisation, 4),
        }

    print("\n  가동률이 N 과 함께 오르는 것은 Kt 가 같이 커져 디스크립터당")
    print("  고정 오버헤드(상태 전이 6 사이클)가 더 잘 분산되기 때문이다.")

    print(f"\n[MEASURED] mac_pe {MAC_PE_CELLS} 셀, 논리 단수 "
          f"{MAC_PE_LOGIC_DEPTH} (2입력 게이트)")
    print("[DERIVED ] MAC 배열 셀 (측정 PE × 배열 크기) 과 FPGA 적합성:")
    dev0 = DEVICES["vck190"]
    seen: set[int] = set()
    for c in sorted(CONFIGS, key=lambda c: c.macs):
        if c.macs in seen:
            continue
        seen.add(c.macs)
        fits = "들어감" if c.macs <= dev0.dsp else \
               f"DSP58 {c.macs/dev0.dsp:.1f}배 초과"
        shape = f"{c.cores}×{c.tile_n}²" if c.cores > 1 else f"{c.tile_n}²"
        print(f"           {shape:<8} MAC {c.macs:>5,}  "
              f"{array_cells(c):>10,} 셀   VC1902: {fits}")
    print("[MEASURED] INT4 가중치: 가중치 비트 정확히 절반, 결과 동일")

    print("\n[DERIVED ] 같은 MAC 수에서 클러스터 vs 단일 타일")
    single = next(c for c in CONFIGS if c.name == "N=64 512b, WS on")
    clus   = next(c for c in CONFIGS if c.name == "4코어 N=32 wr512")
    for c in (single, clus):
        print(f"           {c.name:<20} MAC {c.macs_per_desc:>9,}  "
              f"활성값 {c.act_bytes_per_desc:>6,}B  "
              f"강도 {c.intensity:>4.0f} MAC/B")
    print(f"           → 클러스터가 읽기 대역폭을 "
          f"{single.intensity/clus.intensity:.2f}배만 쓴다 "
          f"(= 1/√CORES). 브로드캐스트 덕이다.")

    # ── 2. 주파수 가정 ───────────────────────────────────────────
    print(f"\n[ASSUMED ] 동작 주파수 — **이 저장소에 측정값이 없다**")
    print("           배치·배선 미실행. 아래부터는 가정에 의존한다.")
    for label, mhz in CLOCK_MHZ.items():
        for tag, c in (("FPGA 최대", FPGA_BEST), ("ASIC 급", BEST)):
            t = peak_tops(c, mhz)
            print(f"           {label} {mhz:.0f} MHz, {tag} ({c.name})"
                  f" → {t:.2f} INT8 TOPS")
            out["assumed"][f"{tag}_{label}_{mhz:.0f}MHz_tops"] = round(t, 4)
    best = FPGA_BEST   # 아래 비교는 실제로 만들 수 있는 구성으로 한다

    # ── 3. H200 과의 격차 ────────────────────────────────────────
    dev = DEVICES["vck190"]
    h200 = DEVICES["h200"]
    print(f"\n[DERIVED ] H200 대비 (ASSUMED 주파수 의존)")
    print(f"  {'항목':<24}{'이 설계':>16}{'H200':>14}{'격차':>12}")
    print("  " + "-" * 68)
    for label, mhz in CLOCK_MHZ.items():
        t = peak_tops(best, mhz)
        print(f"  {'INT8 연산 @' + label:<24}{t:>13.2f} T{H200_INT8_TOPS:>13.0f} T"
              f"{H200_INT8_TOPS/t:>11,.0f}×")
    print(f"  {'메모리 대역폭':<24}{dev.dram_gbps:>11.1f} GB/s"
          f"{h200.dram_gbps:>9,.0f} GB/s{h200.dram_gbps/dev.dram_gbps:>11,.0f}×")
    print(f"  {'메모리 용량':<24}{dev.dram_gb:>13.0f} GB{h200.dram_gb:>12.0f} GB"
          f"{h200.dram_gb/dev.dram_gb:>11,.1f}×")
    out["derived"]["bandwidth_gap"] = round(h200.dram_gbps / dev.dram_gbps, 1)

    # ── 4. LLM 디코드 — 진짜 중요한 숫자 ─────────────────────────
    print("\n" + "=" * 74)
    print(" LLM 디코드 — 연산 격차는 여기서 나타나지 않는다")
    print("=" * 74)
    print("\n디코드는 토큰마다 가중치 전체를 한 번 읽는다. 양쪽 다 메모리에")
    print("묶이므로 실효 격차는 **연산비가 아니라 대역폭비**다.\n")

    model_b = 7.0
    tops_cons = peak_tops(best, CLOCK_MHZ["보수적"])
    print(f"  7B 모델, 배치 1")
    print(f"  {'구성':<28}{'가중치':>10}{'토큰/s':>11}{'묶는 쪽':>10}")
    print("  " + "-" * 62)
    rows = []
    for wbits in WGT_BITS_SUPPORTED:
        r = decode_tokens_per_s(model_b, wbits, dev.dram_gbps, 1, tops_cons)
        print(f"  {'이 설계 INT' + str(wbits):<28}{r['weight_gb']:>8.1f} GB"
              f"{r['tok_s']:>11.1f}{r['bound_by']:>10}")
        rows.append(("this", wbits, r))
        rh = decode_tokens_per_s(model_b, wbits, h200.dram_gbps, 1,
                                 H200_INT8_TOPS)
        print(f"  {'H200 INT' + str(wbits):<28}{rh['weight_gb']:>8.1f} GB"
              f"{rh['tok_s']:>11.1f}{rh['bound_by']:>10}")
        rows.append(("h200", wbits, rh))
        print(f"  {'  → 격차':<28}{'':>11}"
              f"{rh['tok_s']/r['tok_s']:>10.0f}×")
    out["derived"]["decode_7b"] = [
        {"who": w, "wgt_bits": b, **{k: round(v, 3) if isinstance(v, float) else v
                                     for k, v in r.items()}}
        for w, b, r in rows
    ]

    print("\n  INT8 → INT4 로 토큰 속도가 정확히 두 배다 (가중치 바이트가 절반).")
    print("  격차가 연산비(수천 배)가 아니라 대역폭비에 수렴한다.")

    # 배치
    print(f"\n  배치가 메모리 한계를 밀어올린다 (이 설계, 7B INT4, "
          f"{CLOCK_MHZ['보수적']:.0f} MHz 가정)")
    print(f"  {'배치':>6}{'메모리 한계':>14}{'연산 한계':>13}{'실제':>10}{'묶는 쪽':>10}")
    print("  " + "-" * 56)
    for batch in (1, 2, 4, 8, 16, 32):
        r = decode_tokens_per_s(model_b, 4, dev.dram_gbps, batch, tops_cons)
        print(f"  {batch:>6}{r['mem_limit_tok_s']:>14.1f}"
              f"{r['compute_limit_tok_s']:>13.1f}{r['tok_s']:>10.1f}"
              f"{r['bound_by']:>10}")
    print("\n  배치를 키우면 메모리 한계가 올라가다 연산에 부딪힌다.")
    print("  그 지점이 이 칩의 설계 균형점이고, 그 뒤로는 MAC 을 늘려야 한다.")

    # ── 5. SRAM 상주 — 대역폭 벽을 넘는 유일한 길 ────────────────
    print("\n" + "=" * 74)
    print(" SRAM 상주 — Groq / Cerebras 가 간 길")
    print("=" * 74)
    print("\n둘 다 DRAM 대역폭을 키워서 이긴 게 아니라, DRAM 을 임계 경로에서")
    print("뺐다. 대가는 용량이다.\n")
    print(f"  {'장치':<22}{'온칩 SRAM':>12}{'대역폭':>14}{'7B INT4 에 필요한 수':>22}")
    print("  " + "-" * 70)
    for key in ("vck190", "groq", "wse3"):
        d = DEVICES[key]
        if d.sram_mb <= 0:
            continue
        n = sram_resident_devices(model_b, 4, d.sram_mb)
        print(f"  {d.name:<22}{d.sram_mb:>9,.1f} MB{d.dram_gbps:>11,.0f} GB/s"
              f"{n:>20,.1f} 개")
        out["derived"].setdefault("sram_resident", {})[key] = round(n, 2)

    print(f"\n  이 설계를 Groq 처럼 쓰려면 VCK190 "
          f"{sram_resident_devices(model_b, 4, dev.sram_mb):,.0f} 개가 필요하다.")
    print("  이것이 '데이터센터급' 과의 진짜 거리다 — RTL 이 아니라 실리콘 양이다.")

    print("\n" + "=" * 74)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="JSON 으로 출력")
    a = ap.parse_args()
    r = report()
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
