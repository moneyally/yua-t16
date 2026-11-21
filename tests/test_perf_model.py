"""tools/perf_model.py 가 측정값과 일치하는지, 모델 식이 맞는지.

성능 모델은 팔 수 있는 산출물이다. 숫자가 조용히 틀어지면 안 된다.
여기서는 두 가지를 단언한다.

  1) 모델에 적힌 cycles_per_op 가 시뮬레이션 측정값과 같다.
     (측정은 sim/cocotb 쪽에서 하고, 여기서는 상수가 바뀌지 않았는지 본다)
  2) 유도 식이 손으로 계산한 값과 같다.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import perf_model as pm  # noqa: E402


# (이름, Kt, 사이클/op, MAC/사이클, 가동률%) — sim/cocotb 로 측정한 값.
# 재현: spec/memory-bandwidth-analysis.md 와 tools/perf_model.py 주석 참조.
MEASURED = {
    "N=16 128b, WS off": (64, 131, 125, 48.9),
    "N=16 128b, WS on":  (64,  70, 234, 91.4),
    "N=16 256b, WS on":  (32,  38, 216, 84.2),
    "N=32 256b, WS on":  (128, 134, 978, 95.5),
    "N=64 512b, WS on":  (256, 262, 4002, 97.7),
    # 텐서 병렬 클러스터 (run_cluster.py)
    "2코어 N=32 wr256":  (256, 262, 2001, 97.7),
    "4코어 N=32 wr256":  (256, 588.5, 1782, 43.5),
    "4코어 N=32 wr512":  (256, 262, 4002, 97.7),
}


def test_every_config_is_pinned_to_a_measurement():
    """모델의 구성 전부가 측정 기록을 가져야 한다."""
    names = {c.name for c in pm.CONFIGS}
    assert names == set(MEASURED), (
        f"모델에만 있는 구성: {names - set(MEASURED)}, "
        f"기록에만 있는 구성: {set(MEASURED) - names}")


@pytest.mark.parametrize("cfg", pm.CONFIGS, ids=lambda c: c.name)
def test_derived_values_match_measurement(cfg):
    kt, cyc, macs_cyc, util = MEASURED[cfg.name]
    assert cfg.kt == kt
    assert cfg.cycles_per_op == cyc
    assert round(cfg.macs_per_cycle) == macs_cyc
    assert cfg.utilisation * 100 == pytest.approx(util, abs=0.05)
    # MAC 하나가 OPS 둘
    assert cfg.ops_per_cycle == pytest.approx(cfg.macs_per_cycle * 2)


def test_tile_scaling_is_quadratic_in_macs():
    """균형 구성(읽기 폭 = K-step 폭)에서 N 을 2배 하면 처리량이 4배.

    MAC 은 N², 피연산자는 N 이라 산술 강도가 N 에 비례한다. 가동률이
    같이 오르므로 비가 4배를 조금 넘는다.
    """
    by_n = {c.tile_n: c for c in pm.CONFIGS
            if c.weight_stationary and c.cores == 1
            and c.rd_bits == c.tile_n * 8}
    assert set(by_n) == {16, 32, 64}, f"예상 밖 구성: {sorted(by_n)}"
    for lo, hi in ((16, 32), (32, 64)):
        assert by_n[hi].macs == 4 * by_n[lo].macs
        ratio = by_n[hi].macs_per_cycle / by_n[lo].macs_per_cycle
        assert 4.0 <= ratio <= 4.3, f"N={lo}→{hi} 처리량비 {ratio:.2f}"
    # 가동률이 단조 증가해야 한다
    utils = [by_n[n].utilisation for n in (16, 32, 64)]
    assert utils == sorted(utils), f"가동률이 단조가 아니다: {utils}"


def test_fpga_fit_boundary():
    """단일 N=32 는 VC1902 DSP58 예산에 들어가고 그보다 큰 것은 넘는다."""
    dsp = pm.DEVICES["vck190"].dsp
    assert pm.FPGA_BEST.macs <= dsp, "FPGA 설계점이 DSP 예산을 넘는다"
    for c in pm.CONFIGS:
        if c.macs > pm.FPGA_BEST.macs:
            assert c.macs > dsp, \
                f"{c.name} 이 FPGA_BEST 보다 큰데 예산에 들어간다"


def test_cluster_beats_single_tile_on_intensity():
    """같은 MAC 수에서 클러스터의 산술 강도가 √CORES 배 높다.

    활성값을 브로드캐스트하므로 읽기 바이트가 코어 수와 무관하다.
    이것이 "코어를 늘릴까 타일을 키울까" 의 답이다.
    """
    single = next(c for c in pm.CONFIGS if c.name == "N=64 512b, WS on")
    clus = next(c for c in pm.CONFIGS if c.name == "4코어 N=32 wr512")
    assert clus.macs == single.macs, "비교 대상의 MAC 수가 달라졌다"
    assert clus.macs_per_cycle == pytest.approx(single.macs_per_cycle, rel=1e-3)
    assert clus.intensity == pytest.approx(2 * single.intensity), \
        f"강도 {clus.intensity} vs {single.intensity} — √4 = 2배여야 한다"
    # 일반식: 코어 C 개의 N×N 은 강도 C·N
    assert clus.intensity == pytest.approx(clus.cores * clus.tile_n)


def test_cluster_scales_linearly_in_cores():
    """코어 수에 선형. 쓰기 폭이 따라가야 성립한다."""
    one = next(c for c in pm.CONFIGS if c.name == "N=32 256b, WS on")
    two = next(c for c in pm.CONFIGS if c.name == "2코어 N=32 wr256")
    four = next(c for c in pm.CONFIGS if c.name == "4코어 N=32 wr512")
    assert two.macs_per_cycle / one.macs_per_cycle == pytest.approx(2.05, abs=0.05)
    assert four.macs_per_cycle / two.macs_per_cycle == pytest.approx(2.0, abs=0.05)


def test_narrow_write_channel_binds_four_cores():
    """256비트 쓰기로는 4코어가 쓰기에 묶인다. 그 사실이 기록돼 있어야 한다."""
    narrow = next(c for c in pm.CONFIGS if c.name == "4코어 N=32 wr256")
    wide = next(c for c in pm.CONFIGS if c.name == "4코어 N=32 wr512")
    assert narrow.utilisation < 0.5, "쓰기 병목이 사라졌다면 기록을 고쳐야 한다"
    assert wide.utilisation > 0.95
    # 균형 조건: Kt == CORES · N² · 4 / (WR_DATA_W/8)
    need = narrow.cores * narrow.tile_n ** 2 * 4 // (narrow.wr_bits // 8)
    assert need > narrow.kt, "256비트에서 균형 Kt 가 측정 Kt 를 넘어야 한다"
    need_wide = wide.cores * wide.tile_n ** 2 * 4 // (wide.wr_bits // 8)
    assert need_wide == wide.kt, "512비트에서 균형이 맞아야 한다"


def test_int4_doubles_decode_rate():
    """가중치 비트를 반으로 줄이면 메모리에 묶인 토큰 속도가 두 배."""
    bw = pm.DEVICES["vck190"].dram_gbps
    r8 = pm.decode_tokens_per_s(7.0, 8, bw)
    r4 = pm.decode_tokens_per_s(7.0, 4, bw)
    assert r4["tok_s"] == pytest.approx(2 * r8["tok_s"])
    assert r8["bound_by"] == "메모리"


def test_decode_gap_equals_bandwidth_ratio_when_both_memory_bound():
    """양쪽이 메모리에 묶이면 토큰 속도비 = 대역폭비. 연산 격차는 안 나타난다."""
    this = pm.DEVICES["vck190"]
    h200 = pm.DEVICES["h200"]
    a = pm.decode_tokens_per_s(7.0, 4, this.dram_gbps)
    b = pm.decode_tokens_per_s(7.0, 4, h200.dram_gbps)
    assert b["tok_s"] / a["tok_s"] == pytest.approx(
        h200.dram_gbps / this.dram_gbps)


def test_batching_crosses_over_to_compute_bound():
    """배치를 키우면 메모리 한계가 올라가 연산 한계에 부딪힌다."""
    bw = pm.DEVICES["vck190"].dram_gbps
    tops = pm.peak_tops(pm.FPGA_BEST, pm.CLOCK_MHZ["보수적"])
    bounds = [pm.decode_tokens_per_s(7.0, 4, bw, b, tops)["bound_by"]
              for b in (1, 2, 4, 8, 16, 32)]
    assert bounds[0] == "메모리", "배치 1 은 메모리에 묶여야 한다"
    assert bounds[-1] == "연산", "배치 32 는 연산에 묶여야 한다"
    # 한 번만 넘어가야 한다 (단조)
    assert bounds == sorted(bounds, key=lambda x: x != "메모리")


def test_sram_resident_needs_many_devices():
    """SRAM 상주는 실리콘 양의 문제다 — 숫자가 조용히 작아지면 안 된다."""
    n = pm.sram_resident_devices(7.0, 4, pm.DEVICES["vck190"].sram_mb)
    assert n > 100, f"VC1902 {n:.0f}개 — 7B INT4 에는 100개 이상이어야 한다"
    g = pm.sram_resident_devices(7.0, 4, pm.DEVICES["groq"].sram_mb)
    assert 10 < g < 20


def test_clock_is_flagged_as_assumed():
    """주파수가 가정값임이 코드에 남아 있어야 한다."""
    src = (Path(pm.__file__)).read_text()
    i = src.index("CLOCK_MHZ")
    assert "측정" in src[max(0, i - 900):i], \
        "CLOCK_MHZ 앞에 미측정임을 밝히는 주석이 있어야 한다"


def test_report_runs_and_returns_structure():
    r = pm.report()
    assert set(r) == {"measured", "derived", "assumed"}
    assert len(r["measured"]) == len(pm.CONFIGS)
    assert r["derived"]["bandwidth_gap"] > 100
