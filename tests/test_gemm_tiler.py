"""tools/gemm_tiler.py — 타일러가 RTL 과 같은 가정을 하는지, 식이 맞는지.

타일러가 틀리면 RTL 이 맞아도 결과가 틀린다. 그리고 그 틀림은 조용하다
(주소가 어긋나면 0 이 나오고, 플래그가 어긋나면 엉뚱한 가중치를 쓴다).
sim/cocotb/test_gemm_e2e.py 가 실제 RTL 과 대조하고, 여기서는 식과
불변식을 본다.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.gemm_tiler import (  # noqa: E402
    FLAG_ACC_KEEP, FLAG_NO_WB, FLAG_WGT_INT4, FLAG_WGT_KEEP,
    GemmShape, TileConfig, compare_orderings, tile_gemm,
)


def test_flags_match_rtl():
    """플래그 값이 rtl/ctrl_fsm.sv 의 FLAG_* 와 같아야 한다."""
    import re
    src = (Path(__file__).resolve().parents[1] / "rtl" / "ctrl_fsm.sv").read_text()
    rtl_bits = dict(re.findall(r"localparam int (FLAG_\w+)\s*=\s*(\d+);", src))
    for name, bit in (("FLAG_WGT_KEEP", 0), ("FLAG_WGT_INT4", 1),
                      ("FLAG_ACC_KEEP", 2), ("FLAG_NO_WB", 3)):
        assert name in rtl_bits, f"{name} 가 RTL 에 없다"
        assert int(rtl_bits[name]) == bit, \
            f"{name}: RTL 비트 {rtl_bits[name]}, 타일러 {bit}"
    assert FLAG_WGT_KEEP == 1 << 0
    assert FLAG_WGT_INT4 == 1 << 1
    assert FLAG_ACC_KEEP == 1 << 2
    assert FLAG_NO_WB == 1 << 3


def test_covers_every_output_tile_exactly_once():
    """출력 타일을 빠뜨리거나 겹치면 안 된다."""
    T, C = 32, 2
    cfg = TileConfig(tile_n=T, cores=C, max_kt=256)
    shape = GemmShape(M=3 * T, K=128, N=2 * T * C)
    plan = tile_gemm(shape, cfg)
    seen = {}
    for d in plan.descs:
        seen.setdefault((d.ib, d.jb), []).append(d.kb)
    assert len(seen) == 3 * 2, f"출력 타일 {len(seen)}개, 기대 6개"
    for key, kbs in seen.items():
        assert sorted(kbs) == list(range(len(kbs))), \
            f"{key} 의 K 조각이 {kbs} — 0..n-1 이어야 한다"


def test_weight_stationary_loads_once_per_band():
    """밴드 하나에 가중치 적재 한 번."""
    T, C = 32, 2
    cfg = TileConfig(tile_n=T, cores=C, max_kt=256)
    n_ib, n_jb = 4, 3
    plan = tile_gemm(GemmShape(M=n_ib * T, K=64, N=n_jb * T * C), cfg)
    assert plan.weight_loads == n_jb, \
        f"적재 {plan.weight_loads}회, 밴드 {n_jb}개여야 한다"
    assert len(plan.descs) == n_ib * n_jb


def test_single_row_tile_cannot_reuse():
    """행 타일이 하나면 재사용할 상대가 없다 — 절감 1.00×."""
    cfg = TileConfig(tile_n=32, cores=2, max_kt=4096)
    cmp = compare_orderings(GemmShape(M=32, K=4096, N=256), cfg)
    assert cmp["saving"] == pytest.approx(1.0), \
        "행 타일 1개인데 절감이 생겼다 — 식이 틀렸다"


def test_reuse_needs_m_at_least_two_tiles():
    """weight-stationary 는 M >= 2·tile_n 이어야 효과가 난다."""
    cfg = TileConfig(tile_n=32, cores=2, max_kt=4096)
    s1 = compare_orderings(GemmShape(M=32, K=4096, N=256), cfg)["saving"]
    s2 = compare_orderings(GemmShape(M=64, K=4096, N=256), cfg)["saving"]
    s8 = compare_orderings(GemmShape(M=256, K=4096, N=256), cfg)["saving"]
    assert s1 == pytest.approx(1.0)
    assert s2 > s1, "M 을 2타일로 늘렸는데 절감이 안 생긴다"
    assert s8 > s2, "M 을 더 늘렸는데 절감이 안 늘어난다"


def test_decode_padding_efficiency_is_one_over_tile():
    """M=1 이면 패딩 효율이 1/tile_n 이다 — 디코드의 구조적 한계."""
    for T in (16, 32, 64):
        cfg = TileConfig(tile_n=T, cores=2, max_kt=4096)
        plan = tile_gemm(GemmShape(M=1, K=4096, N=T * 2), cfg)
        assert plan.pad_efficiency == pytest.approx(1 / T), \
            f"T={T}: 패딩 효율 {plan.pad_efficiency:.4f}, 기대 {1/T:.4f}"


def test_ksplit_flag_sequence():
    """K 분할 플래그 수열이 RTL 의 요구와 맞아야 한다.

    첫 조각 ACC_KEEP=0 NO_WB=1, 중간 둘 다 1, 마지막 ACC_KEEP=1 NO_WB=0.
    """
    cfg = TileConfig(tile_n=32, cores=1, max_kt=64)
    plan = tile_gemm(GemmShape(M=32, K=200, N=32), cfg)
    chunks = [d for d in plan.descs if (d.ib, d.jb) == (0, 0)]
    assert len(chunks) == 4, f"K=200, max_kt=64 → 4조각, 실제 {len(chunks)}"
    assert sum(d.kt for d in chunks) == 200
    first, *mid, last = chunks
    assert not (first.flags & FLAG_ACC_KEEP), "첫 조각이 누산을 이어받는다"
    assert first.flags & FLAG_NO_WB, "첫 조각이 쓴다"
    for d in mid:
        assert d.flags & FLAG_ACC_KEEP and d.flags & FLAG_NO_WB
    assert last.flags & FLAG_ACC_KEEP, "마지막 조각이 누산을 버린다"
    assert not (last.flags & FLAG_NO_WB), "마지막 조각이 안 쓴다"


def test_ksplit_gives_up_weight_reuse():
    """K > max_kt 면 가중치 재사용이 사라진다 — 누산기가 한 벌이라서.

    이것은 제약이고, 문서화된 사실이다. 조용히 바뀌면 안 된다.
    """
    cfg = TileConfig(tile_n=32, cores=2, max_kt=256)
    plan = tile_gemm(GemmShape(M=128, K=1024, N=256), cfg)
    assert "누산기가 한 벌" in plan.strategy
    assert plan.weight_loads == len(plan.descs), \
        "K 분할인데 가중치를 재사용한다 — RTL 이 지원하지 않는다"


def test_int4_flag_propagates():
    cfg = TileConfig(tile_n=32, cores=1, max_kt=256, wgt_bits=4)
    plan = tile_gemm(GemmShape(M=64, K=128, N=32), cfg)
    assert all(d.flags & FLAG_WGT_INT4 for d in plan.descs)
    # 가중치 바이트가 INT8 의 절반
    plan8 = tile_gemm(GemmShape(M=64, K=128, N=32),
                      TileConfig(tile_n=32, cores=1, max_kt=256, wgt_bits=8))
    assert plan.wgt_bytes * 2 == plan8.wgt_bytes


def test_activation_bytes_shrink_with_cores():
    """활성값 트래픽이 코어 수에 **반비례**한다 — 브로드캐스트의 이득.

    디스크립터 하나가 읽는 활성값은 코어 수와 무관하다(브로드캐스트).
    그런데 코어가 많으면 한 디스크립터가 더 넓은 열 밴드를 덮으므로
    디스크립터 수가 줄고, 총 활성값 트래픽이 1/코어수 로 줄어든다.

    이것이 "코어를 늘릴까 타일을 키울까" 에서 코어가 이기는 이유다.
    """
    shape = GemmShape(M=64, K=128, N=256)
    by_cores = {}
    for cores in (1, 2, 4):
        cfg = TileConfig(tile_n=32, cores=cores, max_kt=256)
        plan = tile_gemm(shape, cfg)
        by_cores[cores] = plan.act_bytes
        # 디스크립터당 활성값은 코어 수와 무관
        per_desc = {d.kt * cfg.tile_n for d in plan.descs}
        assert per_desc == {128 * 32}, f"cores={cores}: {per_desc}"
    assert by_cores[1] == 2 * by_cores[2] == 4 * by_cores[4], \
        f"코어 수에 반비례하지 않는다: {by_cores}"


def test_macs_issued_never_below_useful():
    """발행 MAC 이 필요 MAC 보다 적으면 계산을 빼먹은 것이다."""
    cfg = TileConfig(tile_n=32, cores=2, max_kt=256)
    for M, K, N in ((1, 7, 3), (33, 100, 65), (64, 256, 128), (100, 300, 200)):
        plan = tile_gemm(GemmShape(M=M, K=K, N=N), cfg)
        assert plan.macs_issued >= plan.macs_useful
        assert 0 < plan.pad_efficiency <= 1.0


def test_shape_validation():
    for bad in ({"M": 0, "K": 1, "N": 1}, {"M": 1, "K": -1, "N": 1}):
        with pytest.raises(ValueError):
            GemmShape(**bad)
