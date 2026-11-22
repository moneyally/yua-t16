"""호스트 디스크립터 패커가 RTL 과 같은 바이트를 만드는가.

이번 작업으로 RTL 에 플래그 네 개(weight-stationary / INT4 / K 분할)와
클러스터 스트라이드 두 개가 생겼다. 호스트 패커가 그것을 못 만들면
기능 전체가 cocotb 테스트에서만 닿고 실제 호스트 경로에서는 쓸 수 없다.

여기서 단언하는 것:
  1) DescFlag 비트 위치가 rtl/ctrl_fsm.sv 의 FLAG_* 와 같다.
  2) 오프셋이 RTL 이 읽는 위치와 같다.
  3) 패킹 → 언패킹 라운드트립이 보존되고 CRC 가 맞는다.
  4) 플래그/스트라이드가 CRC 에 포함된다 (빠지면 변조가 통과한다).
  5) 타일러가 자기 복사본이 아니라 SSOT 를 쓴다.
"""
import re
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.orbit_desc import (  # noqa: E402
    crc8, pack_descriptor, pack_gemm, unpack_descriptor,
)
from tools.orbit_mmio_map import (  # noqa: E402
    DESC_FLAGS_OFF, DESC_KT_OFF, DESC_OUT_STRIDE_OFF, DESC_SIZE,
    DESC_WGT_STRIDE_OFF, DescFlag, Opcode,
)

CTRL_FSM = (ROOT / "rtl" / "ctrl_fsm.sv").read_text()


def test_flag_bits_match_rtl():
    """DescFlag 와 RTL 의 FLAG_* 가 같아야 한다. 어긋나면 조용히 오동작."""
    rtl = {m[0]: int(m[1]) for m in
           re.findall(r"localparam int (FLAG_\w+)\s*=\s*(\d+);", CTRL_FSM)}
    want = {
        "FLAG_WGT_KEEP": DescFlag.WGT_KEEP,
        "FLAG_WGT_INT4": DescFlag.WGT_INT4,
        "FLAG_ACC_KEEP": DescFlag.ACC_KEEP,
        "FLAG_NO_WB":    DescFlag.NO_WB,
    }
    assert set(rtl) == set(want), \
        f"RTL 플래그 {sorted(rtl)}, 호스트 {sorted(want)}"
    for name, flag in want.items():
        assert 1 << rtl[name] == int(flag), \
            f"{name}: RTL 비트 {rtl[name]} → {1 << rtl[name]}, 호스트 {int(flag)}"


def test_rtl_reads_flags_from_byte_one():
    """RTL 이 flags 를 디스크립터 바이트 1 에서 읽어야 한다."""
    assert DESC_FLAGS_OFF == 1
    assert re.search(r"flags_r\s*<=\s*latched\[1\]", CTRL_FSM), \
        "ctrl_fsm 이 latched[1] 을 flags 로 쓰지 않는다"


def test_cluster_reads_strides_from_rtl_offsets():
    """gemm_cluster 가 읽는 스트라이드 바이트 위치와 맞아야 한다."""
    cluster = (ROOT / "rtl" / "gemm_cluster.sv").read_text()
    # wgt_stride: 바이트 44..47, out_stride: 52..55 (리틀엔디안)
    for off, name in ((DESC_WGT_STRIDE_OFF, "wgt_stride_r"),
                      (DESC_OUT_STRIDE_OFF, "out_stride_r")):
        m = re.search(name + r"\s*<=\s*\{(.+?)\};", cluster, re.S)
        assert m, f"{name} 대입을 못 찾았다"
        idx = sorted(int(x) for x in re.findall(r"desc_bytes\[(\d+)\]", m.group(1)))
        assert idx == [off, off + 1, off + 2, off + 3], \
            f"{name} 이 바이트 {idx} 를 읽는다, 기대 {off}..{off+3}"


def test_roundtrip_preserves_everything():
    d = pack_gemm(0xDEADBEEF, 0xCAFEBABE, 0x1234_5678_9ABC,
                  kt=251, flags=DescFlag.WGT_KEEP | DescFlag.WGT_INT4,
                  wgt_stride=0x4000, out_stride=0x1000)
    u = unpack_descriptor(d)
    assert len(d) == DESC_SIZE
    assert u["opcode"] == Opcode.GEMM
    assert u["flags"] == DescFlag.WGT_KEEP | DescFlag.WGT_INT4
    assert u["act_addr"] == 0xDEADBEEF
    assert u["wgt_addr"] == 0xCAFEBABE
    assert u["out_addr"] == 0x1234_5678_9ABC
    assert u["kt"] == 251
    assert u["wgt_stride"] == 0x4000
    assert u["out_stride"] == 0x1000
    assert u["crc_ok"]


@pytest.mark.parametrize("flags", list(DescFlag) + [
    DescFlag.ACC_KEEP | DescFlag.NO_WB,
    DescFlag.WGT_KEEP | DescFlag.ACC_KEEP | DescFlag.NO_WB,
])
def test_every_flag_combination_roundtrips(flags):
    u = unpack_descriptor(pack_gemm(0, 0, 0, 1, flags=flags))
    assert u["flags"] == flags
    assert u["crc_ok"]


def test_flags_and_strides_are_covered_by_crc():
    """CRC 가 새 바이트를 덮어야 한다. 안 덮으면 변조가 통과한다."""
    base = pack_gemm(0x1000, 0x2000, 0x3000, 64)
    for off, label in ((DESC_FLAGS_OFF, "flags"),
                       (DESC_WGT_STRIDE_OFF, "wgt_stride"),
                       (DESC_OUT_STRIDE_OFF, "out_stride")):
        tampered = bytearray(base)
        tampered[off] ^= 0xFF
        assert not unpack_descriptor(bytes(tampered))["crc_ok"], \
            f"{label} 바이트를 뒤집었는데 CRC 가 통과한다"


def test_crc_matches_rtl_polynomial():
    """RTL 의 crc8_byte 와 같은 다항식(0x07)을 써야 한다."""
    assert re.search(r"\^\s*8'h07", CTRL_FSM) or \
        re.search(r"\^\s*8'h07", (ROOT / "rtl" / "g3_desc_fsm.sv").read_text()), \
        "RTL 에서 0x07 다항식을 못 찾았다"
    # 알려진 벡터: 빈 데이터는 0
    assert crc8(b"") == 0x00
    # 단일 바이트 0x01 → 다항식 0x07 로 수동 계산한 값
    expect = 0x01
    for _ in range(8):
        expect = ((expect << 1) ^ 0x07) & 0xFF if expect & 0x80 else (expect << 1) & 0xFF
    assert crc8(b"\x01") == expect


def test_defaults_are_backward_compatible():
    """flags/stride 를 안 주면 예전과 같은 바이트가 나와야 한다."""
    new = pack_gemm(0x1000, 0x2000, 0x3000, 64)
    assert new[DESC_FLAGS_OFF] == 0
    assert struct.unpack_from("<I", new, DESC_WGT_STRIDE_OFF)[0] == 0
    assert struct.unpack_from("<I", new, DESC_OUT_STRIDE_OFF)[0] == 0
    assert unpack_descriptor(new)["crc_ok"]


def test_tiler_uses_the_ssot_not_its_own_copy():
    """타일러가 플래그를 다시 정의하면 세 곳이 어긋날 수 있다."""
    src = (ROOT / "tools" / "gemm_tiler.py").read_text()
    assert "from tools.orbit_mmio_map import DescFlag" in src, \
        "타일러가 SSOT 를 가져오지 않는다"
    from tools import gemm_tiler as gt
    assert gt.FLAG_WGT_KEEP == int(DescFlag.WGT_KEEP)
    assert gt.FLAG_NO_WB == int(DescFlag.NO_WB)


def test_scheduler_can_emit_a_whole_shape():
    """add_gemm_shape 가 모양 하나를 디스크립터 수열로 큐에 넣는다."""
    from tools.gemm_tiler import GemmShape, MemLayout, TileConfig, tile_gemm
    from tools.orbit_scheduler import OrbitScheduler

    shape = GemmShape(M=64, K=128, N=128)
    cfg = TileConfig(tile_n=32, cores=2, max_kt=256)
    sched = OrbitScheduler.__new__(OrbitScheduler)   # 디바이스 없이 큐만 쓴다
    sched._ops = []
    n = sched.add_gemm_shape(shape, cfg, 0x100000, 0x400000, 0x800000)

    plan = tile_gemm(shape, cfg)
    layout = MemLayout(shape, cfg, 0x100000, 0x400000, 0x800000)
    assert n == len(plan.descs) == len(sched._ops)
    for op, item in zip(sched._ops, plan.issue(layout)):
        u = unpack_descriptor(op.desc)
        assert u["crc_ok"]
        assert u["act_addr"] == item["act_addr"]
        assert u["wgt_addr"] == item["wgt_addr"]
        assert u["out_addr"] == item["out_addr"]
        assert int(u["flags"]) == item["flags"]
        assert u["wgt_stride"] == item["wgt_stride"]
        assert u["out_stride"] == item["out_stride"]
