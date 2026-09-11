"""기대값 출처: spec/deltarule.md 의 마크다운 표를 **직접 파싱**해서 읽는다 (SSOT).

tests/test_dr1_spec_consistency.py — PLAN W4-1

## SSOT 방향

    docs/DESIGN.md  (수학·숫자 형식)
         ↓
    spec/deltarule.md  (디스크립터·레지스터 바이트 레이아웃)   ← **이 파일이 옳다**
         ↓
    tools/orbit_mmio_map.py  (파이썬 상수)
         ↓
    tools/orbit_desc.py, RTL, 테스트

이 테스트는 문서의 표를 파싱해서 `tools/orbit_mmio_map.py` 상수와 대조한다.
**둘이 다르면 문서가 옳고 상수가 틀린 것이다.** 상수를 고쳐라, 문서를 고치지 말고.
(문서를 바꿔야 한다면 그건 설계 변경이고, `docs/DESIGN.md` 부터 손대야 한다.)

문서를 파싱하는 이유: 상수에서 문서를 생성하면 문서가 상수의 그림자가 된다.
그러면 "문서가 SSOT" 라는 말이 거짓이 된다. 사람이 읽고 고치는 쪽이 문서이므로
문서를 읽어서 코드를 검사하는 방향이 맞다.
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import orbit_mmio_map as M  # noqa: E402

SPEC_PATH = os.path.join(os.path.dirname(__file__), "..", "spec", "deltarule.md")


@pytest.fixture(scope="module")
def spec_text():
    with open(SPEC_PATH, encoding="utf-8") as f:
        return f.read()


def _table_rows(text, section_marker, stop_marker=None):
    """section_marker 뒤의 마크다운 표 행들을 (셀 리스트) 로 돌려준다."""
    start = text.index(section_marker)
    chunk = text[start:]
    if stop_marker and stop_marker in chunk[len(section_marker):]:
        chunk = chunk[: len(section_marker) + chunk[len(section_marker):].index(stop_marker)]
    rows = []
    for line in chunk.split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):   # 구분선
            continue
        rows.append(cells)
    return rows


def _strip_md(s):
    """마크다운 강조·코드 표시 제거."""
    return s.replace("**", "").replace("`", "").strip()


# ---------------------------------------------------------------------------
# opcode
# ---------------------------------------------------------------------------
def test_opcodes_match_spec(spec_text):
    """기대값 출처: spec/deltarule.md 2절 Opcode 표."""
    rows = _table_rows(spec_text, "## 2. Opcode", "## 3.")
    seen = {}
    for cells in rows[1:]:                      # [0] 은 헤더
        val, name = _strip_md(cells[0]), _strip_md(cells[1])
        if not val.startswith("0x"):
            continue
        seen[name] = int(val, 16)

    for name in ("DELTA_INIT", "DELTA_STEP", "DELTA_DUMP"):
        assert name in seen, f"spec 2절 표에 {name} 이 없다"
        got = int(getattr(M.Opcode, name))
        assert got == seen[name], (
            f"{name}: spec={seen[name]:#04x} vs orbit_mmio_map={got:#04x}. "
            "spec 이 옳다 — orbit_mmio_map 을 고쳐라."
        )

    # 기존 opcode 와 충돌이 없어야 한다
    vals = [int(v) for v in M.Opcode]
    assert len(vals) == len(set(vals)), f"Opcode 값이 중복된다: {sorted(vals)}"


def test_fault_codes_match_spec(spec_text):
    """기대값 출처: spec/deltarule.md 4절 fault_code 표."""
    rows = _table_rows(spec_text, "## 4. 실패 조건", "## 5.")
    seen = {}
    for cells in rows[1:]:
        val, name = _strip_md(cells[0]), _strip_md(cells[1])
        if not val.startswith("0x"):
            continue
        seen[name] = int(val, 16)

    for name in ("DR1_BAD_SLOT", "DR1_UNALIGNED"):
        assert name in seen, f"spec 4절 표에 {name} 이 없다"
        assert int(getattr(M.FaultCode, name)) == seen[name], (
            f"{name}: spec={seen[name]:#04x} vs FaultCode={int(getattr(M.FaultCode, name)):#04x}"
        )
    vals = [int(v) for v in M.FaultCode]
    assert len(vals) == len(set(vals)), f"FaultCode 값 중복: {sorted(vals)}"


# ---------------------------------------------------------------------------
# 디스크립터 필드 오프셋
# ---------------------------------------------------------------------------
SPEC_FIELD_TO_CONST = {
    "slot": "DR1_SLOT_OFF",
    "q_addr": "DR1_Q_ADDR_OFF",
    "k_addr": "DR1_K_ADDR_OFF",
    "o_addr": "DR1_O_ADDR_OFF",
    "v_addr": "DR1_V_ADDR_OFF",
    "alpha_uq15": "DR1_ALPHA_OFF",
    "beta_uq15": "DR1_BETA_OFF",
    "crc8": "DESC_CRC_OFF",
    "opcode": "DESC_OPCODE_OFF",
}


def test_delta_step_field_offsets_match_spec(spec_text):
    """기대값 출처: spec/deltarule.md 3.3절 DELTA_STEP 바이트 레이아웃 표."""
    rows = _table_rows(spec_text, "### 3.3 `DELTA_STEP`", "### 3.4")
    found = {}
    for cells in rows[1:]:
        off_s, size_s, field = _strip_md(cells[0]), _strip_md(cells[1]), _strip_md(cells[2])
        if not off_s.isdigit():
            continue
        if field in SPEC_FIELD_TO_CONST:
            found[field] = (int(off_s), int(size_s))

    missing = set(SPEC_FIELD_TO_CONST) - set(found)
    assert not missing, f"spec 3.3 표에서 못 찾은 필드: {sorted(missing)}"

    for field, (off, size) in found.items():
        const = SPEC_FIELD_TO_CONST[field]
        got = getattr(M, const)
        assert got == off, (
            f"{field}: spec 오프셋 {off} vs {const}={got}. "
            "spec 이 옳다 — orbit_mmio_map 을 고쳐라."
        )

    # 폭도 확인: 주소는 8바이트, alpha/beta 는 2바이트, slot/opcode/crc 는 1바이트
    assert found["q_addr"][1] == 8 and found["v_addr"][1] == 8
    assert found["alpha_uq15"][1] == 2 and found["beta_uq15"][1] == 2
    assert found["slot"][1] == 1 and found["opcode"][1] == 1 and found["crc8"][1] == 1


def test_delta_step_fields_do_not_overlap(spec_text):
    """기대값 출처: spec/deltarule.md 3.3절 — 필드가 겹치거나 64바이트를 넘으면 안 된다."""
    rows = _table_rows(spec_text, "### 3.3 `DELTA_STEP`", "### 3.4")
    spans = []
    for cells in rows[1:]:
        off_s, size_s, field = _strip_md(cells[0]), _strip_md(cells[1]), _strip_md(cells[2])
        if not off_s.isdigit() or not size_s.isdigit():
            continue
        spans.append((int(off_s), int(size_s), field))

    spans.sort()
    covered = 0
    for off, size, field in spans:
        assert off >= covered, (
            f"필드 {field}@{off} 가 앞 필드와 겹친다 (앞이 {covered} 까지 씀)"
        )
        assert off + size <= M.DESC_SIZE, (
            f"필드 {field}@{off}+{size} 가 디스크립터 {M.DESC_SIZE}바이트를 넘는다"
        )
        covered = off + size
    assert covered == M.DESC_SIZE, (
        f"3.3 표가 디스크립터 끝까지 덮지 않는다 (마지막 {covered}, 기대 {M.DESC_SIZE}). "
        "예약 영역도 표에 있어야 한다."
    )


def test_dr1_reuses_existing_address_slots():
    """기대값 출처: spec/deltarule.md 3절 서문 — 기존 필드 위치를 재사용한다.

    이게 깨지면 rtl/desc_fsm_v2.sv 의 기존 추출 로직을 고쳐야 한다는 뜻이다.
    """
    assert M.DR1_Q_ADDR_OFF == M.DESC_ACT_ADDR_OFF
    assert M.DR1_K_ADDR_OFF == M.DESC_WGT_ADDR_OFF
    assert M.DR1_O_ADDR_OFF == M.DESC_OUT_ADDR_OFF
    assert M.DESC_CRC_OFF == M.DESC_SIZE - 1


# ---------------------------------------------------------------------------
# 레지스터
# ---------------------------------------------------------------------------
def test_dr1_registers_match_spec(spec_text):
    """기대값 출처: spec/deltarule.md 5절 레지스터 표 (주소·접근·리셋)."""
    rows = _table_rows(spec_text, "## 5. 레지스터 추가분", "### 5.1")
    seen = {}
    for cells in rows[1:]:
        addr_s, name, access, reset_s = (_strip_md(c) for c in cells[:4])
        if not addr_s.startswith("0x"):
            continue
        seen[name] = (int(addr_s.replace("_", ""), 16), access, int(reset_s))

    for name in ("DR1_STATUS", "DR1_SAT_COUNT", "DR1_CLAMP_COUNT", "DR1_CYCLES"):
        assert name in seen, f"spec 5절 표에 {name} 이 없다"
        reg = getattr(M, name)
        addr, access, reset = seen[name]
        assert reg.addr == addr, f"{name} 주소: spec={addr:#010x} vs map={reg.addr:#010x}"
        assert reg.access == access, f"{name} 접근: spec={access} vs map={reg.access}"
        assert reg.reset == reset, f"{name} 리셋: spec={reset} vs map={reg.reset}"

    # DR1 블록 베이스가 기존 블록과 겹치지 않아야 한다
    blocks = sorted((int(b), b.name) for b in M.Block)
    for (a1, n1), (a2, n2) in zip(blocks, blocks[1:]):
        assert a1 != a2, f"블록 주소 충돌: {n1} == {n2} == {a1:#010x}"
    assert int(M.Block.DR1) == 0x8033_0000


def test_dr1_register_addresses_inside_block():
    """기대값 출처: spec/deltarule.md 5절 — DR1 레지스터는 DR1 블록 안에 있어야 한다."""
    base = int(M.Block.DR1)
    for name in ("DR1_STATUS", "DR1_SAT_COUNT", "DR1_CLAMP_COUNT", "DR1_CYCLES"):
        reg = getattr(M, name)
        assert base <= reg.addr < base + 0x1_0000, (
            f"{name}={reg.addr:#010x} 이 DR1 블록 {base:#010x} 밖이다"
        )


# ---------------------------------------------------------------------------
# 트레이스 이벤트 / UQ1.15 상수
# ---------------------------------------------------------------------------
def test_trace_events_match_spec(spec_text):
    """기대값 출처: spec/deltarule.md 5.2절 트레이스 이벤트 표."""
    rows = _table_rows(spec_text, "### 5.2 트레이스 이벤트 추가", "## 6.")
    seen = {}
    for cells in rows[1:]:
        val, name = _strip_md(cells[0]), _strip_md(cells[1])
        if not val.isdigit():
            continue
        seen[name] = int(val)
    for name in ("SAT_EVENT", "CLAMP_EVENT"):
        assert name in seen, f"spec 5.2 표에 {name} 이 없다"
        assert int(getattr(M.TraceType, name)) == seen[name]
    vals = [int(v) for v in M.TraceType]
    assert len(vals) == len(set(vals)), f"TraceType 값 중복: {sorted(vals)}"


def test_uq15_one_is_exact():
    """기대값 출처: spec/deltarule.md 1절 — UQ1.15 에서 0x8000 = 1.0 (정확)."""
    assert M.UQ15_ONE == 0x8000
    from sim.golden.deltarule import UQ15_ONE as GOLDEN_ONE
    assert GOLDEN_ONE == M.UQ15_ONE, "골든 모델과 mmio_map 의 UQ15_ONE 이 다르다"


def test_alignment_and_slots_match_spec(spec_text):
    """기대값 출처: spec/deltarule.md 4절 — 16바이트 정렬, v1 슬롯 1개."""
    assert "16바이트 정렬" in spec_text, "spec 4절에 정렬 요건 문구가 없다"
    assert M.DR1_ADDR_ALIGN == 16
    assert M.DR1_NUM_SLOTS == 1


# ---------------------------------------------------------------------------
# 패커가 스펙대로 채우는지
# ---------------------------------------------------------------------------
# W7 에서 스크래치(1024 원소)가 생기면서 주소에 **범위 상한**이 붙었다.
# 예전 테스트는 0x1000 (원소 2048) 을 썼는데 이제 그것은 범위 밖이다 —
# 기대값을 바꿔 통과시킨 것이 아니라, 하드웨어에 없던 제약이 새로 생긴 것이다.
_LAY = M.dr1_scratch_layout(16)
_Q, _K, _V, _O = (_LAY["q"] * 2, _LAY["k"] * 2, _LAY["v"] * 2, _LAY["o"] * 2)


def test_packers_place_fields_at_spec_offsets():
    """기대값 출처: spec/deltarule.md 3.3 — 패킹한 바이트를 오프셋으로 직접 확인."""
    from tools.orbit_desc import crc8, pack_delta_step, unpack_delta_step

    d = pack_delta_step(
        q_addr=_Q, k_addr=_K, v_addr=_V, o_addr=_O,
        alpha_uq15=M.UQ15_ONE, beta_uq15=0x4000, slot=0,
    )
    assert len(d) == M.DESC_SIZE
    f = unpack_delta_step(d)
    assert f["opcode"] == int(M.Opcode.DELTA_STEP)
    assert (f["q_addr"], f["k_addr"], f["v_addr"], f["o_addr"]) == (_Q, _K, _V, _O)
    assert f["alpha_uq15"] == 0x8000 and f["beta_uq15"] == 0x4000
    assert crc8(d[: M.DESC_CRC_OFF]) == d[M.DESC_CRC_OFF], "CRC-8 이 기존 규칙과 다르다"


def test_packers_reject_spec_violations():
    """기대값 출처: spec/deltarule.md 4절 실패 조건 — 호스트가 먼저 잡아야 한다."""
    from tools.orbit_desc import Dr1FieldError, pack_delta_step

    with pytest.raises(Dr1FieldError, match="정렬"):
        pack_delta_step(_Q + 1, _K, _V, _O, M.UQ15_ONE, 0x4000)
    with pytest.raises(Dr1FieldError, match="slot"):
        pack_delta_step(_Q, _K, _V, _O, M.UQ15_ONE, 0x4000, slot=M.DR1_NUM_SLOTS)
    with pytest.raises(Dr1FieldError, match="UQ1.15"):
        pack_delta_step(_Q, _K, _V, _O, M.UQ15_ONE + 1, 0x4000)
    # 0x08 DR1_ADDR_RANGE 와 같은 조건을 호스트도 잡는다 (W7 신설)
    with pytest.raises(Dr1FieldError, match="스크래치"):
        pack_delta_step((M.DR1_SCRATCH_WORDS - 8) * 2, _K, _V, _O, M.UQ15_ONE, 0x4000)


def test_scratch_layout_is_aligned_and_fits():
    """기대값 출처: spec/deltarule.md 3.6절 — 배치가 정렬·범위 계약을 지키는지."""
    lay = M.dr1_scratch_layout(16)
    assert lay["q"] == 0 and lay["k"] == 16 and lay["v"] == 32 and lay["o"] == 48
    for name, elem in lay.items():
        assert (elem * 2) % M.DR1_ADDR_ALIGN == 0, f"{name} 이 정렬 위반"
    assert lay["dump"] + 16 * 16 <= M.DR1_SCRATCH_WORDS

    # **알려진 한계**: 스크래치 1024 원소로는 d=64 의 덤프(64²=4096 원소)가 안 들어간다.
    # 숨기지 않고 여기서 못 박는다 — d=64 로 확장할 때 스크래치를 먼저 키워야 한다
    # (필요량 = dump 시작 + d², 최소 4608 원소). docs/DESIGN.md 4절.
    with pytest.raises(ValueError, match="스크래치"):
        M.dr1_scratch_layout(64)


def test_golden_rejects_undefined_alpha_beta():
    """기대값 출처: spec/deltarule.md 1절 — 골든은 1.0 초과에 ValueError."""
    import numpy as np
    from sim.golden.deltarule import step

    z2 = np.zeros(2, dtype=np.int64)
    S = np.zeros((2, 2), dtype=np.int64)
    with pytest.raises(ValueError, match="0x8000"):
        step(S, z2, z2, z2, M.UQ15_ONE + 1, 0)
    with pytest.raises(ValueError):
        step(S, z2, z2, z2, M.UQ15_ONE, -1)
    # 정의된 상한은 통과해야 한다
    step(S, z2, z2, z2, M.UQ15_ONE, M.UQ15_ONE)
