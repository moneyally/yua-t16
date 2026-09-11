"""test_mmio_map.py — MMIO register map SSOT verification."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.orbit_mmio_map import (
    ALL_REGS, BASE, Block, Access, Reg,
    G2_ID, G2_VERSION, G2_CAP0, IRQ_MASK,
    TC0_RUNSTATE, TC0_CTRL, PERF_FREEZE,
    BOOT_CAUSE, SW_RESET, WDOG_CTRL,
    DESC_STAGE_BASE, DESC_STAGE_END, DESC_STAGE_WORDS,
    IrqBit, IRQ_FATAL_MASK,
    Opcode, TraceType,
    TRACE_WIN_BASE, TRACE_META_BASE,
)


def test_all_regs_have_unique_addresses():
    addrs = [r.addr for r in ALL_REGS]
    assert len(addrs) == len(set(addrs)), "Duplicate register addresses found"


def test_all_regs_are_4byte_aligned():
    for r in ALL_REGS:
        assert r.addr % 4 == 0, f"{r.name} at {r.addr:#x} not 4-byte aligned"


def test_all_regs_within_register_space():
    for r in ALL_REGS:
        assert 0x8030_0000 <= r.addr <= 0x803F_FFFF, \
            f"{r.name} at {r.addr:#x} outside register space"


def test_offset_computation():
    assert G2_ID.offset == 0x0_0000
    assert TC0_RUNSTATE.offset == 0x4_0000
    assert PERF_FREEZE.offset == 0x6_0018


def test_g2_id_default():
    assert G2_ID.reset == 0x4732_0001


def test_irq_mask_default_all_masked():
    assert IRQ_MASK.reset == 0xFFFF_FFFF


def test_tc0_ctrl_default_enabled():
    assert TC0_CTRL.reset == 0x01


def test_desc_stage_range():
    assert DESC_STAGE_END == DESC_STAGE_BASE + (DESC_STAGE_WORDS - 1) * 4


def test_irq_bits_cover_12_sources():
    assert len(IrqBit) == 12


def test_irq_fatal_mask():
    # bits 2,4,5,6,8,10 are fatal
    expected = (1<<2)|(1<<4)|(1<<5)|(1<<6)|(1<<8)|(1<<10)
    assert IRQ_FATAL_MASK == expected


def test_opcodes():
    assert Opcode.NOP == 0x01
    assert Opcode.GEMM == 0x02


def test_trace_types():
    assert TraceType.DESC_DISPATCH == 1
    assert TraceType.DESC_DONE == 2
    assert TraceType.DESC_FAULT == 3


def test_trace_windows_in_block():
    # Trace block: 0x803A_0000 to 0x803A_FFFF
    assert TRACE_WIN_BASE >= 0x803A_0000
    assert TRACE_META_BASE >= 0x803A_0000
    assert TRACE_META_BASE < 0x803B_0000


def test_access_types():
    assert BOOT_CAUSE.access == Access.RO
    assert SW_RESET.access == Access.WO
    assert WDOG_CTRL.access == Access.RW


# ── Watchdog (SSOT: spec/watchdog.md) ────────────────────────────────
# 기대값 출처: spec/watchdog.md 1절 비트 표와 2절 타임아웃 수식.

def test_wdog_bit_positions_match_spec():
    from tools.orbit_mmio_map import (
        WDOG_EN, WDOG_KICK, WDOG_TEST_FIRE,
        WDOG_PERIOD_SH, WDOG_PERIOD_W, WDOG_PERIOD_MASK,
    )
    assert WDOG_EN == 1 << 0
    assert WDOG_KICK == 1 << 1
    assert WDOG_TEST_FIRE == 1 << 31
    assert WDOG_PERIOD_SH == 8 and WDOG_PERIOD_W == 16
    assert WDOG_PERIOD_MASK == 0x00FF_FF00
    # 예약 비트 [7:2] 와 [30:24] 가 어떤 필드와도 안 겹친다
    used = WDOG_EN | WDOG_KICK | WDOG_TEST_FIRE | WDOG_PERIOD_MASK
    assert used & 0x00_00_00_FC == 0, "예약 [7:2] 를 밟았다"
    assert used & 0x7F00_0000 == 0, "예약 [30:24] 를 밟았다"


def test_wdog_ctrl_word_packs_period_and_enable():
    from tools.orbit_mmio_map import wdog_ctrl_word, WDOG_EN, WDOG_KICK
    # PERIOD 와 EN 을 **한 번에** 쓴다 (spec 3절) — 나눠 쓰면 옛 창으로 터진다
    w = wdog_ctrl_word(99)
    assert (w >> 8) & 0xFFFF == 99
    assert w & WDOG_EN and w & WDOG_KICK
    assert wdog_ctrl_word(0, enable=False, kick=False) == 0


def test_wdog_ctrl_word_rejects_out_of_range_period():
    from tools.orbit_mmio_map import wdog_ctrl_word
    with pytest.raises(ValueError):
        wdog_ctrl_word(1 << 16)
    with pytest.raises(ValueError):
        wdog_ctrl_word(-1)


def test_wdog_timeout_formula():
    from tools.orbit_mmio_map import wdog_timeout_cycles, WDOG_PRESCALE
    assert WDOG_PRESCALE == 1024
    assert wdog_timeout_cycles(0) == 1024
    assert wdog_timeout_cycles(1) == 2048
    # 100MHz 기준 최대 창이 0.67초 근처여야 한다 (spec 1절 근거 숫자)
    assert 0.6 < wdog_timeout_cycles(0xFFFF) / 100e6 < 0.7
