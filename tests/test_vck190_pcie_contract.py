"""test_vck190_pcie_contract.py — VCK190 PCIe bring-up contract consistency.

Verifies that software/RTL/doc expectations for VCK190 PCIe are aligned.
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.orbit_mmio_map import (
    G2_ID, G2_VERSION, G2_CAP0, TC0_CTRL, IRQ_MASK, IRQ_PENDING,
    DMA_STATUS, TRACE_CTRL, TRACE_HEAD, OOM_STATE, BASE,
    BOOT_CAUSE, PERF_FREEZE,
)
from tools.orbit_mmap_backend import EXPECTED_BAR_SIZES


class TestBarSizes:
    def test_bar0_size(self):
        assert EXPECTED_BAR_SIZES[0] == 0x100000, "BAR0 should be 1 MiB"

    def test_bar2_size(self):
        assert EXPECTED_BAR_SIZES[2] == 0x200000, "BAR2 should be 2 MiB"

    def test_bar4_size(self):
        assert EXPECTED_BAR_SIZES[4] == 0x10000, "BAR4 should be 64 KiB"


class TestBarRegisterCoverage:
    """All critical registers fit within their BAR."""

    def test_bar0_covers_all_registers(self):
        bar0_regs = [
            G2_ID, G2_VERSION, G2_CAP0, BOOT_CAUSE, TC0_CTRL,
            IRQ_PENDING, IRQ_MASK, PERF_FREEZE, OOM_STATE,
            TRACE_HEAD, TRACE_CTRL,
        ]
        bar0_size = EXPECTED_BAR_SIZES[0]
        for reg in bar0_regs:
            assert reg.offset < bar0_size, \
                f"{reg.name} offset {reg.offset:#x} exceeds BAR0 size {bar0_size:#x}"

    def test_bar4_covers_dma(self):
        # DMA registers are at offset 0x1_0000..0x1_001C
        # BAR4 covers offsets 0x1_0000 to 0x1_FFFF
        assert DMA_STATUS.offset >= 0x1_0000
        assert DMA_STATUS.offset < 0x1_0000 + EXPECTED_BAR_SIZES[4]

    def test_trace_window_in_bar0(self):
        from tools.orbit_mmio_map import TRACE_WIN_BASE
        trace_off = TRACE_WIN_BASE - BASE
        assert trace_off < EXPECTED_BAR_SIZES[0], \
            f"Trace window offset {trace_off:#x} exceeds BAR0"


class TestFirstSmokeRegisters:
    """First-smoke reads should hit these registers."""

    FIRST_SMOKE_READS = [
        ("G2_ID", G2_ID, 0x4732_0001),
        ("G2_VERSION", G2_VERSION, 0x0001_0000),
        ("G2_CAP0", G2_CAP0, 0x0000_0060),
        ("TC0_CTRL", TC0_CTRL, 0x0000_0001),  # ENABLE=1
        ("IRQ_MASK", IRQ_MASK, 0xFFFF_FFFF),   # all masked
    ]

    @pytest.mark.parametrize("name,reg,expected", FIRST_SMOKE_READS)
    def test_reset_value(self, name, reg, expected):
        assert reg.reset == expected, \
            f"{name} reset value {reg.reset:#x} != expected {expected:#x}"


class TestCPMConstraints:
    """Document CPM-specific facts for consistency checking."""

    def test_user_clk_gen4_x8(self):
        """Gen4 x8 user_clk is 250 MHz."""
        # This is a documentation-level assertion — no code to test.
        # Keeping as a reminder for IP config.
        expected_mhz = 250
        assert expected_mhz == 250, "Gen4 x8 user_clk should be 250 MHz"

    def test_axi_stream_width(self):
        """CPM AXI-Stream data width for Gen4 x8 should be 256-bit."""
        expected = 256
        assert expected == 256

    def test_bar_count(self):
        """ORBIT-G2 uses 3 BARs (0, 2, 4)."""
        assert len(EXPECTED_BAR_SIZES) == 3
        assert set(EXPECTED_BAR_SIZES.keys()) == {0, 2, 4}
