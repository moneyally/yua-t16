"""test_protob_shell_contract.py — Proto-B RTL shell contract verification.

Validates interface contracts between pcie_ep_versal, dma_bridge,
g2_protob_top and the software stack.
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.orbit_mmio_map import (
    G2_ID, TC0_CTRL, IRQ_PENDING, IRQ_MASK, TRACE_CTRL,
    DMA_STATUS, DMA_ERR_CODE, BASE,
    TRACE_WIN_BASE, TRACE_META_BASE,
)
from tools.orbit_mmap_backend import FakeMmap, FakeBarOpener, MmapBackend
from tools.orbit_dma import (
    DMA_SUBMIT_LO_OFF, DMA_SUBMIT_HI_OFF, DMA_LEN_OFF,
    DMA_CTRL_OFF, DMA_STATUS_OFF, DMA_ERR_CODE_OFF,
    DMA_THROTTLE_OFF, DMA_TIMEOUT_OFF,
)


class TestBarAddressMapping:
    """BAR0 offset = REG_SPEC addr - 0x8030_0000."""

    def test_g2_id_at_bar0_zero(self):
        assert G2_ID.offset == 0

    def test_tc0_ctrl_at_bar0_0x4004(self):
        assert TC0_CTRL.offset == 0x4_0004

    def test_irq_pending_at_bar0_0x90000(self):
        assert IRQ_PENDING.offset == 0x9_0000

    def test_dma_status_at_bar4_offset(self):
        """DMA_STATUS is at reg_top offset 0x1_0010 → BAR4 offset 0x0010."""
        assert DMA_STATUS_OFF == 0x1_0010
        bar4_internal = DMA_STATUS_OFF - 0x1_0000
        assert bar4_internal == 0x0010

    def test_trace_window_in_bar0_extended(self):
        """Trace read window (0xA_0100+) is in BAR0 extended range."""
        trace_off = TRACE_WIN_BASE - BASE
        assert trace_off == 0xA_0100
        assert trace_off > 0x1_0000  # past BAR4 range


class TestDmaBridgeRegisterLayout:
    """DMA bridge register offsets match REG_SPEC section 5."""

    def test_submit_lo(self):
        assert (DMA_SUBMIT_LO_OFF - 0x1_0000) == 0x00

    def test_submit_hi(self):
        assert (DMA_SUBMIT_HI_OFF - 0x1_0000) == 0x04

    def test_len(self):
        assert (DMA_LEN_OFF - 0x1_0000) == 0x08

    def test_ctrl(self):
        assert (DMA_CTRL_OFF - 0x1_0000) == 0x0C

    def test_status(self):
        assert (DMA_STATUS_OFF - 0x1_0000) == 0x10

    def test_err_code(self):
        assert (DMA_ERR_CODE_OFF - 0x1_0000) == 0x14

    def test_throttle(self):
        assert (DMA_THROTTLE_OFF - 0x1_0000) == 0x18

    def test_timeout(self):
        assert (DMA_TIMEOUT_OFF - 0x1_0000) == 0x1C


class TestMmapBackendBarRouting:
    """MmapBackend routes offsets to correct BAR regions."""

    def setup_method(self):
        self.opener = FakeBarOpener()
        self.b = MmapBackend.from_opener(self.opener)

    def test_bar0_register_access(self):
        self.b.write(G2_ID.offset, 0x4732_0001)
        assert self.b.read(G2_ID.offset) == 0x4732_0001

    def test_bar4_dma_routing(self):
        """Offset 0x1_0010 routes to BAR4."""
        bar4 = self.opener.open_bar(4)
        bar4.poke32(0x0010, 0xAABB_CCDD)
        val = self.b.read(0x1_0010)
        assert val == 0xAABB_CCDD

    def test_bar0_extended_trace(self):
        """Offset 0xA_0100 routes to BAR0 (extended range)."""
        self.b.write(0xA_0100, 0x1234)
        assert self.b.read(0xA_0100) == 0x1234

    def test_opener_factory(self):
        b = MmapBackend.from_opener(FakeBarOpener())
        assert isinstance(b, MmapBackend)


class TestFakeBarOpener:
    def test_open_bar0(self):
        opener = FakeBarOpener()
        bar0 = opener.open_bar(0)
        assert bar0.size() == 0x100000  # 1 MiB full register space

    def test_open_bar4(self):
        opener = FakeBarOpener()
        bar4 = opener.open_bar(4)
        assert bar4.size() == 0x10000

    def test_open_invalid_bar(self):
        from tools.orbit_mmap_backend import BarNotFoundError
        opener = FakeBarOpener()
        with pytest.raises(BarNotFoundError, match="BAR1 not defined"):
            opener.open_bar(1)

    def test_linux_opener_no_device(self):
        from tools.orbit_mmap_backend import LinuxPciBarOpener, DeviceNotFoundError
        opener = LinuxPciBarOpener("0000:01:00.0")
        with pytest.raises(DeviceNotFoundError):
            opener.open_bar(0)


class TestProtoBMockMatchesRTL:
    """Proto-B mock DMA state machine matches dma_bridge.sv contract."""

    def test_submit_clears_previous_error(self):
        """REG_SPEC: errors cleared on new START (dma_bridge ST_IDLE)."""
        from tools.orbit_protob_mock import ProtoBMockBackend
        from tools.orbit_dma import OrbitDma
        b = ProtoBMockBackend()
        b.inject_error = True
        dma = OrbitDma(b)
        dma.submit_h2d(0x1000, 256)
        assert dma.read_status().err is True
        b.inject_error = False
        dma.submit_h2d(0x2000, 128)
        assert dma.read_status().err is False
        assert dma.read_status().done is True

    def test_status_bits_layout(self):
        """DMA_STATUS bit layout: [0]=BUSY [1]=DONE [2]=ERR [3]=TIMEOUT [15:8]=INFLIGHT."""
        from tools.orbit_protob_mock import ProtoBMockBackend
        from tools.orbit_dma import OrbitDma, DMA_BUSY, DMA_DONE, DMA_ERR, DMA_TIMEOUT
        assert DMA_BUSY == 1
        assert DMA_DONE == 2
        assert DMA_ERR == 4
        assert DMA_TIMEOUT == 8
