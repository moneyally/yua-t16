"""test_mmap_backend_real_integration.py — MmapBackend + OrbitDevice integration.

Verifies that OrbitDevice works with MmapBackend(FakeBarOpener) end-to-end,
proving the real open path produces a usable backend.
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.orbit_mmap_backend import MmapBackend, FakeBarOpener
from tools.orbit_device import OrbitDevice
from tools.orbit_mmio_map import (
    G2_ID, G2_VERSION, G2_CAP0, TC0_CTRL, IRQ_MASK, TRACE_CTRL,
    BOOT_CAUSE,
)
from tools.orbit_desc import pack_nop


@pytest.fixture
def dev():
    """Create OrbitDevice with MmapBackend(FakeBarOpener).
    Pre-populate BAR0 with reset values matching reg_top defaults."""
    opener = FakeBarOpener()
    bar0 = opener.open_bar(0)
    # Set reset values that reg_top would produce
    bar0.poke32(G2_ID.offset, 0x4732_0001)
    bar0.poke32(G2_VERSION.offset, 0x0001_0000)
    bar0.poke32(G2_CAP0.offset, 0x0000_0060)
    bar0.poke32(TC0_CTRL.offset, 0x0000_0001)  # ENABLE=1
    bar0.poke32(IRQ_MASK.offset, 0xFFFF_FFFF)   # all masked
    bar0.poke32(BOOT_CAUSE.offset, 0x0000_0001) # POR

    b = MmapBackend.from_opener(opener)
    return OrbitDevice(b)


class TestMmapOrbitDeviceSmoke:
    def test_connect(self, dev):
        info = dev.connect()
        assert info.device_id == 0x4732_0001
        assert info.version == 0x0001_0000

    def test_tc_status(self, dev):
        tc = dev.read_tc_status()
        assert tc.enable is True

    def test_queue_status(self, dev):
        qs = dev.read_queue_status(0)
        assert qs.depth == 0

    def test_irq_poll(self, dev):
        assert dev.poll_irq() == 0

    def test_trace_enable(self, dev):
        dev.enable_trace()
        val = dev.backend.read(TRACE_CTRL.offset)
        assert val & 1 == 1

    def test_perf_freeze(self, dev):
        dev.freeze_perf()
        dev.unfreeze_perf()

    def test_enqueue_nop(self, dev):
        """Enqueue NOP via HAL — writes staging + doorbell to BAR0."""
        dev.enqueue_desc(pack_nop(), queue=0)
        # No crash = pass. Actual processing needs RTL DUT.

    def test_backend_is_backend_abc(self, dev):
        from tools.orbit_backend import Backend
        assert isinstance(dev.backend, Backend)


class TestMmapBackendBarRouting:
    def test_dma_register_via_bar4(self):
        opener = FakeBarOpener()
        bar4 = opener.open_bar(4)
        bar4.poke32(0x0010, 0x0000_0002)  # DMA_STATUS DONE bit
        b = MmapBackend.from_opener(opener)
        assert b.read(0x1_0010) == 0x0000_0002

    def test_trace_window_via_bar0(self):
        opener = FakeBarOpener()
        bar0 = opener.open_bar(0)
        bar0.poke32(0xA_0100, 0xCAFE)
        b = MmapBackend.from_opener(opener)
        assert b.read(0xA_0100) == 0xCAFE
