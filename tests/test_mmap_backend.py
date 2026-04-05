"""test_mmap_backend.py — MmapBackend with FakeMmap tests."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.orbit_mmap_backend import FakeMmap, MmapBackend
from tools.orbit_mmio_map import G2_ID, G2_VERSION, TC0_CTRL


class TestFakeMmap:
    def test_read_write(self):
        m = FakeMmap(0x1000)
        m.write32(0x100, 0xDEAD_BEEF)
        assert m.read32(0x100) == 0xDEAD_BEEF

    def test_unaligned_read_raises(self):
        m = FakeMmap(0x100)
        with pytest.raises(ValueError, match="Unaligned"):
            m.read32(0x01)

    def test_unaligned_write_raises(self):
        m = FakeMmap(0x100)
        with pytest.raises(ValueError, match="Unaligned"):
            m.write32(0x03, 0)

    def test_out_of_bounds_read(self):
        m = FakeMmap(0x10)
        with pytest.raises(ValueError, match="out of bounds"):
            m.read32(0x10)

    def test_out_of_bounds_write(self):
        m = FakeMmap(0x10)
        with pytest.raises(ValueError, match="out of bounds"):
            m.write32(0x10, 0)

    def test_poke_peek(self):
        m = FakeMmap(0x100)
        m.poke(0x10, b"\x01\x02\x03\x04")
        assert m.peek(0x10, 4) == b"\x01\x02\x03\x04"
        assert m.read32(0x10) == 0x04030201  # little-endian

    def test_initial_zeros(self):
        m = FakeMmap(0x100)
        assert m.read32(0) == 0


class TestMmapBackend:
    def test_bar0_read_write(self):
        bar0 = FakeMmap(0x100000)  # 1 MiB (full register space)
        b = MmapBackend(bar0)
        b.write(G2_ID.offset, 0x4732_0001)
        assert b.read(G2_ID.offset) == 0x4732_0001

    def test_bar4_routing(self):
        bar0 = FakeMmap(0x100000)
        bar4 = FakeMmap(0x10000)  # 64 KiB
        b = MmapBackend(bar0, bar4)
        # DMA_STATUS is at offset 0x1_0010 -> routes to BAR4
        bar4.poke32(0x0010, 0x0000_0002)  # DONE bit
        assert b.read(0x1_0010) == 0x0000_0002

    def test_bar0_extended_range(self):
        """Trace window at 0xA_0100+ routes to BAR0 (extended range)."""
        bar0 = FakeMmap(0x100000)  # 1 MiB (full register space)
        b = MmapBackend(bar0)
        b.write(0xA_0100, 0xCAFE)
        assert b.read(0xA_0100) == 0xCAFE

    def test_backend_contract(self):
        """MmapBackend satisfies Backend ABC."""
        bar0 = FakeMmap(0x20000)
        b = MmapBackend(bar0)
        from tools.orbit_backend import Backend
        assert isinstance(b, Backend)
