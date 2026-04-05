"""test_real_bar_mmap.py — RealBarMmap lifecycle/bounds/alignment tests.

Uses a real mmap'd tempfile to test the RealBarMmap code path
without requiring PCI hardware.
"""
import mmap
import os
import struct
import tempfile
import pytest
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.orbit_mmap_backend import RealBarMmap, MmapError


@pytest.fixture
def real_bar():
    """Create a RealBarMmap backed by a temp file."""
    size = 4096
    fd, path = tempfile.mkstemp()
    os.write(fd, b"\x00" * size)
    os.lseek(fd, 0, os.SEEK_SET)
    mm = mmap.mmap(fd, size, mmap.MAP_SHARED)
    bar = RealBarMmap(mm, fd, size, bar_index=0)
    yield bar
    bar.close()
    try:
        os.unlink(path)
    except OSError:
        pass


class TestRealBarMmapLifecycle:
    def test_read_write(self, real_bar):
        real_bar.write32(0x100, 0xDEAD_BEEF)
        assert real_bar.read32(0x100) == 0xDEAD_BEEF

    def test_size(self, real_bar):
        assert real_bar.size() == 4096

    def test_close_then_read_raises(self, real_bar):
        real_bar.close()
        with pytest.raises(MmapError, match="closed"):
            real_bar.read32(0)

    def test_close_then_write_raises(self, real_bar):
        real_bar.close()
        with pytest.raises(MmapError, match="closed"):
            real_bar.write32(0, 0)

    def test_double_close_safe(self, real_bar):
        real_bar.close()
        real_bar.close()  # should not raise

    def test_repr(self, real_bar):
        s = repr(real_bar)
        assert "bar=0" in s
        assert "open" in s
        real_bar.close()
        s = repr(real_bar)
        assert "closed" in s


class TestRealBarMmapBounds:
    def test_aligned_access(self, real_bar):
        real_bar.write32(0, 0x01020304)
        assert real_bar.read32(0) == 0x01020304

    def test_unaligned_read_raises(self, real_bar):
        with pytest.raises(ValueError, match="Unaligned"):
            real_bar.read32(1)

    def test_unaligned_write_raises(self, real_bar):
        with pytest.raises(ValueError, match="Unaligned"):
            real_bar.write32(3, 0)

    def test_out_of_bounds_read(self, real_bar):
        with pytest.raises(ValueError, match="out of bounds"):
            real_bar.read32(4096)

    def test_out_of_bounds_write(self, real_bar):
        with pytest.raises(ValueError, match="out of bounds"):
            real_bar.write32(4096, 0)

    def test_last_valid_offset(self, real_bar):
        real_bar.write32(4092, 0xABCD)
        assert real_bar.read32(4092) == 0xABCD


class TestRealBarMmapReadonly:
    def test_readonly_write_raises(self):
        size = 4096
        fd, path = tempfile.mkstemp()
        os.write(fd, b"\x00" * size)
        os.lseek(fd, 0, os.SEEK_SET)
        mm = mmap.mmap(fd, size, mmap.MAP_SHARED)
        bar = RealBarMmap(mm, fd, size, bar_index=0, readonly=True)
        with pytest.raises(MmapError, match="read-only"):
            bar.write32(0, 0)
        assert bar.read32(0) == 0  # read still works
        bar.close()
        os.unlink(path)
