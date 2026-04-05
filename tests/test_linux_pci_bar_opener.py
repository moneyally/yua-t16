"""test_linux_pci_bar_opener.py — LinuxPciBarOpener with fake sysfs.

Creates a temporary sysfs-like directory tree to test discovery,
BAR size checking, ID verification, and error paths.
"""
import os
import pytest
import tempfile
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.orbit_mmap_backend import (
    LinuxPciBarOpener, FakeBarOpener,
    DeviceNotFoundError, BarNotFoundError, BarSizeError,
    PermissionDeniedError, MmapOpenError,
    discover_orbit_devices, probe_orbit_device,
    ORBIT_G2_VENDOR_ID, ORBIT_G2_DEVICE_ID,
    EXPECTED_BAR_SIZES,
)


def _make_fake_sysfs(tmp, bdf="0000:01:00.0",
                     vendor=ORBIT_G2_VENDOR_ID, device=ORBIT_G2_DEVICE_ID,
                     bar_sizes=None):
    """Create a fake sysfs PCI device directory."""
    if bar_sizes is None:
        bar_sizes = {0: 0x100000, 2: 0x200000, 4: 0x10000}

    dev_dir = os.path.join(tmp, bdf)
    os.makedirs(dev_dir, exist_ok=True)

    # vendor / device files
    with open(os.path.join(dev_dir, "vendor"), "w") as f:
        f.write(f"0x{vendor:04x}\n")
    with open(os.path.join(dev_dir, "device"), "w") as f:
        f.write(f"0x{device:04x}\n")

    # resource file (6 lines for BAR0-BAR5)
    lines = []
    for i in range(6):
        sz = bar_sizes.get(i, 0)
        if sz > 0:
            start = 0xF0000000 + i * 0x1000000
            end = start + sz - 1
            flags = 0x00040200
        else:
            start = end = flags = 0
        lines.append(f"0x{start:016x} 0x{end:016x} 0x{flags:016x}")
    with open(os.path.join(dev_dir, "resource"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # resource0/2/4 files (empty files, can't actually mmap)
    for bar_idx in bar_sizes:
        if bar_sizes[bar_idx] > 0:
            with open(os.path.join(dev_dir, f"resource{bar_idx}"), "wb") as f:
                f.write(b"\x00" * 16)  # minimal file

    return dev_dir


class TestDiscovery:
    def test_discover_finds_device(self, tmp_path):
        _make_fake_sysfs(str(tmp_path))
        found = discover_orbit_devices(str(tmp_path))
        assert "0000:01:00.0" in found

    def test_discover_empty_when_no_match(self, tmp_path):
        _make_fake_sysfs(str(tmp_path), vendor=0x9999, device=0x1111)
        found = discover_orbit_devices(str(tmp_path))
        assert found == []

    def test_discover_multiple_devices(self, tmp_path):
        _make_fake_sysfs(str(tmp_path), bdf="0000:01:00.0")
        _make_fake_sysfs(str(tmp_path), bdf="0000:02:00.0")
        found = discover_orbit_devices(str(tmp_path))
        assert len(found) == 2

    def test_discover_nonexistent_dir(self):
        found = discover_orbit_devices("/nonexistent/path")
        assert found == []


class TestProbe:
    def test_probe_ok(self, tmp_path):
        _make_fake_sysfs(str(tmp_path))
        info = probe_orbit_device("0000:01:00.0", str(tmp_path))
        assert info["found"] is True
        assert info["vendor"] == ORBIT_G2_VENDOR_ID
        assert info["bars"][0]["ok"] is True
        assert info["bars"][4]["ok"] is True

    def test_probe_not_found(self, tmp_path):
        info = probe_orbit_device("0000:99:00.0", str(tmp_path))
        assert info["found"] is False

    def test_probe_bar_too_small(self, tmp_path):
        _make_fake_sysfs(str(tmp_path), bar_sizes={0: 0x1000, 4: 0x100})
        info = probe_orbit_device("0000:01:00.0", str(tmp_path))
        assert info["bars"][0]["ok"] is False  # too small
        assert info["bars"][4]["ok"] is False


class TestLinuxPciBarOpener:
    def test_device_not_found(self, tmp_path):
        opener = LinuxPciBarOpener("0000:99:00.0", sysfs_base=str(tmp_path))
        with pytest.raises(DeviceNotFoundError, match="not found"):
            opener.open_bar(0)

    def test_id_mismatch(self, tmp_path):
        _make_fake_sysfs(str(tmp_path), vendor=0x1111, device=0x2222)
        opener = LinuxPciBarOpener("0000:01:00.0", sysfs_base=str(tmp_path),
                                   check_ids=True)
        with pytest.raises(DeviceNotFoundError, match="ID mismatch"):
            opener.open_bar(0)

    def test_bar_not_found(self, tmp_path):
        _make_fake_sysfs(str(tmp_path), bar_sizes={0: 0x100000})
        # BAR4 has no resource4 file
        opener = LinuxPciBarOpener("0000:01:00.0", sysfs_base=str(tmp_path))
        with pytest.raises(BarNotFoundError):
            opener.open_bar(4)

    def test_bar_too_small(self, tmp_path):
        _make_fake_sysfs(str(tmp_path), bar_sizes={0: 0x1000, 4: 0x10000})
        opener = LinuxPciBarOpener("0000:01:00.0", sysfs_base=str(tmp_path))
        with pytest.raises(BarSizeError, match="size.*< expected"):
            opener.open_bar(0)

    def test_bar_zero_size(self, tmp_path):
        _make_fake_sysfs(str(tmp_path), bar_sizes={0: 0x100000})
        # BAR2 in resource file is zero
        opener = LinuxPciBarOpener("0000:01:00.0", sysfs_base=str(tmp_path))
        with pytest.raises(BarNotFoundError, match="zero-size"):
            opener.open_bar(2)

    def test_mmap_open_error(self, tmp_path):
        """resource0 file exists but mmap fails (too small for real mmap)."""
        _make_fake_sysfs(str(tmp_path))
        opener = LinuxPciBarOpener("0000:01:00.0", sysfs_base=str(tmp_path))
        # The fake resource0 file is only 16 bytes, mmap of 0x100000 will fail
        with pytest.raises(MmapOpenError, match="mmap failed"):
            opener.open_bar(0)

    def test_repr(self, tmp_path):
        opener = LinuxPciBarOpener("0000:01:00.0", sysfs_base=str(tmp_path))
        assert "0000:01:00.0" in repr(opener)

    def test_close_safe(self, tmp_path):
        opener = LinuxPciBarOpener("0000:01:00.0", sysfs_base=str(tmp_path))
        opener.close()  # should not raise even with nothing opened
        opener.close()  # double close safe


class TestExceptionHierarchy:
    def test_all_inherit_from_mmap_error(self):
        from tools.orbit_mmap_backend import MmapError
        assert issubclass(DeviceNotFoundError, MmapError)
        assert issubclass(BarNotFoundError, MmapError)
        assert issubclass(BarSizeError, MmapError)
        assert issubclass(PermissionDeniedError, MmapError)
        assert issubclass(MmapOpenError, MmapError)

    def test_catch_base(self):
        from tools.orbit_mmap_backend import MmapError
        with pytest.raises(MmapError):
            raise DeviceNotFoundError("test")
