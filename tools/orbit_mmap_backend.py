"""
orbit_mmap_backend.py — MmapBackend for ORBIT-G2

Real MMIO backend using memory-mapped BAR regions.
SSOT: ORBIT_G2_PCIE_BAR_SPEC.md, ORBIT_G2_LINUX_MMIO.md

Architecture:
  BarRegion (ABC)
    ├── FakeMmap         — bytearray-based (unit tests)
    └── RealBarMmap      — real mmap'd file (real hardware)

  BarOpener (ABC)
    ├── FakeBarOpener    — returns FakeMmap (tests)
    └── LinuxPciBarOpener — opens /sys/bus/pci/.../resourceN (real hardware)

  MmapBackend            — routes offsets to BAR0/BAR4 regions
"""
from __future__ import annotations
import os
import struct
from abc import ABC, abstractmethod
from pathlib import Path
from tools.orbit_backend import Backend


# ═══════════════════════════════════════════════════════════════════
# Exception hierarchy
# ═══════════════════════════════════════════════════════════════════

class MmapError(Exception):
    """Base for all MMIO open/access errors."""

class DeviceNotFoundError(MmapError):
    """PCI device not found at specified BDF or by vendor/device filter."""

class BarNotFoundError(MmapError):
    """BAR resource file missing or BAR has zero size."""

class BarSizeError(MmapError):
    """BAR region is smaller than expected."""

class PermissionDeniedError(MmapError):
    """Insufficient permissions to open BAR resource."""

class MmapOpenError(MmapError):
    """mmap syscall failed."""


# ═══════════════════════════════════════════════════════════════════
# BarRegion — abstract memory region
# ═══════════════════════════════════════════════════════════════════

class BarRegion(ABC):
    @abstractmethod
    def read32(self, offset: int) -> int: ...
    @abstractmethod
    def write32(self, offset: int, value: int): ...
    @abstractmethod
    def size(self) -> int: ...

    def _check_access(self, offset: int):
        if offset < 0 or offset + 4 > self.size():
            raise ValueError(f"Access out of bounds: offset={offset:#x}, size={self.size():#x}")
        if offset % 4 != 0:
            raise ValueError(f"Unaligned access: offset={offset:#x}")


class FakeMmap(BarRegion):
    """In-memory fake mmap for unit testing."""
    def __init__(self, size: int):
        self._size = size
        self._buf = bytearray(size)

    def read32(self, offset: int) -> int:
        self._check_access(offset)
        return struct.unpack_from("<I", self._buf, offset)[0]

    def write32(self, offset: int, value: int):
        self._check_access(offset)
        struct.pack_into("<I", self._buf, offset, value & 0xFFFF_FFFF)

    def size(self) -> int:
        return self._size

    def peek(self, offset: int, length: int) -> bytes:
        return bytes(self._buf[offset:offset + length])

    def poke(self, offset: int, data: bytes):
        self._buf[offset:offset + len(data)] = data

    def poke32(self, offset: int, value: int):
        struct.pack_into("<I", self._buf, offset, value & 0xFFFF_FFFF)


class RealBarMmap(BarRegion):
    """Real mmap'd BAR region.

    Owns the file descriptor and mmap object.
    Provides safe close/cleanup with double-close protection.
    """
    def __init__(self, mm, fd: int, bar_size: int, bar_index: int,
                 readonly: bool = False):
        self._mm = mm
        self._fd = fd
        self._size = bar_size
        self._bar_index = bar_index
        self._readonly = readonly
        self._closed = False

    def read32(self, offset: int) -> int:
        if self._closed:
            raise MmapError("BarRegion is closed")
        self._check_access(offset)
        return struct.unpack_from("<I", self._mm, offset)[0]

    def write32(self, offset: int, value: int):
        if self._closed:
            raise MmapError("BarRegion is closed")
        if self._readonly:
            raise MmapError("BarRegion is read-only")
        self._check_access(offset)
        struct.pack_into("<I", self._mm, offset, value & 0xFFFF_FFFF)

    def size(self) -> int:
        return self._size

    def close(self):
        """Close mmap and fd. Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._mm is not None:
                self._mm.close()
        except Exception:
            pass
        try:
            if self._fd >= 0:
                os.close(self._fd)
        except Exception:
            pass

    def __del__(self):
        self.close()

    def __repr__(self):
        state = "closed" if self._closed else "open"
        return f"RealBarMmap(bar={self._bar_index}, size={self._size:#x}, {state})"


# ═══════════════════════════════════════════════════════════════════
# BarOpener — platform shim
# ═══════════════════════════════════════════════════════════════════

class BarOpener(ABC):
    @abstractmethod
    def open_bar(self, bar_index: int) -> BarRegion: ...
    @abstractmethod
    def close(self): ...


class FakeBarOpener(BarOpener):
    """Returns FakeMmap regions."""
    BAR_SIZES = {0: 0x100000, 2: 0x200000, 4: 0x10000}

    def __init__(self, bar_sizes: dict[int, int] | None = None):
        self._sizes = bar_sizes or self.BAR_SIZES
        self._regions: dict[int, FakeMmap] = {}

    def open_bar(self, bar_index: int) -> FakeMmap:
        if bar_index not in self._sizes:
            raise BarNotFoundError(f"BAR{bar_index} not defined")
        if bar_index not in self._regions:
            self._regions[bar_index] = FakeMmap(self._sizes[bar_index])
        return self._regions[bar_index]

    def close(self):
        self._regions.clear()


# ═══════════════════════════════════════════════════════════════════
# Linux sysfs PCI resource parsing
# ═══════════════════════════════════════════════════════════════════

# Single source for sysfs base path (never hardcoded elsewhere)
SYSFS_PCI_DEVICES = "/sys/bus/pci/devices"

# Expected BAR sizes from PCIE_BAR_SPEC
EXPECTED_BAR_SIZES = {0: 0x100000, 2: 0x200000, 4: 0x10000}

# ORBIT-G2 IDs (확인 필요 — will be set when PCIe IP is configured)
ORBIT_G2_VENDOR_ID = 0x10EE  # placeholder (Xilinx vendor ID for Versal)
ORBIT_G2_DEVICE_ID = 0x9038  # placeholder


def _parse_resource_file(resource_path: str | Path) -> list[tuple[int, int, int]]:
    """Parse sysfs 'resource' file → list of (start, end, flags) per BAR line."""
    entries = []
    try:
        text = Path(resource_path).read_text()
    except FileNotFoundError:
        return entries
    for line in text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            start = int(parts[0], 16)
            end = int(parts[1], 16)
            flags = int(parts[2], 16)
            entries.append((start, end, flags))
    return entries


def _bar_size_from_resource(entries: list[tuple[int, int, int]], bar_index: int) -> int:
    """Extract BAR size from parsed resource entries."""
    if bar_index >= len(entries):
        return 0
    start, end, _ = entries[bar_index]
    if start == 0 and end == 0:
        return 0
    return end - start + 1


def discover_orbit_devices(sysfs_base: str = SYSFS_PCI_DEVICES,
                           vendor_id: int = ORBIT_G2_VENDOR_ID,
                           device_id: int = ORBIT_G2_DEVICE_ID) -> list[str]:
    """Scan sysfs for ORBIT-G2 devices. Returns list of BDF strings."""
    base = Path(sysfs_base)
    if not base.is_dir():
        return []
    found = []
    for entry in sorted(base.iterdir()):
        try:
            v = int((entry / "vendor").read_text().strip(), 16)
            d = int((entry / "device").read_text().strip(), 16)
            if v == vendor_id and d == device_id:
                found.append(entry.name)
        except (FileNotFoundError, ValueError):
            continue
    return found


class LinuxPciBarOpener(BarOpener):
    """Opens BAR regions via Linux sysfs PCI resource files.

    Discovery:
      1. BDF-explicit: open_bar() uses self._bdf
      2. Vendor/device filter: use discover_orbit_devices() first

    Open sequence:
      1. Verify device exists at sysfs path
      2. Verify vendor/device IDs (if check_ids=True)
      3. Parse 'resource' file for BAR size
      4. Check BAR size >= expected
      5. Open resourceN with O_RDWR | O_SYNC
      6. mmap(fd, size, MAP_SHARED)

    All failures raise specific exceptions (no silent fallback).
    """

    def __init__(self, bdf: str, sysfs_base: str = SYSFS_PCI_DEVICES,
                 check_ids: bool = False,
                 expected_sizes: dict[int, int] | None = None):
        self._bdf = bdf
        self._sysfs_base = sysfs_base
        self._check_ids = check_ids
        self._expected = expected_sizes or EXPECTED_BAR_SIZES
        self._device_path = Path(sysfs_base) / bdf
        self._opened: list[RealBarMmap] = []

    @property
    def device_path(self) -> Path:
        return self._device_path

    def _verify_device(self):
        """Check device exists and optionally verify IDs."""
        if not self._device_path.is_dir():
            raise DeviceNotFoundError(
                f"PCI device not found: {self._device_path}\n"
                f"Check BDF '{self._bdf}' with 'lspci -s {self._bdf}'"
            )
        if self._check_ids:
            try:
                vendor = int((self._device_path / "vendor").read_text().strip(), 16)
                device = int((self._device_path / "device").read_text().strip(), 16)
            except FileNotFoundError:
                raise DeviceNotFoundError(
                    f"Cannot read vendor/device IDs at {self._device_path}"
                )
            if vendor != ORBIT_G2_VENDOR_ID or device != ORBIT_G2_DEVICE_ID:
                raise DeviceNotFoundError(
                    f"ID mismatch at {self._bdf}: "
                    f"got {vendor:#06x}:{device:#06x}, "
                    f"expected {ORBIT_G2_VENDOR_ID:#06x}:{ORBIT_G2_DEVICE_ID:#06x}"
                )

    def _get_bar_size(self, bar_index: int) -> int:
        """Parse 'resource' file for BAR size."""
        resource_path = self._device_path / "resource"
        entries = _parse_resource_file(resource_path)
        size = _bar_size_from_resource(entries, bar_index)
        return size

    def open_bar(self, bar_index: int) -> RealBarMmap:
        """Open a BAR region.

        Raises:
            DeviceNotFoundError: BDF path missing or ID mismatch
            BarNotFoundError: resourceN missing or zero-size BAR
            BarSizeError: BAR smaller than PCIE_BAR_SPEC requires
            PermissionDeniedError: cannot open resource file
            MmapOpenError: mmap failed
        """
        import mmap as mmap_mod

        self._verify_device()

        # Check BAR size
        bar_size = self._get_bar_size(bar_index)
        if bar_size == 0:
            raise BarNotFoundError(
                f"BAR{bar_index} not present or zero-size at {self._bdf}.\n"
                f"Check: 'lspci -vvs {self._bdf}' for BAR assignments."
            )

        expected = self._expected.get(bar_index, 0)
        if expected and bar_size < expected:
            raise BarSizeError(
                f"BAR{bar_index} size {bar_size:#x} < expected {expected:#x} "
                f"(from PCIE_BAR_SPEC). Check PCIe endpoint configuration."
            )

        # Open resource file
        resource_file = self._device_path / f"resource{bar_index}"
        if not resource_file.exists():
            raise BarNotFoundError(f"Resource file not found: {resource_file}")

        try:
            fd = os.open(str(resource_file), os.O_RDWR | os.O_SYNC)
        except PermissionError:
            raise PermissionDeniedError(
                f"Cannot open {resource_file}: permission denied.\n"
                "Remediation: add user to appropriate group or set udev rules.\n"
                f"  chmod a+rw {resource_file}  (temporary, for debug only)"
            )
        except OSError as e:
            raise MmapOpenError(f"Cannot open {resource_file}: {e}")

        # mmap
        try:
            mm = mmap_mod.mmap(fd, bar_size, mmap_mod.MAP_SHARED)
        except (OSError, ValueError) as e:
            os.close(fd)
            raise MmapOpenError(
                f"mmap failed for BAR{bar_index} (size={bar_size:#x}): {e}\n"
                "Check: IOMMU settings, available memory, BAR enable status."
            )

        region = RealBarMmap(mm, fd, bar_size, bar_index)
        self._opened.append(region)
        return region

    def close(self):
        """Close all opened BAR regions."""
        for region in self._opened:
            region.close()
        self._opened.clear()

    def __repr__(self):
        return f"LinuxPciBarOpener(bdf='{self._bdf}', path={self._device_path})"


# ═══════════════════════════════════════════════════════════════════
# Device probe helper
# ═══════════════════════════════════════════════════════════════════

def probe_orbit_device(bdf: str, sysfs_base: str = SYSFS_PCI_DEVICES) -> dict:
    """Quick probe: read IDs, BAR sizes, capabilities."""
    dev_path = Path(sysfs_base) / bdf
    if not dev_path.is_dir():
        return {"found": False, "bdf": bdf}

    result = {"found": True, "bdf": bdf}
    try:
        result["vendor"] = int((dev_path / "vendor").read_text().strip(), 16)
        result["device"] = int((dev_path / "device").read_text().strip(), 16)
    except (FileNotFoundError, ValueError):
        result["vendor"] = 0
        result["device"] = 0

    entries = _parse_resource_file(dev_path / "resource")
    result["bars"] = {}
    for idx in [0, 2, 4]:
        size = _bar_size_from_resource(entries, idx)
        expected = EXPECTED_BAR_SIZES.get(idx, 0)
        result["bars"][idx] = {
            "size": size,
            "expected": expected,
            "ok": size >= expected if expected else size > 0,
        }
    return result


# ═══════════════════════════════════════════════════════════════════
# MmapBackend
# ═══════════════════════════════════════════════════════════════════

class MmapBackend(Backend):
    """Backend using mmap'd BAR regions.

    Offset routing:
      [0x0_0000, 0x1_0000)  → BAR0 (registers)
      [0x1_0000, 0x2_0000)  → BAR4 (DMA engine)
      [0x2_0000, 0x10_0000) → BAR0 extended (OOM/TC0/Perf/IRQ/Trace)
    """
    BAR4_MIN = 0x1_0000
    BAR4_MAX = 0x2_0000

    def __init__(self, bar0: BarRegion, bar4: BarRegion | None = None):
        self._bar0 = bar0
        self._bar4 = bar4

    @classmethod
    def from_opener(cls, opener: BarOpener) -> "MmapBackend":
        bar0 = opener.open_bar(0)
        try:
            bar4 = opener.open_bar(4)
        except (BarNotFoundError, NotImplementedError):
            bar4 = None
        return cls(bar0, bar4)

    def read(self, offset: int) -> int:
        if self._bar4 is not None and self.BAR4_MIN <= offset < self.BAR4_MAX:
            return self._bar4.read32(offset - self.BAR4_MIN)
        return self._bar0.read32(offset)

    def write(self, offset: int, value: int):
        if self._bar4 is not None and self.BAR4_MIN <= offset < self.BAR4_MAX:
            self._bar4.write32(offset - self.BAR4_MIN, value)
        else:
            self._bar0.write32(offset, value)
