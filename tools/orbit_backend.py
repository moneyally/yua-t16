"""
orbit_backend.py — MMIO Backend Abstraction

Backends:
  SimBackend  — dict-based register simulator (for unit tests / offline)
  MmapBackend — stub interface (Proto-B/ASIC: real mmap)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from tools.orbit_mmio_map import (
    ALL_REGS, Access, BASE,
    G2_ID, G2_VERSION, G2_CAP0, TC0_CTRL, IRQ_MASK,
)


class Backend(ABC):
    @abstractmethod
    def read(self, offset: int) -> int:
        """Read 32-bit value at offset from BASE."""
        ...

    @abstractmethod
    def write(self, offset: int, value: int):
        """Write 32-bit value at offset from BASE."""
        ...

    def read_abs(self, addr: int) -> int:
        return self.read(addr - BASE)

    def write_abs(self, addr: int, value: int):
        self.write(addr - BASE, value)


class SimBackend(Backend):
    """Dict-based register simulator.

    - RO registers return their reset value (or last set_ro value)
    - RW registers are writable
    - WO registers accept writes (stored for observation) but read as 0
    - W1C registers: write-1-to-clear bits
    """

    def __init__(self):
        self._regs: dict[int, int] = {}  # offset -> value
        self._access: dict[int, str] = {}
        self._w1c_regs: set[int] = set()
        self._wo_regs: set[int] = set()

        # Initialize from ALL_REGS
        for r in ALL_REGS:
            off = r.offset
            self._regs[off] = r.reset
            self._access[off] = r.access
            if r.access == Access.W1C:
                self._w1c_regs.add(off)
            if r.access == Access.WO:
                self._wo_regs.add(off)

    def read(self, offset: int) -> int:
        if offset in self._wo_regs:
            return 0  # WO reads as 0
        return self._regs.get(offset, 0)

    def write(self, offset: int, value: int):
        value &= 0xFFFF_FFFF
        if offset in self._w1c_regs:
            # Write-1-to-clear
            current = self._regs.get(offset, 0)
            self._regs[offset] = current & ~value
        elif self._access.get(offset) == Access.RO:
            pass  # ignore writes to RO
        else:
            self._regs[offset] = value

    def set_ro(self, offset: int, value: int):
        """Set a read-only register value (for simulation)."""
        self._regs[offset] = value & 0xFFFF_FFFF

    def set_pending(self, offset: int, bits: int):
        """OR bits into a W1C register (simulates HW setting pending)."""
        current = self._regs.get(offset, 0)
        self._regs[offset] = current | (bits & 0xFFFF_FFFF)


class MmapBackend(Backend):
    """Stub for future real MMIO access. Not implemented in Proto-A."""

    def __init__(self, base_addr: int = BASE):
        raise NotImplementedError(
            "MmapBackend requires real hardware. Use SimBackend for Proto-A."
        )

    def read(self, offset: int) -> int:
        raise NotImplementedError

    def write(self, offset: int, value: int):
        raise NotImplementedError
