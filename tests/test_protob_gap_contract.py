"""test_protob_gap_contract.py — Proto-B contract verification.

Ensures software abstractions are consistent across Proto-A and Proto-B
backend implementations.
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.orbit_backend import SimBackend, Backend
from tools.orbit_protob_mock import ProtoBMockBackend
from tools.orbit_mmap_backend import MmapBackend, FakeMmap
from tools.orbit_device import OrbitDevice
from tools.orbit_mmio_map import G2_ID, BOOT_CAUSE, BOOT_CAUSE_POR


class TestBackendContract:
    """All backends satisfy the same abstract interface."""

    def _make_backends(self):
        sim = SimBackend()
        protob = ProtoBMockBackend()
        bar0 = FakeMmap(0x100000)
        bar0.poke32(G2_ID.offset, 0x4732_0001)
        bar0.poke32(0x4004, 0x01)  # TC0_CTRL default
        mmap = MmapBackend(bar0)
        return [("SimBackend", sim), ("ProtoBMock", protob), ("MmapBackend", mmap)]

    def test_all_are_backend_instances(self):
        for name, b in self._make_backends():
            assert isinstance(b, Backend), f"{name} is not a Backend"

    def test_read_write_cycle(self):
        """Write then read on all writable backends."""
        for name, b in self._make_backends():
            # Write to a RW register (TC0_CTRL at offset 0x4_0004)
            b.write(0x4_0004, 0x03)
            val = b.read(0x4_0004)
            assert val == 0x03, f"{name}: write/read mismatch: {val}"

    def test_g2_id_readable(self):
        for name, b in self._make_backends():
            val = b.read(G2_ID.offset)
            # SimBackend and ProtoBMock have reset defaults
            # MmapBackend has poked value
            assert isinstance(val, int), f"{name}: read returned non-int"


class TestOrbitDevicePortability:
    """OrbitDevice works with any backend."""

    def test_connect_sim(self):
        dev = OrbitDevice(SimBackend())
        info = dev.connect()
        assert info.device_id == 0x4732_0001

    def test_connect_protob_mock(self):
        dev = OrbitDevice(ProtoBMockBackend())
        info = dev.connect()
        assert info.device_id == 0x4732_0001

    def test_connect_mmap(self):
        bar0 = FakeMmap(0x100000)
        bar0.poke32(G2_ID.offset, 0x4732_0001)
        bar0.poke32(0x0004, 0x0001_0000)  # VERSION
        bar0.poke32(0x0008, 0x0000_0060)  # CAP0
        dev = OrbitDevice(MmapBackend(bar0))
        info = dev.connect()
        assert info.device_id == 0x4732_0001


class TestDmaContractConsistency:
    """DMA API works identically on ProtoBMock."""

    def test_submit_poll_complete(self):
        from tools.orbit_dma import OrbitDma
        b = ProtoBMockBackend()
        dma = OrbitDma(b)
        dma.submit_h2d(0x1000, 256)
        comp = dma.wait_done()
        assert comp.ok is True

    def test_submit_error_path(self):
        from tools.orbit_dma import OrbitDma
        b = ProtoBMockBackend()
        b.inject_error = True
        dma = OrbitDma(b)
        dma.submit_h2d(0x1000, 256)
        comp = dma.wait_done()
        assert comp.ok is False
