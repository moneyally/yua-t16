"""test_orbit_bringup.py — Bringup flow tests."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.orbit_backend import SimBackend
from tools.orbit_device import OrbitDevice
from tools.orbit_bringup import bringup, run_gemm_smoke, collect_diagnostics
from tools.orbit_mmio_map import BOOT_CAUSE, BOOT_CAUSE_POR


@pytest.fixture
def dev():
    b = SimBackend()
    b.set_ro(BOOT_CAUSE.offset, BOOT_CAUSE_POR)
    return OrbitDevice(b)


class TestBringup:
    def test_bringup_ok(self, dev):
        report = bringup(dev)
        assert report.info.device_id == 0x4732_0001
        assert "POR" in report.boot_cause
        assert report.queues_ok is True
        assert report.irq_clear is True
        assert report.trace_enabled is True
        assert report.health == "OK"

    def test_bringup_detects_fault(self, dev):
        from tools.orbit_mmio_map import TC0_FAULT_STATUS, TC0_RUNSTATE
        dev.backend.set_ro(TC0_RUNSTATE.offset, 4)  # FAULT
        dev.backend.set_ro(TC0_FAULT_STATUS.offset, 0x01)
        report = bringup(dev)
        assert "fault" in report.health.lower() or "not idle" in report.health.lower()


class TestSmokeGemm:
    def test_nop_smoke(self, dev):
        bringup(dev)
        result = run_gemm_smoke(dev)
        assert result.nop_ok is True
        # GEMM can't complete in SimBackend (no DMA)
        assert result.gemm_ok is False

    def test_smoke_no_errors(self, dev):
        bringup(dev)
        result = run_gemm_smoke(dev)
        assert len(result.errors) == 0


class TestDiagnostics:
    def test_collect_at_idle(self, dev):
        bringup(dev)
        diag = collect_diagnostics(dev)
        assert diag.tc_state == 0  # IDLE
        assert diag.oom_state == "NORMAL"
        assert diag.irq_pending == 0
        assert diag.dma_busy is False

    def test_collect_with_irq(self, dev):
        from tools.orbit_mmio_map import IRQ_PENDING, IrqBit
        bringup(dev)
        dev.backend.set_pending(IRQ_PENDING.offset, 1 << IrqBit.TRACE_WRAP)
        diag = collect_diagnostics(dev)
        assert "TRACE_WRAP" in diag.irq_sources

    def test_collect_boot_cause(self, dev):
        bringup(dev)
        diag = collect_diagnostics(dev)
        assert "POR" in diag.boot_cause
