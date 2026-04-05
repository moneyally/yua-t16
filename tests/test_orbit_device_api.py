"""test_orbit_device_api.py — OrbitDevice HAL API tests."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.orbit_backend import SimBackend
from tools.orbit_device import OrbitDevice, UnsupportedOpcodeError, DeviceError
from tools.orbit_desc import pack_nop, pack_gemm, pack_descriptor
from tools.orbit_mmio_map import (
    BOOT_CAUSE, BOOT_CAUSE_POR, IRQ_PENDING, IrqBit,
    TC0_RUNSTATE, TC0_FAULT_STATUS, OOM_STATE,
    Opcode,
)


@pytest.fixture
def dev():
    b = SimBackend()
    b.set_ro(BOOT_CAUSE.offset, BOOT_CAUSE_POR)
    return OrbitDevice(b)


class TestConnect:
    def test_connect_success(self, dev):
        info = dev.connect()
        assert info.device_id == 0x4732_0001
        assert info.version == 0x0001_0000

    def test_connect_bad_id(self):
        b = SimBackend()
        b.set_ro(0x0_0000, 0xDEAD_BEEF)  # wrong ID
        d = OrbitDevice(b)
        with pytest.raises(DeviceError, match="Not an ORBIT-G2"):
            d.connect()


class TestEnqueue:
    def test_nop_enqueue(self, dev):
        dev.enqueue_desc(pack_nop(), queue=0)

    def test_gemm_enqueue(self, dev):
        dev.enqueue_desc(pack_gemm(0x1000, 0x2000, 0x3000, 4), queue=0)

    def test_unsupported_opcode_rejected(self, dev):
        desc = pack_descriptor(Opcode.KVC_OP)
        with pytest.raises(UnsupportedOpcodeError, match="not supported"):
            dev.enqueue_desc(desc)

    def test_vpu_opcode_rejected(self, dev):
        desc = pack_descriptor(Opcode.VPU_OP)
        with pytest.raises(UnsupportedOpcodeError):
            dev.enqueue_desc(desc)

    def test_invalid_queue(self, dev):
        with pytest.raises(DeviceError, match="Invalid queue"):
            dev.enqueue_desc(pack_nop(), queue=5)

    def test_wrong_size(self, dev):
        with pytest.raises(DeviceError, match="size mismatch"):
            dev.enqueue_desc(b"\x01\x02\x03")


class TestStatus:
    def test_tc_idle_at_start(self, dev):
        tc = dev.read_tc_status()
        assert tc.state == 0  # IDLE
        assert tc.enable is True

    def test_poll_tc_idle(self, dev):
        assert dev.poll_tc_idle(max_polls=10) is True

    def test_queue_status_empty(self, dev):
        qs = dev.read_queue_status(0)
        assert qs.depth == 0
        assert qs.overflow is False

    def test_oom_normal(self, dev):
        assert dev.read_oom_state() == 0


class TestIRQ:
    def test_poll_irq_empty(self, dev):
        assert dev.poll_irq() == 0

    def test_clear_irq(self, dev):
        dev.backend.set_pending(IRQ_PENDING.offset, 1 << IrqBit.DESC_DONE)
        assert dev.poll_irq() & 1
        dev.clear_irq(1 << IrqBit.DESC_DONE)
        assert dev.poll_irq() == 0


class TestFault:
    def test_clear_fault(self, dev):
        dev.backend.set_ro(TC0_FAULT_STATUS.offset, 0x02)
        dev.backend.set_pending(IRQ_PENDING.offset, 1 << IrqBit.TC0_FAULT)
        dev.clear_fault()
        # W1C should clear fault status
        assert dev.backend.read(TC0_FAULT_STATUS.offset) == 0


class TestTrace:
    def test_enable_disable(self, dev):
        dev.enable_trace()
        dev.disable_trace()

    def test_dump_empty(self, dev):
        entries = dev.dump_trace()
        assert entries == []


class TestPerf:
    def test_freeze_unfreeze(self, dev):
        dev.freeze_perf()
        dev.unfreeze_perf()

    def test_mxu_zero_at_start(self, dev):
        assert dev.read_mxu_busy_cycles() == 0
        assert dev.read_tile_count() == 0
