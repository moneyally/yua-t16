"""test_protob_mock_flow.py — Proto-B mock vertical slice tests."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.orbit_protob_mock import ProtoBMockBackend
from tools.orbit_device import OrbitDevice
from tools.orbit_dma import OrbitDma
from tools.orbit_scheduler import OrbitScheduler, FaultPolicy
from tools.orbit_session import OpStatus
from tools.orbit_poll import read_irq_status
from tools.orbit_mmio_map import IrqBit, IRQ_PENDING, IRQ_MASK


@pytest.fixture
def setup():
    b = ProtoBMockBackend()
    dev = OrbitDevice(b)
    dev.connect()
    dev.enable_trace()
    return b, dev


class TestProtoBVerticalSlice:
    def test_mmio_dma_submit_done_irq_clear(self, setup):
        """MMIO write → DMA submit → status DONE → IRQ → clear."""
        b, dev = setup
        dma = OrbitDma(b)

        # Unmask DMA_DONE IRQ
        b.write(IRQ_MASK.offset, 0xFFFF_FFFD)  # unmask bit 1

        # Submit DMA
        dma.submit_h2d(0x1000, 4096, irq=True)

        # Status should be DONE
        st = dma.read_status()
        assert st.done is True
        assert st.busy is False

        # IRQ DMA_DONE should be pending
        irq = read_irq_status(b)
        assert "DMA_DONE" in irq["sources"]

        # Clear
        b.write(IRQ_PENDING.offset, 1 << IrqBit.DMA_DONE)
        irq2 = read_irq_status(b)
        assert "DMA_DONE" not in irq2["sources"]

    def test_dma_error_inject(self, setup):
        b, dev = setup
        b.inject_error = True
        b.inject_error_code = 0xBB

        dma = OrbitDma(b)
        dma.submit_h2d(0x1000, 256)
        st = dma.read_status()
        assert st.err is True
        assert dma.read_error() == 0xBB

    def test_dma_timeout_inject(self, setup):
        b, dev = setup
        b.inject_timeout = True

        dma = OrbitDma(b)
        dma.submit_h2d(0x1000, 256)
        comp = dma.wait_done()
        assert comp.ok is False
        assert comp.timed_out is True

    def test_throttle_timeout_registers(self, setup):
        b, dev = setup
        dma = OrbitDma(b)
        dma.set_throttle(0x00FF)
        dma.set_timeout(100000)
        assert b.read(0x1_0018) == 0x00FF
        assert b.read(0x1_001C) == 100000


class TestProtoBScheduler:
    def test_protob_mode_nop(self, setup):
        b, dev = setup
        sched = OrbitScheduler(dev, mode="protob")
        sched.add_nop()
        result = sched.run()
        assert result.ops_completed == 1

    def test_protob_mode_unsupported(self, setup):
        b, dev = setup
        sched = OrbitScheduler(dev, mode="protob")
        sched.add_unsupported(0x03)  # KVC
        result = sched.run()
        assert result.ops_rejected == 1

    def test_protob_backward_compat(self, setup):
        """Proto-A mode still works on ProtoBMockBackend."""
        b, dev = setup
        sched = OrbitScheduler(dev, mode="protoa")
        sched.add_nop()
        result = sched.run()
        assert result.ops_completed == 1


class TestProtoBDmaLog:
    def test_submit_logged(self, setup):
        b, dev = setup
        dma = OrbitDma(b)
        dma.submit_h2d(0xAAAA, 512, queue=1)
        dma.submit_d2h(0xBBBB, 1024, queue=2)
        assert len(b.dma_log) == 2
        assert b.dma_log[0].iova == 0xAAAA
        assert b.dma_log[0].direction == 0
        assert b.dma_log[1].iova == 0xBBBB
        assert b.dma_log[1].direction == 1
