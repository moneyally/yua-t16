"""test_orbit_dma.py — DMA software contract tests."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.orbit_protob_mock import ProtoBMockBackend
from tools.orbit_dma import OrbitDma, DmaDir


@pytest.fixture
def dma():
    return OrbitDma(ProtoBMockBackend())


class TestDmaSubmit:
    def test_submit_h2d(self, dma):
        dma.submit_h2d(iova=0x1000, length=4096)
        st = dma.read_status()
        assert st.done is True
        assert st.err is False

    def test_submit_d2h(self, dma):
        dma.submit_d2h(iova=0x2000, length=1024, queue=1)
        st = dma.read_status()
        assert st.done is True

    def test_wait_done_ok(self, dma):
        dma.submit_h2d(0x1000, 256)
        comp = dma.wait_done()
        assert comp.ok is True
        assert comp.polls == 1

    def test_submit_logs_request(self, dma):
        backend = dma._b
        dma.submit_h2d(0xDEAD_0000, 512, queue=2, irq=False)
        assert len(backend.dma_log) == 1
        req = backend.dma_log[0]
        assert req.iova == 0xDEAD_0000
        assert req.length == 512
        assert req.direction == DmaDir.H2D
        assert req.queue == 2


class TestDmaError:
    def test_error_inject(self):
        b = ProtoBMockBackend()
        b.inject_error = True
        b.inject_error_code = 0xAB
        dma = OrbitDma(b)
        dma.submit_h2d(0x1000, 256)
        st = dma.read_status()
        assert st.err is True
        assert st.done is False
        assert dma.read_error() == 0xAB

    def test_timeout_inject(self):
        b = ProtoBMockBackend()
        b.inject_timeout = True
        dma = OrbitDma(b)
        dma.submit_h2d(0x1000, 256)
        st = dma.read_status()
        assert st.timeout is True
        comp = dma.wait_done()
        assert comp.ok is False
        assert comp.timed_out is True

    def test_error_cleared_on_new_submit(self):
        b = ProtoBMockBackend()
        b.inject_error = True
        dma = OrbitDma(b)
        dma.submit_h2d(0x1000, 256)
        assert dma.read_status().err is True
        b.inject_error = False
        dma.submit_h2d(0x2000, 128)
        assert dma.read_status().err is False
        assert dma.read_status().done is True


class TestDmaConfig:
    def test_set_throttle(self, dma):
        dma.set_throttle(0x00FF)
        # Readable via backend
        assert dma._b.read(0x1_0018) == 0x00FF

    def test_set_timeout(self, dma):
        dma.set_timeout(50000)
        assert dma._b.read(0x1_001C) == 50000
