"""
orbit_protob_mock.py — Proto-B Mock Backend

Simulates Proto-B DMA engine behavior on top of SimBackend.
Provides: DMA submit queue, completion generation, timeout/error inject.

Purpose: validate software stack against Proto-B contract
         BEFORE real PCIe/DMA RTL exists.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from tools.orbit_backend import SimBackend
from tools.orbit_mmio_map import (
    IRQ_PENDING, IrqBit, BOOT_CAUSE, BOOT_CAUSE_POR,
)
from tools.orbit_dma import (
    DMA_SUBMIT_LO_OFF, DMA_SUBMIT_HI_OFF, DMA_LEN_OFF, DMA_CTRL_OFF,
    DMA_STATUS_OFF, DMA_ERR_CODE_OFF, DMA_THROTTLE_OFF, DMA_TIMEOUT_OFF,
    DMA_BUSY, DMA_DONE, DMA_ERR, DMA_TIMEOUT,
)


@dataclass
class DmaRequest:
    iova: int
    length: int
    direction: int  # 0=H2D, 1=D2H
    queue: int
    irq_en: bool


class ProtoBMockBackend(SimBackend):
    """Extended SimBackend with DMA engine simulation.

    Intercepts DMA register writes to simulate submit/completion cycle.
    Configurable: inject_error, inject_timeout.
    """

    def __init__(self):
        super().__init__()
        self.set_ro(BOOT_CAUSE.offset, BOOT_CAUSE_POR)

        # DMA state
        self._dma_submit_lo = 0
        self._dma_submit_hi = 0
        self._dma_len = 0
        self._dma_busy = False
        self._dma_done = False
        self._dma_err = False
        self._dma_timeout = False
        self._dma_err_code = 0
        self._dma_throttle = 0
        self._dma_timeout_cycles = 0
        self._dma_inflight = 0

        # Submitted requests log
        self.dma_log: list[DmaRequest] = []

        # Inject controls
        self.inject_error: bool = False
        self.inject_timeout: bool = False
        self.inject_error_code: int = 0xEE

    def write(self, offset: int, value: int):
        value &= 0xFFFF_FFFF

        # Intercept DMA registers
        if offset == DMA_SUBMIT_LO_OFF:
            self._dma_submit_lo = value
            return
        elif offset == DMA_SUBMIT_HI_OFF:
            self._dma_submit_hi = value
            return
        elif offset == DMA_LEN_OFF:
            self._dma_len = value
            return
        elif offset == DMA_CTRL_OFF:
            if value & 0x01:  # START bit
                self._handle_dma_submit(value)
            return
        elif offset == DMA_THROTTLE_OFF:
            self._dma_throttle = value
            return
        elif offset == DMA_TIMEOUT_OFF:
            self._dma_timeout_cycles = value
            return

        # All other registers: normal SimBackend
        super().write(offset, value)

    def read(self, offset: int) -> int:
        if offset == DMA_STATUS_OFF:
            return self._build_dma_status()
        elif offset == DMA_ERR_CODE_OFF:
            return self._dma_err_code
        elif offset == DMA_THROTTLE_OFF:
            return self._dma_throttle
        elif offset == DMA_TIMEOUT_OFF:
            return self._dma_timeout_cycles

        return super().read(offset)

    def _handle_dma_submit(self, ctrl: int):
        """Process DMA submit."""
        direction = (ctrl >> 1) & 0x1
        queue = (ctrl >> 2) & 0x3
        irq_en = bool((ctrl >> 8) & 0x1)
        iova = self._dma_submit_lo | (self._dma_submit_hi << 32)

        req = DmaRequest(
            iova=iova, length=self._dma_len,
            direction=direction, queue=queue, irq_en=irq_en,
        )
        self.dma_log.append(req)

        # Clear previous error state on new submit
        self._dma_err = False
        self._dma_timeout = False
        self._dma_err_code = 0

        # Simulate immediate completion (Proto-B mock: no real latency)
        if self.inject_error:
            self._dma_err = True
            self._dma_err_code = self.inject_error_code
            self._dma_done = False
            if irq_en:
                self.set_pending(IRQ_PENDING.offset, 1 << IrqBit.DMA_ERROR)
        elif self.inject_timeout:
            self._dma_timeout = True
            self._dma_done = False
            if irq_en:
                self.set_pending(IRQ_PENDING.offset, 1 << IrqBit.DMA_ERROR)
        else:
            self._dma_done = True
            self._dma_err = False
            if irq_en:
                self.set_pending(IRQ_PENDING.offset, 1 << IrqBit.DMA_DONE)

        self._dma_busy = False  # instant completion for mock

    def _build_dma_status(self) -> int:
        val = 0
        if self._dma_busy: val |= DMA_BUSY
        if self._dma_done: val |= DMA_DONE
        if self._dma_err:  val |= DMA_ERR
        if self._dma_timeout: val |= DMA_TIMEOUT
        val |= (self._dma_inflight & 0xFF) << 8
        return val
