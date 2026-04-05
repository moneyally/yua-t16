"""
orbit_dma.py — ORBIT-G2 DMA Software Contract

Host-side DMA submit/poll/error API.
Based on REG_SPEC section 5 (DMA Engine) and PCIE_BAR_SPEC section 4.

This module defines the software contract for DMA operations.
The actual register writes go through the Backend abstraction.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from tools.orbit_backend import Backend
from tools.orbit_mmio_map import (
    DMA_STATUS, DMA_ERR_CODE,
    DMA_BUSY, DMA_DONE, DMA_ERR, DMA_TIMEOUT,
)

# DMA register offsets within BAR4 (= REG_SPEC offset 0x1_0000+)
# These match REG_SPEC 0x8031_xxxx -> offset 0x1_xxxx from BASE
DMA_SUBMIT_LO_OFF = 0x1_0000
DMA_SUBMIT_HI_OFF = 0x1_0004
DMA_LEN_OFF       = 0x1_0008
DMA_CTRL_OFF      = 0x1_000C
DMA_STATUS_OFF    = 0x1_0010
DMA_ERR_CODE_OFF  = 0x1_0014
DMA_THROTTLE_OFF  = 0x1_0018
DMA_TIMEOUT_OFF   = 0x1_001C


class DmaDir(IntEnum):
    H2D = 0  # host to device
    D2H = 1  # device to host


class DmaError(Exception):
    """DMA operation error."""


@dataclass(frozen=True)
class DmaStatus:
    busy: bool
    done: bool
    err: bool
    timeout: bool
    inflight: int
    raw: int


@dataclass(frozen=True)
class DmaCompletion:
    ok: bool
    error_code: int
    timed_out: bool
    polls: int


class OrbitDma:
    """DMA engine software interface.

    Submit sequence (BAR_SPEC section 4.1):
      1. write DMA_SUBMIT_LO
      2. write DMA_SUBMIT_HI
      3. write DMA_LEN
      4. write DMA_CTRL with START=1 (must be last)

    Completion:
      poll DMA_STATUS until !BUSY, then check DONE/ERR/TIMEOUT.
    """

    def __init__(self, backend: Backend):
        self._b = backend

    def submit_h2d(self, iova: int, length: int, queue: int = 0,
                   qos: int = 0, irq: bool = True):
        """Submit host-to-device DMA transfer."""
        self._submit(iova, length, DmaDir.H2D, queue, qos, irq)

    def submit_d2h(self, iova: int, length: int, queue: int = 0,
                   qos: int = 0, irq: bool = True):
        """Submit device-to-host DMA transfer."""
        self._submit(iova, length, DmaDir.D2H, queue, qos, irq)

    def _submit(self, iova: int, length: int, direction: DmaDir,
                queue: int, qos: int, irq: bool):
        """Write DMA submit registers. START must be last write."""
        self._b.write(DMA_SUBMIT_LO_OFF, iova & 0xFFFF_FFFF)
        self._b.write(DMA_SUBMIT_HI_OFF, (iova >> 32) & 0xFFFF_FFFF)
        self._b.write(DMA_LEN_OFF, length & 0xFFFF_FFFF)

        ctrl = 0x01  # START=1
        ctrl |= (int(direction) & 0x1) << 1
        ctrl |= (queue & 0x3) << 2
        ctrl |= (qos & 0xF) << 4
        ctrl |= (int(irq) & 0x1) << 8
        self._b.write(DMA_CTRL_OFF, ctrl)

    def read_status(self) -> DmaStatus:
        raw = self._b.read(DMA_STATUS_OFF)
        return DmaStatus(
            busy=bool(raw & DMA_BUSY),
            done=bool(raw & DMA_DONE),
            err=bool(raw & DMA_ERR),
            timeout=bool(raw & DMA_TIMEOUT),
            inflight=(raw >> 8) & 0xFF,
            raw=raw,
        )

    def read_error(self) -> int:
        return self._b.read(DMA_ERR_CODE_OFF)

    def clear_error(self):
        """Errors are cleared on next DMA_CTRL.START (REG_SPEC)."""
        pass  # No explicit clear register; next submit clears

    def set_timeout(self, cycles: int):
        self._b.write(DMA_TIMEOUT_OFF, cycles & 0xFFFF_FFFF)

    def set_throttle(self, value: int):
        self._b.write(DMA_THROTTLE_OFF, value & 0xFFFF_FFFF)

    def wait_done(self, max_polls: int = 10000) -> DmaCompletion:
        """Poll DMA_STATUS until not BUSY."""
        for i in range(max_polls):
            st = self.read_status()
            if not st.busy:
                return DmaCompletion(
                    ok=st.done and not st.err and not st.timeout,
                    error_code=self.read_error() if st.err else 0,
                    timed_out=st.timeout,
                    polls=i + 1,
                )
        return DmaCompletion(ok=False, error_code=0, timed_out=True, polls=max_polls)
