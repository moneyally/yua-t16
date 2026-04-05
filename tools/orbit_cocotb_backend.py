"""
orbit_cocotb_backend.py — CocotbBackend for ORBIT-G2

Wraps a cocotb DUT (g2_ctrl_top) so the same Python host stack
(OrbitDevice, scheduler, bringup) works against the RTL simulation.

Usage inside a cocotb test:
    from tools.orbit_cocotb_backend import CocotbBackend
    from tools.orbit_device import OrbitDevice

    backend = CocotbBackend(dut)
    dev = OrbitDevice(backend)
    await backend.reset()
    info = dev.connect()
    dev.enqueue_desc(pack_nop())
    await backend.tick(50)
    assert dev.poll_tc_idle()
"""
from __future__ import annotations
from tools.orbit_backend import Backend


class CocotbBackend(Backend):
    """Backend that drives g2_ctrl_top DUT via cocotb signal access.

    read/write are synchronous (cocotb is event-driven, but register
    bus is single-cycle in Proto-A, so we use blocking access via
    immediate signal assignment + settle).

    For cocotb coroutine-based access, use the async helpers:
        await backend.async_read(offset)
        await backend.async_write(offset, value)
    """

    def __init__(self, dut):
        self._dut = dut
        self._clk = dut.clk

    # ── Synchronous access (for OrbitDevice API compatibility) ──
    # These set signals immediately. Caller must await clock edges
    # for the values to take effect in RTL.
    # For true synchronous reg access, use async_read/async_write.

    def read(self, offset: int) -> int:
        """Set address and return current rd_data (combinational).

        NOTE: In cocotb, this only works if the caller has already
        awaited a clock edge after the last write. For reliable use,
        prefer async_read().
        """
        self._dut.reg_addr.value = offset
        self._dut.reg_wr_en.value = 0
        # Return whatever rd_data shows (combinational mux in reg_top)
        try:
            return int(self._dut.reg_rd_data.value)
        except ValueError:
            return 0  # X/Z during reset

    def write(self, offset: int, value: int):
        """Set address+data+wr_en. Caller must advance clock."""
        self._dut.reg_addr.value = offset
        self._dut.reg_wr_en.value = 1
        self._dut.reg_wr_data.value = value & 0xFFFF_FFFF

    # ── Async helpers (for use inside cocotb coroutines) ────────

    async def async_write(self, offset: int, value: int):
        """Write register and advance one clock edge."""
        from cocotb.triggers import RisingEdge
        self._dut.reg_addr.value = offset
        self._dut.reg_wr_en.value = 1
        self._dut.reg_wr_data.value = value & 0xFFFF_FFFF
        await RisingEdge(self._clk)
        self._dut.reg_wr_en.value = 0

    async def async_read(self, offset: int) -> int:
        """Read register with proper clock synchronization."""
        from cocotb.triggers import RisingEdge
        self._dut.reg_addr.value = offset
        self._dut.reg_wr_en.value = 0
        await RisingEdge(self._clk)
        return int(self._dut.reg_rd_data.value)

    async def reset(self):
        """Apply power-on reset and wait for completion."""
        from cocotb.triggers import RisingEdge, Timer
        self._dut.por_n.value = 0
        self._dut.reg_addr.value = 0
        self._dut.reg_wr_en.value = 0
        self._dut.reg_wr_data.value = 0
        self._dut.rd_req_ready.value = 1
        self._dut.rd_done.value = 0
        self._dut.rd_data_valid.value = 0
        self._dut.rd_data.value = 0
        self._dut.rd_data_last.value = 0
        self._dut.wr_req_ready.value = 1
        self._dut.wr_done.value = 0
        self._dut.wr_data_ready.value = 1
        await Timer(100, unit="ns")
        self._dut.por_n.value = 1
        for _ in range(30):
            await RisingEdge(self._clk)
            if self._dut.reset_active.value == 0:
                break
        await RisingEdge(self._clk)

    async def tick(self, n: int = 1):
        """Advance n clock cycles."""
        from cocotb.triggers import RisingEdge
        for _ in range(n):
            await RisingEdge(self._clk)

    async def enqueue_and_tick(self, dev, desc: bytes, queue: int = 0,
                                wait_cycles: int = 50):
        """Enqueue descriptor via HAL, then tick for processing.

        Uses async_write for each staging register + doorbell.
        """
        from tools.orbit_desc import desc_to_words
        from tools.orbit_mmio_map import DESC_STAGE_BASE, BASE, QUEUE_DOORBELLS

        words = desc_to_words(desc)
        stage_off = DESC_STAGE_BASE - BASE
        for i, w in enumerate(words):
            await self.async_write(stage_off + i * 4, w)
        await self.async_write(QUEUE_DOORBELLS[queue].offset, 0x0001)
        await self.tick(wait_cycles)
