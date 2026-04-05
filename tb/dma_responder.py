"""
dma_responder.py — Test DMA Memory Model for ORBIT-G2 cocotb tests

Provides a cocotb coroutine that responds to DMA read/write requests
from gemm_core (via g2_ctrl_top DMA ports).

Features:
  - Address-keyed test memory (dict-based, initialized to deterministic pattern)
  - Read: returns data from test memory, beat-by-beat
  - Write: captures data into test memory
  - Proper handshake: rd_req_valid/ready, rd_data_valid/last, wr_req_valid/ready, etc.
  - Timeout detection: raises if no request arrives within max_idle cycles
  - Concurrent operation: run as cocotb.start_soon() alongside test logic

Usage:
    mem = DmaTestMemory()
    mem.fill_region(0x1000, 64, pattern=0x01)  # fill act data
    mem.fill_region(0x2000, 64, pattern=0x02)  # fill wgt data

    responder = DmaResponder(dut, mem)
    cocotb.start_soon(responder.run(expect_reads=2, expect_writes=1))
    # ... drive descriptor ...
    await responder.wait_done()
"""
from __future__ import annotations
import cocotb
from cocotb.triggers import RisingEdge

BEAT_BYTES = 16  # 128-bit DMA bus = 16 bytes per beat


class DmaTestMemory:
    """Simple byte-addressable test memory."""

    def __init__(self):
        self._mem: dict[int, int] = {}  # byte address -> byte value
        self.write_log: list[tuple[int, bytes]] = []  # (addr, data)

    def read_byte(self, addr: int) -> int:
        return self._mem.get(addr, 0)

    def write_byte(self, addr: int, val: int):
        self._mem[addr] = val & 0xFF

    def read_beat(self, addr: int) -> int:
        """Read 128-bit (16-byte) beat at addr, little-endian."""
        val = 0
        for i in range(BEAT_BYTES):
            val |= self.read_byte(addr + i) << (8 * i)
        return val

    def write_beat(self, addr: int, val: int):
        """Write 128-bit beat at addr, little-endian."""
        for i in range(BEAT_BYTES):
            self.write_byte(addr + i, (val >> (8 * i)) & 0xFF)

    def fill_region(self, base: int, size_bytes: int, pattern: int = 0xAB):
        """Fill region with repeating byte pattern."""
        for i in range(size_bytes):
            self._mem[base + i] = (pattern + i) & 0xFF

    def read_region(self, base: int, size_bytes: int) -> bytes:
        return bytes(self.read_byte(base + i) for i in range(size_bytes))


class DmaResponder:
    """Cocotb DMA read/write responder for g2_ctrl_top DUT."""

    def __init__(self, dut, mem: DmaTestMemory, max_idle: int = 2000):
        self._dut = dut
        self._mem = mem
        self._max_idle = max_idle
        self._done = False
        self._reads_served = 0
        self._writes_served = 0
        self._error: str | None = None

    @property
    def done(self) -> bool:
        return self._done

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def reads_served(self) -> int:
        return self._reads_served

    @property
    def writes_served(self) -> int:
        return self._writes_served

    async def run(self, expect_reads: int = 0, expect_writes: int = 0):
        """Serve DMA requests until expected counts are met or timeout."""
        total_expected = expect_reads + expect_writes
        served = 0
        idle = 0

        while served < total_expected and idle < self._max_idle:
            await RisingEdge(self._dut.clk)

            rd_req = False
            wr_req = False
            try:
                rd_req = int(self._dut.rd_req_valid.value) == 1
            except ValueError:
                pass
            try:
                wr_req = int(self._dut.wr_req_valid.value) == 1
            except ValueError:
                pass

            if rd_req and self._reads_served < expect_reads:
                await self._serve_read()
                served += 1
                idle = 0
            elif wr_req and self._writes_served < expect_writes:
                await self._serve_write()
                served += 1
                idle = 0
            else:
                idle += 1

        if idle >= self._max_idle:
            self._error = (f"DMA responder timeout: served {served}/{total_expected} "
                          f"(reads={self._reads_served}, writes={self._writes_served})")
        self._done = True

    async def _serve_read(self):
        """Handle one DMA read request."""
        dut = self._dut
        addr = int(dut.rd_req_addr.value)
        length = int(dut.rd_req_len_bytes.value)
        num_beats = max(1, length // BEAT_BYTES)

        # Accept request
        dut.rd_req_ready.value = 1
        await RisingEdge(dut.clk)
        dut.rd_req_ready.value = 0

        # Send data beats
        for beat in range(num_beats):
            beat_addr = addr + beat * BEAT_BYTES
            data = self._mem.read_beat(beat_addr)
            dut.rd_data_valid.value = 1
            dut.rd_data.value = data
            dut.rd_data_last.value = 1 if beat == num_beats - 1 else 0
            await RisingEdge(dut.clk)

        dut.rd_data_valid.value = 0
        dut.rd_data_last.value = 0
        self._reads_served += 1

    async def _serve_write(self):
        """Handle one DMA write request."""
        dut = self._dut
        addr = int(dut.wr_req_addr.value)

        # Accept request
        dut.wr_req_ready.value = 1
        await RisingEdge(dut.clk)
        dut.wr_req_ready.value = 0

        # Receive data beats
        dut.wr_data_ready.value = 1
        captured = bytearray()
        beat_idx = 0

        for _ in range(2000):  # safety limit
            await RisingEdge(dut.clk)
            try:
                valid = int(dut.wr_data_valid.value) == 1
            except ValueError:
                valid = False

            if valid:
                data = int(dut.wr_data.value)
                beat_addr = addr + beat_idx * BEAT_BYTES
                self._mem.write_beat(beat_addr, data)
                for i in range(BEAT_BYTES):
                    captured.append((data >> (8 * i)) & 0xFF)
                beat_idx += 1

                try:
                    last = int(dut.wr_data_last.value) == 1
                except ValueError:
                    last = False
                if last:
                    break

        dut.wr_data_ready.value = 0
        self._mem.write_log.append((addr, bytes(captured)))
        self._writes_served += 1

    async def wait_done(self, timeout_cycles: int = 5000):
        """Wait until responder is done."""
        for _ in range(timeout_cycles):
            if self._done:
                return
            await RisingEdge(self._dut.clk)
        if not self._done:
            self._error = "wait_done timeout"
            self._done = True
