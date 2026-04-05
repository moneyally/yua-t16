"""
tb_g2_ctrl_top_host_e2e.py — Host-driven co-simulation E2E tests

Uses the real Python host stack (CocotbBackend → OrbitDevice) against
the RTL DUT (g2_ctrl_top) with a DMA responder providing memory.

Tests:
  1. GEMM E2E: bringup → enqueue → DMA → completion → trace → IRQ
  2. NOP via host stack
  3. Illegal opcode → fault → clear
  4. Trace/IRQ consistency
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.orbit_cocotb_backend import CocotbBackend
from tools.orbit_device import OrbitDevice, UnsupportedOpcodeError
from tools.orbit_desc import pack_nop, pack_gemm, pack_descriptor
from tools.orbit_mmio_map import (
    G2_ID, TRACE_CTRL, TRACE_TAIL, IRQ_PENDING, IRQ_MASK,
    TC0_RUNSTATE, TC0_FAULT_STATUS, DMA_STATUS,
    TRACE_WIN_BASE, TRACE_META_BASE, BASE,
    IrqBit, TraceType, Opcode,
    BOOT_CAUSE, BOOT_CAUSE_POR,
    TRACE_ENABLE, OOM_STATE,
)
from tb.dma_responder import DmaTestMemory, DmaResponder


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

async def host_bringup(dut) -> tuple[CocotbBackend, OrbitDevice]:
    """Standard bringup: clock + reset + connect + enable trace + unmask IRQ."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    backend = CocotbBackend(dut)
    await backend.reset()

    dev = OrbitDevice(backend)

    # Read device ID (sync read works after reset settled)
    gid = backend.read(G2_ID.offset)
    assert gid == 0x4732_0001, f"Bad G2_ID: {gid:#x}"

    # Enable trace
    await backend.async_write(TRACE_CTRL.offset, TRACE_ENABLE)

    # Unmask DESC_DONE and TC0_FAULT
    await backend.async_write(IRQ_MASK.offset,
                              0xFFFF_FFFF & ~((1 << IrqBit.DESC_DONE) | (1 << IrqBit.TC0_FAULT)))

    return backend, dev


async def host_enqueue(backend: CocotbBackend, desc: bytes, queue: int = 0):
    """Enqueue descriptor via async register writes."""
    from tools.orbit_desc import desc_to_words
    from tools.orbit_mmio_map import DESC_STAGE_BASE, QUEUE_DOORBELLS
    words = desc_to_words(desc)
    stage_off = DESC_STAGE_BASE - BASE
    for i, w in enumerate(words):
        await backend.async_write(stage_off + i * 4, w)
    await backend.async_write(QUEUE_DOORBELLS[queue].offset, 0x0001)


async def poll_until(backend: CocotbBackend, offset: int, mask: int, expect: int,
                     max_cycles: int = 500) -> int:
    """Poll register until (val & mask) == expect. Returns final value."""
    for _ in range(max_cycles):
        val = await backend.async_read(offset)
        if (val & mask) == expect:
            return val
        await backend.tick(1)
    return await backend.async_read(offset)


# ═══════════════════════════════════════════════════════════════════
# Test 1: GEMM E2E (host-driven, real DUT)
# ═══════════════════════════════════════════════════════════════════

@cocotb.test()
async def test_host_gemm_e2e(dut):
    """Full host-driven GEMM: bringup → enqueue → DMA → done → trace → IRQ."""
    backend, dev = await host_bringup(dut)

    # Set up test memory
    kt = 4
    mem = DmaTestMemory()
    mem.fill_region(0x1000, kt * 16, pattern=0x01)  # act: kt beats × 16 bytes
    mem.fill_region(0x2000, kt * 16, pattern=0x02)  # wgt

    # Start DMA responder (GEMM needs: 2 reads + 1 write)
    responder = DmaResponder(dut, mem)
    cocotb.start_soon(responder.run(expect_reads=2, expect_writes=1))

    # Pack and enqueue GEMM descriptor
    desc = pack_gemm(act_addr=0x1000, wgt_addr=0x2000, out_addr=0x3000, kt=kt)
    await host_enqueue(backend, desc, queue=0)

    # Wait for DMA responder to finish
    await responder.wait_done(timeout_cycles=3000)
    assert responder.error is None, f"DMA responder error: {responder.error}"
    assert responder.reads_served == 2, f"Expected 2 reads, got {responder.reads_served}"
    assert responder.writes_served == 1, f"Expected 1 write, got {responder.writes_served}"

    # Wait for TC0 to return to IDLE
    await backend.tick(20)
    rs = await backend.async_read(TC0_RUNSTATE.offset)
    assert (rs & 0x7) == 0, f"TC0 should be IDLE, got state={(rs & 0x7)}"

    # Check IRQ_PENDING: DESC_DONE (bit 0) should be set
    irq = await backend.async_read(IRQ_PENDING.offset)
    assert irq & (1 << IrqBit.DESC_DONE), f"DESC_DONE IRQ not set: {irq:#x}"

    # Check irq_out is asserted (unmasked)
    assert dut.irq_out.value == 1, "irq_out not asserted"

    # Check trace: should have DISPATCH + DONE events
    tail = await backend.async_read(TRACE_TAIL.offset)
    assert tail >= 2, f"Trace tail should be >= 2 (dispatch+done), got {tail}"

    # Read first trace entry via window
    win_off = (TRACE_WIN_BASE - BASE)
    meta_off = (TRACE_META_BASE - BASE)

    entry0_lo = await backend.async_read(win_off)
    entry0_meta = await backend.async_read(meta_off)
    entry0_type = (entry0_meta >> 4) & 0xF
    assert entry0_type == TraceType.DESC_DISPATCH, f"First trace should be DISPATCH, got {entry0_type}"

    # Check writeback happened (non-empty write log)
    assert len(mem.write_log) >= 1, "No DMA write captured"
    wr_addr, wr_data = mem.write_log[0]
    assert wr_addr == 0x3000, f"Write addr should be 0x3000, got {wr_addr:#x}"
    assert len(wr_data) > 0, "Write data empty"

    # No fault
    fault = await backend.async_read(TC0_FAULT_STATUS.offset)
    assert fault == 0, f"Unexpected fault: {fault:#x}"

    # DMA_STATUS should show DONE
    dma_st = await backend.async_read(DMA_STATUS.offset)
    assert dma_st & 0x2, f"DMA_STATUS DONE bit not set: {dma_st:#x}"

    # Clear IRQ
    await backend.async_write(IRQ_PENDING.offset, irq)
    await backend.tick(2)
    irq2 = await backend.async_read(IRQ_PENDING.offset)
    assert (irq2 & (1 << IrqBit.DESC_DONE)) == 0, "IRQ not cleared"

    dut._log.info("GEMM E2E PASS: dispatch → DMA(2R+1W) → done → trace → IRQ → clear")


# ═══════════════════════════════════════════════════════════════════
# Test 2: NOP via host stack
# ═══════════════════════════════════════════════════════════════════

@cocotb.test()
async def test_host_nop(dut):
    """NOP descriptor via host stack: no DMA, just FSM cycle."""
    backend, dev = await host_bringup(dut)

    desc = pack_nop()
    await host_enqueue(backend, desc, queue=0)
    await backend.tick(30)

    rs = await backend.async_read(TC0_RUNSTATE.offset)
    assert (rs & 0x7) == 0, f"TC0 should be IDLE after NOP, got {rs & 0x7}"

    irq = await backend.async_read(IRQ_PENDING.offset)
    assert irq & (1 << IrqBit.DESC_DONE), "DESC_DONE should be set after NOP"

    tail = await backend.async_read(TRACE_TAIL.offset)
    assert tail >= 1, f"Trace should have at least 1 event, got {tail}"


# ═══════════════════════════════════════════════════════════════════
# Test 3: Illegal opcode → fault → clear
# ═══════════════════════════════════════════════════════════════════

@cocotb.test()
async def test_host_illegal_opcode_fault(dut):
    """Bad opcode (0xFF) → TC0 FAULT → IRQ TC0_FAULT → clear."""
    backend, dev = await host_bringup(dut)

    # Pack descriptor with illegal opcode but valid CRC
    from tools.orbit_desc import pack_descriptor
    desc = pack_descriptor(0xFF)
    await host_enqueue(backend, desc, queue=0)
    await backend.tick(30)

    # TC0 should be back to IDLE (fault was processed)
    rs = await backend.async_read(TC0_RUNSTATE.offset)
    # FSM goes FAULT -> DONE -> IDLE, so by now should be IDLE
    assert (rs & 0x7) == 0, f"TC0 should be IDLE after fault cycle, got {rs & 0x7}"

    # FAULT_STATUS should be non-zero
    fault = await backend.async_read(TC0_FAULT_STATUS.offset)
    assert fault != 0, f"FAULT_STATUS should be set, got {fault}"

    # IRQ TC0_FAULT should be pending
    irq = await backend.async_read(IRQ_PENDING.offset)
    assert irq & (1 << IrqBit.TC0_FAULT), f"TC0_FAULT IRQ not set: {irq:#x}"

    # Clear fault
    await backend.async_write(TC0_FAULT_STATUS.offset, fault)
    await backend.async_write(IRQ_PENDING.offset, 1 << IrqBit.TC0_FAULT)
    await backend.tick(2)

    fault2 = await backend.async_read(TC0_FAULT_STATUS.offset)
    assert fault2 == 0, f"FAULT_STATUS not cleared: {fault2:#x}"
    irq2 = await backend.async_read(IRQ_PENDING.offset)
    assert (irq2 & (1 << IrqBit.TC0_FAULT)) == 0, "TC0_FAULT IRQ not cleared"


# ═══════════════════════════════════════════════════════════════════
# Test 4: Trace / IRQ consistency
# ═══════════════════════════════════════════════════════════════════

@cocotb.test()
async def test_host_trace_irq_consistency(dut):
    """Verify trace events match IRQ state: NOP + fault in sequence."""
    backend, dev = await host_bringup(dut)

    # Clear any prior state
    await backend.async_write(IRQ_PENDING.offset, 0xFFFF_FFFF)
    await backend.tick(2)

    # Submit NOP
    await host_enqueue(backend, pack_nop(), queue=0)
    await backend.tick(30)

    # Submit illegal opcode
    await host_enqueue(backend, pack_descriptor(0xFF), queue=0)
    await backend.tick(30)

    # Read trace
    tail = await backend.async_read(TRACE_TAIL.offset)
    assert tail >= 3, f"Expected >= 3 trace events (NOP dispatch/done + fault), got {tail}"

    # Read IRQ
    irq = await backend.async_read(IRQ_PENDING.offset)

    # DESC_DONE should be set (from NOP completion at minimum)
    assert irq & (1 << IrqBit.DESC_DONE), f"DESC_DONE expected: {irq:#x}"
    # TC0_FAULT should be set (from illegal opcode)
    assert irq & (1 << IrqBit.TC0_FAULT), f"TC0_FAULT expected: {irq:#x}"

    # Verify trace has both DONE and FAULT types
    types_found = set()
    win_off = TRACE_WIN_BASE - BASE
    meta_off = TRACE_META_BASE - BASE
    for i in range(min(tail, 8)):
        m = await backend.async_read(meta_off + i * 4)
        t = (m >> 4) & 0xF
        types_found.add(t)

    assert TraceType.DESC_DONE in types_found, f"DONE not in trace: {types_found}"
    assert TraceType.DESC_FAULT in types_found, f"FAULT not in trace: {types_found}"

    dut._log.info(f"Trace/IRQ consistency PASS: types={types_found}, irq={irq:#x}")


# ═══════════════════════════════════════════════════════════════════
# Test 5: Boot cause via watchdog inject
# ═══════════════════════════════════════════════════════════════════

@cocotb.test()
async def test_host_watchdog_boot_cause(dut):
    """Watchdog inject → reset → BOOT_CAUSE[1] set."""
    backend, dev = await host_bringup(dut)

    # Clear boot cause
    await backend.async_write(BOOT_CAUSE.offset, 0)
    await backend.tick(2)

    # Inject watchdog
    from tools.orbit_mmio_map import WDOG_CTRL
    await backend.async_write(WDOG_CTRL.offset, 0x8000_0000)

    # Wait for reset sequence
    for _ in range(30):
        await backend.tick(1)
        try:
            if dut.reset_active.value == 0:
                break
        except ValueError:
            pass
    await backend.tick(5)

    bc = await backend.async_read(BOOT_CAUSE.offset)
    assert bc & 0x2, f"WDOG bit should be set in BOOT_CAUSE: {bc:#x}"
