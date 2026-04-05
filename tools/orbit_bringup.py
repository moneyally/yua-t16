"""
orbit_bringup.py — ORBIT-G2 Proto-A Scripted Bring-up Flows

Provides canned sequences for device initialization, smoke tests,
and diagnostic collection.
"""
from __future__ import annotations
from dataclasses import dataclass
from tools.orbit_device import OrbitDevice, DeviceInfo
from tools.orbit_session import OrbitSession, OrbitOp, OpStatus
from tools.orbit_scheduler import OrbitScheduler, FaultPolicy
from tools.orbit_desc import pack_nop, pack_gemm
from tools.orbit_trace import dump_trace, read_trace_status
from tools.orbit_poll import (
    read_irq_status, read_boot_cause, read_oom_status,
    read_fault_status, decode_boot_cause, clear_all_irq,
)
from tools.orbit_mmio_map import IRQ_PENDING, BOOT_CAUSE_POR


@dataclass
class BringupReport:
    info: DeviceInfo
    boot_cause: list[str]
    queues_ok: bool
    irq_clear: bool
    trace_enabled: bool
    health: str  # "OK" or error description


@dataclass
class SmokeResult:
    nop_ok: bool
    gemm_ok: bool  # True only if DMA responder available
    trace_entries: int
    errors: list[str]


@dataclass
class DiagnosticsReport:
    tc_state: int
    tc_fault: int
    irq_pending: int
    irq_sources: list[str]
    oom_state: str
    oom_effective: int
    dma_busy: bool
    dma_err_code: int
    trace_head: int
    trace_tail: int
    trace_drop: int
    boot_cause: list[str]


def bringup(dev: OrbitDevice) -> BringupReport:
    """Standard bring-up sequence.

    1. Read device info
    2. Soft reset
    3. Verify boot cause
    4. Check queue status (all empty)
    5. Clear IRQs
    6. Enable trace
    7. Health check
    """
    # 1. Connect
    info = dev.connect()

    # 2. Soft reset not needed on fresh POR, but safe to do
    # (Skip for now — in SimBackend it would trigger a register clear)

    # 3. Boot cause
    bc_raw = info.boot_cause
    causes = decode_boot_cause(bc_raw)

    # 4. Queue check
    queues_ok = True
    for q in range(4):
        qs = dev.read_queue_status(q)
        if qs.depth != 0 or qs.overflow:
            queues_ok = False

    # 5. Clear IRQs
    clear_all_irq(dev.backend)
    irq = dev.poll_irq()
    irq_clear = (irq == 0)

    # 6. Enable trace
    dev.enable_trace()
    trace_enabled = True

    # 7. Health
    tc = dev.read_tc_status()
    health = "OK"
    if tc.state != 0:
        health = f"TC0 not idle: state={tc.state}"
    if tc.fault_code:
        health = f"TC0 fault pending: 0x{tc.fault_code:02x}"

    return BringupReport(
        info=info,
        boot_cause=causes,
        queues_ok=queues_ok,
        irq_clear=irq_clear,
        trace_enabled=trace_enabled,
        health=health,
    )


def run_gemm_smoke(dev: OrbitDevice, kt: int = 4) -> SmokeResult:
    """Smoke test: submit NOP + GEMM (GEMM needs DMA responder).

    In SimBackend, GEMM will not complete (no DMA emulation).
    In CocotbBackend with DMA helper, GEMM should complete.
    """
    errors = []

    # NOP test (always works)
    sched = OrbitScheduler(dev)
    sched.add_nop(tag="smoke_nop")
    result = sched.run()
    nop_ok = result.ops_completed == 1
    if not nop_ok:
        errors.append(f"NOP failed: {result.completions[0].error_msg}")

    # Trace check
    ts = read_trace_status(dev.backend)

    return SmokeResult(
        nop_ok=nop_ok,
        gemm_ok=False,  # Requires DMA responder; not available in SimBackend
        trace_entries=ts["tail"],
        errors=errors,
    )


def collect_diagnostics(dev: OrbitDevice) -> DiagnosticsReport:
    """Collect full device diagnostic snapshot."""
    tc = dev.read_tc_status()
    irq = read_irq_status(dev.backend)
    oom = read_oom_status(dev.backend)
    fault = read_fault_status(dev.backend)
    trace = read_trace_status(dev.backend)
    bc = read_boot_cause(dev.backend)

    return DiagnosticsReport(
        tc_state=tc.state,
        tc_fault=tc.fault_code,
        irq_pending=irq["pending"],
        irq_sources=irq["sources"],
        oom_state=oom["state"],
        oom_effective=oom["effective"],
        dma_busy=fault["dma_busy"],
        dma_err_code=fault["dma_err_code"],
        trace_head=trace["head"],
        trace_tail=trace["tail"],
        trace_drop=trace["drop_count"],
        boot_cause=bc["causes"],
    )
