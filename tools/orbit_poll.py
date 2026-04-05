"""
orbit_poll.py — ORBIT-G2 IRQ / Fault / OOM Polling Utilities

Decodes IRQ pending bitmap, boot cause, OOM state.
Provides W1C clear helpers.
"""
from __future__ import annotations
from tools.orbit_mmio_map import (
    IRQ_PENDING, IRQ_MASK, IRQ_CAUSE_LAST, IRQ_FATAL_MASK, IrqBit,
    BOOT_CAUSE, BOOT_CAUSE_POR, BOOT_CAUSE_WDOG, BOOT_CAUSE_SW, BOOT_CAUSE_FLR,
    OOM_STATE, OOM_USAGE_LO, OOM_RESV_LO, OOM_EFF_LO,
    TC0_FAULT_STATUS, TC0_RUNSTATE,
    DMA_STATUS, DMA_ERR_CODE,
)


def decode_irq_pending(val: int) -> list[str]:
    """Decode IRQ pending bitmap to list of source names."""
    names = []
    for bit in IrqBit:
        if val & (1 << bit):
            names.append(bit.name)
    return names


def split_irq_fatal(val: int) -> tuple[int, int]:
    """Split pending into (fatal_bits, nonfatal_bits)."""
    fatal = val & IRQ_FATAL_MASK
    nonfatal = val & ~IRQ_FATAL_MASK
    return fatal, nonfatal


def read_irq_status(backend) -> dict:
    """Read and decode IRQ status."""
    pending = backend.read(IRQ_PENDING.offset)
    mask = backend.read(IRQ_MASK.offset)
    cause = backend.read(IRQ_CAUSE_LAST.offset)
    fatal, nonfatal = split_irq_fatal(pending)
    return {
        "pending": pending,
        "mask": mask,
        "cause_last": cause,
        "active": pending & ~mask,
        "fatal": fatal,
        "nonfatal": nonfatal,
        "sources": decode_irq_pending(pending),
    }


def clear_irq(backend, bits: int):
    """W1C clear specified IRQ pending bits."""
    backend.write(IRQ_PENDING.offset, bits)


def clear_all_irq(backend):
    """Clear all pending IRQs."""
    pending = backend.read(IRQ_PENDING.offset)
    if pending:
        backend.write(IRQ_PENDING.offset, pending)


def decode_boot_cause(val: int) -> list[str]:
    """Decode BOOT_CAUSE to list of cause names."""
    causes = []
    if val & BOOT_CAUSE_POR:  causes.append("POR")
    if val & BOOT_CAUSE_WDOG: causes.append("WDOG")
    if val & BOOT_CAUSE_SW:   causes.append("SW")
    if val & BOOT_CAUSE_FLR:  causes.append("PCIE_FLR")
    return causes


def read_boot_cause(backend) -> dict:
    val = backend.read(BOOT_CAUSE.offset)
    return {"raw": val, "causes": decode_boot_cause(val)}


OOM_STATE_NAMES = {0: "NORMAL", 1: "PRESSURE", 2: "CRITICAL", 3: "EMERG"}

def read_oom_status(backend) -> dict:
    usage = backend.read(OOM_USAGE_LO.offset)
    resv  = backend.read(OOM_RESV_LO.offset)
    eff   = backend.read(OOM_EFF_LO.offset)
    state_reg = backend.read(OOM_STATE.offset)
    state = state_reg & 0x3
    admit_stop = bool(state_reg & (1 << 8))
    pf_clamp = bool(state_reg & (1 << 9))
    return {
        "usage": usage,
        "reserved": resv,
        "effective": eff,
        "state": OOM_STATE_NAMES.get(state, f"UNKNOWN({state})"),
        "state_raw": state,
        "admission_stop": admit_stop,
        "prefetch_clamp": pf_clamp,
    }


def read_fault_status(backend) -> dict:
    fault = backend.read(TC0_FAULT_STATUS.offset)
    rs = backend.read(TC0_RUNSTATE.offset)
    dma_st = backend.read(DMA_STATUS.offset)
    dma_err = backend.read(DMA_ERR_CODE.offset)
    return {
        "tc0_fault": fault,
        "tc0_runstate": rs & 0x7,
        "dma_busy": bool(dma_st & 1),
        "dma_done": bool(dma_st & 2),
        "dma_err": bool(dma_st & 4),
        "dma_timeout": bool(dma_st & 8),
        "dma_err_code": dma_err,
    }


def clear_fault(backend):
    """Clear TC0 fault status (W1C) and IRQ TC0_FAULT bit."""
    fault = backend.read(TC0_FAULT_STATUS.offset)
    if fault:
        backend.write(TC0_FAULT_STATUS.offset, fault)
    pending = backend.read(IRQ_PENDING.offset)
    if pending & (1 << IrqBit.TC0_FAULT):
        backend.write(IRQ_PENDING.offset, 1 << IrqBit.TC0_FAULT)
