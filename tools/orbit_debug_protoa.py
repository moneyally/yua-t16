#!/usr/bin/env python3
"""
orbit_debug_protoa.py — ORBIT-G2 Proto-A Debug CLI

Usage:
  python -m tools.orbit_debug_protoa <command> [options]

Commands:
  info             G2_ID, VERSION, CAP0/CAP1, BOOT_CAUSE
  queue-status     Q0~Q3 head/tail/depth/overflow
  tc-status        RUNSTATE, CTRL, FAULT_STATUS, DESC_PTR
  oom              OOM usage/reserved/effective/state
  irq              IRQ pending/mask/cause_last
  trace-head       trace head/tail/drop
  trace-dump       trace entry decode (--count N)
  doorbell         descriptor enqueue + kick (--queue Q --opcode OP)
  clear-fault      FAULT_STATUS/IRQ W1C clear
  soft-reset       SW_RESET trigger
  watchdog-inject  WDOG stub pulse
  perf             MXU/VPU perf counters

Backend: SimBackend (default). Real MMIO requires MmapBackend (Proto-B+).
"""
from __future__ import annotations
import argparse
import sys

from tools.orbit_mmio_map import (
    G2_ID, G2_VERSION, G2_CAP0, G2_CAP1, BUILD_HASH_LO, BUILD_HASH_HI,
    BOOT_CAUSE, SW_RESET, WDOG_CTRL,
    QUEUE_STATUSES, Q_OVERFLOW, QUEUE_DOORBELLS,
    TC0_RUNSTATE, TC0_CTRL, TC0_FAULT_STATUS, TC0_DESC_PTR_LO, TC0_DESC_PTR_HI,
    TC0_PERF_CYC_LO, TC0_PERF_CYC_HI,
    MXU_BUSY_CYC_LO, MXU_BUSY_CYC_HI, MXU_TILE_COUNT,
    VPU_BUSY_CYC_LO, VPU_OP_COUNT, PERF_FREEZE,
    DMA_STATUS, DMA_ERR_CODE,
    TC_STATE_IDLE, TC_STATE_FETCH, TC_STATE_RUN, TC_STATE_STALL, TC_STATE_FAULT,
    Opcode, IrqBit, BASE,
    CAP0_HAS_TC1, CAP0_HAS_HBM, CAP0_HAS_TRACE_RING, CAP0_HAS_OOM_GUARD,
)
from tools.orbit_backend import Backend, SimBackend
from tools.orbit_desc import pack_nop, pack_gemm, stage_and_doorbell
from tools.orbit_trace import print_trace, read_trace_status
from tools.orbit_poll import (
    read_irq_status, clear_all_irq, decode_boot_cause,
    read_oom_status, read_fault_status, clear_fault,
)


TC_STATE_NAMES = {0: "IDLE", 1: "FETCH", 2: "RUN", 3: "STALL", 4: "FAULT"}


def cmd_info(b: Backend):
    gid = b.read(G2_ID.offset)
    ver = b.read(G2_VERSION.offset)
    cap0 = b.read(G2_CAP0.offset)
    cap1 = b.read(G2_CAP1.offset)
    bc = b.read(BOOT_CAUSE.offset)
    print(f"G2_ID:      {gid:#010x}")
    print(f"VERSION:    {(ver>>16)&0xFF}.{(ver>>8)&0xFF}.{ver&0xFF}")
    print(f"CAP0:       {cap0:#010x}", end="")
    features = []
    if cap0 & CAP0_HAS_TRACE_RING: features.append("TRACE")
    if cap0 & CAP0_HAS_OOM_GUARD:  features.append("OOM")
    if cap0 & CAP0_HAS_TC1:        features.append("TC1")
    if cap0 & CAP0_HAS_HBM:        features.append("HBM")
    print(f"  [{', '.join(features) or 'none'}]")
    print(f"CAP1:       {cap1:#010x}")
    print(f"BOOT_CAUSE: {bc:#010x}  {decode_boot_cause(bc)}")


def cmd_queue_status(b: Backend):
    names = ["Q0(compute)", "Q1(utility)", "Q2(telem)", "Q3(hipri)"]
    for i, (name, sreg) in enumerate(zip(names, QUEUE_STATUSES)):
        val = b.read(sreg.offset)
        head = val & 0xFFFF
        tail = (val >> 16) & 0xFFFF
        depth = (tail - head) & 0xFFFF
        print(f"  {name}: head={head} tail={tail} depth={depth}")
    ovf = b.read(Q_OVERFLOW.offset)
    print(f"  OVERFLOW: {ovf:#06x}")


def cmd_tc_status(b: Backend):
    rs = b.read(TC0_RUNSTATE.offset)
    ctrl = b.read(TC0_CTRL.offset)
    fault = b.read(TC0_FAULT_STATUS.offset)
    dptr_lo = b.read(TC0_DESC_PTR_LO.offset)
    dptr_hi = b.read(TC0_DESC_PTR_HI.offset)
    pcyc_lo = b.read(TC0_PERF_CYC_LO.offset)
    pcyc_hi = b.read(TC0_PERF_CYC_HI.offset)

    state = TC_STATE_NAMES.get(rs & 0x7, f"?({rs & 0x7})")
    wait_dma = bool(rs & (1 << 8))
    print(f"TC0 RUNSTATE:   {state}" + (f" (WAIT_DMA)" if wait_dma else ""))
    print(f"TC0 CTRL:       ENABLE={ctrl&1} HALT={(ctrl>>1)&1}")
    print(f"TC0 FAULT:      {fault:#010x}")
    print(f"TC0 DESC_PTR:   {dptr_hi:08x}_{dptr_lo:08x}")
    print(f"TC0 PERF_CYC:   {(pcyc_hi << 32) | pcyc_lo}")


def cmd_oom(b: Backend):
    st = read_oom_status(b)
    print(f"  State:     {st['state']}")
    print(f"  Usage:     {st['usage']}")
    print(f"  Reserved:  {st['reserved']}")
    print(f"  Effective: {st['effective']}")
    print(f"  AdmitStop: {st['admission_stop']}  PfClamp: {st['prefetch_clamp']}")


def cmd_irq(b: Backend):
    st = read_irq_status(b)
    print(f"  Pending:   {st['pending']:#010x}  {st['sources']}")
    print(f"  Mask:      {st['mask']:#010x}")
    print(f"  Active:    {st['active']:#010x}")
    print(f"  Fatal:     {st['fatal']:#010x}")
    print(f"  CauseLast: {st['cause_last']:#010x}")


def cmd_trace_head(b: Backend):
    st = read_trace_status(b)
    print(f"  Head: {st['head']}  Tail: {st['tail']}  Drop: {st['drop_count']}")


def cmd_trace_dump(b: Backend, count: int):
    print_trace(b, count=count)


def cmd_doorbell(b: Backend, queue: int, opcode: int, kt: int):
    if opcode == Opcode.NOP:
        desc = pack_nop()
    elif opcode == Opcode.GEMM:
        desc = pack_gemm(0, 0, 0, kt)
    else:
        from tools.orbit_desc import pack_descriptor
        desc = pack_descriptor(opcode)
    stage_and_doorbell(b, desc, queue)
    print(f"Submitted opcode={opcode:#04x} to Q{queue}")


def cmd_clear_fault(b: Backend):
    clear_fault(b)
    clear_all_irq(b)
    print("Fault and IRQ cleared")


def cmd_soft_reset(b: Backend):
    b.write(SW_RESET.offset, 0x01)
    print("SW_RESET triggered")


def cmd_watchdog_inject(b: Backend):
    b.write(WDOG_CTRL.offset, 0x8000_0000)
    print("Watchdog test pulse injected")


def cmd_perf(b: Backend):
    mxu_lo = b.read(MXU_BUSY_CYC_LO.offset)
    mxu_hi = b.read(MXU_BUSY_CYC_HI.offset)
    tile = b.read(MXU_TILE_COUNT.offset)
    vpu = b.read(VPU_BUSY_CYC_LO.offset)
    freeze = b.read(PERF_FREEZE.offset)
    print(f"  MXU busy:  {(mxu_hi << 32) | mxu_lo}")
    print(f"  MXU tiles: {tile}")
    print(f"  VPU busy:  {vpu}  (Proto-A: always 0)")
    print(f"  Freeze:    {freeze & 1}")


def cmd_dma_status(b: Backend):
    st = read_fault_status(b)
    print(f"  DMA BUSY={st['dma_busy']} DONE={st['dma_done']} ERR={st['dma_err']} "
          f"TIMEOUT={st['dma_timeout']}")
    print(f"  ERR_CODE: {st['dma_err_code']:#010x}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="ORBIT-G2 Proto-A Debug CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("info")
    sub.add_parser("queue-status")
    sub.add_parser("tc-status")
    sub.add_parser("oom")
    sub.add_parser("irq")
    sub.add_parser("trace-head")
    p_td = sub.add_parser("trace-dump")
    p_td.add_argument("--count", type=int, default=16)
    p_db = sub.add_parser("doorbell")
    p_db.add_argument("--queue", type=int, default=0)
    p_db.add_argument("--opcode", type=lambda x: int(x, 0), default=0x01)
    p_db.add_argument("--kt", type=int, default=4)
    sub.add_parser("clear-fault")
    sub.add_parser("soft-reset")
    sub.add_parser("watchdog-inject")
    sub.add_parser("perf")
    sub.add_parser("dma-status")

    args = parser.parse_args(argv)

    b = SimBackend()

    dispatch = {
        "info": lambda: cmd_info(b),
        "queue-status": lambda: cmd_queue_status(b),
        "tc-status": lambda: cmd_tc_status(b),
        "oom": lambda: cmd_oom(b),
        "irq": lambda: cmd_irq(b),
        "trace-head": lambda: cmd_trace_head(b),
        "trace-dump": lambda: cmd_trace_dump(b, args.count),
        "doorbell": lambda: cmd_doorbell(b, args.queue, args.opcode, args.kt),
        "clear-fault": lambda: cmd_clear_fault(b),
        "soft-reset": lambda: cmd_soft_reset(b),
        "watchdog-inject": lambda: cmd_watchdog_inject(b),
        "perf": lambda: cmd_perf(b),
        "dma-status": lambda: cmd_dma_status(b),
    }

    if args.cmd is None:
        parser.print_help()
        return 1

    dispatch[args.cmd]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
