"""
orbit_trace.py — ORBIT-G2 Trace Ring Dump / Decode

Reads trace entries via MMIO read window and decodes them.
Format derived from RTL: g2_ctrl_top.sv trace payload.
"""
from __future__ import annotations
from dataclasses import dataclass
from tools.orbit_mmio_map import (
    TRACE_HEAD, TRACE_TAIL, TRACE_DROP_CNT, TRACE_WIN_BASE, TRACE_META_BASE,
    BASE, TraceType, Opcode,
)


@dataclass
class TraceEntry:
    index: int
    type_id: int
    fatal: bool
    payload_lo: int
    payload_hi: int

    @property
    def type_name(self) -> str:
        try:
            return TraceType(self.type_id).name
        except ValueError:
            return f"UNKNOWN({self.type_id})"

    @property
    def queue_class(self) -> int:
        """Extract qclass[1:0] from payload bits [17:16]."""
        return (self.payload_lo >> 16) & 0x3

    @property
    def opcode_or_fault(self) -> int:
        """Extract opcode/fault_code[7:0] from payload bits [15:8]."""
        return (self.payload_lo >> 8) & 0xFF

    def format_human(self) -> str:
        fatal_str = " FATAL" if self.fatal else ""
        op = self.opcode_or_fault
        try:
            op_name = Opcode(op).name
        except ValueError:
            op_name = f"0x{op:02x}"
        return (
            f"[{self.index:4d}] {self.type_name:<14s}{fatal_str} "
            f"Q{self.queue_class} op={op_name} "
            f"payload={self.payload_hi:08x}_{self.payload_lo:08x}"
        )

    def format_hex(self) -> str:
        return (
            f"{self.index:4d}: type={self.type_id} fatal={int(self.fatal)} "
            f"lo={self.payload_lo:08x} hi={self.payload_hi:08x}"
        )


def read_trace_status(backend) -> dict:
    """Read trace ring head/tail/drop_count."""
    head = backend.read(TRACE_HEAD.offset)
    tail = backend.read(TRACE_TAIL.offset)
    drop = backend.read(TRACE_DROP_CNT.offset)
    return {"head": head & 0xFFFF, "tail": tail & 0xFFFF, "drop_count": drop}


def read_trace_entry(backend, index: int) -> TraceEntry:
    """Read a single trace entry via MMIO read window."""
    win_off = (TRACE_WIN_BASE - BASE) + index * 8
    meta_off = (TRACE_META_BASE - BASE) + index * 4

    lo = backend.read(win_off)
    hi = backend.read(win_off + 4)
    meta = backend.read(meta_off)

    type_id = (meta >> 4) & 0xF
    fatal = bool(meta & 0x1)

    return TraceEntry(index=index, type_id=type_id, fatal=fatal,
                      payload_lo=lo, payload_hi=hi)


def dump_trace(backend, count: int | None = None) -> list[TraceEntry]:
    """Read all trace entries from head to tail (or up to count)."""
    status = read_trace_status(backend)
    head = status["head"]
    tail = status["tail"]

    if tail == head:
        return []

    entries = []
    idx = head
    limit = count if count is not None else 1024
    while idx != tail and len(entries) < limit:
        entries.append(read_trace_entry(backend, idx))
        idx = (idx + 1) & 0x3FF  # 1024 entries
    return entries


def print_trace(backend, count: int | None = None, hex_mode: bool = False):
    """Print trace dump to stdout."""
    status = read_trace_status(backend)
    print(f"Trace: head={status['head']} tail={status['tail']} "
          f"drop={status['drop_count']}")

    entries = dump_trace(backend, count)
    if not entries:
        print("  (empty)")
        return

    for e in entries:
        print(f"  {e.format_hex() if hex_mode else e.format_human()}")
