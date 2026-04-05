"""test_trace_decode.py — Trace entry decode verification."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.orbit_trace import TraceEntry
from tools.orbit_mmio_map import TraceType, Opcode


def make_entry(type_id, fatal, qclass, opcode, index=0):
    """Build a TraceEntry matching RTL payload format:
    payload = {46'b0, qclass[1:0], opcode[7:0], 8'b0}
    """
    payload_lo = (qclass & 0x3) << 16 | (opcode & 0xFF) << 8
    return TraceEntry(
        index=index,
        type_id=type_id,
        fatal=fatal,
        payload_lo=payload_lo,
        payload_hi=0,
    )


def test_dispatch_decode():
    e = make_entry(TraceType.DESC_DISPATCH, False, 0, Opcode.GEMM)
    assert e.type_name == "DESC_DISPATCH"
    assert not e.fatal
    assert e.queue_class == 0
    assert e.opcode_or_fault == Opcode.GEMM


def test_done_decode():
    e = make_entry(TraceType.DESC_DONE, False, 1, Opcode.NOP)
    assert e.type_name == "DESC_DONE"
    assert e.queue_class == 1
    assert e.opcode_or_fault == Opcode.NOP


def test_fault_decode():
    e = make_entry(TraceType.DESC_FAULT, True, 3, 0x02)  # CRC fault
    assert e.type_name == "DESC_FAULT"
    assert e.fatal
    assert e.queue_class == 3
    assert e.opcode_or_fault == 0x02


def test_human_format():
    e = make_entry(TraceType.DESC_DISPATCH, False, 0, Opcode.GEMM, index=42)
    s = e.format_human()
    assert "DESC_DISPATCH" in s
    assert "Q0" in s
    assert "GEMM" in s
    assert "42" in s


def test_hex_format():
    e = make_entry(TraceType.DESC_DONE, False, 2, Opcode.NOP, index=5)
    s = e.format_hex()
    assert "5:" in s
    assert "type=2" in s


def test_unknown_type():
    e = TraceEntry(0, 15, False, 0, 0)
    assert "UNKNOWN" in e.type_name
