"""
orbit_session.py — ORBIT-G2 Session / Op / Completion abstractions

OrbitSession manages a sequence of operations on an OrbitDevice.
Proto-A: serial execution (1 outstanding op).
Proto-B: API supports future parallelism.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from tools.orbit_device import OrbitDevice, UnsupportedOpcodeError
from tools.orbit_mmio_map import Opcode, IrqBit


class OpStatus(Enum):
    PENDING   = "pending"
    SUBMITTED = "submitted"
    COMPLETED = "completed"
    FAULTED   = "faulted"
    TIMEOUT   = "timeout"
    REJECTED  = "rejected"  # unsupported opcode


@dataclass
class OrbitOp:
    """A single descriptor operation."""
    desc: bytes
    queue: int = 0
    tag: str = ""

    @property
    def opcode(self) -> int:
        return self.desc[0] if self.desc else 0

    @property
    def opcode_name(self) -> str:
        try:
            return Opcode(self.opcode).name
        except ValueError:
            return f"0x{self.opcode:02x}"


@dataclass
class OrbitCompletion:
    """Result of a submitted operation."""
    op: OrbitOp
    status: OpStatus
    fault_code: int = 0
    cycles: int = 0
    error_msg: str = ""


class OrbitSession:
    """Manages a sequence of ops on a device.

    Usage:
        session = OrbitSession(device)
        session.submit(OrbitOp(pack_nop(), queue=0, tag="warmup"))
        session.submit(OrbitOp(pack_gemm(...), queue=0, tag="gemm0"))
        results = session.run_all()
    """

    def __init__(self, device: OrbitDevice, max_polls: int = 10000):
        self._dev = device
        self._max_polls = max_polls
        self._pending: list[OrbitOp] = []
        self._completions: list[OrbitCompletion] = []

    def submit(self, op: OrbitOp):
        """Add op to pending queue."""
        self._pending.append(op)

    def run_all(self, stop_on_fault: bool = True) -> list[OrbitCompletion]:
        """Execute all pending ops serially.

        Proto-A: 1 outstanding at a time.
        Returns list of completions in submission order.
        """
        results = []
        while self._pending:
            op = self._pending.pop(0)
            comp = self._execute_one(op)
            results.append(comp)
            self._completions.append(comp)

            if stop_on_fault and comp.status in (OpStatus.FAULTED, OpStatus.TIMEOUT):
                # Drain remaining as REJECTED
                for remaining in self._pending:
                    results.append(OrbitCompletion(
                        op=remaining, status=OpStatus.REJECTED,
                        error_msg="Aborted due to prior fault"
                    ))
                self._pending.clear()
                break

        return results

    def _execute_one(self, op: OrbitOp) -> OrbitCompletion:
        """Submit one op and poll for completion."""
        # Check opcode support
        try:
            self._dev.enqueue_desc(op.desc, op.queue)
        except UnsupportedOpcodeError as e:
            return OrbitCompletion(op=op, status=OpStatus.REJECTED, error_msg=str(e))

        # Poll for idle/fault
        idle = self._dev.poll_tc_idle(max_polls=self._max_polls)

        # Read status
        tc = self._dev.read_tc_status()

        if tc.state == 4:  # FAULT
            self._dev.clear_fault()
            return OrbitCompletion(
                op=op, status=OpStatus.FAULTED,
                fault_code=tc.fault_code,
                cycles=tc.perf_cycles,
                error_msg=f"TC0 fault: code=0x{tc.fault_code:02x}",
            )

        if not idle:
            return OrbitCompletion(
                op=op, status=OpStatus.TIMEOUT,
                cycles=tc.perf_cycles,
                error_msg=f"Polling timeout after {self._max_polls} polls",
            )

        return OrbitCompletion(
            op=op, status=OpStatus.COMPLETED,
            cycles=tc.perf_cycles,
        )

    @property
    def completions(self) -> list[OrbitCompletion]:
        return list(self._completions)

    def clear(self):
        self._pending.clear()
        self._completions.clear()
