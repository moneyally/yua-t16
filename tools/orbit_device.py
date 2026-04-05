"""
orbit_device.py — ORBIT-G2 Device HAL

Library API that wraps a Backend into a device-level interface.
All register access goes through orbit_mmio_map constants — no hardcoded addresses.
Designed for Proto-A; API contract carries forward to Proto-B/ASIC.

Usage:
    dev = OrbitDevice(SimBackend())
    dev.connect()
    dev.enqueue_desc(pack_nop(), queue=0)
    dev.poll_tc_idle()
"""
from __future__ import annotations
from dataclasses import dataclass
from tools.orbit_backend import Backend
from tools.orbit_mmio_map import (
    G2_ID, G2_VERSION, G2_CAP0, G2_CAP1,
    BOOT_CAUSE, SW_RESET, WDOG_CTRL,
    QUEUE_STATUSES, Q_OVERFLOW, QUEUE_DOORBELLS,
    TC0_RUNSTATE, TC0_CTRL, TC0_FAULT_STATUS,
    TC0_PERF_CYC_LO, TC0_PERF_CYC_HI,
    TC0_DESC_PTR_LO, TC0_DESC_PTR_HI,
    IRQ_PENDING, IRQ_MASK, IRQ_CAUSE_LAST,
    OOM_USAGE_LO, OOM_RESV_LO, OOM_EFF_LO, OOM_STATE,
    TRACE_HEAD, TRACE_TAIL, TRACE_CTRL, TRACE_DROP_CNT,
    DMA_STATUS, DMA_ERR_CODE,
    MXU_BUSY_CYC_LO, MXU_TILE_COUNT, PERF_FREEZE,
    DESC_STAGE_BASE, BASE,
    TC_CTRL_ENABLE, TC_CTRL_HALT, TC_CTRL_CLR_FAULT,
    TRACE_ENABLE, IrqBit, Opcode,
    TC_STATE_IDLE, TC_STATE_FAULT,
    DESC_SIZE,
)
from tools.orbit_desc import desc_to_words


class DeviceError(Exception):
    """Base error for device operations."""


class UnsupportedOpcodeError(DeviceError):
    """Opcode not supported on current Proto-A runtime path."""


@dataclass(frozen=True)
class DeviceInfo:
    device_id: int
    version: int
    cap0: int
    cap1: int
    boot_cause: int


@dataclass(frozen=True)
class QueueStatus:
    head: int
    tail: int
    depth: int
    overflow: bool


@dataclass(frozen=True)
class TCStatus:
    state: int       # 0=IDLE,1=FETCH,2=RUN,3=STALL,4=FAULT
    wait_dma: bool
    enable: bool
    halt: bool
    fault_code: int
    perf_cycles: int
    desc_ptr: int


class OrbitDevice:
    """Device HAL — abstracts Backend into device operations."""

    # Opcodes actually executable in Proto-A RTL
    SUPPORTED_OPCODES = {Opcode.NOP, Opcode.GEMM}

    def __init__(self, backend: Backend):
        self._b = backend
        self._connected = False

    @property
    def backend(self) -> Backend:
        return self._b

    # ── Connection ───────────────────────────���──────────────────
    def connect(self) -> DeviceInfo:
        """Read device identity and verify it's an ORBIT-G2."""
        info = self.read_info()
        if info.device_id != 0x4732_0001:
            raise DeviceError(f"Not an ORBIT-G2: ID={info.device_id:#010x}")
        self._connected = True
        return info

    def read_info(self) -> DeviceInfo:
        return DeviceInfo(
            device_id=self._b.read(G2_ID.offset),
            version=self._b.read(G2_VERSION.offset),
            cap0=self._b.read(G2_CAP0.offset),
            cap1=self._b.read(G2_CAP1.offset),
            boot_cause=self._b.read(BOOT_CAUSE.offset),
        )

    # ── Register access ─────────────────────────────────────────
    def read_reg(self, offset: int) -> int:
        return self._b.read(offset)

    def write_reg(self, offset: int, value: int):
        self._b.write(offset, value)

    # ── Descriptor enqueue ──────────────────────────────────────
    def enqueue_desc(self, desc: bytes, queue: int = 0):
        """Stage descriptor and ring doorbell.

        Raises UnsupportedOpcodeError if opcode is not executable in Proto-A.
        """
        if len(desc) != DESC_SIZE:
            raise DeviceError(f"Descriptor size mismatch: {len(desc)} != {DESC_SIZE}")

        opcode = desc[0]
        if opcode not in (op.value for op in self.SUPPORTED_OPCODES):
            raise UnsupportedOpcodeError(
                f"Opcode 0x{opcode:02x} not supported on Proto-A runtime. "
                f"Supported: {[f'0x{o:02x}' for o in self.SUPPORTED_OPCODES]}"
            )

        if queue not in range(4):
            raise DeviceError(f"Invalid queue: {queue}")

        words = desc_to_words(desc)
        stage_off = DESC_STAGE_BASE - BASE
        for i, w in enumerate(words):
            self._b.write(stage_off + i * 4, w)

        doorbell_off = QUEUE_DOORBELLS[queue].offset
        self._b.write(doorbell_off, 0x0001)

    def ring_doorbell(self, queue: int = 0):
        """Ring doorbell without staging (re-submit last staged descriptor)."""
        self._b.write(QUEUE_DOORBELLS[queue].offset, 0x0001)

    # ── Queue status ────────────────────────────────────────────
    def read_queue_status(self, queue: int) -> QueueStatus:
        val = self._b.read(QUEUE_STATUSES[queue].offset)
        head = val & 0xFFFF
        tail = (val >> 16) & 0xFFFF
        ovf_reg = self._b.read(Q_OVERFLOW.offset)
        return QueueStatus(
            head=head, tail=tail,
            depth=(tail - head) & 0xFFFF,
            overflow=bool(ovf_reg & (1 << queue)),
        )

    # ── TC status ───────────────────────────────────────────────
    def read_tc_status(self) -> TCStatus:
        rs = self._b.read(TC0_RUNSTATE.offset)
        ctrl = self._b.read(TC0_CTRL.offset)
        fault = self._b.read(TC0_FAULT_STATUS.offset)
        pcyc_lo = self._b.read(TC0_PERF_CYC_LO.offset)
        pcyc_hi = self._b.read(TC0_PERF_CYC_HI.offset)
        dptr_lo = self._b.read(TC0_DESC_PTR_LO.offset)
        dptr_hi = self._b.read(TC0_DESC_PTR_HI.offset)
        return TCStatus(
            state=rs & 0x7,
            wait_dma=bool(rs & (1 << 8)),
            enable=bool(ctrl & TC_CTRL_ENABLE),
            halt=bool(ctrl & TC_CTRL_HALT),
            fault_code=fault & 0xFF,
            perf_cycles=(pcyc_hi << 32) | pcyc_lo,
            desc_ptr=(dptr_hi << 32) | dptr_lo,
        )

    def poll_tc_idle(self, max_polls: int = 10000) -> bool:
        """Poll until TC0 is IDLE or FAULT. Returns True if IDLE."""
        for _ in range(max_polls):
            st = self._b.read(TC0_RUNSTATE.offset) & 0x7
            if st == TC_STATE_IDLE:
                return True
            if st == TC_STATE_FAULT:
                return False
        return False

    # ── IRQ ─────────────────────────────────────────────────────
    def poll_irq(self) -> int:
        """Return current IRQ pending bitmap."""
        return self._b.read(IRQ_PENDING.offset)

    def clear_irq(self, bits: int):
        self._b.write(IRQ_PENDING.offset, bits)

    def unmask_irq(self, bits: int):
        """Unmask specified IRQ bits (clear mask bits)."""
        current = self._b.read(IRQ_MASK.offset)
        self._b.write(IRQ_MASK.offset, current & ~bits)

    # ── Fault ───────────────────────────────────────────────────
    def clear_fault(self):
        """Clear TC0 fault status and related IRQ."""
        fault = self._b.read(TC0_FAULT_STATUS.offset)
        if fault:
            self._b.write(TC0_FAULT_STATUS.offset, fault)
        pending = self._b.read(IRQ_PENDING.offset)
        fault_irq = (1 << IrqBit.TC0_FAULT)
        if pending & fault_irq:
            self._b.write(IRQ_PENDING.offset, fault_irq)

    # ── OOM ─────────────────────────────────────────────────────
    def read_oom_state(self) -> int:
        """Return OOM state: 0=NORMAL,1=PRESSURE,2=CRITICAL,3=EMERG."""
        return self._b.read(OOM_STATE.offset) & 0x3

    def read_oom_effective(self) -> int:
        return self._b.read(OOM_EFF_LO.offset)

    # ── Trace ───────────────────────────────────────────────────
    def enable_trace(self):
        self._b.write(TRACE_CTRL.offset, TRACE_ENABLE)

    def disable_trace(self):
        self._b.write(TRACE_CTRL.offset, 0)

    def dump_trace(self, count: int | None = None):
        """Return (head, tail, drop_count) for caller to use with orbit_trace."""
        from tools.orbit_trace import dump_trace
        return dump_trace(self._b, count)

    # ── Reset ───────────────────────────────────────────────────
    def soft_reset(self):
        self._b.write(SW_RESET.offset, 0x01)

    def watchdog_inject(self):
        self._b.write(WDOG_CTRL.offset, 0x8000_0000)

    # ── Perf ────────────────────────────────────────────────────
    def freeze_perf(self):
        self._b.write(PERF_FREEZE.offset, 1)

    def unfreeze_perf(self):
        self._b.write(PERF_FREEZE.offset, 0)

    def read_mxu_busy_cycles(self) -> int:
        lo = self._b.read(MXU_BUSY_CYC_LO.offset)
        return lo  # hi word omitted for Proto-A (32-bit sufficient)

    def read_tile_count(self) -> int:
        return self._b.read(MXU_TILE_COUNT.offset)
