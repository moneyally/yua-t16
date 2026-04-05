"""
orbit_mmio_map.py — ORBIT-G2 MMIO Register Map SSOT

Single source of truth for all register addresses, bitfields, access types.
Derived from: ORBIT_G2_REG_SPEC.md
RTL cross-ref: rtl/reg_top.sv

All addresses are absolute (base 0x8030_0000).
reg_top uses offset = addr - 0x8030_0000 (20-bit).
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum

# ═══════════════════════════════════════════════════════════════════
# Base addresses
# ═══════════════════════════════════════════════════════════════════
BASE = 0x8030_0000

class Block(IntEnum):
    GLOBAL   = 0x8030_0000
    RESET    = 0x8030_1000
    DOORBELL = 0x8030_2000
    Q_STATUS = 0x8030_3000
    DMA      = 0x8031_0000
    OOM      = 0x8032_0000
    TC0      = 0x8034_0000
    TC1      = 0x8035_0000
    PERF     = 0x8036_0000
    HBM      = 0x8038_0000  # Proto-B/ASIC only
    IRQ      = 0x8039_0000
    TRACE    = 0x803A_0000
    ICI      = 0x8040_0000  # ASIC only

# ═══════════════════════════════════════════════════════════════════
# Access types
# ═══════════════════════════════════════════════════════════════════
class Access:
    RO  = "RO"
    WO  = "WO"
    RW  = "RW"
    W1C = "W1C"

# ═══════════════════════════════════════════════════════════════════
# Register definitions
# ═══════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Reg:
    name: str
    addr: int
    access: str
    reset: int = 0
    desc: str = ""

    @property
    def offset(self) -> int:
        """Offset from BASE for reg_top address bus."""
        return self.addr - BASE

# ── Global / Version ─────────────────────────────────────────────
G2_ID        = Reg("G2_ID",        0x8030_0000, Access.RO, 0x4732_0001, "device ID")
G2_VERSION   = Reg("G2_VERSION",   0x8030_0004, Access.RO, 0x0001_0000, "RTL version")
G2_CAP0      = Reg("G2_CAP0",      0x8030_0008, Access.RO, 0x0000_0060, "feature bitmap 0")
G2_CAP1      = Reg("G2_CAP1",      0x8030_000C, Access.RO, 0x0000_0000, "feature bitmap 1")
BUILD_HASH_LO = Reg("BUILD_HASH_LO", 0x8030_0010, Access.RO, 0, "build hash low")
BUILD_HASH_HI = Reg("BUILD_HASH_HI", 0x8030_0014, Access.RO, 0, "build hash high")

# ── Reset / Boot ─────────────────────────────────────────────────
BOOT_CAUSE    = Reg("BOOT_CAUSE",    0x8030_1000, Access.RO,  desc="reset origin")
SW_RESET      = Reg("SW_RESET",      0x8030_1004, Access.WO,  desc="soft reset trigger")
BOOT_VECTOR_LO = Reg("BOOT_VECTOR_LO", 0x8030_1008, Access.RW, desc="boot addr low")
BOOT_VECTOR_HI = Reg("BOOT_VECTOR_HI", 0x8030_100C, Access.RW, desc="boot addr high")
STRAP_STATUS  = Reg("STRAP_STATUS",  0x8030_1010, Access.RO,  desc="sampled straps")
WDOG_CTRL     = Reg("WDOG_CTRL",     0x8030_1014, Access.RW,  desc="watchdog enable/window")

# ── Queue Doorbell ───────────────────────────────────────────────
Q0_DOORBELL = Reg("Q0_DOORBELL", 0x8030_2000, Access.WO, desc="compute queue kick")
Q1_DOORBELL = Reg("Q1_DOORBELL", 0x8030_2004, Access.WO, desc="utility queue kick")
Q2_DOORBELL = Reg("Q2_DOORBELL", 0x8030_2008, Access.WO, desc="telemetry queue kick")
Q3_DOORBELL = Reg("Q3_DOORBELL", 0x8030_200C, Access.WO, desc="hipri queue kick")

# Descriptor staging: 0x8030_2100..0x8030_213C (16 words)
DESC_STAGE_BASE = 0x8030_2100
DESC_STAGE_END  = 0x8030_213C
DESC_STAGE_WORDS = 16

# ── Queue Status ─────────────────────────────────────────────────
Q0_STATUS = Reg("Q0_STATUS", 0x8030_3000, Access.RO, desc="compute head/tail")
Q1_STATUS = Reg("Q1_STATUS", 0x8030_3004, Access.RO, desc="utility head/tail")
Q2_STATUS = Reg("Q2_STATUS", 0x8030_3008, Access.RO, desc="telemetry head/tail")
Q3_STATUS = Reg("Q3_STATUS", 0x8030_300C, Access.RO, desc="hipri head/tail")
Q_OVERFLOW = Reg("Q_OVERFLOW", 0x8030_3010, Access.W1C, desc="overflow flags")

QUEUE_DOORBELLS = [Q0_DOORBELL, Q1_DOORBELL, Q2_DOORBELL, Q3_DOORBELL]
QUEUE_STATUSES  = [Q0_STATUS, Q1_STATUS, Q2_STATUS, Q3_STATUS]

# ── DMA Engine ───────────────────────────────────────────────────
DMA_STATUS   = Reg("DMA_STATUS",   0x8031_0010, Access.RO, desc="busy/error/queue")
DMA_ERR_CODE = Reg("DMA_ERR_CODE", 0x8031_0014, Access.RO, desc="last error")

# ── OOM Guard ────────────────────────────────────────────────────
OOM_USAGE_LO = Reg("OOM_USAGE_LO", 0x8032_0000, Access.RO, desc="allocated bytes low")
OOM_RESV_LO  = Reg("OOM_RESV_LO",  0x8032_0008, Access.RO, desc="reserved bytes low")
OOM_EFF_LO   = Reg("OOM_EFF_LO",   0x8032_0010, Access.RO, desc="effective usage low")
OOM_STATE    = Reg("OOM_STATE",     0x8032_001C, Access.RO, desc="pressure state")

# ── TC0 Control ──────────────────────────────────────────────────
TC0_RUNSTATE  = Reg("TC0_RUNSTATE",  0x8034_0000, Access.RO,  desc="idle/fetch/run/stall/fault")
TC0_CTRL      = Reg("TC0_CTRL",      0x8034_0004, Access.RW, 0x01, "enable/halt/step/clr_fault")
TC0_DESC_PTR_LO = Reg("TC0_DESC_PTR_LO", 0x8034_0008, Access.RW, desc="current desc ptr low")
TC0_DESC_PTR_HI = Reg("TC0_DESC_PTR_HI", 0x8034_000C, Access.RW, desc="current desc ptr high")
TC0_PERF_CYC_LO = Reg("TC0_PERF_CYC_LO", 0x8034_0010, Access.RO, desc="perf cycles low")
TC0_PERF_CYC_HI = Reg("TC0_PERF_CYC_HI", 0x8034_0014, Access.RO, desc="perf cycles high")
TC0_FAULT_STATUS = Reg("TC0_FAULT_STATUS", 0x8034_0018, Access.W1C, desc="local fault cause")

# ── VPU/MXU Perf ─────────────────────────────────────────────────
MXU_BUSY_CYC_LO = Reg("MXU_BUSY_CYC_LO", 0x8036_0000, Access.RO, desc="MXU busy cycles lo")
MXU_BUSY_CYC_HI = Reg("MXU_BUSY_CYC_HI", 0x8036_0004, Access.RO, desc="MXU busy cycles hi")
VPU_BUSY_CYC_LO = Reg("VPU_BUSY_CYC_LO", 0x8036_0008, Access.RO, desc="VPU busy (0 in Proto-A)")
VPU_BUSY_CYC_HI = Reg("VPU_BUSY_CYC_HI", 0x8036_000C, Access.RO, desc="VPU busy (0 in Proto-A)")
MXU_TILE_COUNT  = Reg("MXU_TILE_COUNT",  0x8036_0010, Access.RO, desc="completed GEMM tiles")
VPU_OP_COUNT    = Reg("VPU_OP_COUNT",    0x8036_0014, Access.RO, desc="VPU ops (0 in Proto-A)")
PERF_FREEZE     = Reg("PERF_FREEZE",     0x8036_0018, Access.RW, desc="snapshot freeze")

# ── IRQ / MSI-X ──────────────────────────────────────────────────
IRQ_PENDING   = Reg("IRQ_PENDING",   0x8039_0000, Access.W1C, desc="pending bitmap")
IRQ_MASK      = Reg("IRQ_MASK",      0x8039_0004, Access.RW, 0xFFFF_FFFF, "mask bitmap")
IRQ_FORCE     = Reg("IRQ_FORCE",     0x8039_0008, Access.RW, desc="test inject")
IRQ_CAUSE_LAST = Reg("IRQ_CAUSE_LAST", 0x8039_0010, Access.RO, desc="last fatal cause")

# ── Trace Ring ───────────────────────────────────────────────────
TRACE_HEAD    = Reg("TRACE_HEAD",    0x803A_0000, Access.RO, desc="ring head")
TRACE_TAIL    = Reg("TRACE_TAIL",    0x803A_0004, Access.RO, desc="ring tail")
TRACE_CTRL    = Reg("TRACE_CTRL",    0x803A_0010, Access.RW, desc="enable/freeze/fatal_only")
TRACE_DROP_CNT = Reg("TRACE_DROP_CNT", 0x803A_0014, Access.RO, desc="dropped entries")

# Trace read window (Proto-A extension within allocated block)
TRACE_WIN_BASE  = 0x803A_0100  # entry N lo at +N*8, hi at +N*8+4
TRACE_META_BASE = 0x803A_3000  # entry N meta at +N*4

# ═══════════════════════════════════════════════════════════════════
# Bitfield helpers
# ═══════════════════════════════════════════════════════════════════

# BOOT_CAUSE
BOOT_CAUSE_POR  = 1 << 0
BOOT_CAUSE_WDOG = 1 << 1
BOOT_CAUSE_SW   = 1 << 2
BOOT_CAUSE_FLR  = 1 << 3

# TC0_RUNSTATE
TC_STATE_IDLE  = 0
TC_STATE_FETCH = 1
TC_STATE_RUN   = 2
TC_STATE_STALL = 3
TC_STATE_FAULT = 4
TC_WAIT_DMA    = 1 << 8
TC_WAIT_MEM    = 1 << 9

# TC0_CTRL
TC_CTRL_ENABLE = 1 << 0
TC_CTRL_HALT   = 1 << 1
TC_CTRL_STEP   = 1 << 2
TC_CTRL_CLR_FAULT = 1 << 3

# OOM_STATE
OOM_NORMAL   = 0
OOM_PRESSURE = 1
OOM_CRITICAL = 2
OOM_EMERG    = 3
OOM_ADMISSION_STOP = 1 << 8
OOM_PREFETCH_CLAMP = 1 << 9

# DMA_STATUS
DMA_BUSY    = 1 << 0
DMA_DONE    = 1 << 1
DMA_ERR     = 1 << 2
DMA_TIMEOUT = 1 << 3

# IRQ bitmap (REG_SPEC section 10.1)
class IrqBit(IntEnum):
    DESC_DONE      = 0
    DMA_DONE       = 1
    DMA_ERROR      = 2
    OOM_PRESSURE   = 3
    OOM_EMERGENCY  = 4
    TC0_FAULT      = 5
    TC1_FAULT      = 6
    HBM_ECC_CORR   = 7
    HBM_ECC_UNCORR = 8
    ICI_MAILBOX    = 9
    WATCHDOG       = 10
    TRACE_WRAP     = 11

IRQ_FATAL_MASK = (
    (1 << IrqBit.DMA_ERROR) |
    (1 << IrqBit.OOM_EMERGENCY) |
    (1 << IrqBit.TC0_FAULT) |
    (1 << IrqBit.TC1_FAULT) |
    (1 << IrqBit.HBM_ECC_UNCORR) |
    (1 << IrqBit.WATCHDOG)
)

# TRACE_CTRL
TRACE_ENABLE     = 1 << 0
TRACE_FREEZE     = 1 << 1
TRACE_FATAL_ONLY = 1 << 2

# G2_CAP0
CAP0_HAS_TC1        = 1 << 0
CAP0_HAS_HBM        = 1 << 1
CAP0_HAS_ICI        = 1 << 2
CAP0_HAS_ECC        = 1 << 3
CAP0_HAS_MSIX       = 1 << 4
CAP0_HAS_TRACE_RING = 1 << 5
CAP0_HAS_OOM_GUARD  = 1 << 6

# ═══════════════════════════════════════════════════════════════════
# Descriptor format (RTL-derived from desc_fsm_v2.sv / ctrl_fsm.sv)
# ═══════════════════════════════════════════════════════════════════
DESC_SIZE = 64  # bytes

# Byte offsets
DESC_OPCODE_OFF = 0
DESC_ACT_ADDR_OFF = 16  # u64 LE
DESC_WGT_ADDR_OFF = 24
DESC_OUT_ADDR_OFF = 32
DESC_KT_OFF       = 40  # u32 LE
DESC_CRC_OFF      = 63  # CRC-8 over bytes [0:62]

# Opcodes
class Opcode(IntEnum):
    NOP    = 0x01
    GEMM   = 0x02
    KVC_OP = 0x03
    VPU_OP = 0x04

# ═══════════════════════════════════════════════════════════════════
# Trace entry format (RTL-derived from g2_ctrl_top.sv)
# ═══════════════════════════════════════════════════════════════════
class TraceType(IntEnum):
    DESC_DISPATCH = 1
    DESC_DONE     = 2
    DESC_FAULT    = 3
    Q_OVERFLOW    = 4

# Trace payload layout: {46'b0, qclass[1:0], opcode_or_fault[7:0], 8'b0}
# Meta: {24'b0, type[3:0], 3'b0, fatal}

# ═══════════════════════════════════════════════════════════════════
# All registers list (for enumeration / tests)
# ═══════════════════════════════════════════════════════════════════
ALL_REGS = [
    G2_ID, G2_VERSION, G2_CAP0, G2_CAP1, BUILD_HASH_LO, BUILD_HASH_HI,
    BOOT_CAUSE, SW_RESET, BOOT_VECTOR_LO, BOOT_VECTOR_HI, STRAP_STATUS, WDOG_CTRL,
    Q0_DOORBELL, Q1_DOORBELL, Q2_DOORBELL, Q3_DOORBELL,
    Q0_STATUS, Q1_STATUS, Q2_STATUS, Q3_STATUS, Q_OVERFLOW,
    DMA_STATUS, DMA_ERR_CODE,
    OOM_USAGE_LO, OOM_RESV_LO, OOM_EFF_LO, OOM_STATE,
    TC0_RUNSTATE, TC0_CTRL, TC0_DESC_PTR_LO, TC0_DESC_PTR_HI,
    TC0_PERF_CYC_LO, TC0_PERF_CYC_HI, TC0_FAULT_STATUS,
    MXU_BUSY_CYC_LO, MXU_BUSY_CYC_HI, VPU_BUSY_CYC_LO, VPU_BUSY_CYC_HI,
    MXU_TILE_COUNT, VPU_OP_COUNT, PERF_FREEZE,
    IRQ_PENDING, IRQ_MASK, IRQ_FORCE, IRQ_CAUSE_LAST,
    TRACE_HEAD, TRACE_TAIL, TRACE_CTRL, TRACE_DROP_CNT,
]


def offset(addr: int) -> int:
    """Convert absolute address to reg_top offset."""
    return addr - BASE
