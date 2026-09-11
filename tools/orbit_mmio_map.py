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
    DR1      = 0x8033_0000  # ORBIT-DR1 델타룰 헤드 (spec/deltarule.md)
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

# ── DR1 (ORBIT-DR1 델타룰 헤드) — SSOT: spec/deltarule.md 5절 ──────
DR1_STATUS      = Reg("DR1_STATUS",      0x8033_0000, Access.RO,  0, "[0] busy, [11:4] last_slot")
DR1_SAT_COUNT   = Reg("DR1_SAT_COUNT",   0x8033_0004, Access.W1C, 0, "포화 누적 횟수")
DR1_CLAMP_COUNT = Reg("DR1_CLAMP_COUNT", 0x8033_0008, Access.W1C, 0, "alpha/beta 클램프 누적 횟수")
DR1_CYCLES      = Reg("DR1_CYCLES",      0x8033_000C, Access.RO,  0, "마지막 DELTA_STEP 사이클 수")

DR1_STATUS_BUSY      = 1 << 0
DR1_STATUS_SLOT_SHIFT = 4
DR1_STATUS_SLOT_MASK  = 0xFF << DR1_STATUS_SLOT_SHIFT

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

# WDOG_CTRL — SSOT: spec/watchdog.md 1절
WDOG_EN          = 1 << 0        # 타이머 동작
WDOG_KICK        = 1 << 1        # 쓰기 펄스: 카운터 리로드 (저장 안 됨, 0 으로 읽힘)
WDOG_TEST_FIRE   = 1 << 31       # 쓰기 펄스: 즉시 리셋 (EN 과 무관, 기존 동작)
WDOG_PERIOD_SH   = 8
WDOG_PERIOD_W    = 16
WDOG_PERIOD_MASK = ((1 << WDOG_PERIOD_W) - 1) << WDOG_PERIOD_SH
WDOG_PRESCALE    = 1024          # 타임아웃 = (PERIOD+1) × 1024 사이클


def wdog_ctrl_word(period: int, *, enable: bool = True, kick: bool = True) -> int:
    """`WDOG_CTRL` 에 쓸 32비트 값. **PERIOD 와 EN 을 한 번에 쓴다.**

    두 번 나눠 쓰면 그 사이에 옛 PERIOD 로 터질 수 있다 (spec/watchdog.md 3절).
    기본으로 KICK 을 같이 실어 새 창에서 출발한다.
    """
    if not 0 <= period < (1 << WDOG_PERIOD_W):
        raise ValueError(
            f"WDOG PERIOD 는 0..{(1 << WDOG_PERIOD_W) - 1} 여야 한다 (받은 값 {period})"
        )
    word = (period << WDOG_PERIOD_SH) & WDOG_PERIOD_MASK
    if enable:
        word |= WDOG_EN
    if kick:
        word |= WDOG_KICK
    return word


def wdog_timeout_cycles(period: int) -> int:
    """타임아웃까지의 사이클 수. spec/watchdog.md 1절의 수식 그대로."""
    return (period + 1) * WDOG_PRESCALE

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

# ═══════════════════════════════════════════════════════════════════
# ORBIT-DR1 디스크립터 필드 — SSOT: spec/deltarule.md 3절
#
# 기존 필드 위치를 그대로 재사용한다. rtl/desc_fsm_v2.sv 의 추출 로직을
# 바꾸지 않기 위해서다:
#     q_addr = act_addr(16), k_addr = wgt_addr(24), o_addr = out_addr(32)
# 비어 있던 예약 영역만 새로 쓴다: v_addr(44), alpha(52), beta(54)
# tests/test_dr1_spec_consistency.py 가 spec 문서 표와 아래 상수를 대조한다.
# ═══════════════════════════════════════════════════════════════════
DR1_SLOT_OFF   = 1    # u8
DR1_Q_ADDR_OFF = 16   # u64 LE  (= DESC_ACT_ADDR_OFF)
DR1_K_ADDR_OFF = 24   # u64 LE  (= DESC_WGT_ADDR_OFF)
DR1_O_ADDR_OFF = 32   # u64 LE  (= DESC_OUT_ADDR_OFF)
DR1_V_ADDR_OFF = 44   # u64 LE
DR1_ALPHA_OFF  = 52   # u16 LE, UQ1.15
DR1_BETA_OFF   = 54   # u16 LE, UQ1.15

DR1_NUM_SLOTS   = 1       # v1. slot != 0 이면 DR1_BAD_SLOT
DR1_ADDR_ALIGN  = 16      # 바이트. act_sram 데이터 폭 128비트
UQ15_ONE        = 0x8000  # UQ1.15 에서 1.0 (정확)

# ── DR1 스크래치 (spec/deltarule.md 3.6절) ─────────────────────────
DR1_SCRATCH_BASE  = 0x8033_1000   # MMIO 창. 32비트 워드 하나 = Q1.15 원소 하나

# **두 개의 한계가 우연히 같은 값이다. 헷갈리지 말 것.**
#   DR1_SCRATCH_MMIO_WORDS — 호스트가 **닿을 수 있는** 원소 수. `rtl/reg_top.sv` 의
#       `dr1_scr_addr` 이 10비트이고 `A_DR1_SCR_END = 0x3_1FFC` 가 딱 이만큼을 덮는다.
#   DR1_SCRATCH_WORDS      — RTL 스크래치의 실제 깊이 (`rtl/dr1/dr1_scratch.sv` DEPTH).
#
# d=64 로 키우려면 **둘 다** 늘려야 한다. RTL DEPTH 만 8192 로 올리고 이 값을 따라
# 올리면, 호스트는 1024 원소 너머를 영영 못 읽는데 아무도 모른다 (창 밖 읽기는
# 0 으로 떨어진다). 그래서 `tests/test_mmio_map.py` 가 둘이 같은지 검사한다 —
# 다르게 만들려면 `reg_top` 의 디코드 폭을 먼저 넓히고 그 테스트를 고쳐야 한다.
DR1_SCRATCH_MMIO_WORDS = 1024
DR1_SCRATCH_WORDS = 1024
DR1_SCRATCH_END   = DR1_SCRATCH_BASE + DR1_SCRATCH_MMIO_WORDS * 4 - 4
DR1_DUMP_ELEM     = 512           # DELTA_DUMP 기본 목적지 (원소 인덱스)


def dr1_scratch_layout(d: int, scratch_words: int | None = None) -> dict:
    """DR1 스크래치의 표준 배치. **단일 출처** (spec/deltarule.md 3.6절).

    테스트벤치(`tb/tb_dr1_top.py` CocotbDut)와 호스트 HAL(`tools/orbit_device.py`)이
    둘 다 이 함수만 쓴다. 배치가 두 곳에 있으면 한쪽을 고칠 때 다른 쪽이 조용히 틀린다.

    반환값은 **원소 인덱스**다. 바이트 주소는 ×2.
    모든 벡터 시작점은 16바이트 정렬이어야 한다 (= 원소 인덱스가 8의 배수).

    `scratch_words` 를 주면 그 크기로 검사한다. d=64 는 덤프(64²=4096)가
    기본 스크래치(1024)에 안 들어가므로, 더 큰 스크래치로 파라미터화한
    시뮬레이션에서만 쓴다 (`tb/tb_dr1_d64.py`).
    """
    words = DR1_SCRATCH_WORDS if scratch_words is None else int(scratch_words)
    layout = {
        "q": 0,
        "k": d,
        "v": 2 * d,
        "o": 3 * d,
        "dump": DR1_DUMP_ELEM,
    }
    for name, elem in layout.items():
        if (elem * 2) % DR1_ADDR_ALIGN != 0:
            raise ValueError(
                f"d={d} 에서 {name} 시작 원소 {elem} (바이트 {elem*2}) 가 "
                f"{DR1_ADDR_ALIGN}바이트 정렬이 아니다 — 하드웨어가 0x06 을 낸다"
            )
    if layout["o"] + d > words or layout["dump"] + d * d > words:
        raise ValueError(
            f"d={d} 는 스크래치 {words} 원소에 들어가지 않는다 "
            f"(필요: 덤프 {layout['dump'] + d * d}, 벡터 {layout['o'] + d})"
        )
    return layout



# Opcodes
class Opcode(IntEnum):
    NOP    = 0x01
    GEMM   = 0x02
    KVC_OP = 0x03
    VPU_OP = 0x04
    # G3 경로 (rtl/g3_desc_fsm.sv)
    MXU_FWD    = 0x10
    BACKWARD   = 0x20
    OPTIMIZER  = 0x30
    COLLECTIVE = 0x40
    # ORBIT-DR1 (spec/deltarule.md 2절)
    DELTA_INIT = 0x50
    DELTA_STEP = 0x51
    DELTA_DUMP = 0x52


class FaultCode(IntEnum):
    """desc_fsm_v2 fault_code. spec/deltarule.md 4절."""
    NONE           = 0x00
    ILLEGAL_OPCODE = 0x01
    CRC_MISMATCH   = 0x02
    TIMEOUT        = 0x03
    RESERVED       = 0x04
    DR1_BAD_SLOT   = 0x05
    DR1_UNALIGNED  = 0x06
    DR1_UNIMPL     = 0x07
    DR1_ADDR_RANGE = 0x08

# ═══════════════════════════════════════════════════════════════════
# Trace entry format (RTL-derived from g2_ctrl_top.sv)
# ═══════════════════════════════════════════════════════════════════
class TraceType(IntEnum):
    DESC_DISPATCH = 1
    DESC_DONE     = 2
    DESC_FAULT    = 3
    Q_OVERFLOW    = 4
    SAT_EVENT     = 5   # spec/deltarule.md 5.2
    CLAMP_EVENT   = 6

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
