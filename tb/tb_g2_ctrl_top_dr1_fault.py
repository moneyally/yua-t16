"""기대값 출처: spec/deltarule.md 2·4·5절 + tools/orbit_mmio_map.py (레지스터맵 SSOT).
디스크립터는 tools/orbit_desc.py 의 호스트 패커가 만든다 — 테스트가 바이트를 손으로 쓰지 않는다.

tb_g2_ctrl_top_dr1_fault.py — PLAN W6: BUG-001 회귀를 **DR1 경로로 확장**

tb_g2_ctrl_top_fault_irq.py 는 CRC·illegal opcode 로 "fault 가 DESC_DONE 을 올리지
않는다"를 확인한다. 여기서는 **엔진(dr1_top)이 낸 실패**로 같은 것을 확인한다.
경로가 다르다: desc_fsm_v2 가 스스로 판단한 fault 가 아니라, 디스패치 후에
엔진이 done_err 를 올린 경우다. W6 에서 `core_err` 포트를 새로 뚫었으므로
이 경로가 처음 생겼다.

  D1. DELTA_INIT(slot 0)  → DESC_DONE 뜨고 TC0_FAULT 없음      (정상 경로)
  D2. DELTA_DUMP(정렬됨)  → DESC_DONE 뜨고 TC0_FAULT 없음      (정상 경로)
  D3. slot=1 (호스트 검사 우회) → TC0_FAULT + fault_code 0x05, **DESC_DONE 없음**
  D4. DUMP dst 미정렬     → TC0_FAULT + 0x06, **DESC_DONE 없음**
  D5. DELTA_STEP(0x51)    → W6 에는 없는 경로. ILLEGAL_OPCODE(0x01), DESC_DONE 없음
  D6. DR1 레지스터가 읽히고 W1C 가 동작한다
"""
import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.orbit_desc import (  # noqa: E402
    Dr1FieldError,
    crc8,
    pack_delta_dump,
    pack_delta_init,
    pack_delta_step,
)
from tools.orbit_mmio_map import (  # noqa: E402
    BASE,
    dr1_scratch_layout,
    DESC_SIZE,
    DESC_STAGE_BASE,
    DESC_STAGE_WORDS,
    DR1_CLAMP_COUNT,
    DR1_CYCLES,
    DR1_SAT_COUNT,
    DR1_SCRATCH_WORDS,
    DR1_SLOT_OFF,
    DR1_STATUS,
    IRQ_MASK,
    IRQ_PENDING,
    FaultCode,
    IrqBit,
    Opcode,
    Q0_DOORBELL,
    TC0_FAULT_STATUS,
    TRACE_CTRL,
)

A_DESC_STAGE = DESC_STAGE_BASE - BASE
A_Q0_DOORBELL = Q0_DOORBELL.addr - BASE
A_IRQ_PENDING = IRQ_PENDING.addr - BASE
A_IRQ_MASK = IRQ_MASK.addr - BASE
A_TC0_FAULT = TC0_FAULT_STATUS.addr - BASE
A_TRACE_CTRL = TRACE_CTRL.addr - BASE
A_DR1_STATUS = DR1_STATUS.addr - BASE
A_DR1_SAT = DR1_SAT_COUNT.addr - BASE
A_DR1_CLAMP = DR1_CLAMP_COUNT.addr - BASE
A_DR1_CYCLES = DR1_CYCLES.addr - BASE

BIT_DESC_DONE = 1 << IrqBit.DESC_DONE
BIT_TC0_FAULT = 1 << IrqBit.TC0_FAULT

# 스크래치 배치는 spec/deltarule.md 3.6절의 단일 출처에서 온다
LAY = dr1_scratch_layout(16)
DUMP_ADDR = LAY["dump"] * 2


def force_slot(desc: bytes, slot: int) -> list:
    """호스트 패커의 슬롯 검사를 우회해서 잘못된 슬롯을 박아 넣는다.

    tools/orbit_desc.pack_delta_init 은 slot>=1 이면 Dr1FieldError 를 낸다 (정상).
    하지만 **호스트가 버그면 하드웨어가 잡아야 한다** — 그것을 보는 테스트다.
    CRC 는 다시 계산해서 CRC 오류로 잡히지 않게 한다.
    """
    d = list(desc)
    d[DR1_SLOT_OFF] = slot & 0xFF
    d[DESC_SIZE - 1] = crc8(d[: DESC_SIZE - 1])
    return d


def force_opcode(desc: bytes, opcode: int) -> list:
    d = list(desc)
    d[0] = opcode & 0xFF
    d[DESC_SIZE - 1] = crc8(d[: DESC_SIZE - 1])
    return d


def force_qaddr(desc: bytes, addr: int) -> list:
    """q_addr(바이트 16) 을 호스트 검사를 우회해 바꿔 넣는다."""
    d = list(desc)
    for i in range(8):
        d[16 + i] = (addr >> (8 * i)) & 0xFF
    d[DESC_SIZE - 1] = crc8(d[: DESC_SIZE - 1])
    return d


def force_dst(desc: bytes, addr: int) -> list:
    """out_addr(바이트 32) 를 미정렬 주소로 바꾼다. 패커는 정렬을 강제한다."""
    d = list(desc)
    for i in range(8):
        d[32 + i] = (addr >> (8 * i)) & 0xFF
    d[DESC_SIZE - 1] = crc8(d[: DESC_SIZE - 1])
    return d


async def reset_dut(dut):
    dut.por_n.value = 0
    dut.reg_addr.value = 0
    dut.reg_wr_en.value = 0
    dut.reg_wr_data.value = 0
    dut.rd_req_ready.value = 1
    dut.rd_done.value = 0
    dut.rd_data_valid.value = 0
    dut.rd_data.value = 0
    dut.rd_data_last.value = 0
    dut.wr_req_ready.value = 1
    dut.wr_done.value = 0
    dut.wr_data_ready.value = 1
    await Timer(100, unit="ns")
    dut.por_n.value = 1
    for _ in range(30):
        await RisingEdge(dut.clk)
        if dut.reset_active.value == 0:
            break
    await RisingEdge(dut.clk)


async def reg_write(dut, addr, data):
    dut.reg_addr.value = addr
    dut.reg_wr_en.value = 1
    dut.reg_wr_data.value = data
    await RisingEdge(dut.clk)
    dut.reg_wr_en.value = 0


async def reg_read(dut, addr):
    dut.reg_addr.value = addr
    dut.reg_wr_en.value = 0
    await RisingEdge(dut.clk)
    return int(dut.reg_rd_data.value)


async def stage_and_bell(dut, desc, q=0):
    for i in range(DESC_STAGE_WORDS):
        w = 0
        for b in range(4):
            idx = i * 4 + b
            if idx < DESC_SIZE:
                w |= desc[idx] << (8 * b)
        await reg_write(dut, A_DESC_STAGE + i * 4, w)
    await reg_write(dut, A_Q0_DOORBELL + q * 4, 1)


async def run_descriptor(dut, desc, settle=300):
    await reg_write(dut, A_IRQ_MASK, 0x0000_0000)
    await reg_write(dut, A_TRACE_CTRL, 1)
    await stage_and_bell(dut, list(desc))
    for _ in range(settle):
        await RisingEdge(dut.clk)
    pending = await reg_read(dut, A_IRQ_PENDING)
    fault = await reg_read(dut, A_TC0_FAULT)
    return pending, fault


def describe(pending):
    bits = [b.name for b in IrqBit if pending & (1 << b)]
    return f"{pending:#010x} {bits}"


@cocotb.test()
async def test_d1_delta_init_completes(dut):
    """D1: DELTA_INIT 이 desc_fsm_v2 를 통과해 DESC_DONE 을 올린다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    pending, fault = await run_descriptor(dut, pack_delta_init(slot=0))
    dut._log.info(f"D1 DELTA_INIT IRQ={describe(pending)} TC0_FAULT={fault:#010x}")
    assert pending & BIT_DESC_DONE, f"D1: DELTA_INIT 인데 DESC_DONE 이 없다 {describe(pending)}"
    assert not (pending & BIT_TC0_FAULT), f"D1: fault 가 떴다 {describe(pending)}"


@cocotb.test()
async def test_d2_delta_dump_completes(dut):
    """D2: DELTA_DUMP(16바이트 정렬) 도 정상 완료한다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await run_descriptor(dut, pack_delta_init(slot=0))
    await reg_write(dut, A_IRQ_PENDING, 0xFFFF_FFFF)      # W1C 로 지우고 다시 본다
    # DUMP 는 d행 × (d원소 쓰기) 라 약 305 사이클이다 (F2 실측). 기본 settle 로는 모자란다.
    pending, fault = await run_descriptor(dut, pack_delta_dump(DUMP_ADDR, slot=0), settle=800)
    dut._log.info(f"D2 DELTA_DUMP IRQ={describe(pending)} TC0_FAULT={fault:#010x}")
    assert pending & BIT_DESC_DONE, f"D2: DESC_DONE 이 없다 {describe(pending)}"
    assert not (pending & BIT_TC0_FAULT), f"D2: fault 가 떴다 {describe(pending)}"


@cocotb.test()
async def test_d3_bad_slot_never_raises_desc_done(dut):
    """D3: slot 초과 → TC0_FAULT(0x05) 만. **DESC_DONE 은 뜨면 안 된다** (BUG-001).

    이것이 W6 에서 새로 생긴 경로다: desc_fsm_v2 가 판단한 fault 가 아니라
    **엔진이 done_err 로 올린** fault 다 (core_err 포트).
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    desc = force_slot(pack_delta_init(slot=0), 1)
    pending, fault = await run_descriptor(dut, desc)
    dut._log.info(f"D3 slot=1 IRQ={describe(pending)} TC0_FAULT={fault:#010x}")

    assert pending & BIT_TC0_FAULT, f"D3: 잘못된 슬롯인데 TC0_FAULT 가 없다 {describe(pending)}"
    assert (fault & 0xFF) == int(FaultCode.DR1_BAD_SLOT), (
        f"D3: fault_code={fault & 0xFF:#04x}, {int(FaultCode.DR1_BAD_SLOT):#04x} 이어야 한다"
    )
    assert not (pending & BIT_DESC_DONE), (
        "D3 위반 — 엔진이 실패했는데 DESC_DONE(완료) IRQ 가 떴다. BUG-001 과 같은 종류다. "
        f"IRQ_PENDING={describe(pending)}"
    )


@cocotb.test()
async def test_d4_unaligned_dump_never_raises_desc_done(dut):
    """D4: DUMP dst_addr 미정렬 → TC0_FAULT(0x06), DESC_DONE 없음."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    desc = force_dst(pack_delta_dump(DUMP_ADDR, slot=0), 0x2008)   # 8바이트 정렬 = 미정렬
    pending, fault = await run_descriptor(dut, desc)
    dut._log.info(f"D4 미정렬 IRQ={describe(pending)} TC0_FAULT={fault:#010x}")

    assert pending & BIT_TC0_FAULT, f"D4: TC0_FAULT 가 없다 {describe(pending)}"
    assert (fault & 0xFF) == int(FaultCode.DR1_UNALIGNED), (
        f"D4: fault_code={fault & 0xFF:#04x}, {int(FaultCode.DR1_UNALIGNED):#04x} 이어야 한다"
    )
    assert not (pending & BIT_DESC_DONE), (
        f"D4 위반 — 실패인데 DESC_DONE 이 떴다 {describe(pending)}"
    )


@cocotb.test()
async def test_d5_delta_step_completes_end_to_end(dut):
    """D5: **DELTA_STEP 이 디스크립터 경로로 끝까지 돈다** (W7).

    W6 에서는 이 자리에 "0x51 은 desc_fsm_v2 가 ILLEGAL_OPCODE 로 막는다" 가
    있었다. W7 에서 계산 경로가 생겨서 **검사 대상이 바뀌었다** — 기대값을
    고쳐 통과시킨 것이 아니다.

    여기서는 완료 계약만 본다 (DESC_DONE 뜨고 TC0_FAULT 없음).
    값이 골든과 맞는지는 tb/tb_dr1_harness_rtl.py 와 tb/tb_dr1_host_e2e.py 가 본다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await run_descriptor(dut, pack_delta_init(slot=0))
    await reg_write(dut, A_IRQ_PENDING, 0xFFFF_FFFF)

    desc = pack_delta_step(
        q_addr=LAY["q"] * 2, k_addr=LAY["k"] * 2, v_addr=LAY["v"] * 2,
        o_addr=LAY["o"] * 2, alpha_uq15=0x8000, beta_uq15=0x4000, slot=0,
    )
    pending, fault = await run_descriptor(dut, desc, settle=600)
    dut._log.info(f"D5 STEP IRQ={describe(pending)} TC0_FAULT={fault:#010x}")

    assert pending & BIT_DESC_DONE, f"D5: STEP 이 완료되지 않았다 {describe(pending)}"
    assert not (pending & BIT_TC0_FAULT), (
        f"D5: STEP 이 fault 를 냈다 TC0_FAULT={fault:#010x} {describe(pending)}"
    )


@cocotb.test()
async def test_d5b_step_with_bad_addr_still_faults(dut):
    """D5b: STEP 이 동작해도 **잘못된 주소는 여전히 막는다** (0x08 DR1_ADDR_RANGE).

    경로가 열렸다고 검사가 느슨해지면 안 된다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # 호스트 패커가 먼저 잡는다 — 그것부터 확인한다
    try:
        pack_delta_step(
            q_addr=(DR1_SCRATCH_WORDS - 8) * 2, k_addr=LAY["k"] * 2, v_addr=LAY["v"] * 2,
            o_addr=LAY["o"] * 2, alpha_uq15=0x8000, beta_uq15=0, slot=0,
        )
    except Dr1FieldError:
        pass
    else:
        raise AssertionError("D5b: 호스트 패커가 범위 초과를 안 잡았다")

    # 호스트를 우회해서 하드웨어가 잡는지 본다
    desc = force_qaddr(
        pack_delta_step(
            q_addr=LAY["q"] * 2, k_addr=LAY["k"] * 2, v_addr=LAY["v"] * 2,
            o_addr=LAY["o"] * 2, alpha_uq15=0x8000, beta_uq15=0, slot=0,
        ),
        (DR1_SCRATCH_WORDS - 8) * 2,
    )
    pending, fault = await run_descriptor(dut, desc)
    dut._log.info(f"D5b 범위초과 IRQ={describe(pending)} TC0_FAULT={fault:#010x}")
    assert pending & BIT_TC0_FAULT, f"D5b: TC0_FAULT 가 없다 {describe(pending)}"
    assert (fault & 0xFF) == int(FaultCode.DR1_ADDR_RANGE), (
        f"D5b: fault_code={fault & 0xFF:#04x}"
    )
    assert not (pending & BIT_DESC_DONE), f"D5b: 실패인데 DESC_DONE 이 떴다 {describe(pending)}"


@cocotb.test()
async def test_d6_dr1_registers(dut):
    """D6: DR1 레지스터가 reg_top 에 배선됐고 W1C 가 동작한다 (spec 5절)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await run_descriptor(dut, pack_delta_init(slot=0))

    status = await reg_read(dut, A_DR1_STATUS)
    sat = await reg_read(dut, A_DR1_SAT)
    clamp = await reg_read(dut, A_DR1_CLAMP)
    cycles = await reg_read(dut, A_DR1_CYCLES)
    dut._log.info(
        f"D6 DR1_STATUS={status:#010x} SAT={sat} CLAMP={clamp} CYCLES={cycles}"
    )

    assert status & 1 == 0, "D6: 디스크립터가 끝났는데 busy 가 1 이다"
    # W6 에는 포화/클램프를 올리는 경로가 없다 (STEP 이 W7). 0 이어야 한다.
    assert sat == 0, f"D6: SAT_COUNT={sat}, W6 에는 올리는 경로가 없다"
    assert clamp == 0, f"D6: CLAMP_COUNT={clamp}"
    assert cycles > 0, "D6: INIT 이 끝났는데 DR1_CYCLES 가 0 이다"

    # W1C: 읽은 값을 그대로 다시 쓴다 → 0. 0 을 쓰면 아무 일도 없어야 한다.
    await reg_write(dut, A_DR1_SAT, 0)
    assert await reg_read(dut, A_DR1_SAT) == 0
