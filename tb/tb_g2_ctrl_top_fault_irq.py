"""기대값 출처: tools/orbit_mmio_map.py (레지스터맵 SSOT) + docs/DESIGN.md 9절 완료신호 정의.

tb_g2_ctrl_top_fault_irq.py — PLAN W2-2: fault 디스크립터의 IRQ 계약

docs/AUDIT.md §4 가설의 최종 확인 지점.

  rtl/desc_fsm_v2.sv:331   ST_FAULT -> ST_DONE   (fault 도 ST_DONE 을 거친다)
  rtl/desc_fsm_v2.sv:328   ST_DONE 에서 done_pulse = 1
  rtl/g2_ctrl_top.sv:460   irq_sources = { ..., fsm_fault_valid, ..., fsm_done_pulse }
                           → irq_sources[0] = DESC_DONE = fsm_done_pulse
                             irq_sources[5] = TC0_FAULT = fsm_fault_valid

  ⇒ CRC 나 opcode 가 틀린 디스크립터 하나가 **TC0_FAULT 와 DESC_DONE 을 동시에** 올린다.
     호스트 입장에서는 "실패했는데 완료 인터럽트가 왔다"가 된다.

계약:
  F1. fault 디스크립터는 TC0_FAULT IRQ 를 올려야 한다.        (정상 동작)
  F2. fault 디스크립터는 DESC_DONE IRQ 를 올리면 **안 된다**.  (가설 — 지금 실패하면 버그 확정)
  F3. 정상 디스크립터(NOP)는 DESC_DONE 을 올리고 TC0_FAULT 는 올리지 않는다. (대조군)

주소는 전부 tools/orbit_mmio_map.py 에서 가져온다 (docs/AUDIT.md §3 C급: 기존
테스트벤치들이 주소를 파일마다 재타이핑하는 문제를 여기서는 반복하지 않는다).
"""
import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.orbit_mmio_map import (  # noqa: E402
    BASE,
    DESC_SIZE,
    DESC_STAGE_BASE,
    DESC_STAGE_WORDS,
    IRQ_MASK,
    IRQ_PENDING,
    IrqBit,
    Q0_DOORBELL,
    TC0_FAULT_STATUS,
    TRACE_CTRL,
)

# DUT 의 reg_addr 는 BASE 로부터의 오프셋이다.
A_DESC_STAGE = DESC_STAGE_BASE - BASE
A_Q0_DOORBELL = Q0_DOORBELL.addr - BASE
A_IRQ_PENDING = IRQ_PENDING.addr - BASE
A_IRQ_MASK = IRQ_MASK.addr - BASE
A_TC0_FAULT = TC0_FAULT_STATUS.addr - BASE
A_TRACE_CTRL = TRACE_CTRL.addr - BASE

BIT_DESC_DONE = 1 << IrqBit.DESC_DONE
BIT_TC0_FAULT = 1 << IrqBit.TC0_FAULT


def crc8(data):
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def make_desc(opcode, kt=4, valid_crc=True):
    d = [0] * DESC_SIZE
    d[0] = opcode & 0xFF
    for i in range(8):
        d[8 + i] = (0x1000 >> (8 * i)) & 0xFF
        d[16 + i] = (0x2000 >> (8 * i)) & 0xFF
        d[24 + i] = (0x3000 >> (8 * i)) & 0xFF
    for i in range(4):
        d[40 + i] = (kt >> (8 * i)) & 0xFF
    d[DESC_SIZE - 1] = crc8(d[: DESC_SIZE - 1])
    if not valid_crc:
        d[DESC_SIZE - 1] ^= 0xFF
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


async def run_descriptor(dut, desc, settle=200):
    """디스크립터 하나를 밀어 넣고 IRQ_PENDING 을 읽어 돌려준다."""
    await reg_write(dut, A_IRQ_MASK, 0x0000_0000)  # 전부 언마스크
    await reg_write(dut, A_TRACE_CTRL, 1)
    await stage_and_bell(dut, desc)
    for _ in range(settle):
        await RisingEdge(dut.clk)
    return await reg_read(dut, A_IRQ_PENDING)


def describe(pending):
    bits = [b.name for b in IrqBit if pending & (1 << b)]
    return f"{pending:#010x} {bits}"


@cocotb.test()
async def test_nop_raises_desc_done_only(dut):
    """F3 대조군: 정상 NOP → DESC_DONE 만, TC0_FAULT 는 없어야 한다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    pending = await run_descriptor(dut, make_desc(0x01))
    dut._log.info(f"NOP        IRQ_PENDING = {describe(pending)}")
    assert pending & BIT_DESC_DONE, f"정상 NOP 인데 DESC_DONE 이 없다: {describe(pending)}"
    assert not (pending & BIT_TC0_FAULT), f"정상 NOP 인데 TC0_FAULT 가 떴다: {describe(pending)}"


@cocotb.test()
async def test_illegal_opcode_must_not_raise_desc_done(dut):
    """F1+F2: illegal opcode → TC0_FAULT 는 뜨고 DESC_DONE 은 뜨면 안 된다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    pending = await run_descriptor(dut, make_desc(0xFF))
    dut._log.info(f"ILLEGAL    IRQ_PENDING = {describe(pending)}")
    fault_status = await reg_read(dut, A_TC0_FAULT)
    dut._log.info(f"ILLEGAL    TC0_FAULT_STATUS = {fault_status:#010x}")

    assert pending & BIT_TC0_FAULT, f"F1: illegal opcode 인데 TC0_FAULT 가 없다: {describe(pending)}"
    assert not (pending & BIT_DESC_DONE), (
        "F2 위반 — 실패한 디스크립터에 DESC_DONE(완료) IRQ 가 떴다. "
        f"IRQ_PENDING={describe(pending)}. "
        "원인: rtl/desc_fsm_v2.sv:331 ST_FAULT -> ST_DONE 이라 fault 도 done_pulse 를 내고, "
        "rtl/g2_ctrl_top.sv:460 에서 irq_sources[0]=DESC_DONE=fsm_done_pulse 이다. "
        "docs/AUDIT.md §4 가설 확정 → docs/BUGS.md."
    )


@cocotb.test()
async def test_crc_fail_must_not_raise_desc_done(dut):
    """F1+F2: CRC 불일치도 같다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    pending = await run_descriptor(dut, make_desc(0x02, valid_crc=False))
    dut._log.info(f"CRCFAIL    IRQ_PENDING = {describe(pending)}")

    assert pending & BIT_TC0_FAULT, f"F1: CRC 불일치인데 TC0_FAULT 가 없다: {describe(pending)}"
    assert not (pending & BIT_DESC_DONE), (
        "F2 위반 — CRC 실패 디스크립터에 DESC_DONE(완료) IRQ 가 떴다. "
        f"IRQ_PENDING={describe(pending)}"
    )
