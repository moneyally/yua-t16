"""기대값 출처: rtl/oom_guard.sv 카운터 사양 + docs/DESIGN.md 5.1절 완료 신호 계약 (프로토콜 계약 테스트, 골든 모델 아님).

tb_g2_ctrl_top_oom_fault.py — BUG-001 수정의 보완 검증

BUG-001 을 고칠 때 `ST_FAULT -> ST_DONE` 전이를 **일부러 남겼다.**
`oom_alloc_dec` 가 `fsm_done_pulse`(리타이어)에 걸려 있어서, `ST_IDLE` 로 직행시키면
fault 트랜잭션마다 OOM 사용량이 감소하지 않고 누수되기 때문이다
(docs/DESIGN.md 5.1절 규칙 4, docs/BUGS.md BUG-001).

그 근거를 테스트로 고정한다. `rtl/oom_guard.sv:73` 의 카운터는 **레벨 감지**다 —
`alloc_dec` 가 1인 사이클마다 감소한다. 따라서 "정확히 1사이클, 정확히 1회"가
안 지켜지면 사용량이 어긋난다.

검사:
  O1. fault(done_err) 트랜잭션 후 OOM 사용량이 **정확히 시작값으로 돌아온다.**
      (감소 0회면 누수, 2회 이상이면 과다 반납)
  O2. 성공(done_ok) 트랜잭션도 마찬가지.
  O3. underflow_error 는 끝까지 0.
  O4. fault 를 여러 번 반복해도 사용량이 누적되지 않는다 (누수 없음).

`rtl/g2_ctrl_top.sv` 의 DESC_COST=4096 이 디스크립터 1개당 증감량이다.
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
    OOM_STATE,
    Q0_DOORBELL,
    TRACE_CTRL,
)

A_DESC_STAGE = DESC_STAGE_BASE - BASE
A_Q0_DOORBELL = Q0_DOORBELL.addr - BASE
A_IRQ_MASK = IRQ_MASK.addr - BASE
A_TRACE_CTRL = TRACE_CTRL.addr - BASE
A_OOM_STATE = OOM_STATE.addr - BASE
# OOM_USAGE_LO / OOM_EFF_LO 는 orbit_mmio_map 에 Reg 객체가 아니라 상수로 없다.
# REG_SPEC 상 USAGE 는 OOM 블록 베이스, EFF 는 +0x10 이다 (tb_g2_ctrl_top_oom.py 와 동일).
A_OOM_USAGE = 0x2_0000
A_OOM_EFF = 0x2_0010

DESC_COST = 4096


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


async def run_one(dut, desc, settle=250):
    """디스크립터 1개를 처리하고, 그동안 alloc_dec 가 몇 사이클 high 였는지 센다.

    oom_alloc_dec 는 g2_ctrl_top 내부 신호다. 시뮬레이터가 접근을 허용하면 직접 세고,
    아니면 None 을 돌려준다 (그때는 사용량 차이로만 판정한다).
    """
    await stage_and_bell(dut, desc)
    dec_cycles = None
    try:
        handle = dut.oom_alloc_dec
        dec_cycles = 0
        for _ in range(settle):
            await RisingEdge(dut.clk)
            dec_cycles += int(handle.value)
    except AttributeError:
        for _ in range(settle):
            await RisingEdge(dut.clk)
    return dec_cycles


async def prep(dut):
    await reg_write(dut, A_IRQ_MASK, 0x0000_0000)
    await reg_write(dut, A_TRACE_CTRL, 1)


@cocotb.test()
async def test_fault_descriptor_releases_oom_exactly_once(dut):
    """O1/O3: fault(done_err) 트랜잭션 후 OOM 사용량이 정확히 원래대로 돌아온다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await prep(dut)

    before = await reg_read(dut, A_OOM_USAGE)
    dec = await run_one(dut, make_desc(0xFF))          # illegal opcode -> done_err
    after = await reg_read(dut, A_OOM_USAGE)
    dut._log.info(f"ILLEGAL: usage {before} -> {after}, alloc_dec high cycles = {dec}")

    assert after == before, (
        f"fault 트랜잭션 후 OOM 사용량이 원래대로 돌아오지 않았다: {before} -> {after} "
        f"(차이 {after - before}, DESC_COST={DESC_COST}). "
        "양수면 누수(감소가 안 일어남 — ST_FAULT 가 ST_DONE 을 거치지 않는다는 뜻), "
        "음수면 과다 반납(감소가 2회 이상)."
    )
    if dec is not None:
        assert dec == 1, (
            f"oom_alloc_dec 는 트랜잭션당 정확히 1사이클이어야 한다. 관측: {dec}사이클. "
            "rtl/oom_guard.sv 의 카운터는 레벨 감지라 high 사이클마다 감소한다."
        )
    assert dut.oom_underflow.value == 0 if hasattr(dut, "oom_underflow") else True


@cocotb.test()
async def test_ok_descriptor_releases_oom_exactly_once(dut):
    """O2: 성공(done_ok) 트랜잭션도 정확히 1회 반납한다 (대조군)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await prep(dut)

    before = await reg_read(dut, A_OOM_USAGE)
    dec = await run_one(dut, make_desc(0x01))          # NOP -> done_ok
    after = await reg_read(dut, A_OOM_USAGE)
    dut._log.info(f"NOP:     usage {before} -> {after}, alloc_dec high cycles = {dec}")

    assert after == before, f"NOP 후 OOM 사용량 불일치: {before} -> {after}"
    if dec is not None:
        assert dec == 1, f"oom_alloc_dec 관측 {dec}사이클 (1이어야 한다)"


@cocotb.test()
async def test_repeated_faults_do_not_leak(dut):
    """O4: fault 를 4번 반복해도 사용량이 누적되지 않는다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await prep(dut)

    base = await reg_read(dut, A_OOM_USAGE)
    for i in range(4):
        # illegal opcode 와 CRC 오류를 번갈아 — fault 종류가 달라도 같아야 한다
        desc = make_desc(0xFF) if i % 2 == 0 else make_desc(0x02, valid_crc=False)
        await run_one(dut, desc)
        now = await reg_read(dut, A_OOM_USAGE)
        dut._log.info(f"  fault #{i}: usage = {now}")
        assert now == base, (
            f"fault {i + 1}번째에서 사용량이 어긋났다: {base} -> {now}. "
            f"매 fault 마다 {DESC_COST} 씩 누수되면 {base + DESC_COST * (i + 1)} 이 된다."
        )

    state = await reg_read(dut, A_OOM_STATE)
    dut._log.info(f"최종 OOM_STATE = {state}")
    assert (state & 0x3) == 0, f"반복 fault 후 OOM 상태가 NORMAL 이 아니다: {state}"
