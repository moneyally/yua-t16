"""기대값 출처: sim/golden/deltarule.py — 하네스(tb/tb_dr1_top.py)가 골든과 비교한다.
불변조건 I1(초기 상태 S=0)은 docs/DESIGN.md 7절.

tb_dr1_harness_rtl.py — 하네스를 **실제 RTL(rtl/dr1/dr1_top.sv)** 에 붙인다 (PLAN W6)

지금까지 하네스는 골든-대-골든으로만 돌았다. 여기서 처음으로 DUT 자리에
`CocotbDut` (실 RTL) 이 들어간다.

  R1. I1 — DELTA_INIT 직후 DELTA_DUMP 가 골든의 초기 상태(전부 0)와 비트 일치.
           토큰 0개짜리 하네스 실행이고, 불일치 0 이어야 한다.
  R2. 오류 주입 — 덤프 1워드를 뒤집어서 하네스가 **위치를 찍어** 잡는지.
           이게 없으면 R1 의 "불일치 0" 은 아무 의미가 없다 (하네스가 죽어 있어도 통과).
  R3. slot 초과 → CocotbDut 이 Dr1Fault(0x05) 를 던진다 (done_pulse 가 아니라 done_err 를 본다)
  R4. step 은 W7 — NotImplementedError. 조용히 0 을 돌려주지 않는다.
"""

import os
import sys

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from tb_dr1_top import CocotbDut, Dr1Fault, run_harness_async  # noqa: E402
from tools.orbit_mmio_map import FaultCode  # noqa: E402

D = 16


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.cmd_valid.value = 0
    dut.cmd_opcode.value = 0
    dut.cmd_slot.value = 0
    dut.cmd_dst_addr.value = 0
    dut.sat_count_clr.value = 0
    dut.clamp_count_clr.value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_r1_invariant_i1_on_real_rtl(dut):
    """R1: 실 RTL 의 INIT→DUMP 가 골든 초기 상태와 비트 일치 (불변조건 I1)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    res = await run_harness_async(CocotbDut(dut, D), tokens=[], seed=0)
    dut._log.info(res.report())
    assert res.ok, f"R1: 실 RTL 이 골든과 다르다\n{res.report()}"


@cocotb.test()
async def test_r2_fault_injection_on_rtl_path(dut):
    """R2: 덤프 1워드를 뒤집으면 하네스가 위치를 찍어 잡는가.

    하네스가 실제로 비교를 하는지 확인하는 유일한 방법이다. R1 만 있으면
    "아무것도 안 하고 통과" 와 구분되지 않는다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    class FlipDut(CocotbDut):
        """RTL 에서 받은 덤프의 한 칸만 1 LSB 틀리게 만든다. RTL 은 그대로다."""

        def __init__(self, dut, d, cell):
            super().__init__(dut, d)
            self.cell = cell

        async def dump(self, slot: int = 0):
            S = np.asarray(await super().dump(slot), dtype=np.int64).copy()
            r, c = self.cell
            S[r][c] += 1
            return S

    res = await run_harness_async(FlipDut(dut, D, (3, 11)), tokens=[], seed=0)
    dut._log.info(res.report(max_lines=3))
    assert not res.ok, "R2: 오류를 주입했는데 하네스가 통과시켰다 — 하네스가 죽어 있다"
    m = res.first
    assert (m.row, m.col) == (3, 11), f"R2: 위치를 잘못 찍었다 {m}"
    assert m.expected == 0 and m.actual == 1, f"R2: 값을 잘못 찍었다 {m}"


@cocotb.test()
async def test_r3_bad_slot_raises_fault(dut):
    """R3: slot 초과 → Dr1Fault(0x05). done_pulse 를 완료로 읽지 않는다는 뜻이다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    cd = CocotbDut(dut, D)
    try:
        await cd.init(slot=1)
    except Dr1Fault as e:
        assert int(e.fault_code) == int(FaultCode.DR1_BAD_SLOT), f"R3: fault={e}"
    else:
        raise AssertionError("R3: slot=1 인데 성공했다 — done_ok 가 떴다는 뜻이다")


@cocotb.test()
async def test_r4_step_is_w7(dut):
    """R4: step 은 아직 없다. NotImplementedError 로 확실히 터진다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    cd = CocotbDut(dut, D)
    await cd.init(0)
    try:
        await cd.step(0, None, None, None, 0, 0)
    except NotImplementedError as e:
        assert "W7" in str(e)
    else:
        raise AssertionError("R4: step 이 조용히 성공했다")
