"""기대값 출처: spec/deltarule.md 2·3.4·4·5절 + docs/DESIGN.md 5.1절 완료 신호 계약.
상태 내용은 sim/golden/deltarule.py 의 초기 상태(I1: S=0)와 tools/orbit_pack.pack_state.

tb_dr1_top_fsm.py — rtl/dr1/dr1_top.sv (PLAN W6, INIT/DUMP 골격만)

  F1. DELTA_INIT → done_ok 정확히 1사이클, done_err 0
  F2. DELTA_DUMP → D행 행 우선 스트림, 내용이 골든 I1(S=0)과 일치
  F3. DELTA_STEP → done_err + fault_code 0x07 (DR1_UNIMPL). **done_ok 없음**
  F4. slot 초과 → done_err + 0x05 (DR1_BAD_SLOT). **done_ok 없음** (BUG-001 계열)
  F5. DUMP dst_addr 16바이트 미정렬 → done_err + 0x06 (DR1_UNALIGNED)
  F6. 알 수 없는 opcode → done_err + 0x01 (ILLEGAL_OPCODE)
  F7. 모든 경로에서 done_pulse == done_ok | done_err, ok/err 중 정확히 하나

레지스터 출력은 falling edge 샘플링 (tb/tb_cdc_fifo_async.py 교훈).
"""

import os
import sys

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.orbit_mmio_map import FaultCode, Opcode  # noqa: E402
from tools.orbit_pack import unpack_state  # noqa: E402

D = 16
W = 16


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


async def issue(dut, opcode, slot=0, dst_addr=0, max_cycles=8 * D):
    """디스크립터 1개를 보내고 완료까지 관찰한다.

    반환: dict(ok=, err=, fault=, pulse=, rows=[(row, data), ...], cycles=)
      ok/err/pulse 는 **높았던 사이클 수**다 (1이어야 한다 — 폭 계약).
    """
    dut.cmd_opcode.value = int(opcode)
    dut.cmd_slot.value = int(slot)
    dut.cmd_dst_addr.value = int(dst_addr)
    dut.cmd_valid.value = 1
    await RisingEdge(dut.clk)
    dut.cmd_valid.value = 0

    n_ok = n_err = n_pulse = 0
    fault = None
    rows = []
    cycles = None
    for _ in range(max_cycles):
        await FallingEdge(dut.clk)
        if int(dut.dump_valid.value):
            rows.append((int(dut.dump_row.value), int(dut.dump_data.value)))
        if int(dut.done_pulse.value):
            n_pulse += 1
        if int(dut.done_ok.value):
            n_ok += 1
            cycles = int(dut.dr1_cycles.value)
        if int(dut.done_err.value):
            n_err += 1
            fault = int(dut.fault_code.value)
            cycles = int(dut.dr1_cycles.value)
        if n_pulse and not int(dut.busy.value):
            break
    return dict(ok=n_ok, err=n_err, fault=fault, pulse=n_pulse, rows=rows, cycles=cycles)


def check_completion_contract(r, tag, expect_ok):
    """DESIGN.md 5.1: ok/err 중 정확히 하나, 각각 정확히 1사이클, pulse = ok|err."""
    assert r["ok"] + r["err"] == 1, (
        f"{tag}: done_ok={r['ok']} done_err={r['err']} — 정확히 하나여야 한다"
    )
    assert r["pulse"] == 1, f"{tag}: done_pulse 가 {r['pulse']} 사이클 (1이어야 한다)"
    if expect_ok:
        assert r["ok"] == 1 and r["err"] == 0, f"{tag}: 성공이어야 한다 {r}"
    else:
        assert r["err"] == 1 and r["ok"] == 0, (
            f"{tag}: 실패여야 한다. done_ok 가 떴다면 BUG-001 재발이다 {r}"
        )


@cocotb.test()
async def test_f1_init_done_ok(dut):
    """F1: DELTA_INIT 이 done_ok 를 정확히 1사이클 낸다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    r = await issue(dut, Opcode.DELTA_INIT, slot=0)
    check_completion_contract(r, "F1 INIT", expect_ok=True)
    assert r["rows"] == [], "F1: INIT 은 덤프하지 않는다"
    dut._log.info(f"F1: INIT 완료, dr1_cycles={r['cycles']}")


@cocotb.test()
async def test_f2_dump_row_major_zeros(dut):
    """F2: DELTA_DUMP 가 D행을 행 우선으로 흘리고, 내용이 골든 I1(S=0)과 일치."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await issue(dut, Opcode.DELTA_INIT, slot=0)
    r = await issue(dut, Opcode.DELTA_DUMP, slot=0, dst_addr=0x1000)
    check_completion_contract(r, "F2 DUMP", expect_ok=True)

    assert len(r["rows"]) == D, f"F2: 행이 {len(r['rows'])}개 (D={D} 이어야 한다)"
    order = [row for row, _ in r["rows"]]
    assert order == list(range(D)), f"F2: 행 우선 순서가 아니다 {order}"

    S = unpack_state([data for _, data in r["rows"]], D, W)
    expect = np.zeros((D, D), dtype=np.int64)
    assert np.array_equal(S, expect), (
        f"F2: INIT 직후 상태가 0 이 아니다 (골든 불변조건 I1)\n{S}"
    )
    dut._log.info(f"F2: {D}행 행 우선 덤프, 전부 0 — I1 확인. dr1_cycles={r['cycles']}")


@cocotb.test()
async def test_f3_step_is_unimplemented(dut):
    """F3: DELTA_STEP 은 W6 에 없다 → done_err + DR1_UNIMPL. 조용히 성공하지 않는다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    r = await issue(dut, Opcode.DELTA_STEP, slot=0)
    check_completion_contract(r, "F3 STEP", expect_ok=False)
    assert r["fault"] == int(FaultCode.DR1_UNIMPL), (
        f"F3: fault_code=0x{r['fault']:02X}, 0x{int(FaultCode.DR1_UNIMPL):02X} 이어야 한다"
    )


@cocotb.test()
async def test_f4_bad_slot_never_reports_done_ok(dut):
    """F4: slot 초과 → done_err + DR1_BAD_SLOT. **done_ok 는 절대 뜨지 않는다** (BUG-001).

    모든 opcode 에 대해 확인한다 — 슬롯 검사가 opcode 보다 먼저다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for opc in (Opcode.DELTA_INIT, Opcode.DELTA_STEP, Opcode.DELTA_DUMP):
        r = await issue(dut, opc, slot=1, dst_addr=0x1000)   # NUM_SLOTS=1 → slot 1 은 불법
        check_completion_contract(r, f"F4 slot=1 opcode=0x{int(opc):02X}", expect_ok=False)
        assert r["fault"] == int(FaultCode.DR1_BAD_SLOT), (
            f"F4: opcode 0x{int(opc):02X} fault=0x{r['fault']:02X}, "
            f"0x{int(FaultCode.DR1_BAD_SLOT):02X} 이어야 한다"
        )
        assert r["rows"] == [], "F4: 실패한 디스크립터가 덤프를 흘리면 안 된다"


@cocotb.test()
async def test_f5_unaligned_dump_addr(dut):
    """F5: DUMP dst_addr 이 16바이트 정렬이 아니면 done_err + DR1_UNALIGNED."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for bad in (0x1001, 0x1008, 0x100F):
        r = await issue(dut, Opcode.DELTA_DUMP, slot=0, dst_addr=bad)
        check_completion_contract(r, f"F5 dst=0x{bad:X}", expect_ok=False)
        assert r["fault"] == int(FaultCode.DR1_UNALIGNED), (
            f"F5: dst=0x{bad:X} fault=0x{r['fault']:02X}"
        )

    # 정렬된 주소는 통과해야 한다 (검사가 과하지 않은지)
    r = await issue(dut, Opcode.DELTA_DUMP, slot=0, dst_addr=0x1010)
    check_completion_contract(r, "F5 정렬 주소", expect_ok=True)


@cocotb.test()
async def test_f6_illegal_opcode(dut):
    """F6: DR1 이 모르는 opcode → done_err + ILLEGAL_OPCODE."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    r = await issue(dut, Opcode.GEMM, slot=0)      # 0x02 는 DR1 것이 아니다
    check_completion_contract(r, "F6 GEMM opcode", expect_ok=False)
    assert r["fault"] == int(FaultCode.ILLEGAL_OPCODE), f"F6: fault=0x{r['fault']:02X}"


@cocotb.test()
async def test_f7_back_to_back_and_status(dut):
    """F7: INIT→DUMP→STEP(실패)→INIT 을 연속으로. 매번 계약을 지키는지."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    seq = [
        (Opcode.DELTA_INIT, 0, 0, True),
        (Opcode.DELTA_DUMP, 0, 0x2000, True),
        (Opcode.DELTA_STEP, 0, 0, False),
        (Opcode.DELTA_INIT, 0, 0, True),
        (Opcode.DELTA_DUMP, 0, 0x2000, True),
    ]
    for i, (opc, slot, dst, ok) in enumerate(seq):
        r = await issue(dut, opc, slot=slot, dst_addr=dst)
        check_completion_contract(r, f"F7 step {i} opcode=0x{int(opc):02X}", expect_ok=ok)

    # 마지막 덤프도 여전히 0 이어야 한다 (STEP 이 실패했으니 상태가 변할 리 없다)
    S = unpack_state([data for _, data in r["rows"]], D, W)
    assert np.array_equal(S, np.zeros((D, D), dtype=np.int64)), (
        f"F7: 실패한 STEP 이 상태를 건드렸다\n{S}"
    )
    assert int(dut.dr1_sat_count.value) == 0
    assert int(dut.dr1_clamp_count.value) == 0
