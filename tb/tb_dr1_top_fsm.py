"""기대값 출처: spec/deltarule.md 2·3.4·4·5절 + docs/DESIGN.md 5.1절 완료 신호 계약.
상태 내용은 sim/golden/deltarule.py 의 초기 상태(I1: S=0)와 tools/orbit_pack.

tb_dr1_top_fsm.py — rtl/dr1/dr1_tb_wrap.sv (dr1_top + dr1_scratch), PLAN W7

FSM·fault 계약만 본다. 계산이 골든과 맞는지는 tb_dr1_harness_rtl.py 가 본다.

  F1. DELTA_INIT → done_ok 정확히 1사이클, done_err 0
  F2. DELTA_DUMP → D행 행 우선 스트림, 내용이 골든 I1(S=0)과 일치
  F3. DELTA_STEP 이 **이제 동작한다** (W6 의 0x07 DR1_UNIMPL 은 더 이상 안 난다)
  F4. slot 초과 → done_err + 0x05 (DR1_BAD_SLOT). **done_ok 없음** (BUG-001 계열)
  F5. 주소 16바이트 미정렬 → 0x06 (DR1_UNALIGNED)
  F6. 알 수 없는 opcode → 0x01 (ILLEGAL_OPCODE)
  F7. 주소가 스크래치 밖 → 0x08 (DR1_ADDR_RANGE)
  F8. 연속 실행에서 매번 계약을 지킨다
  F9. α/β > 0x8000 → 클램프 + DR1_CLAMP_COUNT 증가. **fault 가 아니다**

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
SCRATCH = 1024

Q_ELEM, K_ELEM, V_ELEM, O_ELEM = 0, D, 2 * D, 3 * D
DUMP_ELEM = 512


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.cmd_valid.value = 0
    dut.cmd_opcode.value = 0
    dut.cmd_slot.value = 0
    dut.cmd_q_addr.value = Q_ELEM * 2
    dut.cmd_k_addr.value = K_ELEM * 2
    dut.cmd_v_addr.value = V_ELEM * 2
    dut.cmd_dst_addr.value = O_ELEM * 2
    dut.cmd_alpha.value = 0x8000
    dut.cmd_beta.value = 0
    dut.h_en.value = 0
    dut.h_we.value = 0
    dut.h_addr.value = 0
    dut.h_wdata.value = 0
    dut.sat_count_clr.value = 0
    dut.clamp_count_clr.value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def issue(dut, opcode, *, slot=0, q_addr=Q_ELEM * 2, k_addr=K_ELEM * 2,
                v_addr=V_ELEM * 2, dst_addr=O_ELEM * 2, alpha=0x8000, beta=0,
                max_cycles=4096):
    """디스크립터 1개를 보내고 완료까지 관찰한다.

    반환: dict(ok=, err=, fault=, pulse=, rows=, cycles=, clamp=)
      ok/err/pulse 는 **높았던 사이클 수**다 (1이어야 한다 — 폭 계약).
    """
    # **falling edge 에 정렬한 뒤** 입력을 세운다. 아무 때나 valid 를 올리고
    # FallingEdge 를 기다리면, 그 사이의 rising edge 가 이미 명령을 받아버린다
    # → 같은 디스크립터가 두 번 실행된다 (실제로 겪었다: F9 클램프가 4 나왔다).
    await FallingEdge(dut.clk)
    dut.cmd_opcode.value = int(opcode)
    dut.cmd_slot.value = int(slot)
    dut.cmd_q_addr.value = int(q_addr)
    dut.cmd_k_addr.value = int(k_addr)
    dut.cmd_v_addr.value = int(v_addr)
    dut.cmd_dst_addr.value = int(dst_addr)
    dut.cmd_alpha.value = int(alpha) & 0xFFFF
    dut.cmd_beta.value = int(beta) & 0xFFFF
    dut.cmd_valid.value = 1

    for _ in range(max_cycles):
        if int(dut.cmd_ready.value):
            break
        await FallingEdge(dut.clk)
    await RisingEdge(dut.clk)      # 이 edge 에서 정확히 한 번 수락된다
    dut.cmd_valid.value = 0

    n_ok = n_err = n_pulse = 0
    fault = None
    rows = {}
    cycles = None
    for _ in range(max_cycles):
        await FallingEdge(dut.clk)
        if int(dut.dump_valid.value):
            rows[int(dut.dump_row.value)] = int(dut.dump_data.value)
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
    return dict(ok=n_ok, err=n_err, fault=fault, pulse=n_pulse, rows=rows,
                cycles=cycles, clamp=int(dut.dr1_clamp_count.value))


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

    r = await issue(dut, Opcode.DELTA_INIT)
    check_completion_contract(r, "F1 INIT", expect_ok=True)
    assert r["rows"] == {}, "F1: INIT 은 덤프하지 않는다"
    dut._log.info(f"F1: INIT 완료, dr1_cycles={r['cycles']}")


@cocotb.test()
async def test_f2_dump_row_major_zeros(dut):
    """F2: DELTA_DUMP 가 D행을 행 우선으로 흘리고, 내용이 골든 I1(S=0)과 일치."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await issue(dut, Opcode.DELTA_INIT)
    r = await issue(dut, Opcode.DELTA_DUMP, dst_addr=DUMP_ELEM * 2)
    check_completion_contract(r, "F2 DUMP", expect_ok=True)

    assert len(r["rows"]) == D, f"F2: 행이 {len(r['rows'])}개 (D={D} 이어야 한다)"
    assert sorted(r["rows"].keys()) == list(range(D)), f"F2: 행 번호가 빠졌다 {sorted(r['rows'])}"

    S = unpack_state([r["rows"][i] for i in range(D)], D, W)
    assert np.array_equal(S, np.zeros((D, D), dtype=np.int64)), (
        f"F2: INIT 직후 상태가 0 이 아니다 (골든 불변조건 I1)\n{S}"
    )
    dut._log.info(f"F2: {D}행 행 우선 덤프, 전부 0 — I1 확인. dr1_cycles={r['cycles']}")


@cocotb.test()
async def test_f3_step_now_works(dut):
    """F3: DELTA_STEP 이 W7 에서 동작한다. W6 의 0x07 DR1_UNIMPL 은 더 이상 안 난다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await issue(dut, Opcode.DELTA_INIT)
    r = await issue(dut, Opcode.DELTA_STEP, alpha=0x8000, beta=0x4000)
    check_completion_contract(r, "F3 STEP", expect_ok=True)
    assert r["fault"] is None, f"F3: STEP 이 fault 를 냈다 0x{r['fault']:02X}"
    dut._log.info(f"F3: STEP 완료, dr1_cycles={r['cycles']}")


@cocotb.test()
async def test_f4_bad_slot_never_reports_done_ok(dut):
    """F4: slot 초과 → done_err + DR1_BAD_SLOT. **done_ok 는 절대 뜨지 않는다** (BUG-001)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    for opc in (Opcode.DELTA_INIT, Opcode.DELTA_STEP, Opcode.DELTA_DUMP):
        r = await issue(dut, opc, slot=1, dst_addr=DUMP_ELEM * 2)
        check_completion_contract(r, f"F4 slot=1 opcode=0x{int(opc):02X}", expect_ok=False)
        assert r["fault"] == int(FaultCode.DR1_BAD_SLOT), (
            f"F4: opcode 0x{int(opc):02X} fault=0x{r['fault']:02X}"
        )
        assert r["rows"] == {}, "F4: 실패한 디스크립터가 덤프를 흘리면 안 된다"


@cocotb.test()
async def test_f5_unaligned_addr(dut):
    """F5: 주소가 16바이트 정렬이 아니면 done_err + DR1_UNALIGNED."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # DUMP: dst 미정렬
    for bad in (0x1002, 0x1008, 0x100E):
        r = await issue(dut, Opcode.DELTA_DUMP, dst_addr=bad)
        check_completion_contract(r, f"F5 dump dst=0x{bad:X}", expect_ok=False)
        assert r["fault"] == int(FaultCode.DR1_UNALIGNED), f"F5: fault=0x{r['fault']:02X}"

    # STEP: q/k/v/o 중 하나만 틀려도 잡아야 한다
    for kw in ("q_addr", "k_addr", "v_addr", "dst_addr"):
        r = await issue(dut, Opcode.DELTA_STEP, **{kw: 0x22})
        check_completion_contract(r, f"F5 step {kw}", expect_ok=False)
        assert r["fault"] == int(FaultCode.DR1_UNALIGNED), (
            f"F5: {kw} 미정렬인데 fault=0x{r['fault']:02X}"
        )

    # 정렬된 주소는 통과해야 한다 (검사가 과하지 않은지)
    r = await issue(dut, Opcode.DELTA_DUMP, dst_addr=DUMP_ELEM * 2)
    check_completion_contract(r, "F5 정렬 주소", expect_ok=True)


@cocotb.test()
async def test_f6_illegal_opcode(dut):
    """F6: DR1 이 모르는 opcode → done_err + ILLEGAL_OPCODE."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    r = await issue(dut, Opcode.GEMM)      # 0x02 는 DR1 것이 아니다
    check_completion_contract(r, "F6 GEMM opcode", expect_ok=False)
    assert r["fault"] == int(FaultCode.ILLEGAL_OPCODE), f"F6: fault=0x{r['fault']:02X}"


@cocotb.test()
async def test_f7_addr_out_of_range(dut):
    """F7: 스크래치 밖 주소 → DR1_ADDR_RANGE (0x08).

    정렬은 맞지만 범위를 넘는 주소다. 이걸 안 잡으면 다른 벡터를 덮어쓴다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    # **정렬은 맞고 범위만 넘는** 주소를 골라야 한다. 안 그러면 0x06 이 먼저 뜬다
    # (정렬 검사가 범위 검사보다 앞이다 — dr1_top 디코드 순서).
    far = (SCRATCH - 8) * 2          # 원소 1016, 바이트 2032 = 16의 배수 ✔
    assert far % 16 == 0, "테스트가 고른 주소가 정렬돼 있지 않다"
    r = await issue(dut, Opcode.DELTA_STEP, q_addr=far)
    check_completion_contract(r, "F7 step q 범위", expect_ok=False)
    assert r["fault"] == int(FaultCode.DR1_ADDR_RANGE), f"F7: fault=0x{r['fault']:02X}"

    # DUMP 는 d*d 개가 들어가야 하므로 한계가 더 낮다
    dump_far = (SCRATCH - D * D + 8) * 2
    assert dump_far % 16 == 0
    r = await issue(dut, Opcode.DELTA_DUMP, dst_addr=dump_far)
    check_completion_contract(r, "F7 dump 범위", expect_ok=False)
    assert r["fault"] == int(FaultCode.DR1_ADDR_RANGE), f"F7: fault=0x{r['fault']:02X}"

    # 경계 바로 안쪽은 통과해야 한다 (검사가 한 칸 과하지 않은지)
    ok_addr = (SCRATCH - D) * 2
    assert ok_addr % 16 == 0
    r = await issue(dut, Opcode.DELTA_STEP, q_addr=ok_addr)
    check_completion_contract(r, "F7 경계 안쪽", expect_ok=True)


@cocotb.test()
async def test_f8_back_to_back(dut):
    """F8: INIT→STEP→DUMP→(실패)→INIT 을 연속으로. 매번 계약을 지키는지."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    seq = [
        (Opcode.DELTA_INIT, dict(), True),
        (Opcode.DELTA_STEP, dict(beta=0x2000), True),
        (Opcode.DELTA_DUMP, dict(dst_addr=DUMP_ELEM * 2), True),
        (Opcode.DELTA_STEP, dict(slot=3), False),
        (Opcode.DELTA_INIT, dict(), True),
        (Opcode.DELTA_STEP, dict(beta=0x8000), True),
    ]
    for i, (opc, kw, ok) in enumerate(seq):
        r = await issue(dut, opc, **kw)
        check_completion_contract(r, f"F8 step {i} opcode=0x{int(opc):02X}", expect_ok=ok)


@cocotb.test()
async def test_f9_alpha_beta_clamp(dut):
    """F9: α/β > 0x8000 은 **fault 가 아니다**. 클램프하고 센다 (spec 1절).

    골든은 같은 입력에 ValueError 를 낸다 — 그래서 하네스의 GoldenDut 도
    클램프한 뒤 골든을 부른다. 하드웨어와 같은 편에 선다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)

    await issue(dut, Opcode.DELTA_INIT)
    before = int(dut.dr1_clamp_count.value)

    r = await issue(dut, Opcode.DELTA_STEP, alpha=0xFFFF, beta=0x9000)
    check_completion_contract(r, "F9 클램프", expect_ok=True)
    assert r["clamp"] == before + 2, (
        f"F9: α,β 둘 다 클램프됐으니 카운트가 2 늘어야 한다 (before={before} after={r['clamp']})"
    )

    # 정상 범위는 클램프하지 않는다
    r2 = await issue(dut, Opcode.DELTA_STEP, alpha=0x8000, beta=0x0001)
    assert r2["clamp"] == r["clamp"], f"F9: 정상 입력인데 클램프가 늘었다 {r2['clamp']}"
