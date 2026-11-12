"""
tb_oom_guard_thresholds.py — cocotb testbench for oom_guard.sv

Tests:
  1. Threshold crossing: NORMAL -> PRESSURE -> CRITICAL -> EMERG
  2. Release and recovery: EMERG -> CRITICAL -> PRESSURE -> NORMAL
  3. admission_stop asserted only in EMERG
  4. prefetch_clamp asserted in CRITICAL+
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


# Pressure states (match RTL enum)
ST_NORMAL   = 0
ST_PRESSURE = 1
ST_CRITICAL = 2
ST_EMERG    = 3


async def settle(dut):
    """클럭 엣지 직후 등록 출력이 안정되도록 한 델타 넘긴다.

    cocotb 에서 `await RisingEdge(clk)` 직후에 바로 읽으면 Verilator 에서는
    그 엣지로 갱신되기 "전" 값이 보인다. 이 엣지에서 막 바뀐 레지스터 출력을
    검사하려면 한 델타를 넘겨야 한다.

    ReadOnly 가 정석이지만, 이 테스트들은 검사 직후 다시 신호를 쓰기 때문에
    ReadOnly 구간에 들어가면 쓰기가 막힌다. 짧은 Timer 로 넘긴다.
    """
    await Timer(1, unit="ns")


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.alloc_inc.value = 0
    dut.alloc_dec.value = 0
    dut.alloc_bytes.value = 0
    dut.dma_reserve_inc.value = 0
    dut.dma_reserve_dec.value = 0
    dut.dma_rsv_bytes.value = 0
    dut.prefetch_reserve_inc.value = 0
    dut.prefetch_reserve_dec.value = 0
    dut.prefetch_rsv_bytes.value = 0
    dut.thresh_pressure.value = 1000
    dut.thresh_critical.value = 2000
    dut.thresh_emergency.value = 3000
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


async def alloc(dut, amount):
    """Allocate bytes."""
    dut.alloc_inc.value = 1
    dut.alloc_bytes.value = amount
    await RisingEdge(dut.clk)
    dut.alloc_inc.value = 0


async def dealloc(dut, amount):
    """Deallocate bytes."""
    dut.alloc_dec.value = 1
    dut.alloc_bytes.value = amount
    await RisingEdge(dut.clk)
    dut.alloc_dec.value = 0


@cocotb.test()
async def test_threshold_escalation(dut):
    """NORMAL -> PRESSURE -> CRITICAL -> EMERG as usage grows."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Should start NORMAL
    assert int(dut.pressure_state.value) == ST_NORMAL

    # Allocate to cross pressure threshold (1000)
    await alloc(dut, 1100)
    await RisingEdge(dut.clk)
    await settle(dut)
    assert int(dut.pressure_state.value) == ST_PRESSURE, \
        f"Expected PRESSURE, got {int(dut.pressure_state.value)}"
    assert dut.admission_stop.value == 0
    assert dut.prefetch_clamp.value == 0

    # Cross critical threshold (2000)
    await alloc(dut, 1000)
    await RisingEdge(dut.clk)
    await settle(dut)
    assert int(dut.pressure_state.value) == ST_CRITICAL
    assert dut.prefetch_clamp.value == 1
    assert dut.admission_stop.value == 0

    # Cross emergency threshold (3000)
    await alloc(dut, 1000)
    await RisingEdge(dut.clk)
    await settle(dut)
    assert int(dut.pressure_state.value) == ST_EMERG
    assert dut.admission_stop.value == 1
    assert dut.prefetch_clamp.value == 1


@cocotb.test()
async def test_recovery_with_hysteresis(dut):
    """EMERG recovery: must drop below critical to leave EMERG."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Go to EMERG
    await alloc(dut, 3500)
    await RisingEdge(dut.clk)
    await settle(dut)
    assert int(dut.pressure_state.value) == ST_EMERG

    # Drop below emergency but still above critical — should go to CRITICAL
    await dealloc(dut, 1000)  # now at 2500
    await RisingEdge(dut.clk)
    await settle(dut)
    assert int(dut.pressure_state.value) == ST_CRITICAL

    # Drop below critical but above pressure — should go to PRESSURE
    await dealloc(dut, 1000)  # now at 1500
    await RisingEdge(dut.clk)
    await settle(dut)
    assert int(dut.pressure_state.value) == ST_PRESSURE

    # Drop below pressure — back to NORMAL
    await dealloc(dut, 1000)  # now at 500
    await RisingEdge(dut.clk)
    await settle(dut)
    assert int(dut.pressure_state.value) == ST_NORMAL
    assert dut.admission_stop.value == 0
    assert dut.prefetch_clamp.value == 0
