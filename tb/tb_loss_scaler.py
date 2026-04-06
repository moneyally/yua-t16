"""
tb_loss_scaler.py — cocotb testbench for loss_scaler.sv

Tests:
  1. Initial scale value
  2. Overflow scales down (×0.5)
  3. Success interval scales up (×2.0)
  4. Min clamp
  5. Max clamp
  6. Invalid config error
  7. Reset mid-operation
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import struct


def float_to_fp32(f):
    return struct.unpack('>I', struct.pack('>f', f))[0]

def fp32_to_float(b):
    return struct.unpack('>f', struct.pack('>I', b & 0xFFFFFFFF))[0]


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.step_done.value = 0
    dut.overflow_detect.value = 0
    dut.init_scale_fp32.value = float_to_fp32(32768.0)
    dut.growth_factor_fp32.value = float_to_fp32(2.0)
    dut.backoff_factor_fp32.value = float_to_fp32(0.5)
    dut.growth_interval.value = 4
    dut.min_scale_fp32.value = float_to_fp32(1.0)
    dut.max_scale_fp32.value = float_to_fp32(16777216.0)  # 2^24
    await Timer(100, unit="ns")
    dut.rst_n.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)


async def do_step(dut, overflow=False, max_wait=20):
    dut.overflow_detect.value = 1 if overflow else 0
    dut.step_done.value = 1
    await RisingEdge(dut.clk)
    dut.step_done.value = 0
    for _ in range(max_wait):
        await RisingEdge(dut.clk)
        if dut.done_pulse.value == 1:
            return True
    return False


def get_scale(dut):
    return fp32_to_float(int(dut.current_scale_fp32.value))


@cocotb.test()
async def test_init_scale(dut):
    """Initial scale = 32768.0 after reset."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Wait for init
    for _ in range(5):
        await RisingEdge(dut.clk)

    s = get_scale(dut)
    assert abs(s - 32768.0) < 1.0, f"Expected init 32768.0, got {s}"


@cocotb.test()
async def test_overflow_scales_down(dut):
    """Overflow → scale × 0.5."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    for _ in range(5):
        await RisingEdge(dut.clk)

    s_before = get_scale(dut)
    ok = await do_step(dut, overflow=True)
    assert ok
    s_after = get_scale(dut)

    expected = s_before * 0.5
    assert abs(s_after - expected) < 1.0, f"Expected {expected}, got {s_after}"


@cocotb.test()
async def test_success_interval_scales_up(dut):
    """4 successful steps → scale × 2.0."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    for _ in range(5):
        await RisingEdge(dut.clk)

    s_before = get_scale(dut)

    # 4 successful steps (growth_interval=4)
    for i in range(4):
        ok = await do_step(dut, overflow=False)
        assert ok, f"Step {i} done not seen"

    s_after = get_scale(dut)
    expected = s_before * 2.0
    assert abs(s_after - expected) < 1.0, f"Expected {expected}, got {s_after}"


@cocotb.test()
async def test_min_clamp(dut):
    """Repeated overflow → scale clamps at min_scale."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    dut.rst_n.value = 0
    dut.init_scale_fp32.value = float_to_fp32(4.0)
    dut.min_scale_fp32.value = float_to_fp32(1.0)
    dut.max_scale_fp32.value = float_to_fp32(1000.0)
    dut.growth_factor_fp32.value = float_to_fp32(2.0)
    dut.backoff_factor_fp32.value = float_to_fp32(0.5)
    dut.growth_interval.value = 4
    dut.step_done.value = 0
    dut.overflow_detect.value = 0
    await Timer(100, unit="ns")
    dut.rst_n.value = 1
    for _ in range(5):
        await RisingEdge(dut.clk)

    # 4→2→1→1 (clamped at 1.0)
    for _ in range(5):
        await do_step(dut, overflow=True)

    s = get_scale(dut)
    assert abs(s - 1.0) < 0.01, f"Should clamp at 1.0, got {s}"


@cocotb.test()
async def test_max_clamp(dut):
    """Repeated success → scale clamps at max_scale."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    dut.rst_n.value = 0
    dut.init_scale_fp32.value = float_to_fp32(4.0)
    dut.min_scale_fp32.value = float_to_fp32(1.0)
    dut.max_scale_fp32.value = float_to_fp32(32.0)
    dut.growth_factor_fp32.value = float_to_fp32(2.0)
    dut.backoff_factor_fp32.value = float_to_fp32(0.5)
    dut.growth_interval.value = 1  # scale up every step
    dut.step_done.value = 0
    dut.overflow_detect.value = 0
    await Timer(100, unit="ns")
    dut.rst_n.value = 1
    for _ in range(5):
        await RisingEdge(dut.clk)

    # 4→8→16→32→32 (clamped)
    for _ in range(10):
        await do_step(dut, overflow=False)

    s = get_scale(dut)
    assert abs(s - 32.0) < 0.1, f"Should clamp at 32.0, got {s}"


@cocotb.test()
async def test_invalid_config_error(dut):
    """min > max → err_code=1."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    dut.rst_n.value = 0
    dut.min_scale_fp32.value = float_to_fp32(100.0)
    dut.max_scale_fp32.value = float_to_fp32(10.0)  # invalid
    dut.init_scale_fp32.value = float_to_fp32(50.0)
    dut.growth_factor_fp32.value = float_to_fp32(2.0)
    dut.backoff_factor_fp32.value = float_to_fp32(0.5)
    dut.growth_interval.value = 4
    dut.step_done.value = 0
    dut.overflow_detect.value = 0
    await Timer(100, unit="ns")
    dut.rst_n.value = 1
    for _ in range(5):
        await RisingEdge(dut.clk)

    ok = await do_step(dut, overflow=False)
    assert ok
    assert int(dut.err_code.value) == 1, f"Expected err=1, got {int(dut.err_code.value)}"


@cocotb.test()
async def test_reset_mid_operation(dut):
    """Reset returns to IDLE with init scale."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    for _ in range(5):
        await RisingEdge(dut.clk)

    # Do some overflows to change scale
    await do_step(dut, overflow=True)
    await do_step(dut, overflow=True)

    # Reset
    dut.rst_n.value = 0
    for _ in range(2):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(5):
        await RisingEdge(dut.clk)

    assert dut.busy.value == 0
    # Scale should be back to default (32768.0 from reset)
    s = get_scale(dut)
    assert abs(s - 32768.0) < 1.0, f"After reset, scale should be 32768, got {s}"
