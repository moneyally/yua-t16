"""
tb_mxu_bf16_16x16.py — cocotb testbench for BF16 16×16 MXU

Tests:
  1. Single MAC step: A[1×1] × B[1×1] in BF16, verify FP32 accumulator
  2. Full 16×16 identity: A=I, B=values, result=values
  3. Multi-step accumulate: K=4 steps, verify sum
  4. Zero handling: zero × anything = 0
  5. Clear: acc_clr resets accumulators
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import struct
import random


def float_to_bf16(f):
    """Convert Python float to BF16 (16-bit integer)."""
    fp32_bits = struct.unpack('>I', struct.pack('>f', f))[0]
    return (fp32_bits >> 16) & 0xFFFF


def bf16_to_float(bf16):
    """Convert BF16 (16-bit integer) to Python float."""
    fp32_bits = bf16 << 16
    return struct.unpack('>f', struct.pack('>I', fp32_bits))[0]


def fp32_bits_to_float(bits):
    """Convert FP32 bit pattern (32-bit integer) to Python float."""
    return struct.unpack('>f', struct.pack('>I', bits & 0xFFFFFFFF))[0]


# ── packed 포트 ───────────────────────────────────────────────────
# a_row / b_col / acc_out 은 평탄화된 packed 벡터다 (Yosys 가 unpacked
# 배열 포트를 파싱하지 못해서 — rtl/mxu_bf16_16x16.sv 주석 참고).
# 낮은 인덱스가 하위 비트이고, acc_out 은 행 우선이다.
COLS = 16


def lanes(vals, width=16):
    """워드 리스트 → packed 벡터 정수. 적게 주면 나머지는 0 이다."""
    v = 0
    for i, x in enumerate(vals):
        v |= (int(x) & ((1 << width) - 1)) << (width * i)
    return v


def acc_at(dut, r, c):
    """acc_out 의 원소 (r,c) 를 FP32 비트로 돌려준다."""
    return (int(dut.acc_out.value) >> (32 * (r * COLS + c))) & 0xFFFFFFFF


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.en.value = 0
    dut.acc_clr.value = 0
    dut.a_row.value = 0
    dut.b_col.value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_single_multiply(dut):
    """1.0 × 2.0 = 2.0 in position [0][0]."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Clear accumulators
    dut.acc_clr.value = 1
    await RisingEdge(dut.clk)
    dut.acc_clr.value = 0

    # a_row 레인 0 = 1.0, b_col 레인 0 = 2.0, 나머지는 0
    dut.a_row.value = lanes([float_to_bf16(1.0)])
    dut.b_col.value = lanes([float_to_bf16(2.0)])
    dut.en.value = 1
    await RisingEdge(dut.clk)
    dut.en.value = 0
    await RisingEdge(dut.clk)

    # acc_out 원소 (0,0)
    result_bits = acc_at(dut, 0, 0)
    result = fp32_bits_to_float(result_bits)

    assert abs(result - 2.0) < 0.01, f"Expected ~2.0, got {result} (bits={result_bits:#010x})"


@cocotb.test()
async def test_accumulate_4_steps(dut):
    """4 steps of 1.0 × 1.0 = 4.0."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut.acc_clr.value = 1
    await RisingEdge(dut.clk)
    dut.acc_clr.value = 0

    # 4 MAC steps
    for _ in range(4):
        dut.a_row.value = lanes([float_to_bf16(1.0)])
        dut.b_col.value = lanes([float_to_bf16(1.0)])
        dut.en.value = 1
        await RisingEdge(dut.clk)
    dut.en.value = 0
    await RisingEdge(dut.clk)

    result = fp32_bits_to_float(acc_at(dut, 0, 0))
    assert abs(result - 4.0) < 0.1, f"Expected ~4.0, got {result}"


@cocotb.test()
async def test_zero_multiply(dut):
    """0.0 × anything = 0.0."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut.acc_clr.value = 1
    await RisingEdge(dut.clk)
    dut.acc_clr.value = 0

    dut.a_row.value = lanes([float_to_bf16(0.0)])
    dut.b_col.value = lanes([float_to_bf16(42.0)])
    dut.en.value = 1
    await RisingEdge(dut.clk)
    dut.en.value = 0
    await RisingEdge(dut.clk)

    result = fp32_bits_to_float(acc_at(dut, 0, 0))
    assert abs(result) < 0.001, f"Expected 0.0, got {result}"


@cocotb.test()
async def test_clear(dut):
    """acc_clr resets accumulator to 0."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Accumulate something
    dut.acc_clr.value = 1
    await RisingEdge(dut.clk)
    dut.acc_clr.value = 0

    dut.a_row.value = lanes([float_to_bf16(5.0)])
    dut.b_col.value = lanes([float_to_bf16(3.0)])
    dut.en.value = 1
    await RisingEdge(dut.clk)
    dut.en.value = 0
    await RisingEdge(dut.clk)

    # Verify non-zero
    result1 = fp32_bits_to_float(acc_at(dut, 0, 0))
    assert result1 != 0.0, "Should be non-zero before clear"

    # Clear
    dut.acc_clr.value = 1
    await RisingEdge(dut.clk)
    dut.acc_clr.value = 0
    await RisingEdge(dut.clk)

    result2 = fp32_bits_to_float(acc_at(dut, 0, 0))
    assert result2 == 0.0, f"Should be 0 after clear, got {result2}"


@cocotb.test()
async def test_negative_multiply(dut):
    """-2.0 × 3.0 = -6.0."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut.acc_clr.value = 1
    await RisingEdge(dut.clk)
    dut.acc_clr.value = 0

    dut.a_row.value = lanes([float_to_bf16(-2.0)])
    dut.b_col.value = lanes([float_to_bf16(3.0)])
    dut.en.value = 1
    await RisingEdge(dut.clk)
    dut.en.value = 0
    await RisingEdge(dut.clk)

    result = fp32_bits_to_float(acc_at(dut, 0, 0))
    assert abs(result - (-6.0)) < 0.1, f"Expected ~-6.0, got {result}"
