"""
test_gemm_int4.py — cocotb tests for gemm_int4 module
YUA-T16 v2 INT4 GEMM verification
"""
from __future__ import annotations

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TILE = 16


def float_to_fp16_bits(f: float) -> int:
    """Convert Python float to FP16 bit pattern (uint16)."""
    arr = np.array([f], dtype=np.float16)
    return int(arr.view(np.uint16)[0])


def pack_b_flat(b_int4: np.ndarray) -> int:
    """
    Pack 16x16 int4 weight matrix into b_flat integer.
    b_int4: [16,16] int8 (range -8..7)
    Element (i,j) stored at bit offset (i*TILE+j)*4, 4-bit two's complement.
    Returns an integer with TILE*TILE*4 bits = 1024 bits.
    """
    total_bits = TILE * TILE * 4
    result = 0
    for i in range(TILE):
        for j in range(TILE):
            val = int(b_int4[i, j]) & 0xF  # keep lower 4 bits (two's complement)
            bit_offset = (i * TILE + j) * 4
            result |= (val << bit_offset)
    return result


def pack_a_flat(a_int8: np.ndarray) -> int:
    """
    Pack 16x16 int8 activation matrix into a_flat integer.
    Element (i,j) at bit offset (i*TILE+j)*8.
    """
    result = 0
    for i in range(TILE):
        for j in range(TILE):
            val = int(a_int8[i, j]) & 0xFF
            bit_offset = (i * TILE + j) * 8
            result |= (val << bit_offset)
    return result


def pack_scale_flat(scales: np.ndarray) -> int:
    """
    Pack 16 FP16 scales into scale_flat integer.
    scale[j] at bits [j*16+15 : j*16].
    scales: [16] float16 or float32 array
    """
    result = 0
    for j in range(TILE):
        fp16_bits = float_to_fp16_bits(float(scales[j]))
        result |= (fp16_bits << (j * 16))
    return result


def unpack_c_flat(c_flat_val: int) -> np.ndarray:
    """
    Unpack c_flat integer into [16,16] int32 array.
    Element (i,j) at bit offset (i*TILE+j)*32.
    """
    c = np.zeros((TILE, TILE), dtype=np.int32)
    for i in range(TILE):
        for j in range(TILE):
            bit_offset = (i * TILE + j) * 32
            word = (c_flat_val >> bit_offset) & 0xFFFFFFFF
            # Sign-extend 32-bit
            if word & 0x80000000:
                word -= 0x100000000
            c[i, j] = word
    return c


def gemm_int4_ref(a_int8: np.ndarray, b_int4: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """
    NumPy reference implementation.
    a_int8: [16,16] int8
    b_int4: [16,16] int8 (range -8..7, already unpacked)
    scales: [16] float16 — one per column j
    Returns: [16,16] int32
    """
    # Dequantize: b_fp[k,j] = b_int4[k,j] * scales[j]
    b_fp = b_int4.astype(np.float32) * scales.astype(np.float32)  # broadcast over rows
    # GEMM: c[i,j] = sum_k( a[i,k] * b_fp[k,j] )
    c = a_int8.astype(np.float32) @ b_fp
    return np.round(c).astype(np.int32)


async def reset_dut(dut):
    """Apply reset."""
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.a_flat.value = 0
    dut.b_flat.value = 0
    dut.scale_flat.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


async def run_gemm(dut, a_int8: np.ndarray, b_int4: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Drive inputs, pulse start, wait for done, return c matrix."""
    # Pack inputs
    a_val = pack_a_flat(a_int8)
    b_val = pack_b_flat(b_int4)
    s_val = pack_scale_flat(scales)

    dut.a_flat.value = a_val
    dut.b_flat.value = b_val
    dut.scale_flat.value = s_val

    await RisingEdge(dut.clk)

    # Pulse start for one cycle
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for done
    timeout = 2000
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        if dut.done.value == 1:
            break
    else:
        raise RuntimeError("Timeout waiting for done signal")

    # Read c_flat
    c_flat_val = int(dut.c_flat.value)
    return unpack_c_flat(c_flat_val)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_basic_int4(dut):
    """
    16x16 GEMM with INT4 weights and INT8 activations.
    Compare against numpy reference. Tolerance: max absolute error < 2.
    """
    cocotb.log.info("test_basic_int4: starting")

    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Random INT8 activations
    rng = np.random.default_rng(42)
    a_int8 = rng.integers(-128, 128, size=(TILE, TILE), dtype=np.int8)

    # Random INT4 weights (range -8 to 7)
    b_int4 = rng.integers(-8, 8, size=(TILE, TILE), dtype=np.int8)

    # Random positive FP16 scales (small values to keep INT32 in range)
    scales_f32 = rng.uniform(0.5, 2.0, size=(TILE,)).astype(np.float32)
    scales_fp16 = scales_f32.astype(np.float16)

    # Run DUT
    c_dut = await run_gemm(dut, a_int8, b_int4, scales_fp16)

    # Reference
    c_ref = gemm_int4_ref(a_int8, b_int4, scales_fp16)

    cocotb.log.info(f"c_ref sample [0,0..3]: {c_ref[0, :4]}")
    cocotb.log.info(f"c_dut sample [0,0..3]: {c_dut[0, :4]}")

    max_err = int(np.max(np.abs(c_dut.astype(np.int64) - c_ref.astype(np.int64))))
    cocotb.log.info(f"Max absolute error: {max_err}")

    assert max_err < 2, f"test_basic_int4 FAILED: max error {max_err} >= 2\nRef:\n{c_ref}\nDUT:\n{c_dut}"
    cocotb.log.info("test_basic_int4: PASSED")


@cocotb.test()
async def test_scale_effect(dut):
    """
    All weights = +1 (INT4 value 1).
    Output should equal sum of each activation row * scale.
    c[i,j] = sum_k( a[i,k] * 1 * scale[j] ) = scale[j] * sum_k(a[i,k])
    """
    cocotb.log.info("test_scale_effect: starting")

    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # Activations: small positive values to keep sum manageable
    rng = np.random.default_rng(123)
    a_int8 = rng.integers(1, 10, size=(TILE, TILE), dtype=np.int8)

    # All weights = 1
    b_int4 = np.ones((TILE, TILE), dtype=np.int8)

    # Scales: distinct values per column
    scales_f32 = np.arange(1, TILE + 1, dtype=np.float32) * 0.5  # 0.5, 1.0, ..., 8.0
    scales_fp16 = scales_f32.astype(np.float16)

    # Run DUT
    c_dut = await run_gemm(dut, a_int8, b_int4, scales_fp16)

    # Reference
    c_ref = gemm_int4_ref(a_int8, b_int4, scales_fp16)

    cocotb.log.info(f"c_ref[0,:4]: {c_ref[0,:4]}")
    cocotb.log.info(f"c_dut[0,:4]: {c_dut[0,:4]}")

    max_err = int(np.max(np.abs(c_dut.astype(np.int64) - c_ref.astype(np.int64))))
    cocotb.log.info(f"Max absolute error: {max_err}")

    assert max_err < 2, f"test_scale_effect FAILED: max error {max_err} >= 2\nRef:\n{c_ref}\nDUT:\n{c_dut}"
    cocotb.log.info("test_scale_effect: PASSED")


@cocotb.test()
async def test_negative_weights(dut):
    """
    Negative INT4 weights (-1 to -8).
    Verify correct sign handling and accumulation.
    """
    cocotb.log.info("test_negative_weights: starting")

    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    rng = np.random.default_rng(999)

    # Positive activations
    a_int8 = rng.integers(1, 20, size=(TILE, TILE), dtype=np.int8)

    # Negative INT4 weights: range -8 to -1
    b_int4 = rng.integers(-8, 0, size=(TILE, TILE), dtype=np.int8)

    # Positive scales
    scales_f32 = np.ones(TILE, dtype=np.float32)
    scales_fp16 = scales_f32.astype(np.float16)

    # Run DUT
    c_dut = await run_gemm(dut, a_int8, b_int4, scales_fp16)

    # Reference
    c_ref = gemm_int4_ref(a_int8, b_int4, scales_fp16)

    cocotb.log.info(f"c_ref[0,:4]: {c_ref[0,:4]}")
    cocotb.log.info(f"c_dut[0,:4]: {c_dut[0,:4]}")

    # Verify all outputs are negative (positive act * negative weight * positive scale)
    assert np.all(c_ref < 0), "Reference should be all negative for these inputs"
    assert np.all(c_dut < 0), f"DUT output should be all negative, got positives: {c_dut[c_dut >= 0]}"

    max_err = int(np.max(np.abs(c_dut.astype(np.int64) - c_ref.astype(np.int64))))
    cocotb.log.info(f"Max absolute error: {max_err}")

    assert max_err < 2, f"test_negative_weights FAILED: max error {max_err} >= 2\nRef:\n{c_ref}\nDUT:\n{c_dut}"
    cocotb.log.info("test_negative_weights: PASSED")
