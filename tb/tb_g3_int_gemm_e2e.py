"""
tb_g3_int_gemm_e2e.py — G3-INT-001: BF16 GEMM forward E2E

E2E path verified:
  MXU_FWD descriptor → g3_desc_fsm → mxu adapter → mxu_bf16_16x16
  → completion → done_pulse

Tests:
  1. MXU_FWD dispatch + data + done
  2. MXU_FWD result verification (BF16 1.0×2.0 = 2.0)
  3. Illegal opcode fault passthrough
  4. Timeout fault passthrough
  5. Reset during integration
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import struct

DESC_SIZE = 64
OP_MXU_FWD = 0x10
OP_NOP = 0x01


def crc8(data):
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def make_desc(opcode, valid_crc=True):
    desc = [0] * DESC_SIZE
    desc[0] = opcode & 0xFF
    crc_val = crc8(desc[:DESC_SIZE - 1])
    desc[DESC_SIZE - 1] = crc_val if valid_crc else (crc_val ^ 0xFF)
    return desc


def float_to_bf16(f):
    fp32_bits = struct.unpack('>I', struct.pack('>f', f))[0]
    return (fp32_bits >> 16) & 0xFFFF


def fp32_bits_to_float(bits):
    return struct.unpack('>f', struct.pack('>I', bits & 0xFFFFFFFF))[0]


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.int_desc_valid.value = 0
    dut.int_queue_class.value = 0
    dut.mxu_data_valid.value = 0
    for i in range(DESC_SIZE):
        dut.int_desc_bytes[i].value = 0
    for i in range(16):
        dut.mxu_a_row[i].value = 0
        dut.mxu_b_col[i].value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


async def send_desc(dut, desc_bytes):
    """Send descriptor via integration hook."""
    while dut.int_desc_ready.value == 0:
        await RisingEdge(dut.clk)
    dut.int_desc_valid.value = 1
    for i in range(DESC_SIZE):
        dut.int_desc_bytes[i].value = desc_bytes[i]
    await RisingEdge(dut.clk)
    dut.int_desc_valid.value = 0


async def drive_mxu_data(dut, a_vals, b_vals, k_steps=1):
    """Drive BF16 data into MXU for k_steps cycles."""
    for k in range(k_steps):
        dut.mxu_data_valid.value = 1
        for i in range(16):
            dut.mxu_a_row[i].value = float_to_bf16(a_vals[i]) if i < len(a_vals) else 0
            dut.mxu_b_col[i].value = float_to_bf16(b_vals[i]) if i < len(b_vals) else 0
        await RisingEdge(dut.clk)
    dut.mxu_data_valid.value = 0


async def wait_done(dut, max_cycles=200):
    for _ in range(max_cycles):
        await RisingEdge(dut.clk)
        if dut.done_pulse.value == 1:
            return True
    return False


@cocotb.test()
async def test_mxu_forward_dispatch_and_done(dut):
    """MXU_FWD descriptor → dispatch → data → done_pulse."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Send MXU_FWD descriptor
    await send_desc(dut, make_desc(OP_MXU_FWD))

    # Wait a few cycles for FSM to reach DISPATCH → MXU_ACTIVE
    for _ in range(10):
        await RisingEdge(dut.clk)

    # Drive 1 step of data: a[0]=1.0, b[0]=1.0
    a = [1.0] + [0.0] * 15
    b = [1.0] + [0.0] * 15
    await drive_mxu_data(dut, a, b, k_steps=1)

    # Wait for done
    ok = await wait_done(dut)
    assert ok, "done_pulse not seen after MXU forward"
    assert dut.fault_valid.value == 0, "Unexpected fault"


@cocotb.test()
async def test_mxu_forward_result_verification(dut):
    """Verify MXU accumulator: 1.0 × 2.0 = 2.0 at [0][0]."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await send_desc(dut, make_desc(OP_MXU_FWD))
    for _ in range(10):
        await RisingEdge(dut.clk)

    # a[0]=1.0, b[0]=2.0
    a = [1.0] + [0.0] * 15
    b = [2.0] + [0.0] * 15
    await drive_mxu_data(dut, a, b, k_steps=1)

    ok = await wait_done(dut)
    assert ok, "done_pulse not seen"

    # Read accumulator [0][0]
    result_bits = int(dut.mxu_acc[0][0].value)
    result = fp32_bits_to_float(result_bits)
    assert abs(result - 2.0) < 0.1, f"Expected ~2.0, got {result}"


@cocotb.test()
async def test_illegal_opcode_fault(dut):
    """Illegal opcode 0xFF → fault passthrough."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await send_desc(dut, make_desc(0xFF))

    for _ in range(30):
        await RisingEdge(dut.clk)
        if dut.fault_valid.value == 1:
            break
    assert dut.fault_valid.value == 1
    assert int(dut.fault_code.value) == 0x01


@cocotb.test()
async def test_timeout_fault(dut):
    """MXU_FWD but no data → timeout fault."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await send_desc(dut, make_desc(OP_MXU_FWD))
    # Don't drive any data — wait for timeout (1000 cycles)
    for _ in range(1200):
        await RisingEdge(dut.clk)
        if dut.fault_valid.value == 1:
            break

    assert dut.fault_valid.value == 1
    assert int(dut.fault_code.value) == 0x03  # timeout


@cocotb.test()
async def test_nop_passthrough(dut):
    """NOP descriptor works through integration top."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await send_desc(dut, make_desc(OP_NOP))
    ok = await wait_done(dut)
    assert ok, "NOP should complete"
    assert dut.fault_valid.value == 0
