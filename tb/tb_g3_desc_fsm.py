"""
tb_g3_desc_fsm.py — cocotb testbench for g3_desc_fsm.sv

Tests:
  1. NOP descriptor
  2. MXU_FWD dispatch
  3. Illegal opcode fault
  4. CRC fail fault
  5. Timeout fault
  6. Unsupported opcode (BACKWARD) fault
  7. Queue class propagation
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

DESC_SIZE = 64
OP_NOP       = 0x01
OP_GEMM_INT8 = 0x02
OP_MXU_FWD   = 0x10
OP_BACKWARD  = 0x20
OP_OPTIMIZER = 0x30
OP_COLLECTIVE = 0x40


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


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.desc_valid.value = 0
    dut.queue_class.value = 0
    dut.mxu_cmd_ready.value = 1
    dut.gemm_cmd_ready.value = 1
    dut.core_done.value = 0
    dut.timeout_cycles.value = 50  # short for test
    for i in range(DESC_SIZE):
        dut.desc_bytes[i].value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


async def send_desc(dut, desc_bytes):
    while dut.desc_ready.value == 0:
        await RisingEdge(dut.clk)
    dut.desc_valid.value = 1
    for i in range(DESC_SIZE):
        dut.desc_bytes[i].value = desc_bytes[i]
    await RisingEdge(dut.clk)
    dut.desc_valid.value = 0


@cocotb.test()
async def test_nop(dut):
    """NOP: no dispatch, direct done."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await send_desc(dut, make_desc(OP_NOP))

    for _ in range(20):
        await RisingEdge(dut.clk)
        if dut.done_pulse.value == 1:
            break
    assert dut.done_pulse.value == 1, "NOP should produce done_pulse"
    assert dut.fault_valid.value == 0, "NOP should not fault"


@cocotb.test()
async def test_mxu_fwd_dispatch(dut):
    """MXU_FWD: dispatch to MXU, wait for core_done."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await send_desc(dut, make_desc(OP_MXU_FWD))

    # Wait for mxu_cmd_valid
    for _ in range(20):
        await RisingEdge(dut.clk)
        if dut.mxu_cmd_valid.value == 1:
            break
    assert dut.mxu_cmd_valid.value == 1, "MXU cmd should fire"

    # Simulate core completion
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.core_done.value = 1
    await RisingEdge(dut.clk)
    dut.core_done.value = 0

    for _ in range(10):
        await RisingEdge(dut.clk)
        if dut.done_pulse.value == 1:
            break
    assert dut.done_pulse.value == 1, "Should complete after core_done"
    assert dut.fault_valid.value == 0, "No fault expected"


@cocotb.test()
async def test_illegal_opcode(dut):
    """Unknown opcode 0xFF -> fault_code 0x01."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await send_desc(dut, make_desc(0xFF))

    for _ in range(20):
        await RisingEdge(dut.clk)
        if dut.fault_valid.value == 1:
            break
    assert dut.fault_valid.value == 1
    assert int(dut.fault_code.value) == 0x01


@cocotb.test()
async def test_crc_fail(dut):
    """Bad CRC -> fault_code 0x02."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await send_desc(dut, make_desc(OP_MXU_FWD, valid_crc=False))

    for _ in range(20):
        await RisingEdge(dut.clk)
        if dut.fault_valid.value == 1:
            break
    assert dut.fault_valid.value == 1
    assert int(dut.fault_code.value) == 0x02


@cocotb.test()
async def test_timeout(dut):
    """core_done never arrives -> fault_code 0x03."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)
    dut.timeout_cycles.value = 10  # very short

    await send_desc(dut, make_desc(OP_MXU_FWD))

    for _ in range(50):
        await RisingEdge(dut.clk)
        if dut.fault_valid.value == 1:
            break
    assert dut.fault_valid.value == 1
    assert int(dut.fault_code.value) == 0x03


@cocotb.test()
async def test_unsupported_backward(dut):
    """BACKWARD opcode -> fault_code 0x04 (unsupported target)."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    await send_desc(dut, make_desc(OP_BACKWARD))

    for _ in range(20):
        await RisingEdge(dut.clk)
        if dut.fault_valid.value == 1:
            break
    assert dut.fault_valid.value == 1
    assert int(dut.fault_code.value) == 0x04


@cocotb.test()
async def test_queue_class_propagation(dut):
    """Queue class is captured and visible."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut.queue_class.value = 3  # hipri
    await send_desc(dut, make_desc(OP_NOP))

    for _ in range(20):
        await RisingEdge(dut.clk)
        if dut.done_pulse.value == 1:
            break
    assert int(dut.current_qclass.value) == 3
