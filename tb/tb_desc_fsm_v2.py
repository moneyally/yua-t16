"""
tb_desc_fsm_v2.py — cocotb testbench for desc_fsm_v2.sv

Tests:
  1. Valid GEMM descriptor — normal dispatch and completion
  2. Illegal opcode — fault_code 0x01
  3. CRC mismatch — fault_code 0x02
  4. NOP descriptor — no dispatch, direct done
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


DESC_SIZE = 64


def crc8(data_bytes):
    """CRC-8 with polynomial 0x07 (matches RTL)."""
    crc = 0x00
    for byte in data_bytes:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def make_descriptor(opcode, act=0, wgt=0, out=0, kt=0, valid_crc=True):
    """Build a DESC_SIZE byte descriptor with CRC."""
    desc = [0] * DESC_SIZE
    desc[0] = opcode & 0xFF

    # Little-endian encode addresses
    for i in range(8):
        desc[16 + i] = (act >> (8 * i)) & 0xFF
        desc[24 + i] = (wgt >> (8 * i)) & 0xFF
        desc[32 + i] = (out >> (8 * i)) & 0xFF

    for i in range(4):
        desc[40 + i] = (kt >> (8 * i)) & 0xFF

    # CRC over bytes 0..62, stored in byte 63
    crc_val = crc8(desc[:DESC_SIZE - 1])
    desc[DESC_SIZE - 1] = crc_val if valid_crc else (crc_val ^ 0xFF)
    return desc


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.desc_valid.value = 0
    dut.queue_class.value = 0
    dut.cmd_ready.value = 0
    dut.core_done.value = 0
    dut.timeout_cycles.value = 100_000
    for i in range(DESC_SIZE):
        dut.desc_bytes[i].value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


async def send_descriptor(dut, desc_bytes):
    """Send descriptor when FSM is ready."""
    while dut.desc_ready.value == 0:
        await RisingEdge(dut.clk)
    dut.desc_valid.value = 1
    for i in range(DESC_SIZE):
        dut.desc_bytes[i].value = desc_bytes[i]
    await RisingEdge(dut.clk)
    dut.desc_valid.value = 0


@cocotb.test()
async def test_valid_gemm_descriptor(dut):
    """GEMM (0x02) descriptor: dispatch, wait for core_done, done_pulse."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    desc = make_descriptor(opcode=0x02, act=0x1000, wgt=0x2000, out=0x3000, kt=16)
    await send_descriptor(dut, desc)

    # Wait for cmd_valid
    for _ in range(20):
        await RisingEdge(dut.clk)
        if dut.cmd_valid.value == 1:
            assert dut.cmd_opcode.value == 0x02
            dut.cmd_ready.value = 1
            break
    await RisingEdge(dut.clk)
    dut.cmd_ready.value = 0

    # Simulate core completion after some cycles
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.core_done.value = 1
    await RisingEdge(dut.clk)
    dut.core_done.value = 0

    # Wait for done_pulse
    for _ in range(10):
        await RisingEdge(dut.clk)
        if dut.done_pulse.value == 1:
            break

    assert dut.done_pulse.value == 1, "done_pulse not asserted"
    assert dut.fault_valid.value == 0, "Unexpected fault"


@cocotb.test()
async def test_illegal_opcode(dut):
    """Illegal opcode (0xFF) should trigger fault_code 0x01."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    desc = make_descriptor(opcode=0xFF)
    await send_descriptor(dut, desc)

    # Wait for fault
    for _ in range(20):
        await RisingEdge(dut.clk)
        if dut.fault_valid.value == 1:
            break

    assert dut.fault_valid.value == 1, "Fault not raised for illegal opcode"
    assert int(dut.fault_code.value) == 0x01, \
        f"Expected fault_code 0x01, got {int(dut.fault_code.value):#x}"


@cocotb.test()
async def test_crc_mismatch(dut):
    """Bad CRC should trigger fault_code 0x02."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    desc = make_descriptor(opcode=0x02, valid_crc=False)
    await send_descriptor(dut, desc)

    for _ in range(20):
        await RisingEdge(dut.clk)
        if dut.fault_valid.value == 1:
            break

    assert dut.fault_valid.value == 1, "Fault not raised for CRC mismatch"
    assert int(dut.fault_code.value) == 0x02, \
        f"Expected fault_code 0x02, got {int(dut.fault_code.value):#x}"


@cocotb.test()
async def test_nop_descriptor(dut):
    """NOP (0x01) should skip dispatch and go straight to done."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    desc = make_descriptor(opcode=0x01)
    await send_descriptor(dut, desc)

    # Should get done_pulse without cmd_valid ever being asserted
    cmd_was_valid = False
    for _ in range(20):
        await RisingEdge(dut.clk)
        if dut.cmd_valid.value == 1:
            cmd_was_valid = True
        if dut.done_pulse.value == 1:
            break

    assert dut.done_pulse.value == 1, "done_pulse not asserted for NOP"
    assert not cmd_was_valid, "cmd_valid should not be asserted for NOP"
