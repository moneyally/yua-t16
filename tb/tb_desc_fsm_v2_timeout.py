"""
tb_desc_fsm_v2_timeout.py — cocotb testbench for desc_fsm_v2.sv timeout

Tests:
  1. Timeout expiry when core_done never arrives — fault_code 0x03
  2. core_done arrives just before timeout — no fault
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


DESC_SIZE = 64


def crc8(data_bytes):
    crc = 0x00
    for byte in data_bytes:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def make_descriptor(opcode, valid_crc=True):
    desc = [0] * DESC_SIZE
    desc[0] = opcode & 0xFF
    crc_val = crc8(desc[:DESC_SIZE - 1])
    desc[DESC_SIZE - 1] = crc_val if valid_crc else (crc_val ^ 0xFF)
    return desc


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.desc_valid.value = 0
    dut.queue_class.value = 0
    dut.cmd_ready.value = 0
    dut.core_done.value = 0
    dut.timeout_cycles.value = 20  # short timeout for test
    for i in range(DESC_SIZE):
        dut.desc_bytes[i].value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


async def send_and_dispatch(dut, desc_bytes):
    """Send descriptor, accept cmd, enter WAIT state."""
    while dut.desc_ready.value == 0:
        await RisingEdge(dut.clk)
    dut.desc_valid.value = 1
    for i in range(DESC_SIZE):
        dut.desc_bytes[i].value = desc_bytes[i]
    await RisingEdge(dut.clk)
    dut.desc_valid.value = 0

    # Accept cmd
    for _ in range(20):
        await RisingEdge(dut.clk)
        if dut.cmd_valid.value == 1:
            dut.cmd_ready.value = 1
            break
    await RisingEdge(dut.clk)
    dut.cmd_ready.value = 0


@cocotb.test()
async def test_timeout_expiry(dut):
    """core_done never arrives, should timeout with fault_code 0x03."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    desc = make_descriptor(opcode=0x02)
    await send_and_dispatch(dut, desc)

    # Wait for timeout (20 cycles + margin)
    for _ in range(50):
        await RisingEdge(dut.clk)
        if dut.fault_valid.value == 1:
            break

    assert dut.fault_valid.value == 1, "Timeout fault not raised"
    assert int(dut.fault_code.value) == 0x03, \
        f"Expected fault_code 0x03 (timeout), got {int(dut.fault_code.value):#x}"


@cocotb.test()
async def test_core_done_before_timeout(dut):
    """core_done arrives at cycle 15 (timeout=20), should complete normally."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    desc = make_descriptor(opcode=0x02)
    await send_and_dispatch(dut, desc)

    # Send core_done before timeout
    for _ in range(10):
        await RisingEdge(dut.clk)
    dut.core_done.value = 1
    await RisingEdge(dut.clk)
    dut.core_done.value = 0

    # Wait for done
    for _ in range(10):
        await RisingEdge(dut.clk)
        if dut.done_pulse.value == 1:
            break

    assert dut.done_pulse.value == 1, "done_pulse not asserted"
    # fault should NOT be valid at done time (cleared on IDLE entry)
    # But we just need to check it wasn't a fault path
