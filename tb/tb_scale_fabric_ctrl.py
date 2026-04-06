"""
tb_scale_fabric_ctrl.py — cocotb testbench for scale_fabric_ctrl.sv

Half-duplex 2-peer fabric controller. 32-bit payload, framed with `last`.

Tests:
  1. Send single frame (4 beats)
  2. Recv single frame (4 beats)
  3. Send then recv back-to-back
  4. Illegal start while busy
  5. Reset mid-frame
  6. Last boundary preserved
  7. Done pulse + busy
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.send_start.value = 0
    dut.recv_start.value = 0
    dut.local_tx_valid.value = 0
    dut.local_tx_data.value = 0
    dut.local_tx_last.value = 0
    dut.fabric_tx_ready.value = 1
    dut.fabric_rx_valid.value = 0
    dut.fabric_rx_data.value = 0
    dut.fabric_rx_last.value = 0
    dut.peer_rx_ready.value = 1
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)


async def send_frame(dut, data_list):
    """Send a frame of data words via local_tx."""
    for i, d in enumerate(data_list):
        dut.local_tx_valid.value = 1
        dut.local_tx_data.value = d
        dut.local_tx_last.value = 1 if i == len(data_list) - 1 else 0
        await RisingEdge(dut.clk)
        while dut.local_tx_ready.value == 0:
            await RisingEdge(dut.clk)
    dut.local_tx_valid.value = 0
    dut.local_tx_last.value = 0


async def recv_frame(dut, data_list, max_wait=100):
    """Drive fabric_rx with data_list, capture from peer_rx."""
    captured = []
    for i, d in enumerate(data_list):
        dut.fabric_rx_valid.value = 1
        dut.fabric_rx_data.value = d
        dut.fabric_rx_last.value = 1 if i == len(data_list) - 1 else 0
        await RisingEdge(dut.clk)
        while dut.fabric_rx_ready.value == 0:
            await RisingEdge(dut.clk)
        if dut.peer_rx_valid.value == 1:
            captured.append(int(dut.peer_rx_data.value))
    dut.fabric_rx_valid.value = 0
    dut.fabric_rx_last.value = 0
    return captured


async def wait_done(dut, max_wait=100):
    for _ in range(max_wait):
        await RisingEdge(dut.clk)
        if dut.done_pulse.value == 1:
            return True
    return False


@cocotb.test()
async def test_send_single_frame(dut):
    """Send 4-beat frame, verify fabric_tx output."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    data = [0xDEAD0000, 0xDEAD0001, 0xDEAD0002, 0xDEAD0003]

    # Capture fabric_tx output
    captured = []

    async def capture_tx():
        for _ in range(20):
            await RisingEdge(dut.clk)
            if dut.fabric_tx_valid.value == 1:
                captured.append(int(dut.fabric_tx_data.value))
                if dut.fabric_tx_last.value == 1:
                    break

    dut.send_start.value = 1
    await RisingEdge(dut.clk)
    dut.send_start.value = 0

    cocotb.start_soon(capture_tx())
    await send_frame(dut, data)
    ok = await wait_done(dut)
    assert ok, "done_pulse not seen"

    assert captured == data, f"TX data mismatch: {captured} vs {data}"
    assert int(dut.err_code.value) == 0


@cocotb.test()
async def test_recv_single_frame(dut):
    """Recv 4-beat frame, verify peer_rx output."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    data = [0xBEEF0000, 0xBEEF0001, 0xBEEF0002, 0xBEEF0003]

    dut.recv_start.value = 1
    await RisingEdge(dut.clk)
    dut.recv_start.value = 0

    captured = await recv_frame(dut, data)
    ok = await wait_done(dut)
    assert ok

    assert captured == data, f"RX data mismatch: {captured} vs {data}"
    assert int(dut.err_code.value) == 0


@cocotb.test()
async def test_send_recv_back_to_back(dut):
    """Send frame, then recv frame, sequentially."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # SEND
    tx_data = [0xAA, 0xBB]
    dut.send_start.value = 1
    await RisingEdge(dut.clk)
    dut.send_start.value = 0
    await send_frame(dut, tx_data)
    ok = await wait_done(dut)
    assert ok, "Send done not seen"

    # RECV
    rx_data = [0xCC, 0xDD]
    dut.recv_start.value = 1
    await RisingEdge(dut.clk)
    dut.recv_start.value = 0
    captured = await recv_frame(dut, rx_data)
    ok = await wait_done(dut)
    assert ok, "Recv done not seen"

    assert captured == rx_data


@cocotb.test()
async def test_illegal_start_while_busy(dut):
    """Start send during active send → err_code=0x01."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Start send
    dut.send_start.value = 1
    await RisingEdge(dut.clk)
    dut.send_start.value = 0

    # Feed one beat but don't send last
    dut.local_tx_valid.value = 1
    dut.local_tx_data.value = 0x1234
    dut.local_tx_last.value = 0
    await RisingEdge(dut.clk)
    dut.local_tx_valid.value = 0

    # Try to start again while busy
    dut.send_start.value = 1
    await RisingEdge(dut.clk)
    dut.send_start.value = 0

    for _ in range(10):
        await RisingEdge(dut.clk)
        if dut.done_pulse.value == 1:
            break

    assert int(dut.err_code.value) == 0x01, f"Expected err 0x01, got {int(dut.err_code.value):#x}"


@cocotb.test()
async def test_reset_mid_frame(dut):
    """Reset during send → idle, no hang."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut.send_start.value = 1
    await RisingEdge(dut.clk)
    dut.send_start.value = 0

    # Start sending
    dut.local_tx_valid.value = 1
    dut.local_tx_data.value = 0xABCD
    dut.local_tx_last.value = 0
    await RisingEdge(dut.clk)
    dut.local_tx_valid.value = 0

    # Reset mid-frame
    dut.rst_n.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(3):
        await RisingEdge(dut.clk)

    assert dut.busy.value == 0, "Should be idle after reset"


@cocotb.test()
async def test_last_boundary_preserved(dut):
    """fabric_tx_last mirrors local_tx_last exactly."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut.send_start.value = 1
    await RisingEdge(dut.clk)
    dut.send_start.value = 0

    last_seen = []

    # 3 beats: no-last, no-last, last
    for i in range(3):
        dut.local_tx_valid.value = 1
        dut.local_tx_data.value = i
        dut.local_tx_last.value = 1 if i == 2 else 0
        await RisingEdge(dut.clk)
        last_seen.append(int(dut.fabric_tx_last.value))

    dut.local_tx_valid.value = 0
    await wait_done(dut)

    assert last_seen == [0, 0, 1], f"Last pattern should be [0,0,1], got {last_seen}"


@cocotb.test()
async def test_done_pulse_and_busy(dut):
    """busy high during transfer, done_pulse at end."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    dut.send_start.value = 1
    await RisingEdge(dut.clk)
    dut.send_start.value = 0

    saw_busy = False

    # Send 2-beat frame
    for i in range(2):
        dut.local_tx_valid.value = 1
        dut.local_tx_data.value = i
        dut.local_tx_last.value = 1 if i == 1 else 0
        await RisingEdge(dut.clk)
        if dut.busy.value == 1:
            saw_busy = True
    dut.local_tx_valid.value = 0

    ok = await wait_done(dut)
    assert saw_busy, "busy should have been high"
    assert ok, "done_pulse not seen"
