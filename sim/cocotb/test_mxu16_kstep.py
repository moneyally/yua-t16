"""16x16 타일 단독 K>1 누적 검증 — 128x128 래퍼의 X가 타일에서 오는지 격리."""
import struct
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

N = 16

def to_bf16(f):
    bits = struct.unpack(">I", struct.pack(">f", f))[0]
    lsb = (bits >> 16) & 1
    bits += 0x7FFF + lsb
    return (bits >> 16) & 0xFFFF

def bf16_to_f(b):
    return struct.unpack(">f", struct.pack(">I", (b & 0xFFFF) << 16))[0]

def raw(sig):
    """X 포함 가능한 신호를 문자열로."""
    return str(sig.value)

@cocotb.test()
async def test_tile_k_accumulate(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.en.value = 0
    dut.acc_clr.value = 0
    for i in range(N):
        dut.a_row[i].value = 0
        dut.b_col[i].value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    dut.acc_clr.value = 1
    await RisingEdge(dut.clk)
    dut.acc_clr.value = 0
    await RisingEdge(dut.clk)

    dut._log.info(f"clear 직후 acc_out[0][0] = {raw(dut.acc_flat[0])}")

    # K=4 누적: 1*1 + 1*1 + 1*1 + 1*1 = 4.0
    for k in range(4):
        for i in range(N):
            dut.a_row[i].value = to_bf16(1.0 if i == 0 else 0.0)
            dut.b_col[i].value = to_bf16(1.0 if i == 0 else 0.0)
        dut.en.value = 1
        await RisingEdge(dut.clk)
        dut.en.value = 0
        await RisingEdge(dut.clk)
        dut._log.info(f"K={k+1} 후 acc_out[0][0] = {raw(dut.acc_flat[0])}")

    s = raw(dut.acc_flat[0])
    assert "x" not in s.lower(), f"타일 누적기에 X 발생: {s}"
    got = struct.unpack(">f", struct.pack(">I", int(s, 2)))[0]
    assert abs(got - 4.0) < 1e-3, f"타일 K=4 누적 오류: 기대 4.0, 실제 {got}"


@cocotb.test()
async def test_tile_zero_input_acc(dut):
    """0 입력이 섞여도 누적기가 오염되지 않아야 한다 (fp32_add 의 zero 경로)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.en.value = 0
    dut.acc_clr.value = 0
    for i in range(N):
        dut.a_row[i].value = 0
        dut.b_col[i].value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    dut.acc_clr.value = 1
    await RisingEdge(dut.clk)
    dut.acc_clr.value = 0
    await RisingEdge(dut.clk)

    # 전부 0 입력으로 en 한 번 → 0 + 0 = 0 이어야 함
    dut.en.value = 1
    await RisingEdge(dut.clk)
    dut.en.value = 0
    await RisingEdge(dut.clk)

    s = raw(dut.acc_flat[0])
    dut._log.info(f"0x0 누적 후 acc_out[0][0] = {s}")
    assert "x" not in s.lower(), f"0 입력 누적에서 X 발생: {s}"
    assert int(s, 2) == 0, f"0 입력인데 누적기가 {int(s,2):#010x}"
