"""test_mxu128_stream_rd.py — 공유 변환기 순차 읽기 검증.

STREAM_RD=1 로 빌드한 DUT 에서 rd_idx/rd_fp32 로 읽은 값이,
기대 내적값과 일치하는지 본다. 또 FP32_OUT=0 이므로 acc_flat 은
전부 0 이어야 한다 (변환기가 제거됐다는 구조적 확인).
"""
import struct
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

DIM = 128


def to_bf16(f):
    bits = struct.unpack(">I", struct.pack(">f", f))[0]
    lsb = (bits >> 16) & 1
    bits += 0x7FFF + lsb
    return (bits >> 16) & 0xFFFF


def bf16_to_f(b):
    return struct.unpack(">f", struct.pack(">I", (b & 0xFFFF) << 16))[0]


def bits_to_f(bits):
    return struct.unpack(">f", struct.pack(">I", bits & 0xFFFFFFFF))[0]


async def reset(dut):
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.acc_clr.value = 0
    dut.data_valid.value = 0
    dut.k_steps.value = 1
    dut.rd_idx.value = 0
    for i in range(DIM):
        dut.a_col[i].value = 0
        dut.b_row[i].value = 0
    await Timer(100, unit="ns")
    dut.rst_n.value = 1
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.acc_clr.value = 1
    await RisingEdge(dut.clk)
    dut.acc_clr.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)


async def matmul(dut, steps):
    dut.k_steps.value = len(steps)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    for a_vec, b_vec in steps:
        for _ in range(5000):
            await RisingEdge(dut.clk)
            if dut.ready.value == 1:
                break
        dut.data_valid.value = 1
        for i in range(DIM):
            dut.a_col[i].value = to_bf16(a_vec[i])
            dut.b_row[i].value = to_bf16(b_vec[i])
        await RisingEdge(dut.clk)
        dut.data_valid.value = 0
    for _ in range(5000):
        await RisingEdge(dut.clk)
        if dut.done_pulse.value == 1:
            return
    raise TimeoutError("done_pulse 없음")


async def read_at(dut, row, col):
    dut.rd_idx.value = row * DIM + col
    await Timer(1, unit="ns")
    return bits_to_f(int(dut.rd_fp32.value))


@cocotb.test()
async def test_stream_read_matches_expected(dut):
    """여러 위치를 순차 읽기로 확인한다 — 타일 경계 포함."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    # a[i] = i 번째가 0 이 아닌 희소 벡터 몇 개로 K=3 누산
    picks = [(0, 0), (0, 16), (16, 0), (17, 33), (127, 127), (64, 65)]
    vals = [(1.0, 2.0), (0.5, 4.0), (2.0, 1.5)]   # 합 = 2 + 2 + 3 = 7.0

    steps = []
    for av, bv in vals:
        a = [av] * DIM
        b = [bv] * DIM
        steps.append((a, b))

    await matmul(dut, steps)

    exp = sum(bf16_to_f(to_bf16(av)) * bf16_to_f(to_bf16(bv)) for av, bv in vals)
    for r, c in picks:
        got = await read_at(dut, r, c)
        assert abs(got - exp) < 1e-3, f"[{r}][{c}]: 기대 {exp}, 실제 {got}"


@cocotb.test()
async def test_distinct_values_per_position(dut):
    """위치마다 다른 값이 나오는지 — 읽기 라우팅이 실제로 동작하는 확인."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    # a[i] = (i % 8) + 1, b[j] = (j % 4) + 1  →  C[i][j] = a[i] * b[j]
    a = [float((i % 8) + 1) for i in range(DIM)]
    b = [float((j % 4) + 1) for j in range(DIM)]
    await matmul(dut, [(a, b)])

    for r, c in [(0, 0), (1, 2), (15, 15), (16, 16), (33, 70), (127, 126)]:
        got = await read_at(dut, r, c)
        exp = bf16_to_f(to_bf16(a[r])) * bf16_to_f(to_bf16(b[c]))
        assert abs(got - exp) < 1e-3, f"[{r}][{c}]: 기대 {exp}, 실제 {got}"


@cocotb.test()
async def test_acc_flat_tied_off(dut):
    """FP32_OUT=0 이면 PE 별 변환기가 없으므로 acc_flat 은 전부 0 이다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    a = [1.0] * DIM
    b = [1.0] * DIM
    await matmul(dut, [(a, b)])

    # 읽기 경로는 1.0 을 내야 한다
    assert abs(await read_at(dut, 0, 0) - 1.0) < 1e-3

    # 반면 acc_flat 은 묶여 있다
    for idx in (0, 1, DIM * 64 + 7, DIM * DIM - 1):
        assert int(dut.acc_flat[idx].value) == 0, \
            f"acc_flat[{idx}] 가 0 이 아니다 — FP32_OUT=0 이 반영되지 않았다"
