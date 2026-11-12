"""
test_mxu128_kstep.py — K>1 누적 정확성 테스트 (mxu_bf16_128x128)

기존 tb_mxu_bf16_128x128.py의 5개 테스트는 전부 k_steps=1이다.
그래서 S_STORE_TILE의 누적 경로가 한 번도 검증된 적이 없다.

이 테스트는 K>1에서 C[i][j] = sum_k a_k[i] * b_k[j] 가 맞는지 본다.
"""
import struct
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

DIM = 128


def to_bf16(f: float) -> int:
    """FP32 → BF16 (round-to-nearest-even, DUT 입력 포맷)."""
    bits = struct.unpack(">I", struct.pack(">f", f))[0]
    lsb = (bits >> 16) & 1
    bits += 0x7FFF + lsb
    return (bits >> 16) & 0xFFFF


def bf16_to_f(b: int) -> float:
    return struct.unpack(">f", struct.pack(">I", (b & 0xFFFF) << 16))[0]


def bits_to_f(bits: int) -> float:
    return struct.unpack(">f", struct.pack(">I", bits & 0xFFFFFFFF))[0]


async def reset(dut):
    dut.rst_n.value = 0
    dut.start.value = 0
    dut.acc_clr.value = 0
    dut.data_valid.value = 0
    dut.k_steps.value = 1
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


async def matmul(dut, steps, timeout=200_000):
    """steps = [(a_vec, b_vec), ...] 를 K-step 으로 흘린다."""
    dut.k_steps.value = len(steps)
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    for a_vec, b_vec in steps:
        for _ in range(timeout):
            await RisingEdge(dut.clk)
            if dut.ready.value == 1:
                break
        else:
            raise TimeoutError("ready 가 뜨지 않음")
        dut.data_valid.value = 1
        for i in range(DIM):
            dut.a_col[i].value = to_bf16(a_vec[i])
            dut.b_row[i].value = to_bf16(b_vec[i])
        await RisingEdge(dut.clk)
        dut.data_valid.value = 0

    cycles = 0
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        cycles += 1
        if dut.done_pulse.value == 1:
            return cycles
    raise TimeoutError("done_pulse 가 뜨지 않음")


def expected(steps, i, j):
    """BF16으로 양자화된 입력 기준 기대 FP32 누적값."""
    acc = 0.0
    for a_vec, b_vec in steps:
        acc += bf16_to_f(to_bf16(a_vec[i])) * bf16_to_f(to_bf16(b_vec[j]))
    return acc


@cocotb.test()
async def test_k2_accumulate(dut):
    """K=2: C[0][0] = 1*1 + 1*1 = 2.0 — 가장 단순한 누적."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    a = [0.0] * DIM; a[0] = 1.0
    b = [0.0] * DIM; b[0] = 1.0
    steps = [(a, b), (a, b)]

    await matmul(dut, steps)
    got = bits_to_f(int(dut.acc_flat[0].value))
    exp = expected(steps, 0, 0)
    assert abs(got - exp) < 1e-3, f"K=2 누적 오류: 기대 {exp}, 실제 {got}"


@cocotb.test()
async def test_k4_accumulate_mixed(dut):
    """K=4, 서로 다른 값. 정수 덧셈이면 여기서 크게 틀어진다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    vals = [(1.0, 2.0), (0.5, 4.0), (2.0, 1.0), (1.5, 2.0)]  # 합 = 2+2+2+3 = 9.0
    steps = []
    for av, bv in vals:
        a = [0.0] * DIM; a[0] = av
        b = [0.0] * DIM; b[0] = bv
        steps.append((a, b))

    await matmul(dut, steps)
    got = bits_to_f(int(dut.acc_flat[0].value))
    exp = expected(steps, 0, 0)
    assert abs(got - exp) < 1e-2, f"K=4 누적 오류: 기대 {exp}, 실제 {got}"


@cocotb.test()
async def test_k3_cross_tile(dut):
    """K=3, 타일 경계를 넘는 위치(row 0, col 16 → tile[0][1])에서도 누적이 맞아야 한다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    steps = []
    for av, bv in [(1.0, 3.0), (2.0, 1.0), (1.0, 1.0)]:  # 3 + 2 + 1 = 6.0
        a = [0.0] * DIM; a[0] = av
        b = [0.0] * DIM; b[16] = bv
        steps.append((a, b))

    await matmul(dut, steps)
    idx = 0 * DIM + 16
    got = bits_to_f(int(dut.acc_flat[idx].value))
    exp = expected(steps, 0, 16)
    assert abs(got - exp) < 1e-2, f"타일 경계 K=3 누적 오류: 기대 {exp}, 실제 {got}"


@cocotb.test()
async def test_k8_cycle_cost(dut):
    """K=8 완료까지의 사이클 수를 기록한다 (성능 기준선)."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset(dut)

    a = [0.0] * DIM; a[0] = 1.0
    b = [0.0] * DIM; b[0] = 1.0
    steps = [(a, b)] * 8

    cycles = await matmul(dut, steps)
    dut._log.info(f"[PERF] K=8 완료 사이클: {cycles}  (K-step 당 {cycles/8:.1f})")

    got = bits_to_f(int(dut.acc_flat[0].value))
    exp = expected(steps, 0, 0)
    assert abs(got - exp) < 1e-2, f"K=8 누적 오류: 기대 {exp}, 실제 {got}"
