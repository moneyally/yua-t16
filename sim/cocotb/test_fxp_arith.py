"""test_fxp_arith.py — 고정소수점 누산 산술의 랜덤 차등 테스트.

Python 부동소수점 기준값과 비교한다. 입력이 BF16 이므로 가수는 8비트뿐이고,
고정소수점 분해능(2^-16)은 그보다 훨씬 곱다. 따라서 범위 안에서는
기준값과 맞아야 한다.
"""
import random
import struct
import cocotb
from cocotb.triggers import Timer

ACC_W = 48
ACC_F = 16
FXP_MAX = (1 << (ACC_W - 1)) - 1
FXP_MIN = -(1 << (ACC_W - 1))


def to_bf16(f: float) -> int:
    bits = struct.unpack(">I", struct.pack(">f", f))[0]
    lsb = (bits >> 16) & 1
    bits += 0x7FFF + lsb
    return (bits >> 16) & 0xFFFF


def bf16_to_f(b: int) -> float:
    return struct.unpack(">f", struct.pack(">I", (b & 0xFFFF) << 16))[0]


def bits_to_f(bits: int) -> float:
    return struct.unpack(">f", struct.pack(">I", bits & 0xFFFFFFFF))[0]


def signed(v: int, w: int = ACC_W) -> int:
    return v - (1 << w) if v & (1 << (w - 1)) else v


def ref_mul_fxp(fa: float, fb: float) -> int:
    """기준: 곱을 2^F 로 스케일해 0 방향으로 버림."""
    prod = bf16_to_f(to_bf16(fa)) * bf16_to_f(to_bf16(fb))
    scaled = prod * (1 << ACC_F)
    v = int(scaled)              # 0 방향 절삭
    return max(FXP_MIN, min(FXP_MAX, v))


async def settle(dut):
    await Timer(1, unit="ns")


@cocotb.test()
async def test_mul_fxp_known(dut):
    """알려진 값 몇 개."""
    cases = [
        (1.0, 1.0,   1 << ACC_F),
        (1.0, 2.0,   2 << ACC_F),
        (2.0, 2.0,   4 << ACC_F),
        (0.5, 0.5,   1 << (ACC_F - 2)),
        (-1.0, 1.0, -(1 << ACC_F)),
        (-2.0, -3.0, 6 << ACC_F),
        (0.0, 5.0,   0),
        (3.0, 0.0,   0),
    ]
    for fa, fb, exp in cases:
        dut.a.value = to_bf16(fa)
        dut.b.value = to_bf16(fb)
        await settle(dut)
        got = signed(int(dut.mul_fxp.value))
        assert got == exp, f"{fa} × {fb}: 기대 {exp}, 실제 {got}"


@cocotb.test()
async def test_mul_fxp_random(dut):
    """랜덤 BF16 쌍 2000개를 Python 기준값과 비교."""
    rng = random.Random(0xC0FFEE)
    bad = 0
    for _ in range(2000):
        fa = rng.uniform(-64.0, 64.0)
        fb = rng.uniform(-64.0, 64.0)
        dut.a.value = to_bf16(fa)
        dut.b.value = to_bf16(fb)
        await settle(dut)
        got = signed(int(dut.mul_fxp.value))
        exp = ref_mul_fxp(fa, fb)
        # 우하향 시프트 절삭 방향 차이로 1 LSB 가 날 수 있다
        if abs(got - exp) > 1:
            bad += 1
            if bad <= 3:
                dut._log.error(f"{fa:.6f} × {fb:.6f}: 기대 {exp}, 실제 {got}")
    assert bad == 0, f"{bad}/2000 건 불일치"


@cocotb.test()
async def test_add_sat(dut):
    """포화 덧셈: 정상 구간과 양/음 양쪽 넘침."""
    cases = [
        (0, 0, 0),
        (100, 200, 300),
        (-100, 50, -50),
        (FXP_MAX, 1, FXP_MAX),          # 양 넘침
        (FXP_MAX, FXP_MAX, FXP_MAX),
        (FXP_MIN, -1, FXP_MIN),         # 음 넘침
        (FXP_MIN, FXP_MIN, FXP_MIN),
        (FXP_MAX, FXP_MIN, -1),         # 부호 다름 → 넘침 아님
    ]
    for x, y, exp in cases:
        dut.add_x.value = x & ((1 << ACC_W) - 1)
        dut.add_y.value = y & ((1 << ACC_W) - 1)
        await settle(dut)
        got = signed(int(dut.add_sat.value))
        assert got == exp, f"{x} + {y}: 기대 {exp}, 실제 {got}"


@cocotb.test()
async def test_to_fp32_known(dut):
    for fixed, exp in [
        (0, 0.0),
        (1 << ACC_F, 1.0),
        (2 << ACC_F, 2.0),
        (-(1 << ACC_F), -1.0),
        (3 << ACC_F, 3.0),
        (1 << (ACC_F - 1), 0.5),
        (-(6 << ACC_F), -6.0),
    ]:
        dut.acc_in.value = fixed & ((1 << ACC_W) - 1)
        await settle(dut)
        got = bits_to_f(int(dut.to_fp32.value))
        assert abs(got - exp) < 1e-6, f"fxp {fixed}: 기대 {exp}, 실제 {got}"


@cocotb.test()
async def test_to_fp32_random(dut):
    """랜덤 고정소수점 값 → FP32. 상대오차가 FP32 가수 한계 안이어야 한다."""
    rng = random.Random(0xBEEF)
    worst = 0.0
    for _ in range(2000):
        v = rng.randint(-(1 << 40), 1 << 40)
        dut.acc_in.value = v & ((1 << ACC_W) - 1)
        await settle(dut)
        got = bits_to_f(int(dut.to_fp32.value))
        exp = v / (1 << ACC_F)
        if exp == 0:
            assert got == 0
            continue
        rel = abs(got - exp) / abs(exp)
        worst = max(worst, rel)
        assert rel < 2e-7, f"fxp {v}: 기대 {exp}, 실제 {got}, 상대오차 {rel:.3e}"
    dut._log.info(f"최대 상대오차 {worst:.3e} (FP32 가수 한계 ≈ 1.19e-07)")


@cocotb.test()
async def test_end_to_end_dot(dut):
    """곱 → 누산 → FP32 변환 전 과정을 K=64 내적으로 검증."""
    rng = random.Random(7)
    for trial in range(20):
        acc = 0
        ref = 0.0
        for _ in range(64):
            fa = rng.uniform(-8.0, 8.0)
            fb = rng.uniform(-8.0, 8.0)
            dut.a.value = to_bf16(fa)
            dut.b.value = to_bf16(fb)
            await settle(dut)
            incr = signed(int(dut.mul_fxp.value))

            dut.add_x.value = acc & ((1 << ACC_W) - 1)
            dut.add_y.value = incr & ((1 << ACC_W) - 1)
            await settle(dut)
            acc = signed(int(dut.add_sat.value))

            ref += bf16_to_f(to_bf16(fa)) * bf16_to_f(to_bf16(fb))

        dut.acc_in.value = acc & ((1 << ACC_W) - 1)
        await settle(dut)
        got = bits_to_f(int(dut.to_fp32.value))
        assert abs(got - ref) < 1e-3, \
            f"trial {trial}: 기대 {ref}, 실제 {got}, 차 {abs(got-ref):.3e}"
