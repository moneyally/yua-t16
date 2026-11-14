"""test_gemm_rd256.py — RD_DATA_W=256 에서 결과가 같은지, 비트 수가 주는지.

폭을 넓혀도 계산 결과는 바뀌면 안 된다. 바뀌는 것은 DMA 읽기 비트 수뿐이다.
한 K-step 은 INT8 16개 = 128비트이므로 256비트 비트 하나가 K-step 두 개를
나른다. SRAM 은 워드당 K-step 두 개를 담고, 연산 쪽에서 절반을 고른다.
"""
import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from dma_stub import SimpleMem, start_dma

import os
RD_BEAT_BYTES = int(os.environ.get("RD_BEAT_BYTES", "32"))  # RD_DATA_W/8


def u32_le(x): return int(x).to_bytes(4, "little", signed=False)
def u64_le(x): return int(x).to_bytes(8, "little", signed=False)


async def boot(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.desc_valid.value = 0
    for i in range(64):
        dut.desc_bytes[i].value = 0
    dut.rd_req_ready.value = 1
    dut.wr_req_ready.value = 1
    dut.rd_data_valid.value = 0
    dut.rd_data.value = 0
    dut.rd_data_last.value = 0
    dut.rd_done.value = 0
    dut.wr_data_ready.value = 1
    dut.wr_done.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(5):
        await RisingEdge(dut.clk)


async def run(dut, Kt, A, B, mem):
    """GEMM 한 번. DMA 스텁은 호출자가 테스트당 한 번만 띄운다.

    매 호출마다 start_dma 를 부르면 스텁 코루틴이 쌓여 같은 신호를
    여러 개가 동시에 구동한다.
    """
    act_base, wgt_base, out_base = 0x1000, 0x2000, 0x3000
    for k in range(Kt):
        mem.write_bytes(act_base + 16 * k,
                        bytes((int(A[i, k]) & 0xFF) for i in range(16)))
        mem.write_bytes(wgt_base + 16 * k,
                        bytes((int(B[k, j]) & 0xFF) for j in range(16)))

    d = bytearray(64)
    d[0:1] = bytes([0x02])
    d[16:24] = u64_le(act_base)
    d[24:32] = u64_le(wgt_base)
    d[32:40] = u64_le(out_base)
    d[40:44] = u32_le(Kt)

    for _ in range(2000):
        await Timer(1, unit="ns")
        if int(dut.desc_ready.value) == 1:
            break
        await RisingEdge(dut.clk)
    for i in range(64):
        dut.desc_bytes[i].value = d[i]
    dut.desc_valid.value = 1
    await RisingEdge(dut.clk)
    dut.desc_valid.value = 0

    total = rd_beats = 0
    for _ in range(100_000):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        total += 1
        if int(dut.rd_data_valid.value) == 1 and int(dut.rd_data_ready.value) == 1:
            rd_beats += 1
        if int(dut.done_pulse.value) == 1:
            return total, rd_beats, mem, out_base
    raise TimeoutError("done_pulse 없음")


def expected(A, B, Kt):
    return (A[:, :Kt].astype(np.int32) @ B[:Kt, :].astype(np.int32))


def read_result(mem, out_base):
    out = np.zeros((16, 16), dtype=np.int64)
    raw = mem.read_bytes(out_base, 16 * 16 * 4)
    for i in range(16):
        for j in range(16):
            v = int.from_bytes(raw[(i * 16 + j) * 4:(i * 16 + j) * 4 + 4],
                               "little", signed=True)
            out[i, j] = v
    return out


@cocotb.test()
async def test_result_matches_reference(dut):
    """256비트 읽기에서도 결과가 numpy 기준값과 일치한다."""
    await boot(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT_BYTES, wr_beat_bytes=16)
    rng = np.random.default_rng(1234)
    for Kt in (1, 2, 3, 8, 16):
        A = rng.integers(-128, 128, size=(16, Kt), dtype=np.int8)
        B = rng.integers(-128, 128, size=(Kt, 16), dtype=np.int8)
        _, beats, _, out_base = await run(dut, Kt, A, B, mem)
        got = read_result(mem, out_base)
        exp = expected(A, B, Kt)
        assert (got == exp).all(), \
            f"Kt={Kt}: 결과 불일치 (첫 오차 {np.argwhere(got != exp)[:1]})"
        dut._log.info(f"  Kt={Kt:>3}  읽기 비트 {beats:>4}  결과 일치")


@cocotb.test()
async def test_read_beats_halved(dut):
    """읽기 비트 수가 128비트 대비 절반이어야 한다.

    Kt=64 면 ACT 2048바이트 + WGT 2048바이트 = 4096바이트.
    32바이트 비트면 128 비트 (128비트 폭이면 256 비트였다).
    """
    await boot(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT_BYTES, wr_beat_bytes=16)
    rng = np.random.default_rng(7)
    Kt = 64
    A = rng.integers(-128, 128, size=(16, Kt), dtype=np.int8)
    B = rng.integers(-128, 128, size=(Kt, 16), dtype=np.int8)
    total, beats, _, out_base = await run(dut, Kt, A, B, mem)

    exp_beats = 2 * (Kt * 16) // RD_BEAT_BYTES     # ACT + WGT
    dut._log.info(f"  Kt={Kt}  총 {total} 사이클  읽기 비트 {beats} "
                  f"(기대 {exp_beats}, 128비트였다면 {exp_beats*2})")
    assert beats == exp_beats, f"읽기 비트 {beats}, 기대 {exp_beats}"

    got = read_result(mem, out_base)
    exp = expected(A, B, Kt)
    assert (got == exp).all(), "Kt=64 결과 불일치"
