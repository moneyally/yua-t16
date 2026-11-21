"""test_wr256.py — WR_DATA_W=256 에서 결과가 같고 쓰기 비트가 절반인가.

디스크립터당 출력은 타일 크기의 제곱(16×16 = 256개 × 4바이트 = 1,024바이트)
이고 폭과 무관하다. 폭을 넓히면 비트 수가 절반이 된다.

쓰기 폭을 넓히는 이유는 타일 확장 때문이다. 연산은 Kt 사이클인데 출력은
N² 라서, N 을 키우면 쓰기가 묶는 쪽이 된다. 먼저 폭을 맞춰 둔다.

리틀엔디안 워드 순서가 폭에 따라 깨지기 쉬운 지점이다 — 256비트 비트
하나에 누산기 8개가 들어가고, 하위 32비트가 낮은 인덱스여야 한다.
결과를 numpy 와 대조하면 그게 바로 잡힌다.
"""
import os
import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from dma_stub import SimpleMem, start_dma

def probe_beats(dut):
    """비트 바이트 수를 포트 폭에서 읽는다 (환경변수 불일치 방지)."""
    return len(dut.rd_data) // 8, len(dut.wr_data) // 8


def u32_le(x): return int(x).to_bytes(4, "little", signed=False)
def u64_le(x): return int(x).to_bytes(8, "little", signed=False)


def make_desc(act, wgt, out, Kt, flags=0):
    d = bytearray(64)
    d[0:1] = bytes([0x02])
    d[1:2] = bytes([flags])
    d[16:24] = u64_le(act)
    d[24:32] = u64_le(wgt)
    d[32:40] = u64_le(out)
    d[40:44] = u32_le(Kt)
    return d


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


async def push(dut, desc):
    for _ in range(4000):
        await Timer(1, unit="ns")
        if int(dut.desc_ready.value) == 1:
            break
        await RisingEdge(dut.clk)
    else:
        raise TimeoutError("desc_ready 안 섬")
    for i in range(64):
        dut.desc_bytes[i].value = desc[i]
    dut.desc_valid.value = 1
    await RisingEdge(dut.clk)
    dut.desc_valid.value = 0

    wr_beats = 0
    for _ in range(100_000):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.wr_data_valid.value) and int(dut.wr_data_ready.value):
            wr_beats += 1
        if int(dut.done_pulse.value) == 1:
            return wr_beats
    raise TimeoutError("done_pulse 없음")


def read_result(mem, out_base):
    raw = mem.read_bytes(out_base, 16 * 16 * 4)
    out = np.zeros((16, 16), dtype=np.int64)
    for i in range(16):
        for j in range(16):
            o = (i * 16 + j) * 4
            out[i, j] = int.from_bytes(raw[o:o + 4], "little", signed=True)
    return out


@cocotb.test()
async def test_result_matches_reference(dut):
    """폭을 넓혀도 결과가 numpy 기준값과 같다 (워드 순서 포함)."""
    await boot(dut)
    RD_BEAT, WR_BEAT = probe_beats(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(31337)

    for Kt in (1, 4, 16, 64):
        A = rng.integers(-128, 128, size=(16, Kt), dtype=np.int8)
        B = rng.integers(-128, 128, size=(Kt, 16), dtype=np.int8)
        act, wgt, out = 0x1000, 0x8000, 0x30000
        for k in range(Kt):
            mem.write_bytes(act + 16 * k,
                            bytes((int(A[i, k]) & 0xFF) for i in range(16)))
            mem.write_bytes(wgt + 16 * k,
                            bytes((int(B[k, j]) & 0xFF) for j in range(16)))

        beats = await push(dut, make_desc(act, wgt, out, Kt))
        got = read_result(mem, out)
        exp = A[:, :Kt].astype(np.int32) @ B[:Kt, :].astype(np.int32)
        assert (got == exp).all(), \
            f"Kt={Kt}: 결과 불일치 (워드 순서?), 첫 오차 {np.argwhere(got != exp)[:1]}"

        exp_beats = (16 * 16 * 4) // WR_BEAT
        assert beats == exp_beats, f"Kt={Kt}: 쓰기 비트 {beats}, 기대 {exp_beats}"
        dut._log.info(f"  Kt={Kt:>3}  쓰기 비트 {beats:>3}  결과 일치")


@cocotb.test()
async def test_write_bytes_counted_right(dut):
    """perf_bytes 가 폭에 맞춰 바이트를 센다 (비트 수가 아니라)."""
    await boot(dut)
    RD_BEAT, WR_BEAT = probe_beats(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(8)
    Kt = 16
    A = rng.integers(-128, 128, size=(16, Kt), dtype=np.int8)
    B = rng.integers(-128, 128, size=(Kt, 16), dtype=np.int8)
    for k in range(Kt):
        mem.write_bytes(0x1000 + 16 * k,
                        bytes((int(A[i, k]) & 0xFF) for i in range(16)))
        mem.write_bytes(0x8000 + 16 * k,
                        bytes((int(B[k, j]) & 0xFF) for j in range(16)))
    await push(dut, make_desc(0x1000, 0x8000, 0x30000, Kt))

    rd_bytes = 2 * Kt * 16          # ACT + WGT
    wr_bytes = 16 * 16 * 4          # 1,024
    got = int(dut.perf_bytes.value)
    dut._log.info(f"  perf_bytes={got} (기대 읽기 {rd_bytes} + 쓰기 {wr_bytes} "
                  f"= {rd_bytes + wr_bytes})")
    assert got == rd_bytes + wr_bytes, \
        f"perf_bytes {got}, 기대 {rd_bytes + wr_bytes}"
