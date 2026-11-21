"""test_weight_stationary.py — 가중치 재적재를 생략하면 트래픽이 절반이 되는가.

LLM 추론에서 한 레이어의 가중치는 그 레이어를 지나는 모든 활성값 타일에
대해 같다. 디스크립터마다 다시 읽으면 DRAM 트래픽의 절반이 낭비다.

디스크립터 바이트 1 bit0 (FLAG_WGT_KEEP) 이 서면 로드 FSM 이 WGT 단계를
건너뛴다. 이 테스트는 두 가지를 단언한다.

  1) 건너뛰어도 결과가 numpy 기준값과 같다 (상주 가중치가 살아 있다).
  2) 읽기 비트 수가 실제로 절반으로 준다.

그리고 첫 디스크립터가 wgt_keep=1 로 와도 안전해야 한다 — 가중치를
한 번도 적재하지 않았으면 플래그를 무시하고 적재해야 한다.
"""
import os
import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from dma_stub import SimpleMem, start_dma

RD_BEAT = int(os.environ.get("RD_BEAT_BYTES", "16"))
FLAG_WGT_KEEP = 0x01


def u32_le(x): return int(x).to_bytes(4, "little", signed=False)
def u64_le(x): return int(x).to_bytes(8, "little", signed=False)


def make_desc(act, wgt, out, Kt, flags=0):
    d = bytearray(64)
    d[0:1] = bytes([0x02])          # OP_GEMM_INT8
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
    """디스크립터 하나를 밀어 넣고, 완료까지 돌면서 읽기 비트를 센다."""
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

    beats = 0
    for _ in range(100_000):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.rd_data_valid.value) and int(dut.rd_data_ready.value):
            beats += 1
        if int(dut.done_pulse.value) == 1:
            return beats
    raise TimeoutError("done_pulse 없음")


def write_operands(mem, act_base, wgt_base, A, B, Kt):
    for k in range(Kt):
        mem.write_bytes(act_base + 16 * k,
                        bytes((int(A[i, k]) & 0xFF) for i in range(16)))
        mem.write_bytes(wgt_base + 16 * k,
                        bytes((int(B[k, j]) & 0xFF) for j in range(16)))


def read_result(mem, out_base):
    raw = mem.read_bytes(out_base, 16 * 16 * 4)
    out = np.zeros((16, 16), dtype=np.int64)
    for i in range(16):
        for j in range(16):
            o = (i * 16 + j) * 4
            out[i, j] = int.from_bytes(raw[o:o + 4], "little", signed=True)
    return out


def expect(A, B, Kt):
    return A[:, :Kt].astype(np.int32) @ B[:Kt, :].astype(np.int32)


@cocotb.test()
async def test_resident_weights_give_same_result(dut):
    """가중치를 한 번 적재하고 N 개 디스크립터가 재사용해도 결과가 같다.

    같은 가중치 B 에 서로 다른 활성값 A0..A3 을 통과시킨다. 2번째부터
    FLAG_WGT_KEEP 을 세우고, 네 결과 전부 numpy 와 맞는지 본다.
    """
    await boot(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=16)
    rng = np.random.default_rng(99)

    Kt = 16
    B = rng.integers(-128, 128, size=(Kt, 16), dtype=np.int8)
    wgt_base = 0x8000
    As = [rng.integers(-128, 128, size=(16, Kt), dtype=np.int8) for _ in range(4)]

    beats = []
    for n, A in enumerate(As):
        act_base = 0x1000 + 0x1000 * n
        out_base = 0x20000 + 0x1000 * n
        write_operands(mem, act_base, wgt_base, A, B, Kt)
        flags = FLAG_WGT_KEEP if n > 0 else 0
        b = await push(dut, make_desc(act_base, wgt_base, out_base, Kt, flags))
        beats.append(b)

        got = read_result(mem, out_base)
        exp = expect(A, B, Kt)
        assert (got == exp).all(), \
            f"#{n} (flags={flags:#02x}): 결과 불일치, 첫 오차 {np.argwhere(got != exp)[:1]}"

    dut._log.info(f"  읽기 비트: {beats}  (첫 번째만 가중치 적재)")
    # 첫 번째는 ACT+WGT, 나머지는 ACT 만 → 정확히 절반
    assert beats[0] == 2 * beats[1], \
        f"재사용 디스크립터가 절반이 아니다: {beats[0]} vs {beats[1]}"
    for b in beats[1:]:
        assert b == beats[1], f"재사용 비트 수가 들쭉날쭉: {beats}"

    total = sum(beats)
    naive = beats[0] * len(beats)
    dut._log.info(f"  총 {total} 비트 vs 매번 재적재 {naive} 비트 "
                  f"→ {naive/total:.2f}× 절감")

    reuse = int(dut.perf_wgt_reuse.value)
    load = int(dut.perf_wgt_load.value)
    dut._log.info(f"  계측 카운터: 재사용 {reuse}, 적재 {load}")
    assert (reuse, load) == (3, 1), f"카운터가 틀렸다: 재사용 {reuse}, 적재 {load}"


@cocotb.test()
async def test_first_descriptor_cannot_skip(dut):
    """첫 디스크립터가 wgt_keep=1 로 와도 가중치를 적재해야 한다.

    적재 전 SRAM 은 0 이므로, 플래그를 그대로 따르면 결과가 0 이 된다.
    wgt_valid_r 가 그것을 막는다.
    """
    await boot(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=16)
    rng = np.random.default_rng(5)

    Kt = 8
    A = rng.integers(1, 128, size=(16, Kt), dtype=np.int8)
    B = rng.integers(1, 128, size=(Kt, 16), dtype=np.int8)
    write_operands(mem, 0x1000, 0x8000, A, B, Kt)

    # 플래그를 세운 채로 첫 디스크립터를 보낸다
    await push(dut, make_desc(0x1000, 0x8000, 0x3000, Kt, FLAG_WGT_KEEP))
    got = read_result(mem, 0x3000)
    exp = expect(A, B, Kt)
    assert (got != 0).any(), "결과가 전부 0 — 가중치를 적재하지 않았다"
    assert (got == exp).all(), "첫 디스크립터 결과 불일치"

    assert int(dut.perf_wgt_load.value) == 1, "적재 카운터가 1 이 아니다"
    assert int(dut.perf_wgt_reuse.value) == 0, "플래그를 그대로 따랐다"
    dut._log.info("  첫 디스크립터는 플래그를 무시하고 적재했다")


@cocotb.test()
async def test_weight_change_midstream(dut):
    """가중치를 중간에 바꿔도 각 디스크립터가 자기 가중치를 쓴다.

    B0 적재 → B0 재사용 → B1 적재 → B1 재사용. 가중치 뱅크 포인터가
    실제 적재 때만 전진하므로, 재사용 디스크립터는 직전에 적재된 뱅크를
    가리켜야 한다. 포인터가 틀리면 이 테스트가 잡는다.
    """
    await boot(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=16)
    rng = np.random.default_rng(777)

    Kt = 16
    B0 = rng.integers(-128, 128, size=(Kt, 16), dtype=np.int8)
    B1 = rng.integers(-128, 128, size=(Kt, 16), dtype=np.int8)
    assert (B0 != B1).any()

    plan = [(B0, 0x8000, 0), (B0, 0x8000, FLAG_WGT_KEEP),
            (B1, 0x9000, 0), (B1, 0x9000, FLAG_WGT_KEEP)]

    for n, (Bm, wbase, flags) in enumerate(plan):
        A = rng.integers(-128, 128, size=(16, Kt), dtype=np.int8)
        act_base = 0x1000 + 0x1000 * n
        out_base = 0x30000 + 0x1000 * n
        write_operands(mem, act_base, wbase, A, Bm, Kt)
        await push(dut, make_desc(act_base, wbase, out_base, Kt, flags))
        got = read_result(mem, out_base)
        exp = expect(A, Bm, Kt)
        assert (got == exp).all(), \
            f"#{n} (flags={flags:#02x}): 가중치 뱅크가 틀렸다, " \
            f"첫 오차 {np.argwhere(got != exp)[:1]}"
        dut._log.info(f"  #{n} flags={flags:#02x} 정상")

    assert int(dut.perf_wgt_load.value) == 2
    assert int(dut.perf_wgt_reuse.value) == 2
