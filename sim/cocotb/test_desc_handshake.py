"""test_desc_handshake.py — 디스크립터 포트가 AXI-Stream 규약을 지키는가.

규약: valid && ready 가 성립한 사이클에 데이터가 소비된다. 그 다음
사이클부터 생산자는 데이터를 바꿀 수 있다.

ctrl_fsm 은 desc_ready 를 ST_IDLE 에서 세우고 desc_bytes 를 ST_LATCH
(한 사이클 뒤)에 읽었다. 그래서 규약을 지키는 호스트가 핸드셰이크 직후
다음 디스크립터를 올리면, 이전 디스크립터 자리에 **다음** 것의 바이트가
들어갔다. 디스크립터를 하나씩만 보내면 드러나지 않고, 연속으로 밀어 넣을
때만 조용히 틀린다.

이 테스트는 **핸드셰이크 직후 desc_bytes 를 0xFF 로 덮어쓴다.** 코어가
그걸 읽으면 주소가 엉뚱해지고 결과가 안 맞는다.
"""
import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from dma_stub import SimpleMem, start_dma


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


def read_result(mem, out_base):
    raw = mem.read_bytes(out_base, 16 * 16 * 4)
    out = np.zeros((16, 16), dtype=np.int64)
    for i in range(16):
        for j in range(16):
            o = (i * 16 + j) * 4
            out[i, j] = int.from_bytes(raw[o:o + 4], "little", signed=True)
    return out


@cocotb.test()
async def test_bytes_consumed_on_handshake(dut):
    """핸드셰이크 직후 desc_bytes 를 쓰레기로 덮어도 결과가 맞아야 한다."""
    await boot(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False)
    rng = np.random.default_rng(4242)

    Kt = 16
    A = rng.integers(-128, 128, size=(16, Kt), dtype=np.int8)
    B = rng.integers(-128, 128, size=(Kt, 16), dtype=np.int8)
    act, wgt, out = 0x1000, 0x8000, 0x30000
    for k in range(Kt):
        mem.write_bytes(act + 16 * k,
                        bytes((int(A[i, k]) & 0xFF) for i in range(16)))
        mem.write_bytes(wgt + 16 * k,
                        bytes((int(B[k, j]) & 0xFF) for j in range(16)))

    d = make_desc(act, wgt, out, Kt)

    # ready 를 기다린다
    for _ in range(2000):
        await Timer(1, unit="ns")
        if int(dut.desc_ready.value) == 1:
            break
        await RisingEdge(dut.clk)
    else:
        raise TimeoutError("desc_ready 안 섬")

    for i in range(64):
        dut.desc_bytes[i].value = d[i]
    dut.desc_valid.value = 1

    # 핸드셰이크가 성립하는 엣지
    await Timer(1, unit="ns")
    assert int(dut.desc_ready.value) == 1, "ready 가 내려갔다"
    await RisingEdge(dut.clk)

    # 규약대로: 이 사이클부터 데이터는 우리 것이 아니다. 쓰레기를 넣는다.
    dut.desc_valid.value = 0
    for i in range(64):
        dut.desc_bytes[i].value = 0xFF
    dut._log.info("  핸드셰이크 직후 desc_bytes 를 0xFF 로 덮었다")

    for _ in range(100_000):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.done_pulse.value) == 1:
            break
    else:
        raise TimeoutError("done_pulse 없음")

    got = read_result(mem, out)
    exp = A[:, :Kt].astype(np.int32) @ B[:Kt, :].astype(np.int32)
    assert (got != 0).any(), \
        "결과가 전부 0 — 덮어쓴 바이트를 읽어서 엉뚱한 주소에 썼다"
    assert (got == exp).all(), \
        f"결과 불일치, 첫 오차 {np.argwhere(got != exp)[:1]}"
    dut._log.info("  핸드셰이크 사이클에 캡처했다 — 결과 정상")


@cocotb.test()
async def test_back_to_back_descriptors_land_correctly(dut):
    """연속 디스크립터가 각자의 out_addr 에 자기 결과를 쓴다.

    버그가 있을 때 정확히 이 모양으로 틀렸다 — 첫 번째 주소가 비고
    결과가 한 칸씩 밀렸다.
    """
    await boot(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False)
    rng = np.random.default_rng(5150)

    Kt = 8
    n = 6
    wgt = 0x8000
    B = rng.integers(-128, 128, size=(Kt, 16), dtype=np.int8)
    for k in range(Kt):
        mem.write_bytes(wgt + 16 * k,
                        bytes((int(B[k, j]) & 0xFF) for j in range(16)))

    As, outs = [], []
    for i in range(n):
        A = rng.integers(-128, 128, size=(16, Kt), dtype=np.int8)
        a_base = 0x1000 + 0x400 * i
        for k in range(Kt):
            mem.write_bytes(a_base + 16 * k,
                            bytes((int(A[r, k]) & 0xFF) for r in range(16)))
        As.append((A, a_base))
        outs.append(0x30000 + 0x1000 * i)

    issued = done_seen = 0
    for _ in range(500_000):
        want = issued < n
        if want:
            A, a_base = As[issued]
            d = make_desc(a_base, wgt, outs[issued], Kt,
                          flags=0x01 if issued > 0 else 0)
            for i in range(64):
                dut.desc_bytes[i].value = d[i]
        else:
            # 더 보낼 것이 없으면 쓰레기를 둔다
            for i in range(64):
                dut.desc_bytes[i].value = 0xFF
        dut.desc_valid.value = 1 if want else 0

        await Timer(1, unit="ns")
        accept = want and int(dut.desc_ready.value) == 1
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")

        if accept:
            issued += 1
        if int(dut.done_pulse.value) == 1:
            done_seen += 1
            if done_seen == n:
                break

    dut.desc_valid.value = 0
    assert done_seen == n, f"{done_seen}/{n} 완료"

    for i, ((A, _), ob) in enumerate(zip(As, outs)):
        got = read_result(mem, ob)
        exp = A[:, :Kt].astype(np.int32) @ B[:Kt, :].astype(np.int32)
        assert (got != 0).any(), f"#{i} 가 비었다 — 디스크립터가 한 칸 밀렸다"
        assert (got == exp).all(), \
            f"#{i} 결과 불일치, 첫 오차 {np.argwhere(got != exp)[:1]}"
    dut._log.info(f"  연속 {n}개가 각자 주소에 정확히 기록됐다")
