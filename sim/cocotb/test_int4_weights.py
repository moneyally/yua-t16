"""test_int4_weights.py — INT4 가중치가 맞는 값을 내고 트래픽을 절반으로 줄이는가.

LLM 단일 스트림 디코드는 토큰마다 가중치 전체를 한 번 읽는 데 묶인다.
가중치를 INT4 로 두면 그 바이트가 절반이 되고, 메모리에 묶인 경로에서
토큰 속도가 그대로 두 배가 된다. 현대 서빙이 전부 4비트 가중치를 쓰는
이유다.

MAC 배열은 그대로 8비트를 받는다. 니블을 2의 보수로 부호확장해 넣는다.
깨지기 쉬운 지점이 둘이다.

  1) **부호확장** — INT4 는 [-8, 7] 이다. 0xF 는 -1 이어야 하고 15 가
     되면 안 된다.
  2) **니블 순서** — 바이트 하나에 K 인덱스가 연속한 두 값이 들어간다.
     하위 니블이 낮은 인덱스여야 한다.

둘 다 numpy 기준값과 대조하면 잡힌다. 특히 음수 가중치를 반드시 넣는다 —
전부 양수면 부호확장 버그가 통과한다.
"""
import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from dma_stub import SimpleMem, start_dma

FLAG_WGT_KEEP = 0x01
FLAG_WGT_INT4 = 0x02


def u32_le(x): return int(x).to_bytes(4, "little", signed=False)
def u64_le(x): return int(x).to_bytes(8, "little", signed=False)


def probe_config(dut):
    accs = len(dut.u_core.acc_out_lat)
    n = int(round(accs ** 0.5))
    assert n * n == accs
    return n, len(dut.rd_data) // 8, len(dut.wr_data) // 8


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
    for _ in range(8000):
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

    rd = 0
    for _ in range(200_000):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.rd_data_valid.value) and int(dut.rd_data_ready.value):
            rd += 1
        if int(dut.done_pulse.value) == 1:
            return rd
    raise TimeoutError("done_pulse 없음")


def pack_int4_weights(B, Kt, N):
    """B(Kt×N) 를 K-step 당 N 니블 = N/2 바이트로 패킹한다.

    바이트 b 의 하위 니블이 열 2b, 상위 니블이 열 2b+1 이다.
    """
    out = bytearray()
    for k in range(Kt):
        for b in range(N // 2):
            lo = int(B[k, 2 * b]) & 0x0F
            hi = int(B[k, 2 * b + 1]) & 0x0F
            out.append(lo | (hi << 4))
    return bytes(out)


def read_result(mem, out_base, N):
    raw = mem.read_bytes(out_base, N * N * 4)
    out = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(N):
            o = (i * N + j) * 4
            out[i, j] = int.from_bytes(raw[o:o + 4], "little", signed=True)
    return out


@cocotb.test()
async def test_int4_matches_reference(dut):
    """INT4 가중치 결과가 numpy 기준값과 일치하고 읽기 비트가 줄어든다."""
    await boot(dut)
    N, RD_BEAT, WR_BEAT = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(606)

    act, wgt, out = 0x1000, 0x40000, 0x80000
    for Kt in (1, 2, 3, 16, 64):
        A = rng.integers(-128, 128, size=(N, Kt), dtype=np.int8)
        # INT4 범위 [-8, 7]. 음수를 반드시 포함시킨다.
        B = rng.integers(-8, 8, size=(Kt, N)).astype(np.int8)
        assert (B < 0).any(), "음수 가중치가 없으면 부호확장 버그를 놓친다"

        for k in range(Kt):
            mem.write_bytes(act + N * k,
                            bytes((int(A[i, k]) & 0xFF) for i in range(N)))
        mem.write_bytes(wgt, pack_int4_weights(B, Kt, N))

        rd = await push(dut, make_desc(act, wgt, out, Kt, FLAG_WGT_INT4))
        got = read_result(mem, out, N)
        exp = A[:, :Kt].astype(np.int32) @ B[:Kt, :].astype(np.int32)
        bad = np.argwhere(got != exp)
        if len(bad):
            i, j = bad[0]
            raise AssertionError(
                f"Kt={Kt}: {len(bad)}/{N*N} 불일치. [{i}][{j}] "
                f"got={got[i,j]} exp={exp[i,j]}  "
                f"(부호확장이면 음수 가중치에서만, 니블 순서면 열 짝/홀에서 틀린다)")

        act_beats = -(-Kt * N // RD_BEAT)              # 올림
        wgt_beats = -(-(Kt * N // 2) // RD_BEAT)
        dut._log.info(f"  Kt={Kt:>3}  읽기 {rd:>3} 비트 "
                      f"(활성값 {act_beats} + 가중치 {wgt_beats})  결과 일치")


@cocotb.test()
async def test_int4_halves_weight_traffic(dut):
    """같은 Kt 에서 INT4 가중치 비트가 INT8 의 절반이다."""
    await boot(dut)
    N, RD_BEAT, WR_BEAT = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(707)

    Kt = 64
    A = rng.integers(-128, 128, size=(N, Kt), dtype=np.int8)
    B = rng.integers(-8, 8, size=(Kt, N)).astype(np.int8)
    act = 0x1000
    for k in range(Kt):
        mem.write_bytes(act + N * k,
                        bytes((int(A[i, k]) & 0xFF) for i in range(N)))

    # INT8 판: 같은 값을 바이트로 펼친다 → 결과가 동일해야 한다
    w8 = 0x40000
    for k in range(Kt):
        mem.write_bytes(w8 + N * k,
                        bytes((int(B[k, j]) & 0xFF) for j in range(N)))
    rd8 = await push(dut, make_desc(act, w8, 0x80000, Kt))
    got8 = read_result(mem, 0x80000, N)

    # INT4 판
    w4 = 0x50000
    mem.write_bytes(w4, pack_int4_weights(B, Kt, N))
    rd4 = await push(dut, make_desc(act, w4, 0x90000, Kt, FLAG_WGT_INT4))
    got4 = read_result(mem, 0x90000, N)

    exp = A.astype(np.int32) @ B.astype(np.int32)
    assert (got8 == exp).all(), "INT8 기준판이 틀렸다"
    assert (got4 == exp).all(), "INT4 결과가 INT8 과 다르다"

    act_beats = Kt * N // RD_BEAT
    w8_beats = rd8 - act_beats
    w4_beats = rd4 - act_beats
    dut._log.info(f"  활성값 {act_beats} 비트 공통. "
                  f"가중치 INT8 {w8_beats} → INT4 {w4_beats} 비트")
    assert w4_beats * 2 == w8_beats, \
        f"가중치 비트가 절반이 아니다: {w8_beats} → {w4_beats}"
    dut._log.info(f"  전체 읽기 {rd8} → {rd4} 비트 "
                  f"({rd8/rd4:.2f}× 절감, 결과 동일)")


@cocotb.test()
async def test_int4_resident_across_descriptors(dut):
    """INT4 포맷이 상주 가중치에 딸려 간다.

    포맷은 디스크립터의 속성이 아니라 **적재된 가중치의 속성**이다.
    건너뛴 디스크립터가 flags 에 INT4 를 안 적어도 상주 가중치가 INT4 면
    INT4 로 읽어야 한다. 그래서 w4 를 가중치 뱅크마다 보관한다.
    """
    await boot(dut)
    N, RD_BEAT, WR_BEAT = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(808)

    Kt = 16
    B = rng.integers(-8, 8, size=(Kt, N)).astype(np.int8)
    wgt = 0x40000
    mem.write_bytes(wgt, pack_int4_weights(B, Kt, N))

    for n in range(4):
        A = rng.integers(-128, 128, size=(N, Kt), dtype=np.int8)
        a_base = 0x1000 + 0x2000 * n
        for k in range(Kt):
            mem.write_bytes(a_base + N * k,
                            bytes((int(A[i, k]) & 0xFF) for i in range(N)))
        ob = 0x80000 + 0x8000 * n
        # 첫 개만 INT4 로 적재. 나머지는 KEEP 만 세우고 INT4 는 안 적는다.
        flags = FLAG_WGT_INT4 if n == 0 else FLAG_WGT_KEEP
        await push(dut, make_desc(a_base, wgt, ob, Kt, flags))

        got = read_result(mem, ob, N)
        exp = A.astype(np.int32) @ B.astype(np.int32)
        assert (got == exp).all(), \
            f"#{n} (flags={flags:#02x}): 상주 가중치의 INT4 포맷을 잃었다, " \
            f"첫 오차 {np.argwhere(got != exp)[:1]}"
        dut._log.info(f"  #{n} flags={flags:#02x} 정상")

    assert int(dut.perf_wgt_load.value) == 1
    assert int(dut.perf_wgt_reuse.value) == 3
