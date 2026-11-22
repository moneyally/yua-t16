"""test_ksplit.py — 축소 차원을 쪼개 누산하면 실제 LLM 레이어 모양이 돌아간다.

축소 차원은 디스크립터당 MAX_KT(256) 로 제한된다. 실제 LLM 레이어는
K=4096 이므로 쪼개 누산해야 한다. 그게 안 되면 이 가속기는 장난감이다.

flags bit2 (ACC_KEEP) / bit3 (NO_WB) 로 조각을 엮는다.

    조각 0      acc_keep=0  no_wb=1    누산기 클리어, 쓰기 없음
    조각 1..n-2 acc_keep=1  no_wb=1    이어받고 쓰기 없음
    조각 n-1    acc_keep=1  no_wb=0    이어받고 쓰기

단언하는 것:

  1) K 를 쪼갠 결과가 쪼개지 않은 결과와 **비트 단위로** 같다.
  2) 중간 조각은 쓰기를 전혀 하지 않는다 (쓰기 대역폭을 안 쓴다).
  3) MAX_KT 를 넘는 K (여기서는 1024) 가 실제로 돌아간다.
  4) weight-stationary 와 같이 쓸 수 있다.
"""
import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from dma_stub import SimpleMem, start_dma

FLAG_WGT_KEEP = 0x01
FLAG_WGT_INT4 = 0x02
FLAG_ACC_KEEP = 0x04
FLAG_NO_WB    = 0x08


def u32_le(x): return int(x).to_bytes(4, "little", signed=False)
def u64_le(x): return int(x).to_bytes(8, "little", signed=False)


def probe_config(dut):
    accs = len(dut.u_core.acc_out_flat)
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
    """디스크립터 하나. (읽기 비트, 쓰기 비트) 를 돌려준다."""
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

    rd = wr = 0
    for _ in range(400_000):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.rd_data_valid.value) and int(dut.rd_data_ready.value):
            rd += 1
        if int(dut.wr_data_valid.value) and int(dut.wr_data_ready.value):
            wr += 1
        if int(dut.done_pulse.value) == 1:
            return rd, wr
    raise TimeoutError("done_pulse 없음")


def write_act(mem, base, A, k0, kt, N):
    for k in range(kt):
        mem.write_bytes(base + N * k,
                        bytes((int(A[i, k0 + k]) & 0xFF) for i in range(N)))


def write_wgt(mem, base, B, k0, kt, N):
    for k in range(kt):
        mem.write_bytes(base + N * k,
                        bytes((int(B[k0 + k, j]) & 0xFF) for j in range(N)))


def read_result(mem, base, N):
    raw = mem.read_bytes(base, N * N * 4)
    out = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(N):
            o = (i * N + j) * 4
            out[i, j] = int.from_bytes(raw[o:o + 4], "little", signed=True)
    return out


async def run_ksplit(dut, mem, A, B, K, chunk, N, act0, wgt0, out,
                     wgt_keep_after_first=False):
    """K 를 chunk 단위로 쪼개 누산한다. (쓰기 비트 합, 조각 수) 를 돌려준다."""
    n_chunk = (K + chunk - 1) // chunk
    wr_total = 0
    for c in range(n_chunk):
        k0 = c * chunk
        kt = min(chunk, K - k0)
        a_base = act0 + 0x8000 * c
        w_base = wgt0 + 0x8000 * c
        write_act(mem, a_base, A, k0, kt, N)
        write_wgt(mem, w_base, B, k0, kt, N)

        flags = 0
        if c > 0:
            flags |= FLAG_ACC_KEEP
        if c < n_chunk - 1:
            flags |= FLAG_NO_WB
        if wgt_keep_after_first and c > 0:
            flags |= FLAG_WGT_KEEP

        rd, wr = await push(dut, make_desc(a_base, w_base, out, kt, flags))
        wr_total += wr
        if c < n_chunk - 1:
            assert wr == 0, \
                f"조각 {c} 가 쓰기를 {wr} 비트 했다 — NO_WB 가 안 먹었다"
    return wr_total, n_chunk


@cocotb.test()
async def test_ksplit_equals_unsplit(dut):
    """K 를 쪼갠 결과가 쪼개지 않은 결과와 같다."""
    await boot(dut)
    N, RD_BEAT, WR_BEAT = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(9090)

    K = 64
    A = rng.integers(-128, 128, size=(N, K), dtype=np.int8)
    B = rng.integers(-128, 128, size=(K, N), dtype=np.int8)
    exp = A.astype(np.int32) @ B.astype(np.int32)

    # 쪼개지 않은 기준판
    write_act(mem, 0x1000, A, 0, K, N)
    write_wgt(mem, 0x40000, B, 0, K, N)
    await push(dut, make_desc(0x1000, 0x40000, 0x80000, K))
    base = read_result(mem, 0x80000, N)
    assert (base == exp).all(), "기준판이 틀렸다"

    # 여러 분할 크기로 쪼갠다
    for chunk in (32, 16, 8, 7):
        wr, n_chunk = await run_ksplit(
            dut, mem, A, B, K, chunk, N, 0x100000, 0x200000, 0x300000)
        got = read_result(mem, 0x300000, N)
        bad = np.argwhere(got != base)
        assert len(bad) == 0, (
            f"chunk={chunk} ({n_chunk}조각): {len(bad)}/{N*N} 불일치. "
            f"첫 오차 {bad[:1]} got={got[tuple(bad[0])]} "
            f"exp={base[tuple(bad[0])]}")
        assert wr == (N * N * 4) // WR_BEAT, \
            f"chunk={chunk}: 쓰기 {wr} 비트 — 마지막 조각만 써야 한다"
        dut._log.info(f"  K={K} chunk={chunk:>3} → {n_chunk:>2}조각  "
                      f"쓰기 {wr} 비트 (마지막만)  결과 일치")


@cocotb.test()
async def test_k_beyond_max_kt(dut):
    """MAX_KT 를 넘는 K 가 실제로 돌아간다 — 실제 LLM 레이어 모양."""
    await boot(dut)
    N, RD_BEAT, WR_BEAT = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(1357)

    max_kt = len(dut.u_core.u_act.mem) * (RD_BEAT // N) // 2
    K = 4 * max_kt           # MAX_KT 를 4배 넘는다
    A = rng.integers(-128, 128, size=(N, K), dtype=np.int8)
    B = rng.integers(-128, 128, size=(K, N), dtype=np.int8)
    exp = A.astype(np.int32) @ B.astype(np.int32)

    wr, n_chunk = await run_ksplit(
        dut, mem, A, B, K, max_kt, N, 0x100000, 0x400000, 0x800000)
    got = read_result(mem, 0x800000, N)
    bad = np.argwhere(got != exp)
    assert len(bad) == 0, (
        f"K={K}: {len(bad)}/{N*N} 불일치. 첫 오차 {bad[:1]} "
        f"got={got[tuple(bad[0])]} exp={exp[tuple(bad[0])]}")
    assert wr == (N * N * 4) // WR_BEAT

    # 32비트 누산기가 넘치지 않는지도 같이 본다
    assert np.abs(exp).max() < 2 ** 31, "기준값이 int32 를 넘었다"
    dut._log.info(f"  MAX_KT={max_kt}, K={K} ({n_chunk}조각)  "
                  f"최대 누산값 {np.abs(exp).max():,}  결과 일치")
    dut._log.info(f"  쓰기 {wr} 비트 — {n_chunk}조각 중 마지막 1개만")


@cocotb.test()
async def test_ksplit_with_weight_stationary(dut):
    """K 분할과 weight-stationary 를 같이 쓴다.

    두 플래그가 서로 다른 자원을 건드리므로(누산기 vs 가중치 SRAM)
    같이 써도 간섭하지 않아야 한다.
    """
    await boot(dut)
    N, RD_BEAT, WR_BEAT = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(2468)

    # 모든 조각이 같은 가중치를 쓰는 경우 (B 가 K 방향으로 반복)
    K, chunk = 64, 16
    A = rng.integers(-128, 128, size=(N, K), dtype=np.int8)
    Bc = rng.integers(-128, 128, size=(chunk, N), dtype=np.int8)
    B = np.tile(Bc, (K // chunk, 1))
    exp = A.astype(np.int32) @ B.astype(np.int32)

    n_chunk = K // chunk
    for c in range(n_chunk):
        a_base = 0x100000 + 0x8000 * c
        write_act(mem, a_base, A, c * chunk, chunk, N)
        if c == 0:
            write_wgt(mem, 0x400000, Bc, 0, chunk, N)
        flags = 0
        if c > 0:
            flags |= FLAG_ACC_KEEP | FLAG_WGT_KEEP
        if c < n_chunk - 1:
            flags |= FLAG_NO_WB
        rd, wr = await push(
            dut, make_desc(a_base, 0x400000, 0x900000, chunk, flags))
        if c < n_chunk - 1:
            assert wr == 0
        if c > 0:
            # 활성값만 읽어야 한다
            assert rd == chunk * N // RD_BEAT, \
                f"조각 {c}: 읽기 {rd} 비트, 활성값만이면 {chunk*N//RD_BEAT}"

    got = read_result(mem, 0x900000, N)
    assert (got == exp).all(), \
        f"불일치, 첫 오차 {np.argwhere(got != exp)[:1]}"
    assert int(dut.perf_wgt_load.value) == 1, "가중치를 두 번 적재했다"
    assert int(dut.perf_wgt_reuse.value) == n_chunk - 1
    dut._log.info(f"  K={K} {n_chunk}조각, 가중치 적재 1회, "
                  f"중간 조각 쓰기 0 — 결과 일치")
