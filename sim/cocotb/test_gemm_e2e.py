"""test_gemm_e2e.py — 실제 (M, K, N) 행렬곱을 호스트 타일러로 쪼개 RTL 에서 돌린다.

여기까지가 "가속기" 라고 부를 수 있는 최소 조건이다. 타일 하나가 맞는
것과 임의 크기 행렬곱이 맞는 것은 다른 문제다 — 타일 좌표, 주소 계산,
플래그 순서, 코어 간 열 분할이 전부 맞아야 한다.

tools/gemm_tiler.py 가 만든 디스크립터 수열을 그대로 쓴다. 타일러가
RTL 과 다른 가정을 하고 있으면 여기서 깨진다.

검증:
  1) 전체 C[M,N] 이 numpy 기준값과 **비트 단위로** 일치한다.
  2) 타일러가 보고한 가중치 적재 횟수가 하드웨어 계측과 일치한다.
  3) weight-stationary 가 실제로 DRAM 읽기를 줄인다 (요청 수로 확인).
"""
import sys
from pathlib import Path

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from dma_stub import SimpleMem, start_dma

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.gemm_tiler import GemmShape, TileConfig, tile_gemm  # noqa: E402


def u32_le(x): return int(x).to_bytes(4, "little", signed=False)
def u64_le(x): return int(x).to_bytes(8, "little", signed=False)


def probe_config(dut):
    cores = len(dut.CORE)
    accs = len(dut.CORE[0].u_core.acc_out_lat)
    n = int(round(accs ** 0.5))
    assert n * n == accs
    max_kt = len(dut.CORE[0].u_core.u_act.mem) * (len(dut.rd_data) // 8 // n) // 2
    return cores, n, len(dut.rd_data) // 8, len(dut.wr_data) // 8, max_kt


def make_desc(act, wgt, out, kt, wgt_stride, out_stride, flags):
    d = bytearray(64)
    d[0:1] = bytes([0x02])
    d[1:2] = bytes([flags])
    d[16:24] = u64_le(act)
    d[24:32] = u64_le(wgt)
    d[32:40] = u64_le(out)
    d[40:44] = u32_le(kt)
    d[44:48] = u32_le(wgt_stride)
    d[52:56] = u32_le(out_stride)
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
    """디스크립터 하나. 읽기 요청 수를 돌려준다."""
    for _ in range(20_000):
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

    reqs = 0
    for _ in range(500_000):
        await Timer(1, unit="ns")
        if int(dut.rd_req_valid.value) and int(dut.rd_req_ready.value):
            reqs += 1
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.done_pulse.value) == 1:
            return reqs
    raise TimeoutError("done_pulse 없음")


@cocotb.test()
async def test_full_gemm_matches_numpy(dut):
    """임의 (M, K, N) 행렬곱 전체가 numpy 와 일치한다."""
    await boot(dut)
    CORES, T, RD_BEAT, WR_BEAT, MAX_KT = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(606060)

    # 행 타일 2개 × 열 밴드 2개가 나오도록 고른다 — weight-stationary 가
    # 실제로 작동하는 최소 모양이다 (행 타일이 1개면 재사용할 상대가 없다).
    M = 2 * T
    N = 2 * T * CORES
    K = min(MAX_KT, 128)

    shape = GemmShape(M=M, K=K, N=N)
    cfg = TileConfig(tile_n=T, cores=CORES, max_kt=MAX_KT, wgt_bits=8)
    plan = tile_gemm(shape, cfg)

    dut._log.info(f"  CORES={CORES} T={T} MAX_KT={MAX_KT}")
    dut._log.info(f"  GEMM M={M} K={K} N={N} → 디스크립터 {len(plan.descs)}개, "
                  f"가중치 적재 {plan.weight_loads}개")
    dut._log.info(f"  전략: {plan.strategy}")

    A = rng.integers(-128, 128, size=(M, K), dtype=np.int8)
    B = rng.integers(-128, 128, size=(K, N), dtype=np.int8)
    exp = A.astype(np.int32) @ B.astype(np.int32)

    # ── 메모리 레이아웃 ──────────────────────────────────────────
    # 활성값: 행 타일 ib 마다 A[ib*T:(ib+1)*T, :] 를 K-step 당 T 바이트로.
    ACT0 = 0x100000
    act_tile_bytes = K * T
    for ib in range(-(-M // T)):
        base = ACT0 + act_tile_bytes * ib
        for k in range(K):
            mem.write_bytes(base + T * k, bytes(
                (int(A[ib * T + i, k]) & 0xFF) if ib * T + i < M else 0
                for i in range(T)))

    # 가중치: 열 밴드 jb, 코어 c 마다 B[:, jb*T*CORES + c*T : +T].
    # 코어 간 간격이 wgt_stride 다.
    WGT0 = 0x400000
    wgt_slice_bytes = K * T
    wgt_band_bytes = wgt_slice_bytes * CORES
    for jb in range(-(-N // (T * CORES))):
        for c in range(CORES):
            col0 = jb * T * CORES + c * T
            base = WGT0 + wgt_band_bytes * jb + wgt_slice_bytes * c
            for k in range(K):
                mem.write_bytes(base + T * k, bytes(
                    (int(B[k, col0 + j]) & 0xFF) if col0 + j < N else 0
                    for j in range(T)))

    # 출력: 디스크립터마다 T × (T*CORES), 코어 간 간격 T*T*4
    OUT0 = 0x800000
    out_desc_bytes = T * T * 4 * CORES

    # ── 실행 ─────────────────────────────────────────────────────
    total_reqs = 0
    for n, d in enumerate(plan.descs):
        act = ACT0 + act_tile_bytes * d.ib
        wgt = WGT0 + wgt_band_bytes * d.jb
        out = OUT0 + out_desc_bytes * n
        total_reqs += await push(dut, make_desc(
            act, wgt, out, d.kt, wgt_slice_bytes, T * T * 4, d.flags))

    # ── 결과 조립 ────────────────────────────────────────────────
    got = np.zeros((M, N), dtype=np.int64)
    for n, d in enumerate(plan.descs):
        base = OUT0 + out_desc_bytes * n
        for c in range(CORES):
            raw = mem.read_bytes(base + T * T * 4 * c, T * T * 4)
            for i in range(T):
                r = d.ib * T + i
                if r >= M:
                    continue
                for j in range(T):
                    col = d.jb * T * CORES + c * T + j
                    if col >= N:
                        continue
                    o = (i * T + j) * 4
                    got[r, col] = int.from_bytes(raw[o:o + 4], "little",
                                                 signed=True)

    bad = np.argwhere(got != exp)
    if len(bad):
        i, j = bad[0]
        raise AssertionError(
            f"{len(bad)}/{M*N} 불일치. [{i}][{j}] got={got[i,j]} "
            f"exp={exp[i,j]}  (행타일 {i//T}, 열밴드 {j//(T*CORES)}, "
            f"코어 {(j%(T*CORES))//T})")
    dut._log.info(f"  C[{M},{N}] 전체 {M*N:,}개 일치")

    # ── 계측 대조 ────────────────────────────────────────────────
    hw_loads = int(dut.perf_wgt_load.value)
    hw_reuse = int(dut.perf_wgt_reuse.value)
    # 하드웨어는 코어마다 센다
    assert hw_loads == plan.weight_loads * CORES, (
        f"가중치 적재: 하드웨어 {hw_loads}, 타일러 예상 "
        f"{plan.weight_loads}×{CORES}={plan.weight_loads*CORES}")
    assert hw_reuse == (len(plan.descs) - plan.weight_loads) * CORES
    dut._log.info(f"  가중치 적재 {hw_loads} / 재사용 {hw_reuse} "
                  f"— 타일러 예상과 일치")
    dut._log.info(f"  읽기 요청 총 {total_reqs}개 "
                  f"(활성값 {len(plan.descs)} + 가중치 "
                  f"{plan.weight_loads*CORES})")
    assert total_reqs == len(plan.descs) + plan.weight_loads * CORES


@cocotb.test()
async def test_weight_stationary_order_saves_reads(dut):
    """순서를 거꾸로 하면 같은 하드웨어로 읽기가 늘어난다.

    타일러가 만드는 순서(밴드 고정, 행 순회)와 그 반대를 둘 다 돌려
    가중치 적재 횟수를 비교한다. RTL 은 같고 순서만 다르다.
    """
    await boot(dut)
    CORES, T, RD_BEAT, WR_BEAT, MAX_KT = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(717171)

    M, N = 4 * T, 2 * T * CORES
    K = min(MAX_KT, 32)
    n_ib, n_jb = M // T, N // (T * CORES)

    A = rng.integers(-128, 128, size=(M, K), dtype=np.int8)
    B = rng.integers(-128, 128, size=(K, N), dtype=np.int8)
    exp = A.astype(np.int32) @ B.astype(np.int32)

    ACT0, WGT0, OUT0 = 0x100000, 0x400000, 0x800000
    act_tile = K * T
    wgt_slice = K * T
    wgt_band = wgt_slice * CORES
    for ib in range(n_ib):
        for k in range(K):
            mem.write_bytes(ACT0 + act_tile * ib + T * k,
                            bytes(int(A[ib * T + i, k]) & 0xFF
                                  for i in range(T)))
    for jb in range(n_jb):
        for c in range(CORES):
            col0 = jb * T * CORES + c * T
            for k in range(K):
                mem.write_bytes(WGT0 + wgt_band * jb + wgt_slice * c + T * k,
                                bytes(int(B[k, col0 + j]) & 0xFF
                                      for j in range(T)))

    results = {}
    for label, order in (("밴드 고정 (타일러)",
                          [(ib, jb) for jb in range(n_jb)
                           for ib in range(n_ib)]),
                         ("행 고정 (반대)",
                          [(ib, jb) for ib in range(n_ib)
                           for jb in range(n_jb)])):
        await boot(dut)
        mem2 = SimpleMem()
        mem2.data = dict(mem.data) if hasattr(mem, "data") else None
        # 같은 메모리를 다시 쓴다 (스텁은 테스트당 한 번만 띄운다)
        last_band = None
        loads = 0
        out_map = {}
        for n, (ib, jb) in enumerate(order):
            flags = 0
            if jb == last_band:
                flags |= 0x01     # FLAG_WGT_KEEP
            else:
                loads += 1
                last_band = jb
            out = OUT0 + T * T * 4 * CORES * n
            out_map[(ib, jb)] = out
            await push(dut, make_desc(
                ACT0 + act_tile * ib, WGT0 + wgt_band * jb, out,
                K, wgt_slice, T * T * 4, flags))

        got = np.zeros((M, N), dtype=np.int64)
        for (ib, jb), base in out_map.items():
            for c in range(CORES):
                raw = mem.read_bytes(base + T * T * 4 * c, T * T * 4)
                for i in range(T):
                    for j in range(T):
                        o = (i * T + j) * 4
                        got[ib * T + i, jb * T * CORES + c * T + j] = \
                            int.from_bytes(raw[o:o + 4], "little", signed=True)
        assert (got == exp).all(), f"{label}: 결과 불일치"
        hw = int(dut.perf_wgt_load.value)
        results[label] = (loads, hw)
        dut._log.info(f"  {label:<20} 가중치 적재 {loads}회 "
                      f"(하드웨어 {hw} = {loads}×{CORES})  결과 일치")
        assert hw == loads * CORES

    ws, naive = results["밴드 고정 (타일러)"][0], results["행 고정 (반대)"][0]
    dut._log.info(f"  순서만 바꿔 가중치 적재 {naive} → {ws}회 "
                  f"({naive/ws:.2f}× 절감). RTL 은 동일하다.")
    assert ws < naive, (
        f"타일러 순서가 더 많이 적재한다 ({ws} vs {naive}) — "
        f"n_ib={n_ib} n_jb={n_jb} 이면 밴드 고정이 {n_jb}, 행 고정이 "
        f"{n_ib*n_jb} 여야 한다")
