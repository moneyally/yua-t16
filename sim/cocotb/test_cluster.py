"""test_cluster.py — 텐서 병렬 클러스터가 맞는 값을 내고 활성값을 한 번만 읽는가.

가중치를 열 방향으로 쪼개 코어마다 다른 조각을 주고, 같은 활성값을
모든 코어에 흘린다. 각 코어는 출력의 다른 열 묶음을 만든다.

    코어 0:  A · W[:, 0:N]    → OUT[:, 0:N]
    코어 1:  A · W[:, N:2N]   → OUT[:, N:2N]

단언하는 것:

  1) 두 조각을 이어 붙이면 A · W (N × 2N) 전체와 같다.
  2) **활성값을 한 번만 읽는다.** 코어마다 따로 읽으면 CORES 배가 되어
     아무것도 못 번다 — 이 설계의 요점이 바로 그 브로드캐스트다.
  3) weight-stationary 와 함께 돌아간다 (가중치 적재는 코어당 1회).

읽기 비트 수를 세어 (2) 를 직접 확인한다. 이것이 틀리면 결과는 맞는데
대역폭 이득이 0 인 상태로 조용히 지나간다.
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
    """구성을 DUT 에서 읽는다 (환경변수 불일치 방지)."""
    cores = len(dut.CORE)
    accs = len(dut.CORE[0].u_core.acc_out_lat)
    n = int(round(accs ** 0.5))
    assert n * n == accs
    return cores, n, len(dut.rd_data) // 8, len(dut.wr_data) // 8


def make_desc(act, wgt, out, Kt, wgt_stride, out_stride, flags=0):
    d = bytearray(64)
    d[0:1] = bytes([0x02])
    d[1:2] = bytes([flags])
    d[16:24] = u64_le(act)
    d[24:32] = u64_le(wgt)
    d[32:40] = u64_le(out)
    d[40:44] = u32_le(Kt)
    d[44:48] = u32_le(wgt_stride)   # 코어 간 가중치 간격
    d[52:56] = u32_le(out_stride)   # 코어 간 출력 간격
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
    """디스크립터 하나. 읽기/쓰기 비트와 고유 읽기 주소 수를 돌려준다."""
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
    reqs = []
    for _ in range(400_000):
        await Timer(1, unit="ns")
        if int(dut.rd_req_valid.value) and int(dut.rd_req_ready.value):
            reqs.append((int(dut.rd_req_addr.value),
                         int(dut.rd_req_len_bytes.value)))
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.rd_data_valid.value) and int(dut.rd_data_ready.value):
            rd += 1
        if int(dut.wr_data_valid.value) and int(dut.wr_data_ready.value):
            wr += 1
        if int(dut.done_pulse.value) == 1:
            return rd, wr, reqs
    raise TimeoutError("done_pulse 없음")


def read_slice(mem, base, N):
    raw = mem.read_bytes(base, N * N * 4)
    out = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(N):
            o = (i * N + j) * 4
            out[i, j] = int.from_bytes(raw[o:o + 4], "little", signed=True)
    return out


@cocotb.test()
async def test_tensor_parallel_matches_reference(dut):
    """조각들을 이어 붙이면 A · W 전체와 같다."""
    await boot(dut)
    CORES, N, RD_BEAT, WR_BEAT = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(1212)

    dut._log.info(f"  CORES={CORES}  TILE_N={N}  "
                  f"출력 {N}×{N*CORES}  MAC {N*N*CORES:,}")

    Kt = 32
    A = rng.integers(-128, 128, size=(N, Kt), dtype=np.int8)
    # 전체 가중치는 (Kt × N*CORES). 코어 c 가 열 [c*N, (c+1)*N) 을 맡는다.
    W = rng.integers(-128, 128, size=(Kt, N * CORES), dtype=np.int8)

    act = 0x1000
    for k in range(Kt):
        mem.write_bytes(act + N * k,
                        bytes((int(A[i, k]) & 0xFF) for i in range(N)))

    # 코어 c 의 가중치 조각을 자기 영역에 K-step 당 N 바이트로 둔다.
    wgt = 0x40000
    wgt_stride = Kt * N          # 조각 하나의 바이트 수
    for c in range(CORES):
        for k in range(Kt):
            mem.write_bytes(wgt + wgt_stride * c + N * k,
                            bytes((int(W[k, c * N + j]) & 0xFF)
                                  for j in range(N)))

    out = 0x80000
    out_stride = N * N * 4
    rd, wr, reqs = await push(
        dut, make_desc(act, wgt, out, Kt, wgt_stride, out_stride))

    got = np.concatenate(
        [read_slice(mem, out + out_stride * c, N) for c in range(CORES)],
        axis=1)
    exp = A.astype(np.int32) @ W.astype(np.int32)
    bad = np.argwhere(got != exp)
    if len(bad):
        i, j = bad[0]
        raise AssertionError(
            f"{len(bad)}/{N*N*CORES} 불일치. [{i}][{j}] "
            f"got={got[i,j]} exp={exp[i,j]} (코어 {j//N} 의 조각)")
    dut._log.info(f"  {N}×{N*CORES} 결과 전체 일치 "
                  f"(읽기 {rd} / 쓰기 {wr} 비트)")

    # 쓰기는 코어마다 자기 조각을 쓴다 → CORES 배
    assert wr == CORES * (N * N * 4) // WR_BEAT, f"쓰기 비트 {wr}"


@cocotb.test()
async def test_activations_are_read_once(dut):
    """활성값을 코어 수와 무관하게 한 번만 읽는다.

    이 설계의 요점이다. 코어마다 따로 읽으면 결과는 맞지만 대역폭 이득이
    0 이 되어 조용히 지나간다. 그래서 요청 주소를 직접 센다.
    """
    await boot(dut)
    CORES, N, RD_BEAT, WR_BEAT = probe_config(dut)
    if CORES < 2:
        dut._log.info("  CORES=1 — 브로드캐스트할 상대가 없다. 건너뛴다.")
        return

    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(3434)

    Kt = 32
    A = rng.integers(-128, 128, size=(N, Kt), dtype=np.int8)
    W = rng.integers(-128, 128, size=(Kt, N * CORES), dtype=np.int8)
    act, wgt, out = 0x1000, 0x40000, 0x80000
    wgt_stride, out_stride = Kt * N, N * N * 4
    for k in range(Kt):
        mem.write_bytes(act + N * k,
                        bytes((int(A[i, k]) & 0xFF) for i in range(N)))
    for c in range(CORES):
        for k in range(Kt):
            mem.write_bytes(wgt + wgt_stride * c + N * k,
                            bytes((int(W[k, c * N + j]) & 0xFF)
                                  for j in range(N)))

    rd, wr, reqs = await push(
        dut, make_desc(act, wgt, out, Kt, wgt_stride, out_stride))

    act_reqs = [r for r in reqs if r[0] == act]
    wgt_reqs = [r for r in reqs if r[0] != act]
    act_bytes = sum(r[1] for r in act_reqs)
    wgt_bytes = sum(r[1] for r in wgt_reqs)

    dut._log.info(f"  읽기 요청 {len(reqs)}개: "
                  f"활성값 {len(act_reqs)}개({act_bytes}B) + "
                  f"가중치 {len(wgt_reqs)}개({wgt_bytes}B)")

    assert len(act_reqs) == 1, \
        f"활성값을 {len(act_reqs)}번 읽었다 — 브로드캐스트가 안 됐다"
    assert len(wgt_reqs) == CORES, \
        f"가중치 요청 {len(wgt_reqs)}개, 코어 {CORES}개여야 한다"
    assert act_bytes == Kt * N, f"활성값 {act_bytes}B, 기대 {Kt*N}B"

    naive = act_bytes * CORES + wgt_bytes
    total = act_bytes + wgt_bytes
    dut._log.info(f"  총 {total}B vs 코어별 개별 읽기 {naive}B "
                  f"→ {naive/total:.2f}× 절감 (활성값 브로드캐스트)")


@cocotb.test()
async def test_cluster_with_weight_stationary(dut):
    """연속 디스크립터에서 코어마다 가중치가 상주한다.

    첫 디스크립터만 가중치를 적재하고(코어당 1회), 나머지는 활성값만
    읽는다. 그러면 읽기 요청이 디스크립터당 **1개**(활성값)로 떨어진다.
    """
    await boot(dut)
    CORES, N, RD_BEAT, WR_BEAT = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(5656)

    Kt = 32
    W = rng.integers(-128, 128, size=(Kt, N * CORES), dtype=np.int8)
    wgt = 0x40000
    wgt_stride, out_stride = Kt * N, N * N * 4
    for c in range(CORES):
        for k in range(Kt):
            mem.write_bytes(wgt + wgt_stride * c + N * k,
                            bytes((int(W[k, c * N + j]) & 0xFF)
                                  for j in range(N)))

    n_desc = 4
    for d in range(n_desc):
        A = rng.integers(-128, 128, size=(N, Kt), dtype=np.int8)
        act = 0x1000 + 0x4000 * d
        for k in range(Kt):
            mem.write_bytes(act + N * k,
                            bytes((int(A[i, k]) & 0xFF) for i in range(N)))
        out = 0x80000 + 0x20000 * d
        flags = FLAG_WGT_KEEP if d > 0 else 0
        rd, wr, reqs = await push(
            dut, make_desc(act, wgt, out, Kt, wgt_stride, out_stride, flags))

        got = np.concatenate(
            [read_slice(mem, out + out_stride * c, N) for c in range(CORES)],
            axis=1)
        exp = A.astype(np.int32) @ W.astype(np.int32)
        assert (got == exp).all(), \
            f"#{d} (flags={flags:#02x}) 불일치, 첫 오차 {np.argwhere(got != exp)[:1]}"

        exp_reqs = 1 if (d > 0 and CORES > 1) else (1 + CORES if d == 0 else 1)
        dut._log.info(f"  #{d} flags={flags:#02x}  읽기 요청 {len(reqs)}개 "
                      f"({rd} 비트)  결과 일치")
        if d > 0:
            assert len(reqs) == 1, \
                f"#{d}: 상주 가중치인데 요청이 {len(reqs)}개 (활성값 1개여야 한다)"

    # 가중치 적재는 코어당 정확히 1회
    assert int(dut.perf_wgt_load.value) == CORES, \
        f"가중치 적재 {int(dut.perf_wgt_load.value)}, 기대 {CORES}"
    assert int(dut.perf_wgt_reuse.value) == CORES * (n_desc - 1)
    dut._log.info(f"  가중치 적재 {CORES}회(코어당 1회), "
                  f"재사용 {CORES*(n_desc-1)}회")


@cocotb.test()
async def test_cluster_streamed_throughput(dut):
    """연속 처리량과 집계 MAC/사이클. 쓰기 채널이 어디서 묶는지도 본다.

    출력량은 코어 수에 비례하는데(코어마다 자기 조각을 쓴다) 연산은
    코어당 Kt 사이클이다. 쓰기 채널이 하나이므로 코어를 늘리면 쓰기가
    묶는 쪽이 된다. 균형점은

        Kt = CORES · N² · 4 / (WR_DATA_W/8)

    이고, 그것을 넘지 못하면(MAX_KT 한계) 코어 확장이 쓰기에 묶인다.
    숫자로 남긴다.
    """
    await boot(dut)
    CORES, N, RD_BEAT, WR_BEAT = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(7878)

    max_kt = len(dut.CORE[0].u_core.u_act.mem) * (RD_BEAT // N) // 2
    balance_kt = CORES * N * N * 4 // WR_BEAT
    Kt = min(max_kt, balance_kt)

    W = rng.integers(-128, 128, size=(Kt, N * CORES), dtype=np.int8)
    wgt = 0x100000
    wgt_stride, out_stride = Kt * N, N * N * 4
    for c in range(CORES):
        for k in range(Kt):
            mem.write_bytes(wgt + wgt_stride * c + N * k,
                            bytes((int(W[k, c * N + j]) & 0xFF)
                                  for j in range(N)))
    A = rng.integers(-128, 128, size=(N, Kt), dtype=np.int8)
    act = 0x1000
    for k in range(Kt):
        mem.write_bytes(act + N * k,
                        bytes((int(A[i, k]) & 0xFF) for i in range(N)))
    exp = A.astype(np.int32) @ W.astype(np.int32)

    n_ops = 16
    out0 = 0x200000
    issued = done_seen = total = 0
    started = False
    done_at = []
    wr_addrs = []
    issued_out = []

    for _ in range(4_000_000):
        want = issued < n_ops
        if want:
            d = make_desc(act, wgt, out0 + 0x40000 * issued, Kt,
                          wgt_stride, out_stride,
                          FLAG_WGT_KEEP if issued > 0 else 0)
            for i in range(64):
                dut.desc_bytes[i].value = d[i]
        dut.desc_valid.value = 1 if want else 0

        await Timer(1, unit="ns")
        accept = want and int(dut.desc_ready.value) == 1
        if int(dut.wr_req_valid.value) and int(dut.wr_req_ready.value):
            wr_addrs.append(int(dut.wr_req_addr.value))
        if accept:
            issued_out.append(int(dut.desc_bytes[32].value)
                              | int(dut.desc_bytes[33].value) << 8
                              | int(dut.desc_bytes[34].value) << 16
                              | int(dut.desc_bytes[35].value) << 24)
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")

        if accept:
            issued += 1
            started = True
        if started:
            total += 1
            if int(dut.done_pulse.value) == 1:
                done_seen += 1
                done_at.append(total)
                if done_seen == n_ops:
                    break

    dut.desc_valid.value = 0
    assert done_seen == n_ops, f"{done_seen}/{n_ops} 완료"

    # 전부 검사하고 어떤 모양으로 틀리는지 남긴다
    fails = []
    for n in range(n_ops):
        base = out0 + 0x40000 * n
        slices = [read_slice(mem, base + out_stride * c, N)
                  for c in range(CORES)]
        got = np.concatenate(slices, axis=1)
        if not (got == exp).all():
            per_core = [int((s_ != exp[:, c*N:(c+1)*N]).sum())
                        for c, s_ in enumerate(slices)]
            zeros = [int((s_ == 0).all()) for s_ in slices]
            fails.append((n, per_core, zeros))
    dut._log.info(f"  발행된 out_addr(하위32): "
                  f"{[hex(a) for a in issued_out[:6]]} ...")
    dut._log.info(f"  실제 쓰기 요청 {len(wr_addrs)}개 (기대 "
                  f"{n_ops*CORES}개): {[hex(a) for a in wr_addrs[:8]]} ...")
    uniq = sorted(set(wr_addrs))
    dut._log.info(f"  고유 쓰기 주소 {len(uniq)}개: "
                  f"{[hex(a) for a in uniq[:8]]} ...")
    if fails:
        dut._log.error(f"  불일치 디스크립터 {len(fails)}/{n_ops}개")
        for n, per_core, zeros in fails[:6]:
            dut._log.error(f"    #{n}: 코어별 오차수 {per_core}  "
                           f"전부0인 코어 {zeros}")
        dut._log.error(f"  done_pulse {done_seen}개, 총 {total} 사이클")
        dut._log.error(f"  emitted={int(dut.emitted.value)} "
                       f"c_dcnt={[int(dut.c_dcnt[k].value) for k in range(CORES)]}")
    assert not fails, f"{len(fails)}/{n_ops} 디스크립터 불일치"

    gaps = [b - a for a, b in zip(done_at, done_at[1:])]
    steady = gaps[2:-2] if len(gaps) > 6 else gaps
    per_op = sum(steady) / len(steady)
    macs = N * N * CORES * Kt

    wr_beats = CORES * (N * N * 4) // WR_BEAT
    rd_beats = Kt * N // RD_BEAT          # 활성값만 (가중치 상주)

    dut._log.info("")
    dut._log.info(f"  CORES={CORES}  TILE_N={N}  Kt={Kt}  "
                  f"(균형 Kt={balance_kt}, 상한 {max_kt})")
    dut._log.info(f"  MAC/디스크립터 {macs:,}")
    dut._log.info(f"  정상상태 {min(steady)}~{max(steady)}, 평균 "
                  f"{per_op:.1f} 사이클/op")
    dut._log.info(f"  → {macs/per_op:,.0f} MAC/사이클 "
                  f"({macs/per_op*2:,.0f} INT8 OPS/사이클)")
    dut._log.info(f"  → 배열 가동률 {macs/per_op/(N*N*CORES)*100:.1f}%")
    dut._log.info(f"  단계 소요: 연산 {Kt}, 쓰기 {wr_beats}, "
                  f"읽기 {rd_beats} 사이클분")
    bound = max(("연산", Kt), ("쓰기", wr_beats), ("읽기", rd_beats),
                key=lambda t: t[1])
    dut._log.info(f"  → 묶는 쪽: {bound[0]} ({bound[1]})")
    dut._log.info("")

    # 묶는 쪽이 연산이면 간격이 일정하다. 쓰기에 묶이면 코어들의 쓰기가
    # 직렬화되며 **주기적으로** 흔들린다 (예: 135, 1290, 258 반복).
    # 그래서 "일정" 이 아니라 "주기적" 을 단언한다 — 주기성이 깨지면
    # 중재가 공정하지 않거나 교착에 가까운 상태라는 뜻이다.
    if min(steady) == max(steady):
        dut._log.info(f"  간격 일정 ({per_op:.0f}) — 연산이 묶는 쪽")
    else:
        period = next((q for q in range(1, len(steady) // 2 + 1)
                       if all(steady[i] == steady[i % q]
                              for i in range(len(steady)))), None)
        dut._log.info(f"  간격 주기 {period} 로 반복: "
                      f"{steady[:period] if period else steady}")
        assert period is not None, f"간격이 주기적이지 않다: {steady}"
        assert period <= CORES, \
            f"주기 {period} 가 코어 수 {CORES} 보다 크다 — 중재 불공정"
