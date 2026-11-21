"""test_tile32.py — TILE_N 을 키워도 맞는 값을 내는가, 그리고 얼마나 빠른가.

N×N 외적 누산이므로 MAC 은 N², 피연산자 바이트는 N 이다. 산술 강도가
N 에 비례하고, 그것이 대역폭 대비 연산을 늘리는 정석 경로다.

N=16/RD=256/WS 에서 로드 채널이 45.5% 만 바빴다 (2.19× 여유). N=32 는
MAC 4배 / 피연산자 2배이므로 그 여유를 정확히 소진해야 한다.

TILE_N 은 환경변수로 받아 같은 테스트를 N=16 과 N=32 에 모두 돌린다.
N=16 에서 통과해야 파라미터화가 기존 동작을 바꾸지 않았음이 확인된다.
"""
import os
import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from dma_stub import SimpleMem, start_dma

FLAG_WGT_KEEP = 0x01


def probe_config(dut):
    """구성을 DUT 에서 직접 읽는다 — 환경변수와 RTL 이 어긋날 수 없게.

    TILE_N 을 환경변수로만 받던 초기 판에서 러너가 Verilator 에 그 값을
    넘기지 않아, RTL 은 N=16 인데 테스트는 N=32 를 가정했다. 조용히
    1,008/1,024 개가 틀렸고 원인을 찾는 데 시간이 걸렸다. 누산기 배열
    길이와 포트 폭에서 역산하면 그런 불일치가 원천적으로 불가능하다.
    """
    accs = len(dut.u_core.acc_out_lat)
    n = int(round(accs ** 0.5))
    assert n * n == accs, f"누산기 {accs}개가 제곱수가 아니다"
    rd_beat = len(dut.rd_data) // 8
    wr_beat = len(dut.wr_data) // 8
    assert rd_beat % n == 0, \
        f"읽기 비트 {rd_beat}바이트가 K-step {n}바이트의 배수가 아니다"
    return n, rd_beat, wr_beat


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


def write_operands(mem, act_base, wgt_base, A, B, Kt, N):
    """A(N×Kt) 는 K-step 당 N 바이트 열, B(Kt×N) 는 K-step 당 N 바이트 행."""
    for k in range(Kt):
        mem.write_bytes(act_base + N * k,
                        bytes((int(A[i, k]) & 0xFF) for i in range(N)))
        mem.write_bytes(wgt_base + N * k,
                        bytes((int(B[k, j]) & 0xFF) for j in range(N)))


def read_result(mem, out_base, N):
    raw = mem.read_bytes(out_base, N * N * 4)
    out = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(N):
            o = (i * N + j) * 4
            out[i, j] = int.from_bytes(raw[o:o + 4], "little", signed=True)
    return out


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

    rd = wr = 0
    for _ in range(200_000):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.rd_data_valid.value) and int(dut.rd_data_ready.value):
            rd += 1
        if int(dut.wr_data_valid.value) and int(dut.wr_data_ready.value):
            wr += 1
        if int(dut.done_pulse.value) == 1:
            return rd, wr
    raise TimeoutError("done_pulse 없음")


@cocotb.test()
async def test_result_matches_reference(dut):
    """N×N 결과가 numpy 기준값과 일치한다."""
    await boot(dut)
    N, RD_BEAT, WR_BEAT = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(2024)

    dut._log.info(f"  TILE_N={N}  MAC={N*N}  K-step={N}바이트  "
                  f"읽기 비트 {RD_BEAT}B  쓰기 비트 {WR_BEAT}B")
    for Kt in (1, 2, 5, 16, 64):
        A = rng.integers(-128, 128, size=(N, Kt), dtype=np.int8)
        B = rng.integers(-128, 128, size=(Kt, N), dtype=np.int8)
        act, wgt, out = 0x1000, 0x20000, 0x40000
        write_operands(mem, act, wgt, A, B, Kt, N)
        rd, wr = await push(dut, make_desc(act, wgt, out, Kt))

        got = read_result(mem, out, N)
        exp = A[:, :Kt].astype(np.int32) @ B[:Kt, :].astype(np.int32)
        bad = np.argwhere(got != exp)
        assert len(bad) == 0, \
            f"Kt={Kt}: {len(bad)}/{N*N} 개 불일치, 첫 오차 {bad[:1]} " \
            f"got={got[tuple(bad[0])]} exp={exp[tuple(bad[0])]}"

        exp_wr = (N * N * 4) // WR_BEAT
        assert wr == exp_wr, f"Kt={Kt}: 쓰기 비트 {wr}, 기대 {exp_wr}"
        dut._log.info(f"  Kt={Kt:>3}  읽기 {rd:>3} / 쓰기 {wr:>3} 비트  "
                      f"MAC {N*N*Kt:>6}  결과 일치")


@cocotb.test()
async def test_streamed_throughput(dut):
    """weight-stationary 로 연속 처리량을 재고 MAC/사이클을 낸다.

    산술 강도는 N 에 비례하므로, N 을 키우면 같은 읽기 대역폭으로 더 많은
    MAC 을 돌릴 수 있어야 한다. 그것을 숫자로 남긴다.
    """
    await boot(dut)
    N, RD_BEAT, WR_BEAT = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(11)

    # 쓰기 비트 = N²/WPB, 연산 = Kt. 균형점은 Kt = N²·4/WR_BEAT.
    Kt = min(256, (N * N * 4) // WR_BEAT)
    n_ops = 24
    A = rng.integers(-128, 128, size=(N, Kt), dtype=np.int8)
    B = rng.integers(-128, 128, size=(Kt, N), dtype=np.int8)
    act, wgt = 0x1000, 0x20000
    write_operands(mem, act, wgt, A, B, Kt, N)
    exp = A[:, :Kt].astype(np.int32) @ B[:Kt, :].astype(np.int32)

    issued = done_seen = total = 0
    started = False
    done_at = []
    out_base = 0x40000

    for _ in range(2_000_000):
        want = issued < n_ops
        if want:
            d = make_desc(act, wgt, out_base + 0x4000 * issued, Kt,
                          FLAG_WGT_KEEP if issued > 0 else 0)
            for i in range(64):
                dut.desc_bytes[i].value = d[i]
        dut.desc_valid.value = 1 if want else 0

        await Timer(1, unit="ns")
        accept = want and int(dut.desc_ready.value) == 1

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

    # 결과 정합성도 같이 본다 — 빠르지만 틀리면 의미가 없다.
    for n in (0, n_ops // 2, n_ops - 1):
        got = read_result(mem, out_base + 0x4000 * n, N)
        bad = np.argwhere(got != exp)
        if len(bad):
            i, j = bad[0]
            dut._log.error(
                f"스트림 #{n}: {len(bad)}/{N*N} 불일치. "
                f"[{i}][{j}] got={got[i, j]} exp={exp[i, j]} "
                f"비율={got[i, j] / exp[i, j] if exp[i, j] else float('nan'):.4f}")
            dut._log.error(f"  got[0][:4]={got[0][:4].tolist()}")
            dut._log.error(f"  exp[0][:4]={exp[0][:4].tolist()}")
        assert len(bad) == 0, f"스트림 #{n} 결과 불일치 ({len(bad)}개)"

    gaps = [b - a for a, b in zip(done_at, done_at[1:])]
    steady = gaps[2:-2] if len(gaps) > 6 else gaps
    per_op = sum(steady) / len(steady)
    macs = N * N * Kt
    dut._log.info("")
    dut._log.info(f"  TILE_N={N}  Kt={Kt}  MAC/디스크립터={macs:,}")
    dut._log.info(f"  정상상태 간격 {min(steady)}~{max(steady)}, "
                  f"평균 {per_op:.1f} 사이클/op")
    dut._log.info(f"  → {macs/per_op:,.0f} MAC/사이클  "
                  f"({macs/per_op*2:,.0f} INT8 OPS/사이클)")
    dut._log.info(f"  → MAC 배열 가동률 {macs/per_op/(N*N)*100:.1f}%")
    dut._log.info("")
    assert min(steady) == max(steady), f"간격이 흔들린다: {steady}"
