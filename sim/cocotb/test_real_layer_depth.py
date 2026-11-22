"""test_real_layer_depth.py — 실제 LLM 레이어 깊이가 한 디스크립터에 들어가는가.

K 분할(10단계)은 축소 차원이 MAX_KT 를 넘을 때의 대비책이다. 그런데
K 분할을 쓰면 누산기가 한 벌이라 가중치 재사용이 사라진다 — 실제 레이어의
절반(K가 깊은 쪽)에서 2.27× 절감을 잃는다.

대안이 둘 있다.

  (A) 누산 SRAM 을 둬서 출력 타일 여러 개의 부분합을 담는다.
      Gemmini 가 이 방식이다 (입력에 가산기가 붙은 듀얼포트 SRAM,
      쓰기 요청의 `acc` 비트로 add/overwrite). 복잡하다 — Gemmini
      자신도 같은 주소 연속 접근 해저드를 TODO 로 남겨 뒀다.

  (B) **MAX_KT 를 K 이상으로 키운다.** 피연산자 SRAM 만 커지고 설계는
      그대로다.

7B 레이어 기준 (B) 의 비용:

      K= 4096  4코어  2.00 MB
      K=11008  4코어  5.38 MB     ← 가장 깊은 레이어 (ffn down)
      VC1902 온칩 20.5 MB

**(B) 로 충분하다.** 이 테스트가 그것을 계산이 아니라 RTL 로 확인한다.
MAX_KT 를 실제 레이어 깊이로 올려 결과 정확성과 weight-stationary 가
그대로 동작하는지 본다.

Kt 가 크면 시뮬레이션이 느리므로 디스크립터 수를 줄인다 — 여기서 재는
것은 처리량이 아니라 **깊은 K 에서 맞는 값이 나오는가** 다.
"""
import os

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from dma_stub import SimpleMem, start_dma

FLAG_WGT_KEEP = 0x01


def u32_le(x): return int(x).to_bytes(4, "little", signed=False)
def u64_le(x): return int(x).to_bytes(8, "little", signed=False)


def probe_config(dut):
    accs = len(dut.u_core.acc_out_lat)
    n = int(round(accs ** 0.5))
    assert n * n == accs
    rd_beat = len(dut.rd_data) // 8
    wr_beat = len(dut.wr_data) // 8
    # 뱅크 2개를 나눠 쓰므로 뱅크당 깊이는 절반
    max_kt = len(dut.u_core.u_act.mem) * (rd_beat // n) // 2
    return n, rd_beat, wr_beat, max_kt


def make_desc(act, wgt, out, kt, flags=0):
    d = bytearray(64)
    d[0:1] = bytes([0x02])
    d[1:2] = bytes([flags])
    d[16:24] = u64_le(act)
    d[24:32] = u64_le(wgt)
    d[32:40] = u64_le(out)
    d[40:44] = u32_le(kt)
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


async def push(dut, desc, budget=2_000_000):
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

    rd = cyc = 0
    reqs = []
    for _ in range(budget):
        await Timer(1, unit="ns")
        if int(dut.rd_req_valid.value) and int(dut.rd_req_ready.value):
            reqs.append((int(dut.rd_req_addr.value),
                         int(dut.rd_req_len_bytes.value)))
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        cyc += 1
        if int(dut.rd_data_valid.value) and int(dut.rd_data_ready.value):
            rd += 1
        if int(dut.done_pulse.value) == 1:
            return rd, cyc, reqs
    raise TimeoutError("done_pulse 없음")


def read_result(mem, base, N):
    raw = mem.read_bytes(base, N * N * 4)
    out = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(N):
            o = (i * N + j) * 4
            out[i, j] = int.from_bytes(raw[o:o + 4], "little", signed=True)
    return out


@cocotb.test()
async def test_full_layer_depth_in_one_descriptor(dut):
    """MAX_KT 를 실제 레이어 깊이로 올려도 맞는 값이 나온다."""
    await boot(dut)
    N, RD_BEAT, WR_BEAT, MAX_KT = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(40960)

    Kt = MAX_KT
    dut._log.info("")
    dut._log.info(f"  TILE_N={N}  MAX_KT={Kt}  "
                  f"(피연산자 SRAM {2*Kt*N*2/1024:.0f}KB/코어)")

    A = rng.integers(-128, 128, size=(N, Kt), dtype=np.int8)
    B = rng.integers(-128, 128, size=(Kt, N), dtype=np.int8)
    exp = A.astype(np.int32) @ B.astype(np.int32)

    # int32 누산기가 넘치지 않는지 먼저 확인한다. K 가 깊으면 실제 위험이다.
    peak = int(np.abs(exp).max())
    bound = Kt * 128 * 128
    dut._log.info(f"  누산 최대 {peak:,} (이론 상한 {bound:,}, "
                  f"int32 한계 {2**31-1:,})")
    assert bound < 2 ** 31, (
        f"K={Kt} 에서 int32 누산기가 넘칠 수 있다 (상한 {bound:,}). "
        f"실제 데이터로는 안 넘쳐도 설계 한계로 기록해야 한다.")

    act, wgt, out = 0x100000, 0x400000, 0x800000
    for k in range(Kt):
        mem.write_bytes(act + N * k,
                        bytes((int(A[i, k]) & 0xFF) for i in range(N)))
        mem.write_bytes(wgt + N * k,
                        bytes((int(B[k, j]) & 0xFF) for j in range(N)))

    rd, cyc, reqs = await push(dut, make_desc(act, wgt, out, Kt))

    # 16비트 길이 필드 때문에 한 요청이 65,504바이트를 넘지 못한다.
    # 긴 전송은 쪼개져야 하고, 조각들이 주소 순서대로 이어져야 한다.
    REQ_MAX = (65535 // RD_BEAT) * RD_BEAT
    total_bytes = Kt * N
    dut._log.info(f"  읽기 요청 {len(reqs)}개 (한 요청 최대 {REQ_MAX:,}B)")
    for i, (a, ln) in enumerate(reqs):
        assert ln <= REQ_MAX, f"요청 {i} 길이 {ln} > {REQ_MAX}"
        assert ln % RD_BEAT == 0, f"요청 {i} 길이 {ln} 가 비트 배수가 아니다"
    act_reqs = [r for r in reqs if r[0] < wgt]
    wgt_reqs = [r for r in reqs if r[0] >= wgt]
    for label, group, base in (("활성값", act_reqs, act), ("가중치", wgt_reqs, wgt)):
        assert sum(ln for _, ln in group) == total_bytes, \
            f"{label} 합계 {sum(ln for _, ln in group):,}B, 기대 {total_bytes:,}B"
        addr = base
        for a, ln in group:
            assert a == addr, f"{label} 주소가 끊겼다: {a:#x}, 기대 {addr:#x}"
            addr += ln
        dut._log.info(f"    {label} {len(group)}개 요청, "
                      f"합계 {total_bytes:,}B, 주소 연속")

    got = read_result(mem, out, N)
    bad = np.argwhere(got != exp)
    assert len(bad) == 0, (
        f"K={Kt}: {len(bad)}/{N*N} 불일치. 첫 오차 {bad[:1]} "
        f"got={got[tuple(bad[0])]} exp={exp[tuple(bad[0])]}")
    dut._log.info(f"  단일 디스크립터 K={Kt}: {cyc:,} 사이클, "
                  f"읽기 {rd:,} 비트, 결과 일치")
    dut._log.info(f"  → K 분할 없이 레이어 하나의 축소 차원 전체를 처리한다")


@cocotb.test()
async def test_weight_stationary_at_layer_depth(dut):
    """깊은 K 에서도 weight-stationary 가 동작한다 — 이것이 (B) 의 요점.

    K 분할을 쓰면 가중치 재사용이 사라진다. MAX_KT 를 키우면 그대로
    남는다. 읽기 비트 수로 확인한다.
    """
    await boot(dut)
    N, RD_BEAT, WR_BEAT, MAX_KT = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(11008)

    Kt = MAX_KT
    B = rng.integers(-128, 128, size=(Kt, N), dtype=np.int8)
    wgt = 0x400000
    for k in range(Kt):
        mem.write_bytes(wgt + N * k,
                        bytes((int(B[k, j]) & 0xFF) for j in range(N)))

    beats = []
    for n in range(3):
        A = rng.integers(-128, 128, size=(N, Kt), dtype=np.int8)
        act = 0x100000 + 0x200000 * n
        for k in range(Kt):
            mem.write_bytes(act + N * k,
                            bytes((int(A[i, k]) & 0xFF) for i in range(N)))
        out = 0x800000 + 0x10000 * n
        rd, cyc, _ = await push(dut, make_desc(act, wgt, out, Kt,
                                              FLAG_WGT_KEEP if n else 0))
        beats.append(rd)
        got = read_result(mem, out, N)
        exp = A.astype(np.int32) @ B.astype(np.int32)
        assert (got == exp).all(), \
            f"#{n}: 불일치, 첫 오차 {np.argwhere(got != exp)[:1]}"
        dut._log.info(f"  #{n} {'WS' if n else '적재'}  읽기 {rd:,} 비트  "
                      f"{cyc:,} 사이클  결과 일치")

    assert beats[0] == 2 * beats[1], \
        f"재사용 디스크립터가 절반이 아니다: {beats}"
    assert beats[1] == beats[2]
    dut._log.info(f"  가중치 적재 {int(dut.perf_wgt_load.value)}회 / "
                  f"재사용 {int(dut.perf_wgt_reuse.value)}회")
    dut._log.info(f"  → 깊은 K 에서도 weight-stationary 가 그대로 동작한다. "
                  f"K 분할을 쓰면 잃는 것이 이것이다.")
