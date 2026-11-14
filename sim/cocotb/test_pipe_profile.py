"""test_pipe_profile.py — 3단 파이프라인에서 각 FSM 이 얼마나 비어 있나.

연속 스트림을 돌리면서 로드/연산/쓰기 FSM 의 상태 점유를 센다.
어느 단계가 묶는 쪽이고, 남은 거품이 어디 있는지 숫자로 본다.
측정만 하고 실패시키지 않는다.
"""
import os
import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from dma_stub import SimpleMem, start_dma

RD_BEAT = int(os.environ.get("RD_BEAT_BYTES", "16"))

LSTATE = ["L_IDLE", "L_ACT_REQ", "L_ACT_RECV", "L_WGT_REQ", "L_WGT_RECV"]
CSTATE = ["C_IDLE", "C_CLR", "C_COMPUTE", "C_POST", "C_LATCH"]
WSTATE = ["W_IDLE", "W_REQ", "W_SEND", "W_DONE"]


def u64(x): return int(x).to_bytes(8, "little", signed=False)


async def boot(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.cmd_valid.value = 0
    dut.act_addr.value = 0
    dut.wgt_addr.value = 0
    dut.out_addr.value = 0
    dut.Kt.value = 0
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


@cocotb.test()
async def test_pipeline_occupancy(dut):
    await boot(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=16)

    Kt = int(os.environ.get("PROF_KT", "64"))
    n_ops = int(os.environ.get("PROF_OPS", "24"))
    rng = np.random.default_rng(3)
    A = rng.integers(-128, 128, size=(16, Kt), dtype=np.int8)
    B = rng.integers(-128, 128, size=(Kt, 16), dtype=np.int8)
    act, wgt, out = 0x1000, 0x2000, 0x3000
    for k in range(Kt):
        mem.write_bytes(act + 16 * k, bytes((int(A[i, k]) & 0xFF) for i in range(16)))
        mem.write_bytes(wgt + 16 * k, bytes((int(B[k, j]) & 0xFF) for j in range(16)))

    lc = {s: 0 for s in LSTATE}
    cc = {s: 0 for s in CSTATE}
    wc = {s: 0 for s in WSTATE}
    issued = done_seen = total = 0
    started = False
    done_at = []

    for _ in range(500_000):
        want = issued < n_ops
        dut.act_addr.value = act
        dut.wgt_addr.value = wgt
        dut.out_addr.value = out + 0x1000 * issued
        dut.Kt.value = Kt
        dut.cmd_valid.value = 1 if want else 0

        await Timer(1, unit="ns")
        accept = want and int(dut.cmd_ready.value) == 1

        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")

        if accept:
            issued += 1
            started = True
        if started:
            total += 1
            lc[LSTATE[int(dut.lst.value)]] += 1
            cc[CSTATE[int(dut.cst.value)]] += 1
            wc[WSTATE[int(dut.wst.value)]] += 1
            if int(dut.done.value) == 1:
                done_seen += 1
                done_at.append(total)
                if done_seen == n_ops:
                    break

    dut.cmd_valid.value = 0
    assert done_seen == n_ops, f"{done_seen}/{n_ops}"

    gaps = [b - a for a, b in zip(done_at, done_at[1:])]
    # 앞뒤 2개는 파이프라인 채움/배출이라 버린다.
    steady = gaps[2:-2] if len(gaps) > 6 else gaps
    dut._log.info("")
    dut._log.info(f"Kt={Kt}, {n_ops} ops, 전체 {total} 사이클 "
                  f"({total/n_ops:.1f}/op 평균, 채움/배출 포함)")
    dut._log.info(f"  완료 간격: {gaps}")
    dut._log.info(f"  정상상태 간격 {min(steady)}~{max(steady)}, "
                  f"평균 {sum(steady)/len(steady):.1f} 사이클/op  "
                  f"→ 가동률 {Kt/(sum(steady)/len(steady))*100:.1f}%")
    dut._log.info("")
    for name, c in (("로드 lst", lc), ("연산 cst", cc), ("쓰기 wst", wc)):
        busy = total - c[list(c)[0]]      # IDLE 이 아닌 사이클
        dut._log.info(f"  {name:>10}  바쁨 {busy:>5} ({busy/total*100:>5.1f}%)  "
                      f"유휴 {c[list(c)[0]]:>5}")
        for s, v in c.items():
            if v:
                dut._log.info(f"               {s:<12} {v:>5} ({v/n_ops:>6.1f}/op)")
    dut._log.info("")
