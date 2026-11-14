"""test_gemm_core_throughput.py — gemm_core 를 직접 구동한 연속 처리량.

gemm_top 경유로는 측정되지 않는다. ctrl_fsm 이 ST_WAIT 에서 core_done 을
기다린 뒤에야 다음 디스크립터를 받으므로, 코어가 아무리 빨리 cmd_ready 를
올려도 위에서 직렬화된다. 여기서는 코어에 직접 명령을 밀어 넣어
"코어가 낼 수 있는 처리량" 을 잰다.
"""
import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

import os
from dma_stub import SimpleMem, start_dma

RD_BEAT = int(os.environ.get("RD_BEAT_BYTES", "16"))

VERBOSE = False

CSTATE = ["C_IDLE", "C_CLR", "C_COMPUTE", "C_POST", "C_LATCH"]
LSTATE = ["L_IDLE", "L_ACT_REQ", "L_ACT_RECV", "L_WGT_REQ", "L_WGT_RECV"]


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


async def stream(dut, Kt: int, n_ops: int) -> dict:
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=16)

    A = np.random.randint(-128, 128, size=(16, max(Kt, 1)), dtype=np.int8)
    B = np.random.randint(-128, 128, size=(max(Kt, 1), 16), dtype=np.int8)
    act_base, wgt_base, out_base = 0x1000, 0x2000, 0x3000
    for k in range(Kt):
        mem.write_bytes(act_base + 16 * k,
                        bytes((int(A[i, k]) & 0xFF) for i in range(16)))
        mem.write_bytes(wgt_base + 16 * k,
                        bytes((int(B[k, j]) & 0xFF) for j in range(16)))

    issued = done_seen = total = comp = 0
    started = False

    # 샘플링 규칙: RisingEdge 직후에는 엣지가 "반영되기 전" 값이 보인다.
    # Timer 로 한 델타 넘긴 뒤에 읽어야 방금 엣지의 결과를 본다.
    # (이걸 빼면 cmd_ready 를 한 사이클 늦게 읽어 같은 명령을 두 번
    #  수용한 것으로 세게 된다.)
    for _ in range(300_000):
        want = issued < n_ops
        dut.cmd_valid.value = 1 if want else 0
        if want:
            dut.act_addr.value = act_base
            dut.wgt_addr.value = wgt_base
            dut.out_addr.value = out_base + 0x1000 * issued
            dut.Kt.value = Kt

        await Timer(1, unit="ns")          # 구동값 전파
        accept = want and int(dut.cmd_ready.value) == 1

        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")          # 엣지 반영

        if accept:
            issued += 1
            started = True
            if VERBOSE:
                dut._log.info(f"  [{total:5d}] accept #{issued}")

        if started:
            total += 1
            if CSTATE[int(dut.cst.value)] == "C_COMPUTE":
                comp += 1
            if int(dut.done.value) == 1:
                done_seen += 1
                if VERBOSE:
                    dut._log.info(f"  [{total:5d}] done #{done_seen}")
                if done_seen == n_ops:
                    break

    dut.cmd_valid.value = 0
    if done_seen != n_ops:
        dut._log.error(
            f"issued={issued} done={done_seen} total={total} "
            f"lst={LSTATE[int(dut.lst.value)]} cst={CSTATE[int(dut.cst.value)]} "
            f"wst={int(dut.wst.value)} "
            f"cmd_ready={int(dut.cmd_ready.value)} busy={int(dut.busy.value)}")
    assert done_seen == n_ops, f"{done_seen}/{n_ops} 만 완료 (total={total})"
    return {"total": total, "compute": comp, "ops": n_ops}


@cocotb.test()
async def test_debug_small(dut):
    """소량 연속 — accept/done 짝이 맞는지 본다."""
    global VERBOSE
    await boot(dut)
    VERBOSE = True
    try:
        await stream(dut, 1, n_ops=3)
    finally:
        VERBOSE = False


@cocotb.test()
async def test_core_streamed_throughput(dut):
    await boot(dut)
    dut._log.info("")
    dut._log.info(f"{'Kt':>5} {'ops':>5} {'전체':>8} {'연산':>8} {'op당':>9} {'가동률':>8}")
    dut._log.info("-" * 54)
    for Kt in (1, 16, 64):
        r = await stream(dut, Kt, n_ops=8)
        per = r["total"] / r["ops"]
        util = r["compute"] / r["total"]
        dut._log.info(f"{Kt:>5} {r['ops']:>5} {r['total']:>8} {r['compute']:>8} "
                      f"{per:>9.1f} {util*100:>7.1f}%")
    dut._log.info("")
    dut._log.info("쓰기 FSM 이 분리돼 다음 디스크립터의 페치/연산과 겹친다.")
    dut._log.info("피연산자 이중 버퍼링까지 들어가 페치도 연산과 겹친다.")
