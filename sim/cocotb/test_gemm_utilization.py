"""test_gemm_utilization.py — MAC 배열이 실제로 얼마나 일하는지 계측한다.

gemm_core 의 FSM 은 ACT 읽기 → WGT 읽기 → 연산 → 쓰기 순으로 완전히
직렬이다. 페치 동안 MAC 배열이 통째로 놀고, 연산 동안 DMA 가 논다.
이 테스트는 상태별 사이클을 세어 그 비율을 숫자로 남긴다.

측정만 하고 실패시키지 않는다 — 개선 전후를 같은 자로 재기 위한 기준선이다.
"""
import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from dma_stub import SimpleMem, start_dma

# gemm_core.sv 의 state_t 순서
# gemm_core.sv 의 state_t / wstate_t 순서
STATE = ["IDLE", "RD_ACT_REQ", "RD_ACT_RECV", "RD_WGT_REQ", "RD_WGT_RECV",
         "ACC_CLR", "COMPUTE", "POST_MAC", "LATCH"]
WSTATE = ["W_IDLE", "W_REQ", "W_SEND", "W_DONE"]
FETCH = {"RD_ACT_REQ", "RD_ACT_RECV", "RD_WGT_REQ", "RD_WGT_RECV"}


def u32_le(x): return int(x).to_bytes(4, "little", signed=False)
def u64_le(x): return int(x).to_bytes(8, "little", signed=False)


async def tick(dut, n=1):
    for _ in range(n):
        await RisingEdge(dut.clk)


async def run_gemm(dut, Kt: int) -> dict:
    """Kt 스텝 GEMM 을 한 번 돌리고 상태별 사이클 수를 돌려준다."""
    mem = SimpleMem()
    start_dma(dut, mem, log=False)

    A = np.random.randint(-128, 128, size=(16, Kt), dtype=np.int8)
    B = np.random.randint(-128, 128, size=(Kt, 16), dtype=np.int8)
    act_base, wgt_base, out_base = 0x1000, 0x2000, 0x3000
    for k in range(Kt):
        mem.write_bytes(act_base + 16 * k,
                        bytes((int(A[i, k]) & 0xFF) for i in range(16)))
        mem.write_bytes(wgt_base + 16 * k,
                        bytes((int(B[k, j]) & 0xFF) for j in range(16)))

    desc = bytearray(64)
    desc[0:1] = bytes([0x02])
    desc[16:24] = u64_le(act_base)
    desc[24:32] = u64_le(wgt_base)
    desc[32:40] = u64_le(out_base)
    desc[40:44] = u32_le(Kt)

    for c in range(2000):
        if int(dut.desc_ready.value) == 1:
            break
        await RisingEdge(dut.clk)

    for i in range(64):
        dut.desc_bytes[i].value = desc[i]
    dut.desc_valid.value = 1
    await RisingEdge(dut.clk)
    dut.desc_valid.value = 0

    counts = {s: 0 for s in STATE}
    counts.update({s: 0 for s in WSTATE})
    total = 0
    seen_work = False
    for _ in range(200_000):
        await RisingEdge(dut.clk)
        st = STATE[int(dut.u_core.st.value)]
        wst = WSTATE[int(dut.u_core.wst.value)]
        idle = (st == "IDLE") and (wst == "W_IDLE")
        if not idle:
            seen_work = True
        if seen_work:
            counts[st] += 1
            counts[wst] += 1
            total += 1
        if seen_work and idle:
            break
    counts["_total"] = total
    return counts


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
    await tick(dut, 5)
    dut.rst_n.value = 1
    await tick(dut, 5)


@cocotb.test()
async def test_utilization_profile(dut):
    """여러 Kt 에서 MAC 배열 가동률을 잰다."""
    await boot(dut)

    dut._log.info("")
    dut._log.info(f"{'Kt':>5} {'전체':>8} {'연산':>8} {'페치':>8} {'쓰기':>8} {'가동률':>8}")
    dut._log.info("-" * 52)

    for Kt in (1, 4, 16, 64):
        c = await run_gemm(dut, Kt)
        total = c["_total"]
        comp = c["COMPUTE"]
        fetch = sum(c[s] for s in FETCH)
        write = c["W_REQ"] + c["W_SEND"]
        util = comp / total if total else 0.0
        dut._log.info(f"{Kt:>5} {total:>8} {comp:>8} {fetch:>8} {write:>8} "
                      f"{util*100:>7.1f}%")
        assert total > 0, f"Kt={Kt}: FSM 이 움직이지 않았다"

    dut._log.info("")
    dut._log.info("쓰기는 별도 FSM(wst)이라 메인 FSM 과 겹쳐 돈다. 합이 전체를")
    dut._log.info("넘는 것은 그 때문이고 정상이다.")
    dut._log.info("페치는 아직 연산과 겹치지 않는다 — 피연산자 이중 버퍼링이 남았다.")


async def run_gemm_stream(dut, Kt: int, n_ops: int) -> dict:
    """디스크립터 n_ops 개를 연속으로 밀어 넣고 전체 사이클을 잰다.

    파이프라인의 값어치는 단일 연산이 아니라 처리량에서 나온다.
    디스크립터 하나만 돌리면 쓰기가 겹칠 상대가 없어 이득이 0이다.
    """
    mem = SimpleMem()
    start_dma(dut, mem, log=False)

    A = np.random.randint(-128, 128, size=(16, Kt), dtype=np.int8)
    B = np.random.randint(-128, 128, size=(Kt, 16), dtype=np.int8)
    act_base, wgt_base, out_base = 0x1000, 0x2000, 0x3000
    for k in range(Kt):
        mem.write_bytes(act_base + 16 * k,
                        bytes((int(A[i, k]) & 0xFF) for i in range(16)))
        mem.write_bytes(wgt_base + 16 * k,
                        bytes((int(B[k, j]) & 0xFF) for j in range(16)))

    def make_desc(i):
        d = bytearray(64)
        d[0:1] = bytes([0x02])
        d[16:24] = u64_le(act_base)
        d[24:32] = u64_le(wgt_base)
        d[32:40] = u64_le(out_base + 0x1000 * i)
        d[40:44] = u32_le(Kt)
        return d

    comp = 0
    total = 0
    issued = 0
    done_seen = 0
    started = False

    # RisingEdge 직후에는 엣지 반영 전 값이 보인다. Timer 로 한 델타
    # 넘긴 뒤에 읽어야 핸드셰이크를 한 번만 센다.
    for _ in range(500_000):
        want = issued < n_ops
        if want:
            d = make_desc(issued)
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
            if STATE[int(dut.u_core.st.value)] == "COMPUTE":
                comp += 1
            if int(dut.done_pulse.value) == 1:
                done_seen += 1
                if done_seen == n_ops:
                    break

    dut.desc_valid.value = 0
    assert done_seen == n_ops, f"{done_seen}/{n_ops} 만 완료"
    return {"total": total, "compute": comp, "ops": n_ops}


@cocotb.test()
async def test_throughput_streamed(dut):
    """gemm_top 경유 연속 처리량 — ctrl_fsm 이 직렬화하는 것을 보인다.

    gemm_core 자체는 쓰기 FSM 분리로 다음 디스크립터를 일찍 받는다.
    그러나 ctrl_fsm 이 ST_WAIT 에서 core_done 을 기다린 뒤에야
    desc_ready 를 올리므로, 시스템 레벨에서는 그 이득이 보이지 않는다.
    코어가 낼 수 있는 처리량은 test_gemm_core_throughput.py 에서 잰다.
    """
    await boot(dut)

    dut._log.info("")
    dut._log.info(f"{'Kt':>5} {'ops':>5} {'전체':>8} {'연산':>8} {'op당':>8} {'가동률':>8}")
    dut._log.info("-" * 52)

    for Kt in (1, 16, 64):
        r = await run_gemm_stream(dut, Kt, n_ops=4)
        per = r["total"] / r["ops"]
        util = r["compute"] / r["total"]
        dut._log.info(f"{Kt:>5} {r['ops']:>5} {r['total']:>8} {r['compute']:>8} "
                      f"{per:>8.1f} {util*100:>7.1f}%")

    dut._log.info("")
    dut._log.info("op 당 사이클이 단일 연산 표와 거의 같다 — ctrl_fsm 이 다음")
    dut._log.info("디스크립터를 core_done 이후에만 받기 때문이다.")
