"""test_gemm_utilization.py — MAC 배열이 실제로 얼마나 일하는지 계측한다.

gemm_core 의 FSM 은 ACT 읽기 → WGT 읽기 → 연산 → 쓰기 순으로 완전히
직렬이다. 페치 동안 MAC 배열이 통째로 놀고, 연산 동안 DMA 가 논다.
이 테스트는 상태별 사이클을 세어 그 비율을 숫자로 남긴다.

측정만 하고 실패시키지 않는다 — 개선 전후를 같은 자로 재기 위한 기준선이다.
"""
import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from dma_stub import SimpleMem, start_dma

# gemm_core.sv 의 state_t 순서
STATE = ["IDLE", "RD_ACT_REQ", "RD_ACT_RECV", "RD_WGT_REQ", "RD_WGT_RECV",
         "ACC_CLR", "COMPUTE", "POST_MAC", "LATCH", "WR_REQ", "WR_SEND", "DONE"]
FETCH = {"RD_ACT_REQ", "RD_ACT_RECV", "RD_WGT_REQ", "RD_WGT_RECV"}
WRITE = {"WR_REQ", "WR_SEND"}


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
    total = 0
    seen_work = False
    for _ in range(200_000):
        await RisingEdge(dut.clk)
        st = STATE[int(dut.u_core.st.value)]
        if st != "IDLE":
            seen_work = True
        if seen_work:
            counts[st] += 1
            total += 1
        if seen_work and st == "IDLE":
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
        write = sum(c[s] for s in WRITE)
        util = comp / total if total else 0.0
        dut._log.info(f"{Kt:>5} {total:>8} {comp:>8} {fetch:>8} {write:>8} "
                      f"{util*100:>7.1f}%")
        assert total > 0, f"Kt={Kt}: FSM 이 움직이지 않았다"

    dut._log.info("")
    dut._log.info("페치와 연산이 겹치지 않으므로 Kt 가 커져도 페치 사이클이")
    dut._log.info("그대로 가동률을 깎는다. 이중 버퍼링의 기준선이다.")
