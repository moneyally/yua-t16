import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import numpy as np

from dma_stub import SimpleMem, start_dma


def u16_le(x): return int(x).to_bytes(2, "little", signed=False)
def u32_le(x): return int(x).to_bytes(4, "little", signed=False)
def u64_le(x): return int(x).to_bytes(8, "little", signed=False)


async def tick(dut, n=1):
    for _ in range(n):
        await RisingEdge(dut.clk)


async def wait_sig_high(dut, sig, max_cycles, name):
    for c in range(max_cycles):
        if int(sig.value) == 1:
            return c
        await RisingEdge(dut.clk)
    raise AssertionError(f"TIMEOUT: {name} did not go high within {max_cycles} cycles")


# 🔥 핵심 디버그 로그 함수
async def log_core_signals(dut, cycles=200):
    dut._log.info("=== START CORE SIGNAL TRACE ===")
    for c in range(cycles):
        st = int(dut.u_core.st.value)

        def fmt128(sig):
            bs = sig.value.binstr.lower()
            if ("x" in bs) or ("z" in bs):
                return bs
            return f"0x{int(sig.value):032x}"

        acc_flat0 = dut.u_core.acc_out_flat[0].value.binstr
        acc_lat0  = dut.u_core.acc_out_lat[0].value.binstr

        mac_en = int(dut.u_core.mac_en.value)
        act_re = int(dut.u_core.act_re.value)
        wgt_re = int(dut.u_core.wgt_re.value)
        raddr  = int(dut.u_core.act_raddr.value)
        act_d  = fmt128(dut.u_core.act_rdata)
        wgt_d  = fmt128(dut.u_core.wgt_rdata)

        wr_req_v = int(dut.wr_req_valid.value)
        wr_req_r = int(dut.wr_req_ready.value)
        wr_v     = int(dut.wr_data_valid.value)
        wr_r     = int(dut.wr_data_ready.value)
        wr_last  = int(dut.wr_data_last.value)

        dut._log.info(
            f"[C{c:03d}] "
            f"st={st:02d} "
            f"issue_k_addr={int(dut.u_core.issue_k_addr.value):02d} "
            f"issue_k_mac={int(dut.u_core.issue_k_mac.value):02d} | "
            f"act_issue={int(dut.u_core.act_issue.value)} "
            f"act_issue_d={int(dut.u_core.act_issue_d.value)} "
            f"mac_en={mac_en} | "
            f"re(a/w)={act_re}/{wgt_re} raddr={raddr:02d} | "
            f"act_dout={act_d} wgt_dout={wgt_d} | "
            f"wr_req(v/r)={wr_req_v}/{wr_req_r} wr(v/r/last)={wr_v}/{wr_r}/{wr_last} | "
            f"acc_flat0={acc_flat0} "
            f"acc_lat0={acc_lat0}"
        )
        await RisingEdge(dut.clk)
    dut._log.info("=== END CORE SIGNAL TRACE ===")



@cocotb.test()
async def test_gemm_top_basic(dut):
    # clock
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    # reset & defaults
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

    # ✅ RESET CHECK: acc regs really become 0? (if not, reset polarity/connection first)
    for idx in [0, 1, 15, 16, 255]:
        bs = dut.u_core.acc_out_flat[idx].value.binstr.lower()
        assert ("x" not in bs) and ("z" not in bs), f"acc_out_flat[{idx}] has X/Z after reset: {bs}"
        assert int(dut.u_core.acc_out_flat[idx].value) == 0, f"acc_out_flat[{idx}] != 0 after reset"

    mem = SimpleMem()
    dma = start_dma(dut, mem, log=False)

    dut._log.info("=== START test_gemm_top_basic ===")

    # 🔥 문제 고정: Kt=1만
    Kt = 1
    dut._log.info(f"--- CASE Kt={Kt} ---")

    A = np.random.randint(-128, 128, size=(16, Kt), dtype=np.int8)
    B = np.random.randint(-128, 128, size=(Kt, 16), dtype=np.int8)

    act_base = 0x1000
    wgt_base = 0x2000
    out_base = 0x3000

    for k in range(Kt):
        mem.write_bytes(act_base + 16 * k,
                        bytes((int(A[i, k]) & 0xFF) for i in range(16)))
        mem.write_bytes(wgt_base + 16 * k,
                        bytes((int(B[k, j]) & 0xFF) for j in range(16)))

    desc = bytearray(64)
    desc[0:1]   = bytes([0x02])
    desc[16:24] = u64_le(act_base)
    desc[24:32] = u64_le(wgt_base)
    desc[32:40] = u64_le(out_base)
    desc[40:44] = u32_le(Kt)

    await wait_sig_high(dut, dut.desc_ready, 2000, "desc_ready")

    for i in range(64):
        dut.desc_bytes[i].value = desc[i]

    dut.desc_valid.value = 1
    await tick(dut, 1)
    dut.desc_valid.value = 0

    dut._log.info("descriptor pushed, waiting done_pulse...")

    # 🔥 done 기다리는 동안 신호 로그
    log_task = cocotb.start_soon(log_core_signals(dut, cycles=200))

    await wait_sig_high(dut, dut.done_pulse, 200000, "done_pulse")
    await log_task

    out = mem.read_bytes(out_base, 1024)
    C_hw = np.frombuffer(out, dtype=np.int32).reshape(16, 16)
    C_ref = A.astype(np.int32) @ B.astype(np.int32)

    dut._log.info(f"C_hw[0,:8]={C_hw[0,:8]}")
    dut._log.info(f"C_ref[0,:8]={C_ref[0,:8]}")

    assert np.all(C_hw == C_ref), "Mismatch GEMM Kt=1"
    dma.report_errors()
