import cocotb
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
import numpy as np

@cocotb.test()
async def test_mac_array_basic(dut):
    # 100MHz
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    clk = dut.clk

    async def tick(n=1):
        for _ in range(n):
            await RisingEdge(clk)

    dut.rst_n.value = 0
    dut.en.value = 0
    dut.acc_clr.value = 0
    await tick(2)
    dut.rst_n.value = 1
    await tick(2)

    for Kt in [1, 2, 7, 16]:
        A = np.random.randint(-128, 128, size=(16, Kt), dtype=np.int8)
        B = np.random.randint(-128, 128, size=(Kt, 16), dtype=np.int8)

        # clear accumulators
        dut.acc_clr.value = 1
        await tick(1)
        dut.acc_clr.value = 0

        # K-loop
        for k in range(Kt):
            for i in range(16):
                dut.a_row[i].value = int(A[i, k])
            for j in range(16):
                dut.b_col[j].value = int(B[k, j])

            dut.en.value = 1
            await tick(1)

        dut.en.value = 0
        await tick(1)

        # acc_out_flat is a packed vector [32*256-1:0]
        C_hw = np.zeros((16, 16), dtype=np.int32)
        flat_val = int(dut.acc_out_flat.value)
        for i in range(16):
            for j in range(16):
                idx = (i * 16) + j
                raw = (flat_val >> (idx * 32)) & 0xFFFFFFFF
                # sign extend 32-bit
                if raw >= 0x80000000:
                    raw -= 0x100000000
                C_hw[i, j] = raw

        C_ref = A.astype(np.int32) @ B.astype(np.int32)
        assert np.all(C_hw == C_ref), f"Mismatch Kt={Kt}"

        dut._log.info(f"PASS mac_array: Kt={Kt}")
