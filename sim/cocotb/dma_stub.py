import cocotb
from cocotb.triggers import RisingEdge, ReadOnly
from cocotb.utils import get_sim_time


class SimpleMem:
    def __init__(self):
        self.mem = {}

    def write_bytes(self, addr: int, data: bytes):
        for i, b in enumerate(data):
            self.mem[addr + i] = b

    def read_bytes(self, addr: int, n: int) -> bytes:
        return bytes(self.mem.get(addr + i, 0) for i in range(n))


def has_xz(binstr: str):
    bad = []
    width = len(binstr)
    for i in range(width):
        ch = binstr[width - 1 - i].lower()
        if ch not in ("0", "1"):
            bad.append(i)
    return (len(bad) > 0), bad


class DMAStub:
    def __init__(self, dut, mem: SimpleMem, beat_bytes=16, log=False):
        self.dut = dut
        self.mem = mem
        self.beat_bytes = beat_bytes
        self.log = log
        self.errors = []   # ✅ 에러 누적 버퍼

    async def run_read(self):
        dut = self.dut

        rd_active = False
        rd_base = 0
        rd_beats = 0
        rd_idx = 0
        rd_done_pulse = False

        dut.rd_req_ready.value = 1
        dut.rd_data_valid.value = 0
        dut.rd_data_last.value = 0
        dut.rd_data.value = 0
        dut.rd_done.value = 0

        while True:
            await RisingEdge(dut.clk)

            dut.rd_req_ready.value = 1
            dut.rd_done.value = 1 if rd_done_pulse else 0
            rd_done_pulse = False

            if rd_active:
                data = self.mem.read_bytes(
                    rd_base + rd_idx * self.beat_bytes,
                    self.beat_bytes,
                )
                dut.rd_data.value = int.from_bytes(data, "little")
                dut.rd_data_valid.value = 1
                dut.rd_data_last.value = (rd_idx == rd_beats - 1)
            else:
                dut.rd_data_valid.value = 0
                dut.rd_data_last.value = 0

            await ReadOnly()

            if (not rd_active and
                int(dut.rd_req_valid.value) == 1 and
                int(dut.rd_req_ready.value) == 1):

                rd_base = int(dut.rd_req_addr.value)
                length = int(dut.rd_req_len_bytes.value)
                assert length % self.beat_bytes == 0

                rd_beats = length // self.beat_bytes
                rd_idx = 0
                rd_active = True

            if rd_active:
                if (int(dut.rd_data_valid.value) == 1 and
                    int(dut.rd_data_ready.value) == 1):

                    if rd_idx == rd_beats - 1:
                        rd_active = False
                        rd_done_pulse = True
                    else:
                        rd_idx += 1

    async def run_write(self):
        dut = self.dut

        wr_active = False
        wr_base = 0
        wr_beats = 0
        wr_idx = 0
        wr_done_pulse = False

        dut.wr_req_ready.value = 1
        dut.wr_data_ready.value = 1
        dut.wr_done.value = 0

        while True:
            await RisingEdge(dut.clk)

            dut.wr_req_ready.value = 1
            dut.wr_data_ready.value = 1
            dut.wr_done.value = 1 if wr_done_pulse else 0
            wr_done_pulse = False

            await ReadOnly()

            if (not wr_active and
                int(dut.wr_req_valid.value) == 1 and
                int(dut.wr_req_ready.value) == 1):

                wr_base = int(dut.wr_req_addr.value)
                length = int(dut.wr_req_len_bytes.value)
                assert length % self.beat_bytes == 0

                wr_beats = length // self.beat_bytes
                wr_idx = 0
                wr_active = True

            if wr_active:
                if (int(dut.wr_data_valid.value) == 1 and
                    int(dut.wr_data_ready.value) == 1):

                    v = dut.wr_data.value
                    bx = v.binstr
                    bad, bits = has_xz(bx)

                    if bad:
                        self.errors.append({
                            "time_ns": get_sim_time("ns"),
                            "type": "WR_DATA_X",
                            "bits": bits,
                            "data": bx,
                            "addr": wr_base + wr_idx * self.beat_bytes,
                            "beat": wr_idx,
                        })

                    # Write the actual beat data to the correct address
                    data = int(v).to_bytes(self.beat_bytes, "little")
                    self.mem.write_bytes(
                        wr_base + wr_idx * self.beat_bytes, data
                    )

                    if wr_idx == wr_beats - 1:
                        wr_active = False
                        wr_done_pulse = True
                    else:
                        wr_idx += 1

    def report_errors(self):
        if not self.errors:
            return

        self.dut._log.error("========== DMA ERRORS ==========")
        for e in self.errors:
            self.dut._log.error(
                f"[t={e['time_ns']}ns] "
                f"{e['type']} addr=0x{e['addr']:x} "
                f"beat={e['beat']} bits={e['bits']}"
            )
        assert False, f"{len(self.errors)} DMA errors detected"


def start_dma(dut, mem: SimpleMem, log=False):
    dma = DMAStub(dut, mem, log=log)
    cocotb.start_soon(dma.run_read())
    cocotb.start_soon(dma.run_write())
    return dma
