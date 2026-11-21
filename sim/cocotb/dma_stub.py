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
    def __init__(self, dut, mem: SimpleMem, beat_bytes=16, log=False,
                 rd_beat_bytes=None, wr_beat_bytes=None,
                 rd_latency=0, wr_latency=0):
        """읽기와 쓰기는 독립 채널이라 폭이 다를 수 있다.
        gemm_core 의 RD_DATA_W 를 넓히면 읽기 비트만 넓어진다.

        rd_latency / wr_latency: 요청 수락과 첫 비트 사이의 사이클 수.
        기본 0 은 **낙관적 상한**이다 — 실제 DDR4 는 요청에서 첫 데이터까지
        수십 사이클이 걸린다. 지연을 넣어도 파이프라인이 그것을 흡수해야
        한다 (흡수하지 못하면 설계가 지연에 민감하다는 뜻이고, 그건
        보드에서 처음 드러나면 늦다).
        """
        self.dut = dut
        self.mem = mem
        self.beat_bytes = beat_bytes
        self.rd_beat_bytes = rd_beat_bytes or beat_bytes
        self.wr_beat_bytes = wr_beat_bytes or beat_bytes
        self.rd_latency = rd_latency
        self.wr_latency = wr_latency
        self.log = log
        self.errors = []   # ✅ 에러 누적 버퍼
        self._tasks = []   # 중지할 수 있도록 보관

    async def run_read(self):
        dut = self.dut

        rd_active = False
        rd_base = 0
        rd_beats = 0
        rd_idx = 0
        rd_done_pulse = False
        rd_wait = 0          # 남은 지연 사이클

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

            if rd_active and rd_wait == 0:
                data = self.mem.read_bytes(
                    rd_base + rd_idx * self.rd_beat_bytes,
                    self.rd_beat_bytes,
                )
                dut.rd_data.value = int.from_bytes(data, "little")
                dut.rd_data_valid.value = 1
                dut.rd_data_last.value = (rd_idx == rd_beats - 1)
            else:
                dut.rd_data_valid.value = 0
                dut.rd_data_last.value = 0
            if rd_active and rd_wait > 0:
                rd_wait -= 1

            await ReadOnly()

            if (not rd_active and
                int(dut.rd_req_valid.value) == 1 and
                int(dut.rd_req_ready.value) == 1):

                rd_base = int(dut.rd_req_addr.value)
                length = int(dut.rd_req_len_bytes.value)
                assert length % self.rd_beat_bytes == 0

                rd_beats = length // self.rd_beat_bytes
                rd_idx = 0
                rd_active = True
                rd_wait = self.rd_latency

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
                assert length % self.wr_beat_bytes == 0

                wr_beats = length // self.wr_beat_bytes
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
                            "addr": wr_base + wr_idx * self.wr_beat_bytes,
                            "beat": wr_idx,
                        })

                    # Write the actual beat data to the correct address
                    data = int(v).to_bytes(self.wr_beat_bytes, "little")
                    self.mem.write_bytes(
                        wr_base + wr_idx * self.wr_beat_bytes, data
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


    def stop(self):
        """구동 코루틴을 멈춘다. 같은 테스트에서 설정을 바꿔 다시 띄울 때 필요."""
        for t in self._tasks:
            try:
                t.kill()
            except Exception:
                pass
        self._tasks = []
        _ACTIVE.pop(id(self.dut), None)


# dut 하나에 스텁 하나. 두 개가 같은 신호를 구동하면 조용히 틀린다 —
# 이 저장소에서 여러 번 당했다 (한 번은 "지연을 넣었는데 손실 0%" 로
# 나타났다. 지연 0 짜리 첫 스텁이 계속 rd_data_valid 를 세우고 있었다).
# 그래서 중복 기동을 **에러로 만든다.**
_ACTIVE: dict[int, "DMAStub"] = {}


def _alive(task) -> bool:
    """코루틴이 아직 도는가. cocotb 버전에 따라 API 가 다르므로 방어적으로."""
    for attr in ("done", "_finished"):
        v = getattr(task, attr, None)
        if v is None:
            continue
        try:
            finished = v() if callable(v) else bool(v)
        except Exception:
            continue
        return not finished
    return True   # 알 수 없으면 살아 있다고 본다 (안전한 쪽)


def start_dma(dut, mem: SimpleMem, log=False, rd_beat_bytes=None,
              wr_beat_bytes=None, rd_latency=0, wr_latency=0):
    key = id(dut)
    prev = _ACTIVE.get(key)
    if prev is not None:
        # cocotb 는 테스트가 끝나면 코루틴을 죽인다. 같은 모듈의 다음
        # 테스트가 다시 띄우는 것은 정상이므로, 살아 있는지 확인한다.
        if any(_alive(t) for t in prev._tasks):
            raise RuntimeError(
                "이 dut 에 DMA 스텁이 이미 돌고 있다. 두 스텁이 같은 신호를 "
                "구동하면 측정이 조용히 틀린다. 설정을 바꾸려면 먼저 "
                "기존 스텁의 .stop() 을 부르고 다시 start_dma() 하라."
            )
        _ACTIVE.pop(key, None)
    dma = DMAStub(dut, mem, log=log, rd_beat_bytes=rd_beat_bytes,
                  wr_beat_bytes=wr_beat_bytes,
                  rd_latency=rd_latency, wr_latency=wr_latency)
    dma._tasks = [cocotb.start_soon(dma.run_read()),
                  cocotb.start_soon(dma.run_write())]
    _ACTIVE[key] = dma
    return dma
