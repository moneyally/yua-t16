"""기대값 출처: AMBA AXI4 버스트 규칙 + 파이썬 메모리 모델(왕복한 바이트가 기대값).
계산이 없는 어댑터라 골든 모델이 없다 — **프로토콜과 데이터 왕복이 기대값**이다.

tb_axi4_master_adapter.py — rtl/axi4_master_adapter.sv (외부 메모리 경로)

AXI4 슬레이브 메모리 모델을 붙여서 본다. 모델은 **프로토콜 위반을 잡는다** —
봐주지 않는다. 실제 DDR 컨트롤러가 안 봐주기 때문이다.

  M1. 짧은 읽기 (1 beat) 왕복
  M2. **256 beat 초과** 읽기 → 버스트가 쪼개지고 데이터는 이어진다
  M3. **4KB 경계를 넘는** 읽기 → 경계에서 쪼개진다 (안 쪼개면 모델이 잡는다)
  M4. 쓰기 왕복 + 읽기로 확인
  M5. 4KB 경계를 넘는 쓰기
  M6. 소비자 backpressure (rd_data_ready 를 띄엄띄엄) 에도 데이터가 안 샌다
  M7. RRESP/BRESP 오류가 err 로 올라온다
  M8. len=0 요청이 멈추지 않는다 (done 이 뜬다)
"""

import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_W = 128
BYTES = DATA_W // 8      # 16
PAGE = 4096


class AxiSlaveMemory:
    """AXI4 슬레이브 메모리 모델. **프로토콜을 검사한다.**

    봐주지 않는 것들:
      - 버스트가 4KB 경계를 넘으면 즉시 AssertionError
      - AxLEN > 255 (하드웨어적으로 불가능하지만 계산 실수를 잡는다)
      - WLAST 가 예고한 beat 수와 다르면 AssertionError
    """

    def __init__(self, dut, resp_read=0, resp_write=0):
        self.dut = dut
        self.mem = {}              # beat 주소 -> 128비트 정수
        self.resp_read = resp_read
        self.resp_write = resp_write
        self.read_bursts = []      # (addr, beats) 기록 — 쪼개짐을 확인한다
        self.write_bursts = []

    def _check_burst(self, addr, beats, tag):
        assert beats >= 1, f"{tag}: beats={beats}"
        assert beats <= 256, f"{tag}: AxLEN 초과 beats={beats} (AXI4 최대 256)"
        first_page = addr // PAGE
        last_page = (addr + beats * BYTES - 1) // PAGE
        assert first_page == last_page, (
            f"{tag}: **4KB 경계를 넘는 버스트** addr={addr:#x} beats={beats} "
            f"(끝 {addr + beats*BYTES - 1:#x}). AXI4 위반이다."
        )

    async def run_read(self):
        dut = self.dut
        dut.m_axi_arready.value = 1
        dut.m_axi_rvalid.value = 0
        dut.m_axi_rlast.value = 0
        dut.m_axi_rresp.value = 0
        dut.m_axi_rid.value = 0
        while True:
            await RisingEdge(dut.clk)
            if not (int(dut.m_axi_arvalid.value) and int(dut.m_axi_arready.value)):
                continue
            addr = int(dut.m_axi_araddr.value)
            beats = int(dut.m_axi_arlen.value) + 1
            self._check_burst(addr, beats, "AR")
            self.read_bursts.append((addr, beats))

            dut.m_axi_arready.value = 0
            for i in range(beats):
                dut.m_axi_rvalid.value = 1
                dut.m_axi_rdata.value = self.mem.get(addr // BYTES + i, 0)
                dut.m_axi_rlast.value = 1 if i == beats - 1 else 0
                dut.m_axi_rresp.value = self.resp_read
                # rready 를 볼 때까지 유지한다 (AXI 규약)
                while True:
                    await RisingEdge(dut.clk)
                    if int(dut.m_axi_rready.value):
                        break
            dut.m_axi_rvalid.value = 0
            dut.m_axi_rlast.value = 0
            dut.m_axi_arready.value = 1

    async def run_write(self):
        dut = self.dut
        dut.m_axi_awready.value = 1
        dut.m_axi_wready.value = 0
        dut.m_axi_bvalid.value = 0
        dut.m_axi_bresp.value = 0
        dut.m_axi_bid.value = 0
        while True:
            await RisingEdge(dut.clk)
            if not (int(dut.m_axi_awvalid.value) and int(dut.m_axi_awready.value)):
                continue
            addr = int(dut.m_axi_awaddr.value)
            beats = int(dut.m_axi_awlen.value) + 1
            self._check_burst(addr, beats, "AW")
            self.write_bursts.append((addr, beats))

            dut.m_axi_awready.value = 0
            dut.m_axi_wready.value = 1
            got = 0
            while got < beats:
                await RisingEdge(dut.clk)
                if int(dut.m_axi_wvalid.value) and int(dut.m_axi_wready.value):
                    self.mem[addr // BYTES + got] = int(dut.m_axi_wdata.value)
                    last = int(dut.m_axi_wlast.value)
                    got += 1
                    assert last == (1 if got == beats else 0), (
                        f"W: WLAST 가 어긋났다 (beat {got}/{beats}, wlast={last})"
                    )
            dut.m_axi_wready.value = 0

            dut.m_axi_bvalid.value = 1
            dut.m_axi_bresp.value = self.resp_write
            while True:
                await RisingEdge(dut.clk)
                if int(dut.m_axi_bready.value):
                    break
            dut.m_axi_bvalid.value = 0
            # **다음 버스트를 받을 준비로 되돌린다.** 안 되돌리면 쪼개진 두 번째
            # 버스트의 AW 가 영영 안 받아들여져서 테스트가 "RTL 이 멈췄다"고 거짓말한다
            # (M5 가 실제로 그렇게 실패했다 — 읽기 쪽에는 이 복원이 있었다).
            dut.m_axi_awready.value = 1


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.rd_req_valid.value = 0
    dut.rd_req_addr.value = 0
    dut.rd_req_len_bytes.value = 0
    dut.rd_data_ready.value = 1
    dut.wr_req_valid.value = 0
    dut.wr_req_addr.value = 0
    dut.wr_req_len_bytes.value = 0
    dut.wr_data_valid.value = 0
    dut.wr_data.value = 0
    dut.wr_data_last.value = 0
    dut.m_axi_arready.value = 0
    dut.m_axi_rvalid.value = 0
    dut.m_axi_rdata.value = 0
    dut.m_axi_rresp.value = 0
    dut.m_axi_rlast.value = 0
    dut.m_axi_rid.value = 0
    dut.m_axi_awready.value = 0
    dut.m_axi_wready.value = 0
    dut.m_axi_bvalid.value = 0
    dut.m_axi_bresp.value = 0
    dut.m_axi_bid.value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def setup(dut, **kw):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    mem = AxiSlaveMemory(dut, **kw)
    cocotb.start_soon(mem.run_read())
    cocotb.start_soon(mem.run_write())
    await RisingEdge(dut.clk)
    return mem


async def do_read(dut, addr, n_beats, *, gap=0, guard=200000):
    """읽기 요청 하나. gap>0 이면 소비자가 띄엄띄엄 받는다 (backpressure)."""
    await FallingEdge(dut.clk)
    dut.rd_req_addr.value = addr
    dut.rd_req_len_bytes.value = n_beats * BYTES
    dut.rd_req_valid.value = 1
    while not int(dut.rd_req_ready.value):
        await FallingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rd_req_valid.value = 0

    out = []
    saw_last = 0
    err = 0
    tick = 0
    for _ in range(guard):
        if gap:
            tick += 1
            dut.rd_data_ready.value = 1 if (tick % (gap + 1) == 0) else 0
        await FallingEdge(dut.clk)
        if int(dut.rd_data_valid.value) and int(dut.rd_data_ready.value):
            out.append(int(dut.rd_data.value))
            if int(dut.rd_data_last.value):
                saw_last += 1
        if int(dut.err.value):
            err = 1
        if int(dut.rd_done.value):
            dut.rd_data_ready.value = 1
            return out, saw_last, err
        await RisingEdge(dut.clk)
    raise AssertionError(f"읽기가 끝나지 않았다 addr={addr:#x} beats={n_beats}")


async def do_write(dut, addr, beats_data, *, guard=200000):
    """쓰기 요청 하나. beats_data 는 128비트 정수 리스트."""
    await FallingEdge(dut.clk)
    dut.wr_req_addr.value = addr
    dut.wr_req_len_bytes.value = len(beats_data) * BYTES
    dut.wr_req_valid.value = 1
    while not int(dut.wr_req_ready.value):
        await FallingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.wr_req_valid.value = 0

    i = 0
    err = 0
    for _ in range(guard):
        await FallingEdge(dut.clk)
        if i < len(beats_data):
            dut.wr_data_valid.value = 1
            dut.wr_data.value = beats_data[i]
            dut.wr_data_last.value = 1 if i == len(beats_data) - 1 else 0
        else:
            dut.wr_data_valid.value = 0
        accepted = int(dut.wr_data_valid.value) and int(dut.wr_data_ready.value)
        if int(dut.err.value):
            err = 1
        done = int(dut.wr_done.value)
        await RisingEdge(dut.clk)
        if accepted:
            i += 1
        if done:
            dut.wr_data_valid.value = 0
            return err
    raise AssertionError(f"쓰기가 끝나지 않았다 addr={addr:#x}")


def pattern(i):
    """beat i 의 시험 데이터. 128비트를 꽉 채워서 잘림을 잡는다."""
    return ((0xA5A5_0000 + i) << 96) | ((0x1234_0000 + i) << 64) \
        | ((0xDEAD_0000 + i) << 32) | (0xBEEF_0000 + i)


@cocotb.test()
async def test_m1_short_read(dut):
    """M1: 1 beat 읽기 왕복."""
    mem = await setup(dut)
    mem.mem[0x1000 // BYTES] = pattern(7)

    out, last, err = await do_read(dut, 0x1000, 1)
    assert out == [pattern(7)], f"M1: {out}"
    assert last == 1, f"M1: last 가 {last}번 (1이어야 한다)"
    assert err == 0
    assert mem.read_bursts == [(0x1000, 1)], f"M1: 버스트 {mem.read_bursts}"


@cocotb.test()
async def test_m2_long_read_splits(dut):
    """M2: 300 beat 읽기 → **256 상한**에 맞춰 쪼개지고 데이터는 이어진다."""
    mem = await setup(dut)
    base = 0x10_0000                      # 4KB 정렬
    n = 300
    for i in range(n):
        mem.mem[base // BYTES + i] = pattern(i)

    out, last, err = await do_read(dut, base, n)
    assert err == 0
    assert len(out) == n, f"M2: beat 수 {len(out)} != {n}"
    assert out == [pattern(i) for i in range(n)], "M2: 데이터가 어긋났다"
    assert last == 1, f"M2: last 가 {last}번 — 버스트마다 뜨면 안 된다"
    assert len(mem.read_bursts) >= 2, f"M2: 안 쪼개졌다 {mem.read_bursts}"
    assert all(b <= 256 for _, b in mem.read_bursts), f"M2: {mem.read_bursts}"


@cocotb.test()
async def test_m3_read_crosses_4k(dut):
    """M3: 4KB 경계를 넘는 읽기 → 경계에서 쪼개진다.

    안 쪼개면 슬레이브 모델이 AssertionError 를 던진다 (봐주지 않는다).
    """
    mem = await setup(dut)
    base = 0x2000 - 5 * BYTES             # 경계 5 beat 앞
    n = 12                                # 경계를 확실히 넘는다
    for i in range(n):
        mem.mem[base // BYTES + i] = pattern(100 + i)

    out, last, err = await do_read(dut, base, n)
    assert err == 0
    assert out == [pattern(100 + i) for i in range(n)], "M3: 데이터가 어긋났다"
    assert last == 1
    assert len(mem.read_bursts) == 2, f"M3: 경계에서 2개로 쪼개져야 한다 {mem.read_bursts}"
    assert mem.read_bursts[0] == (base, 5), f"M3: 첫 버스트 {mem.read_bursts[0]}"
    assert mem.read_bursts[1] == (0x2000, 7), f"M3: 둘째 버스트 {mem.read_bursts[1]}"


@cocotb.test()
async def test_m4_write_then_read(dut):
    """M4: 쓰기 왕복 — 쓴 뒤 읽어서 같은 값이 나온다."""
    await setup(dut)
    base = 0x20_0000
    data = [pattern(200 + i) for i in range(8)]

    err = await do_write(dut, base, data)
    assert err == 0
    out, last, rerr = await do_read(dut, base, len(data))
    assert rerr == 0 and last == 1
    assert out == data, f"M4: 쓴 것과 읽은 것이 다르다\n  쓴 것={data}\n  읽은 것={out}"


@cocotb.test()
async def test_m5_write_crosses_4k(dut):
    """M5: 4KB 경계를 넘는 쓰기도 쪼개진다."""
    mem = await setup(dut)
    base = 0x3000 - 3 * BYTES
    data = [pattern(300 + i) for i in range(10)]

    err = await do_write(dut, base, data)
    assert err == 0
    assert len(mem.write_bursts) == 2, f"M5: {mem.write_bursts}"
    assert mem.write_bursts[0] == (base, 3), f"M5: 첫 버스트 {mem.write_bursts[0]}"

    out, _, _ = await do_read(dut, base, len(data))
    assert out == data, "M5: 경계를 넘어 쓴 데이터가 어긋났다"


@cocotb.test()
async def test_m6_consumer_backpressure(dut):
    """M6: 소비자가 띄엄띄엄 받아도 beat 가 새거나 겹치지 않는다."""
    mem = await setup(dut)
    base = 0x30_0000
    n = 40
    for i in range(n):
        mem.mem[base // BYTES + i] = pattern(400 + i)

    out, last, err = await do_read(dut, base, n, gap=2)
    assert err == 0 and last == 1
    assert out == [pattern(400 + i) for i in range(n)], (
        f"M6: backpressure 에서 데이터가 어긋났다 (받은 {len(out)}/{n})"
    )


@cocotb.test()
async def test_m7_error_response(dut):
    """M7: RRESP/BRESP 가 OKAY 가 아니면 err 로 올라온다. **조용히 넘어가지 않는다.**"""
    mem = await setup(dut, resp_read=2, resp_write=2)     # 2 = SLVERR
    mem.mem[0x40_0000 // BYTES] = pattern(1)

    _, _, err = await do_read(dut, 0x40_0000, 1)
    assert err == 1, "M7: RRESP=SLVERR 인데 err 가 안 떴다"

    werr = await do_write(dut, 0x40_1000, [pattern(2)])
    assert werr == 1, "M7: BRESP=SLVERR 인데 err 가 안 떴다"


@cocotb.test()
async def test_m8_zero_length(dut):
    """M8: 길이 0 요청이 멈추지 않는다 (done 이 뜬다).

    길이 0 은 호스트 버그지만, 하드웨어가 **멈추면** 그게 더 나쁘다.
    """
    await setup(dut)
    out, last, err = await do_read(dut, 0x50_0000, 0)
    assert out == [] and last == 0 and err == 0, f"M8 읽기: {out} {last} {err}"

    werr = await do_write(dut, 0x50_0000, [])
    assert werr == 0, "M8 쓰기"
