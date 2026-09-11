"""기대값 출처: AMBA AXI4-Lite 핸드셰이크 규약 + reg_top 버스 규약(이 레포 기존 인터페이스).
계산이 없는 어댑터라 골든 모델이 없다 — **프로토콜 자체가 기대값**이다.

tb_axil_reg_bridge.py — rtl/axil_reg_bridge.sv (PLAN W11, 보드 前 작업)

  A1. 쓰기: AW/W 가 같은 사이클에 와도 동작하고 BVALID 가 정확히 뜬다
  A2. 쓰기: AW 가 W 보다 **먼저** 와도 동작한다 (AXI4-Lite 는 순서를 강제하지 않는다)
  A3. 쓰기: W 가 AW 보다 **먼저** 와도 동작한다
  A4. 읽기: 주소를 걸고 RD_LAT 사이클 뒤 rd_data 를 잡는다 → **한 번만 읽으면 된다**
  A5. 읽기 중에는 reg_wr_en 이 절대 뜨지 않는다 (쓰기 오염 방지)
  A6. BREADY/RREADY 를 늦게 줘도 응답이 유지된다 (VALID 는 READY 를 기다린다)
  A7. 연속 트랜잭션이 서로 섞이지 않는다

DUT 뒤에 가짜 레지스터 파일(파이썬 dict)을 붙여서 왕복을 본다.
"""

import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RD_LAT = 2


class FakeRegFile:
    """reg_top 자리에 놓는 가짜 레지스터 파일.

    **1사이클 지연**을 흉내 낸다 — 스크래치/트레이스 창이 그렇기 때문이다.
    지연이 없는 레지스터만 흉내 내면 브리지의 대기 로직을 검사하지 못한다.
    """

    def __init__(self, dut):
        self.dut = dut
        self.mem = {}
        self._pending = 0

    async def run(self):
        while True:
            await RisingEdge(self.dut.clk)
            # 쓰기: wr_en 이 뜬 사이클의 주소/데이터를 받는다
            if int(self.dut.reg_wr_en.value):
                self.mem[int(self.dut.reg_addr.value)] = int(self.dut.reg_wr_data.value)
            # 읽기: 주소를 받고 **다음 사이클**에 데이터를 내보낸다
            self.dut.reg_rd_data.value = self._pending
            self._pending = self.mem.get(int(self.dut.reg_addr.value), 0)


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.s_axil_awaddr.value = 0
    dut.s_axil_awvalid.value = 0
    dut.s_axil_wdata.value = 0
    dut.s_axil_wstrb.value = 0xF
    dut.s_axil_wvalid.value = 0
    dut.s_axil_bready.value = 1
    dut.s_axil_araddr.value = 0
    dut.s_axil_arvalid.value = 0
    dut.s_axil_rready.value = 1
    dut.reg_rd_data.value = 0
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def _send_aw(dut, addr, delay):
    """AW 채널 하나를 보낸다. **ready 를 falling edge 에서 확인**한 뒤 한 edge 만 보낸다.

    처음에 `RisingEdge` 직후에 ready 를 봤다가 틀렸다 — 그 시점엔 이미 상태가
    바뀌어 ready 가 0 이라 VALID 를 못 내렸고, 다음 트랜잭션이 통째로 막혔다.
    (cdc_fifo 에서 배운 것과 같은 종류다: 등록된 신호는 falling edge 에서 본다.)
    """
    for _ in range(delay):
        await RisingEdge(dut.clk)
    dut.s_axil_awaddr.value = addr
    dut.s_axil_awvalid.value = 1
    while True:
        await FallingEdge(dut.clk)
        if int(dut.s_axil_awready.value):
            break
    await RisingEdge(dut.clk)      # 이 edge 에서 수락된다
    dut.s_axil_awvalid.value = 0


async def _send_w(dut, data, delay):
    for _ in range(delay):
        await RisingEdge(dut.clk)
    dut.s_axil_wdata.value = data
    dut.s_axil_wstrb.value = 0xF
    dut.s_axil_wvalid.value = 1
    while True:
        await FallingEdge(dut.clk)
        if int(dut.s_axil_wready.value):
            break
    await RisingEdge(dut.clk)
    dut.s_axil_wvalid.value = 0


async def axil_write(dut, addr, data, *, aw_delay=0, w_delay=0, guard=64):
    """AXI4-Lite 쓰기. AW 와 W 를 **독립 코루틴**으로 보낸다.

    AXI4-Lite 는 두 채널의 순서를 강제하지 않는다 — delay 로 순서를 바꿔 가며
    브리지가 둘 다 견디는지 본다.
    """
    cocotb.start_soon(_send_aw(dut, addr, aw_delay))
    cocotb.start_soon(_send_w(dut, data, w_delay))

    for _ in range(guard):
        await FallingEdge(dut.clk)
        if int(dut.s_axil_bvalid.value):
            resp = int(dut.s_axil_bresp.value)
            await RisingEdge(dut.clk)      # BREADY=1 이므로 여기서 소비된다
            return resp
    raise AssertionError(f"쓰기 응답(BVALID)이 오지 않았다 addr={addr:#x}")


async def axil_read(dut, addr, guard=64):
    """AXI4-Lite 읽기. **한 번만** 읽는다 (브리지가 지연을 흡수한다)."""
    dut.s_axil_araddr.value = addr
    dut.s_axil_arvalid.value = 1
    while True:
        await FallingEdge(dut.clk)
        if int(dut.s_axil_arready.value):
            break
    await RisingEdge(dut.clk)
    dut.s_axil_arvalid.value = 0

    for _ in range(guard):
        await FallingEdge(dut.clk)
        if int(dut.s_axil_rvalid.value):
            data = int(dut.s_axil_rdata.value)
            resp = int(dut.s_axil_rresp.value)
            await RisingEdge(dut.clk)
            return data, resp
    raise AssertionError(f"읽기 응답(RVALID)이 오지 않았다 addr={addr:#x}")


async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    rf = FakeRegFile(dut)
    cocotb.start_soon(rf.run())
    return rf


@cocotb.test()
async def test_a1_write_read_same_cycle(dut):
    """A1: AW/W 동시 → 쓰기, 그리고 읽어서 같은 값이 나온다."""
    await setup(dut)

    assert await axil_write(dut, 0x100, 0xDEAD_BEEF) == 0, "A1: BRESP 가 OKAY 가 아니다"
    data, resp = await axil_read(dut, 0x100)
    assert resp == 0, "A1: RRESP 가 OKAY 가 아니다"
    assert data == 0xDEAD_BEEF, f"A1: 0xDEADBEEF 를 썼는데 {data:#x} 를 읽었다"


@cocotb.test()
async def test_a2_aw_before_w(dut):
    """A2: AW 가 W 보다 먼저 와도 쓰기가 된다 (AXI4-Lite 는 순서를 강제하지 않는다)."""
    await setup(dut)

    await axil_write(dut, 0x200, 0x1234_5678, w_delay=3)
    data, _ = await axil_read(dut, 0x200)
    assert data == 0x1234_5678, f"A2: {data:#x}"


@cocotb.test()
async def test_a3_w_before_aw(dut):
    """A3: W 가 AW 보다 먼저 와도 쓰기가 된다."""
    await setup(dut)

    await axil_write(dut, 0x300, 0x0BAD_F00D, aw_delay=3)
    data, _ = await axil_read(dut, 0x300)
    assert data == 0x0BAD_F00D, f"A3: {data:#x}"


@cocotb.test()
async def test_a4_read_absorbs_latency(dut):
    """A4: **한 번만 읽어도** 1사이클 지연 뒤의 값이 나온다.

    MMIO 규약(두 번 읽기)은 호스트 쪽 편의였다. AXI 경로는 브리지가 흡수한다.
    이게 안 되면 보드에서 스크래치 읽기가 전부 한 칸씩 밀린다.
    """
    rf = await setup(dut)
    rf.mem[0x400] = 0xAAAA_5555
    rf.mem[0x404] = 0x5555_AAAA

    # 서로 다른 주소를 번갈아 읽어도 각각 제 값이 나와야 한다 (밀리면 여기서 걸린다)
    for addr, want in ((0x400, 0xAAAA_5555), (0x404, 0x5555_AAAA),
                       (0x400, 0xAAAA_5555), (0x404, 0x5555_AAAA)):
        data, _ = await axil_read(dut, addr)
        assert data == want, f"A4: addr={addr:#x} 기대 {want:#x} 실제 {data:#x} — 한 칸 밀렸다"


@cocotb.test()
async def test_a5_read_never_writes(dut):
    """A5: 읽기 트랜잭션 동안 reg_wr_en 이 절대 뜨지 않는다."""
    rf = await setup(dut)
    rf.mem[0x500] = 0x1111_2222

    saw_wr = 0
    dut.s_axil_araddr.value = 0x500
    dut.s_axil_arvalid.value = 1
    for _ in range(32):
        await FallingEdge(dut.clk)
        if int(dut.reg_wr_en.value):
            saw_wr += 1
        if int(dut.s_axil_arready.value):
            dut.s_axil_arvalid.value = 0
        if int(dut.s_axil_rvalid.value):
            break
    await RisingEdge(dut.clk)
    assert saw_wr == 0, f"A5: 읽기 중에 reg_wr_en 이 {saw_wr}번 떴다 — 레지스터가 오염된다"


@cocotb.test()
async def test_a6_slow_ready(dut):
    """A6: BREADY/RREADY 를 늦게 줘도 VALID 가 유지된다 (AXI 규약)."""
    rf = await setup(dut)
    rf.mem[0x600] = 0x9999_0000

    # 읽기: RREADY 를 5사이클 늦춘다
    dut.s_axil_rready.value = 0
    dut.s_axil_araddr.value = 0x600
    dut.s_axil_arvalid.value = 1
    for _ in range(32):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.s_axil_arready.value):
            dut.s_axil_arvalid.value = 0
        if int(dut.s_axil_rvalid.value):
            break
    held = []
    for _ in range(5):
        await FallingEdge(dut.clk)
        held.append((int(dut.s_axil_rvalid.value), int(dut.s_axil_rdata.value)))
    assert all(v == 1 for v, _ in held), f"A6: RVALID 가 내려갔다 {held}"
    assert all(d == 0x9999_0000 for _, d in held), f"A6: RDATA 가 흔들렸다 {held}"
    dut.s_axil_rready.value = 1
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_a7_back_to_back(dut):
    """A7: 연속 트랜잭션이 섞이지 않는다."""
    await setup(dut)

    vals = {0x700 + i * 4: 0x1000_0000 + i for i in range(8)}
    for a, v in vals.items():
        await axil_write(dut, a, v)
    for a, v in vals.items():
        data, _ = await axil_read(dut, a)
        assert data == v, f"A7: addr={a:#x} 기대 {v:#x} 실제 {data:#x}"
