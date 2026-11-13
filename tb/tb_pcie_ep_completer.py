"""tb_pcie_ep_completer.py — pcie_ep_versal 완결자 경로 검증.

검증 대상은 "로직" 이다: BAR 라우팅, 읽기/쓰기 처리, 완료 생성,
백프레셔, 지원하지 않는 요청의 처리.

검증 대상이 "아니다": CQ/CC 디스크립터의 실제 비트 위치. 이 저장소에는
PG343 이 없어 확인할 수 없다. 자극을 만드는 상수는 RTL 의 localparam 과
같은 값을 쓰므로, 형식이 틀렸다면 양쪽을 같이 고치면 되고 아래 테스트는
그대로 유효하다.
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# rtl/pcie_ep_versal.sv 의 CQ_*/CC_* 와 같은 값
CQ_ADDR_LSB, CQ_ADDR_W = 2, 62
CQ_DWCNT_LSB = 64
CQ_REQTYPE_LSB = 75
CQ_REQID_LSB = 80
CQ_TAG_LSB = 96
CQ_BARID_LSB = 112
CQ_PAYLOAD_LSB = 128

CC_BYTECNT_LSB = 16
CC_DWCNT_LSB = 32
CC_STATUS_LSB = 43
CC_REQID_LSB = 48
CC_TAG_LSB = 64
CC_PAYLOAD_LSB = 96

REQ_MEM_READ = 0b0000
REQ_MEM_WRITE = 0b0001
CPL_SUCCESS = 0b000
CPL_UNSUPPORTED = 0b001

BARS = {0: 0, 2: 2, 4: 4}


def field(word, lsb, width):
    return (word >> lsb) & ((1 << width) - 1)


def make_cq(addr, dwcnt=1, reqtype=REQ_MEM_READ, reqid=0x0100,
            tag=0x5A, barid=0, payload=0):
    w = 0
    w |= (addr >> 2) << CQ_ADDR_LSB
    w |= dwcnt << CQ_DWCNT_LSB
    w |= reqtype << CQ_REQTYPE_LSB
    w |= reqid << CQ_REQID_LSB
    w |= tag << CQ_TAG_LSB
    w |= barid << CQ_BARID_LSB
    w |= payload << CQ_PAYLOAD_LSB
    return w


async def settle(dut):
    await Timer(1, unit="ns")


async def boot(dut):
    cocotb.start_soon(Clock(dut.user_clk, 10, unit="ns").start())
    dut.user_reset.value = 1
    dut.user_lnk_up.value = 1
    dut.cfg_phy_link_down.value = 0
    dut.cfg_flr_active.value = 0
    dut.m_axis_cq_tdata.value = 0
    dut.m_axis_cq_tvalid.value = 0
    dut.m_axis_cq_tlast.value = 0
    dut.m_axis_cq_tkeep.value = 0
    dut.m_axis_cq_tuser.value = 0xF        # first BE = 0xF
    dut.s_axis_cc_tready.value = 1
    dut.m_axis_rc_tdata.value = 0
    dut.m_axis_rc_tvalid.value = 0
    dut.m_axis_rc_tlast.value = 0
    dut.s_axis_rq_tready.value = 1
    dut.msix_req.value = 0
    dut.msix_vector.value = 0
    for b in (0, 2, 4):
        getattr(dut, f"bar{b}_req_ready").value = 1
        getattr(dut, f"bar{b}_rsp_valid").value = 0
        getattr(dut, f"bar{b}_rsp_rdata").value = 0
    for _ in range(3):
        await RisingEdge(dut.user_clk)
    dut.user_reset.value = 0
    for _ in range(3):
        await RisingEdge(dut.user_clk)


async def send_cq(dut, word, last=True):
    dut.m_axis_cq_tdata.value = word
    dut.m_axis_cq_tvalid.value = 1
    dut.m_axis_cq_tlast.value = 1 if last else 0
    for _ in range(50):
        await settle(dut)
        if int(dut.m_axis_cq_tready.value) == 1:
            break
        await RisingEdge(dut.user_clk)
    await RisingEdge(dut.user_clk)
    dut.m_axis_cq_tvalid.value = 0
    dut.m_axis_cq_tlast.value = 0
    await settle(dut)


async def await_cc(dut, timeout=60):
    for _ in range(timeout):
        await settle(dut)
        if int(dut.s_axis_cc_tvalid.value) == 1:
            w = int(dut.s_axis_cc_tdata.value)
            await RisingEdge(dut.user_clk)
            return w
        await RisingEdge(dut.user_clk)
    raise TimeoutError("완료(CC)가 오지 않았다")


@cocotb.test()
async def test_write_routes_to_bar0(dut):
    """BAR0 32비트 쓰기 — 주소/데이터/BE 가 그대로 전달된다."""
    await boot(dut)
    await send_cq(dut, make_cq(0x1234, reqtype=REQ_MEM_WRITE,
                               barid=0, payload=0xDEADBEEF))

    for _ in range(20):
        await settle(dut)
        if int(dut.bar0_req_valid.value) == 1:
            assert int(dut.bar0_req_wr.value) == 1
            assert int(dut.bar0_req_addr.value) == 0x1234
            assert int(dut.bar0_req_wdata.value) == 0xDEADBEEF
            assert int(dut.bar0_req_be.value) == 0xF
            return
        await RisingEdge(dut.user_clk)
    raise AssertionError("bar0_req_valid 가 뜨지 않았다")


@cocotb.test()
async def test_read_returns_completion(dut):
    """BAR0 읽기 — 응답 데이터가 완료 payload 로 돌아온다."""
    await boot(dut)
    await send_cq(dut, make_cq(0x40, reqtype=REQ_MEM_READ,
                               barid=0, tag=0x33, reqid=0x0200))

    for _ in range(20):
        await settle(dut)
        if int(dut.bar0_req_valid.value) == 1:
            assert int(dut.bar0_req_wr.value) == 0
            break
        await RisingEdge(dut.user_clk)
    else:
        raise AssertionError("읽기 요청이 BAR0 로 가지 않았다")

    await RisingEdge(dut.user_clk)
    dut.bar0_rsp_rdata.value = 0x47320001       # G2_ID
    dut.bar0_rsp_valid.value = 1
    await RisingEdge(dut.user_clk)
    dut.bar0_rsp_valid.value = 0

    cc = await await_cc(dut)
    assert field(cc, CC_PAYLOAD_LSB, 32) == 0x47320001, "payload 불일치"
    assert field(cc, CC_STATUS_LSB, 3) == CPL_SUCCESS
    assert field(cc, CC_DWCNT_LSB, 11) == 1
    assert field(cc, CC_BYTECNT_LSB, 13) == 4
    assert field(cc, CC_TAG_LSB, 8) == 0x33, "태그가 그대로 돌아와야 한다"
    assert field(cc, CC_REQID_LSB, 16) == 0x0200, "Requester ID 가 그대로여야 한다"


@cocotb.test()
async def test_bar_routing(dut):
    """BAR ID 에 따라 서로 다른 BAR 포트로 간다."""
    for barid, port in BARS.items():
        await boot(dut)
        await send_cq(dut, make_cq(0x20, reqtype=REQ_MEM_WRITE,
                                   barid=barid, payload=0xA5A5A5A5))
        hit = None
        for _ in range(20):
            await settle(dut)
            for b in (0, 2, 4):
                if int(getattr(dut, f"bar{b}_req_valid").value) == 1:
                    hit = b
            if hit is not None:
                break
            await RisingEdge(dut.user_clk)
        assert hit == port, f"BAR ID {barid} → bar{hit} (기대 bar{port})"


@cocotb.test()
async def test_unknown_bar_is_unsupported(dut):
    """정의되지 않은 BAR ID 는 어떤 BAR 포트도 건드리지 않고 UR 로 완료한다."""
    await boot(dut)
    await send_cq(dut, make_cq(0x10, reqtype=REQ_MEM_READ, barid=5))

    cc = await await_cc(dut)
    assert field(cc, CC_STATUS_LSB, 3) == CPL_UNSUPPORTED
    for b in (0, 2, 4):
        assert int(getattr(dut, f"bar{b}_req_valid").value) == 0, \
            f"bar{b} 요청이 나가면 안 된다"


@cocotb.test()
async def test_multi_dword_is_unsupported(dut):
    """2 DW 이상 읽기는 UR 로 완료한다 (호스트는 32비트만 쓴다)."""
    await boot(dut)
    await send_cq(dut, make_cq(0x80, dwcnt=4, reqtype=REQ_MEM_READ, barid=0))

    cc = await await_cc(dut)
    assert field(cc, CC_STATUS_LSB, 3) == CPL_UNSUPPORTED
    assert field(cc, CC_DWCNT_LSB, 11) == 0


@cocotb.test()
async def test_backpressure_while_busy(dut):
    """BAR 가 받아주지 않는 동안 요청 valid 가 유지되고, CQ 는 더 받지 않는다."""
    await boot(dut)
    dut.bar0_req_ready.value = 0            # BAR 가 받아주지 않게 막는다
    await send_cq(dut, make_cq(0x100, reqtype=REQ_MEM_READ, barid=0))

    # 처리 중에는 CQ 를 더 받지 않는다
    await RisingEdge(dut.user_clk)
    await settle(dut)
    assert int(dut.m_axis_cq_tready.value) == 0, \
        "요청 처리 중인데 CQ 를 더 받고 있다"

    # 그리고 요청 valid 는 ready 가 올 때까지 내려가면 안 된다
    for c in range(8):
        await settle(dut)
        assert int(dut.bar0_req_valid.value) == 1, \
            f"백프레셔 {c} 사이클째에 bar0_req_valid 가 내려갔다"
        await RisingEdge(dut.user_clk)

    # 풀어주면 진행해서 완료까지 간다
    dut.bar0_req_ready.value = 1
    await RisingEdge(dut.user_clk)
    await RisingEdge(dut.user_clk)
    dut.bar0_rsp_rdata.value = 0xCAFEBABE
    dut.bar0_rsp_valid.value = 1
    await RisingEdge(dut.user_clk)
    dut.bar0_rsp_valid.value = 0

    cc = await await_cc(dut)
    assert field(cc, CC_PAYLOAD_LSB, 32) == 0xCAFEBABE


@cocotb.test()
async def test_cc_waits_for_tready(dut):
    """CC 는 s_axis_cc_tready 가 올라올 때까지 유지된다."""
    await boot(dut)
    dut.s_axis_cc_tready.value = 0
    await send_cq(dut, make_cq(0x4, reqtype=REQ_MEM_READ, barid=0))

    for _ in range(20):
        await settle(dut)
        if int(dut.bar0_req_valid.value) == 1:
            break
        await RisingEdge(dut.user_clk)
    await RisingEdge(dut.user_clk)
    dut.bar0_rsp_rdata.value = 0x11223344
    dut.bar0_rsp_valid.value = 1
    await RisingEdge(dut.user_clk)
    dut.bar0_rsp_valid.value = 0

    # tready=0 인 동안 tvalid 가 유지돼야 한다
    held = 0
    for _ in range(8):
        await settle(dut)
        if int(dut.s_axis_cc_tvalid.value) == 1:
            held += 1
        await RisingEdge(dut.user_clk)
    assert held >= 4, f"tready=0 인데 CC 가 유지되지 않았다 (held={held})"

    dut.s_axis_cc_tready.value = 1
    cc = await await_cc(dut)
    assert field(cc, CC_PAYLOAD_LSB, 32) == 0x11223344
