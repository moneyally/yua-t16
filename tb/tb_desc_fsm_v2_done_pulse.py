"""기대값 출처: docs/DESIGN.md 9절 + rtl/desc_fsm_v2.sv FSM 사양 (골든 모델 아님 — 프로토콜 계약 테스트).

tb_desc_fsm_v2_done_pulse.py — PLAN W2-2: done_pulse 계약 검증

docs/AUDIT.md §4 가 남긴 발견 두 가지를 테스트로 바꾼 것이다:

  1. 레포 전체에 done_pulse 의 **폭**이나 **개수**를 검사하는 테스트가 0개였다.
     기존 테스트는 전부 "N 사이클 안에 1이 된 적이 있는가"만 본다.

  2. rtl/desc_fsm_v2.sv:331 이 ST_FAULT -> ST_DONE 으로 가므로,
     **fault 가 난 디스크립터도 done_pulse 를 낸다**는 가설.
     rtl/g2_ctrl_top.sv:460 에서 irq_sources[0](DESC_DONE) = fsm_done_pulse 이므로
     이 가설이 참이면 실패한 디스크립터에 DESC_DONE IRQ 가 뜬다.
     (상위 레벨 검증은 tb_g2_ctrl_top_fault_irq.py)

검사 계약:
  C1. 정상 디스크립터(NOP, GEMM): done_pulse 는 **정확히 1사이클**, 트랜잭션당 **정확히 1회**.
  C2. fault 디스크립터(illegal opcode / CRC / timeout): done_pulse 가 나온다면 그것도
      정확히 1사이클 1회여야 한다. 그리고 fault_valid 와의 관계를 기록한다.
  C3. fault 디스크립터에서 done_pulse 가 **아예 나오지 않아야 한다** — 이것이 가설.
      지금 실패하면 버그 확정, 통과하면 가설 기각. 어느 쪽이든 결과를 docs/BUGS.md 에 적는다.
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

DESC_SIZE = 64


def crc8(data_bytes):
    """CRC-8, polynomial 0x07 — rtl/desc_fsm_v2.sv 의 crc8_byte 와 같은 다항식."""
    crc = 0x00
    for byte in data_bytes:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def make_descriptor(opcode, valid_crc=True, kt=4):
    d = [0] * DESC_SIZE
    d[0] = opcode & 0xFF
    for i in range(8):
        d[8 + i] = (0x1000 >> (8 * i)) & 0xFF   # act_addr
        d[16 + i] = (0x2000 >> (8 * i)) & 0xFF  # wgt_addr
        d[24 + i] = (0x3000 >> (8 * i)) & 0xFF  # out_addr
    for i in range(4):
        d[32 + i] = (kt >> (8 * i)) & 0xFF
    d[DESC_SIZE - 1] = crc8(d[: DESC_SIZE - 1])
    if not valid_crc:
        d[DESC_SIZE - 1] ^= 0xFF
    return d


def set_desc_bytes(dut, desc):
    """desc_bytes 포트에 디스크립터를 쓴다.

    이 포트는 원본 SV 에서 `logic [7:0] desc_bytes [0:DESC_SIZE-1]` (unpacked) 인데,
    sv2v 를 거치면 `[(DESC_SIZE*8)-1:0]` packed 벡터가 되고 원소 i 는
    비트 오프셋 ((DESC_SIZE-1)-i)*8 에 놓인다 (byte 0 이 MSB 쪽).
    verilator/questa 로 원본 SV 를 직접 돌릴 때와 icarus+sv2v 로 돌릴 때
    같은 테스트가 동작하도록 두 모양을 모두 지원한다.
    """
    h = dut.desc_bytes
    n = len(desc)
    width = len(h)
    if width == n:                      # unpacked array of bytes (네이티브 SV)
        for i, b in enumerate(desc):
            h[i].value = b
    elif width == n * 8:                # packed vector (sv2v 변환본)
        v = 0
        for i, b in enumerate(desc):
            v |= (b & 0xFF) << (((n - 1) - i) * 8)
        h.value = v
    else:
        raise AssertionError(
            f"desc_bytes 폭이 예상 밖이다: len={width}, DESC_SIZE={n}"
        )


async def reset_dut(dut):
    dut.rst_n.value = 0
    dut.desc_valid.value = 0
    dut.queue_class.value = 0
    dut.cmd_ready.value = 1
    dut.core_done.value = 0
    dut.timeout_cycles.value = 200
    set_desc_bytes(dut, [0] * DESC_SIZE)
    await Timer(50, unit="ns")
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


async def send_descriptor(dut, desc):
    set_desc_bytes(dut, desc)
    dut.desc_valid.value = 1
    # desc_ready 가 뜰 때까지 유지
    for _ in range(20):
        await RisingEdge(dut.clk)
        if dut.desc_ready.value == 1:
            break
    await RisingEdge(dut.clk)
    dut.desc_valid.value = 0


async def observe(dut, cycles, feed_core_done=False):
    """cycles 사이클 동안 done_pulse / fault_valid 를 사이클 단위로 기록.

    returns dict:
      pulses      : [(start_cycle, width), ...]  done_pulse 가 1이었던 연속 구간
      total_high  : done_pulse 가 1이었던 총 사이클 수
      fault_cycle : fault_valid 가 처음 1이 된 사이클 (없으면 None)
    """
    trace_done, trace_fault = [], []
    fed = False
    for c in range(cycles):
        await RisingEdge(dut.clk)
        # cmd_valid 가 뜨면 코어 완료를 한 번 돌려준다 (정상 경로용)
        if feed_core_done and not fed and int(dut.cmd_valid.value) == 1:
            await RisingEdge(dut.clk)
            dut.core_done.value = 1
            await RisingEdge(dut.clk)
            dut.core_done.value = 0
            fed = True
            trace_done.extend([0, 0])
            trace_fault.extend([0, 0])
            continue
        trace_done.append(int(dut.done_pulse.value))
        trace_fault.append(int(dut.fault_valid.value))

    pulses, start = [], None
    for i, v in enumerate(trace_done):
        if v and start is None:
            start = i
        elif not v and start is not None:
            pulses.append((start, i - start))
            start = None
    if start is not None:
        pulses.append((start, len(trace_done) - start))

    fault_cycle = next((i for i, v in enumerate(trace_fault) if v), None)
    return {
        "pulses": pulses,
        "total_high": sum(trace_done),
        "fault_cycle": fault_cycle,
        "trace_done": trace_done,
    }


def assert_single_1cycle_pulse(dut, obs, label):
    """C1/C2: done_pulse 는 1사이클 폭, 1회."""
    assert len(obs["pulses"]) == 1, (
        f"[{label}] done_pulse 는 트랜잭션당 정확히 1회여야 한다. "
        f"관측: {len(obs['pulses'])}회 {obs['pulses']}"
    )
    start, width = obs["pulses"][0]
    assert width == 1, (
        f"[{label}] done_pulse 폭은 정확히 1사이클이어야 한다. "
        f"관측: {width}사이클 (시작 사이클 {start})"
    )


@cocotb.test()
async def test_nop_done_pulse_is_exactly_one_cycle(dut):
    """C1: NOP 디스크립터 — done_pulse 1사이클 1회."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await send_descriptor(dut, make_descriptor(0x01))
    obs = await observe(dut, 40)
    dut._log.info(f"NOP: pulses={obs['pulses']} total_high={obs['total_high']} fault={obs['fault_cycle']}")
    assert obs["fault_cycle"] is None, "NOP 은 fault 를 내면 안 된다"
    assert_single_1cycle_pulse(dut, obs, "NOP")


@cocotb.test()
async def test_gemm_done_pulse_is_exactly_one_cycle(dut):
    """C1: 정상 GEMM 디스크립터 — done_pulse 1사이클 1회."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await send_descriptor(dut, make_descriptor(0x02))
    obs = await observe(dut, 60, feed_core_done=True)
    dut._log.info(f"GEMM: pulses={obs['pulses']} total_high={obs['total_high']} fault={obs['fault_cycle']}")
    assert obs["fault_cycle"] is None, "정상 GEMM 은 fault 를 내면 안 된다"
    assert_single_1cycle_pulse(dut, obs, "GEMM")


@cocotb.test()
async def test_illegal_opcode_must_not_produce_done_pulse(dut):
    """C3 가설: illegal opcode 는 fault 만 내고 done_pulse 를 내면 안 된다.

    실패하면 rtl/desc_fsm_v2.sv:331 (ST_FAULT -> ST_DONE) 가 버그로 확정된다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await send_descriptor(dut, make_descriptor(0xFF))
    obs = await observe(dut, 40)
    dut._log.info(f"ILLEGAL: pulses={obs['pulses']} total_high={obs['total_high']} fault={obs['fault_cycle']}")
    assert obs["fault_cycle"] is not None, "illegal opcode 는 fault_valid 를 내야 한다"
    if obs["pulses"]:
        assert_single_1cycle_pulse(dut, obs, "ILLEGAL(width)")
    assert obs["total_high"] == 0, (
        "fault 난 디스크립터가 done_pulse 를 냈다. "
        f"pulses={obs['pulses']}, fault_cycle={obs['fault_cycle']}. "
        "g2_ctrl_top.sv:460 에서 irq_sources[0]=DESC_DONE=fsm_done_pulse 이므로 "
        "실패한 디스크립터에 완료 IRQ 가 뜬다. docs/AUDIT.md §4 가설 확정."
    )


@cocotb.test()
async def test_crc_fail_must_not_produce_done_pulse(dut):
    """C3 가설: CRC 불일치도 마찬가지."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await send_descriptor(dut, make_descriptor(0x02, valid_crc=False))
    obs = await observe(dut, 40)
    dut._log.info(f"CRCFAIL: pulses={obs['pulses']} total_high={obs['total_high']} fault={obs['fault_cycle']}")
    assert obs["fault_cycle"] is not None, "CRC 불일치는 fault_valid 를 내야 한다"
    if obs["pulses"]:
        assert_single_1cycle_pulse(dut, obs, "CRCFAIL(width)")
    assert obs["total_high"] == 0, (
        "CRC fault 디스크립터가 done_pulse 를 냈다. "
        f"pulses={obs['pulses']}, fault_cycle={obs['fault_cycle']}"
    )


@cocotb.test()
async def test_timeout_must_not_produce_done_pulse(dut):
    """C3 가설: timeout fault 도 마찬가지. core_done 을 끝까지 주지 않는다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    dut.timeout_cycles.value = 30
    await send_descriptor(dut, make_descriptor(0x02))
    obs = await observe(dut, 80)
    dut._log.info(f"TIMEOUT: pulses={obs['pulses']} total_high={obs['total_high']} fault={obs['fault_cycle']}")
    assert obs["fault_cycle"] is not None, "timeout 은 fault_valid 를 내야 한다"
    if obs["pulses"]:
        assert_single_1cycle_pulse(dut, obs, "TIMEOUT(width)")
    assert obs["total_high"] == 0, (
        "timeout fault 디스크립터가 done_pulse 를 냈다. "
        f"pulses={obs['pulses']}, fault_cycle={obs['fault_cycle']}"
    )


@cocotb.test()
async def test_diag_fault_cycle_table(dut):
    """진단용 (assertion 없음): illegal opcode 트랜잭션의 사이클별 신호 표.

    PLAN W2-2 가 요구하는 "어느 사이클에 어떤 신호가 기대와 다른지"의 근거.
    이 표를 docs/BUGS.md 에 그대로 붙인다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await reset_dut(dut)
    await send_descriptor(dut, make_descriptor(0xFF))

    def probe(name):
        try:
            return int(getattr(dut, name).value)
        except Exception:
            return None

    dut._log.info("cyc | state | busy | fault_valid | fault_code | done_pulse")
    dut._log.info("----+-------+------+-------------+------------+-----------")
    for c in range(12):
        await RisingEdge(dut.clk)
        st = probe("state")
        dut._log.info(
            "%3d | %5s | %4d | %11d | %#10x | %10d"
            % (
                c,
                "?" if st is None else str(st),
                int(dut.busy.value),
                int(dut.fault_valid.value),
                int(dut.fault_code.value),
                int(dut.done_pulse.value),
            )
        )
