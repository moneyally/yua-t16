"""test_dma_latency.py — 실제 DRAM 지연을 넣어도 파이프라인이 흡수하는가.

DMA 스텁은 기본값으로 지연 0 에 응답한다. 그래서 이 저장소의 모든 사이클
수는 **낙관적 상한**이다 — 그 단서를 숫자로 바꾼다.

실제 DDR4 는 요청 수락에서 첫 데이터까지 수십 사이클이 걸린다. 설계가
그것을 흡수하지 못하면 보드에서 처음 드러나고, 그때는 늦다.

파이프라인이 흡수해야 하는 이유: 로드 FSM 은 디스크립터 N+1 을 받는 동안
연산 FSM 이 N 을 돌리고 있다. 읽기 지연은 로드 쪽에만 얹히므로, 로드가
연산보다 빠른 구성에서는 여유가 지연을 먹어 준다.

단언하는 것:
  1) 지연이 있어도 결과가 맞는다.
  2) 로드에 여유가 있는 구성(weight-stationary ON)에서는 처리량이
     거의 안 떨어진다.
  3) 떨어지는 양을 기록한다 — 그게 "낙관적 상한" 의 실제 크기다.
"""
import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from dma_stub import SimpleMem, start_dma

FLAG_WGT_KEEP = 0x01

# DDR4-3200 CL22 기준 대략적인 요청→첫데이터 지연을 가속기 클럭으로 환산.
# 정확한 값은 컨트롤러와 주파수에 따라 달라지므로 범위를 재 본다.
LATENCIES = (0, 8, 24, 48)


def u32_le(x): return int(x).to_bytes(4, "little", signed=False)
def u64_le(x): return int(x).to_bytes(8, "little", signed=False)


def probe_config(dut):
    accs = len(dut.acc_out_flat)
    n = int(round(accs ** 0.5))
    assert n * n == accs
    return n, len(dut.rd_data) // 8, len(dut.wr_data) // 8


async def boot(dut):
    dut.rst_n.value = 0
    dut.cmd_valid.value = 0
    dut.act_addr.value = 0
    dut.wgt_addr.value = 0
    dut.out_addr.value = 0
    dut.Kt.value = 0
    dut.wgt_keep.value = 0
    dut.wgt_int4.value = 0
    dut.acc_keep.value = 0
    dut.no_wb.value = 0
    dut.rd_req_ready.value = 1
    dut.wr_req_ready.value = 1
    dut.rd_data_valid.value = 0
    dut.rd_data.value = 0
    dut.rd_data_last.value = 0
    dut.rd_done.value = 0
    dut.wr_data_ready.value = 1
    dut.wr_done.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(5):
        await RisingEdge(dut.clk)


async def measure(dut, mem, N, RD_BEAT, Kt, n_ops, exp, ws=True):
    """연속 n_ops 개의 정상상태 간격을 잰다."""
    act, wgt, out0 = 0x1000, 0x40000, 0x100000
    issued = done_seen = total = 0
    started = False
    done_at = []

    for _ in range(4_000_000):
        want = issued < n_ops
        dut.act_addr.value = act
        dut.wgt_addr.value = wgt
        dut.out_addr.value = out0 + 0x2000 * issued
        dut.Kt.value = Kt
        dut.wgt_keep.value = 1 if (ws and issued > 0) else 0
        dut.cmd_valid.value = 1 if want else 0

        await Timer(1, unit="ns")
        accept = want and int(dut.cmd_ready.value) == 1
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")

        if accept:
            issued += 1
            started = True
        if started:
            total += 1
            if int(dut.done.value) == 1:
                done_seen += 1
                done_at.append(total)
                if done_seen == n_ops:
                    break

    dut.cmd_valid.value = 0
    assert done_seen == n_ops, f"{done_seen}/{n_ops} 완료"
    gaps = [b - a for a, b in zip(done_at, done_at[1:])]
    steady = gaps[2:-2] if len(gaps) > 6 else gaps
    return sum(steady) / len(steady)


async def _sweep(dut, N, RD_BEAT, WR_BEAT, Kt, A, B, exp, ws, side="rd"):
    """side="rd" 는 읽기 지연, "wr" 는 쓰기 지연을 쓸어본다."""
    base = None
    rows = []
    for lat in LATENCIES:
        await boot(dut)
        mem = SimpleMem()
        # 설정을 바꿀 때마다 스텁을 교체한다. 두 개가 같이 돌면 지연 0 짜리
        # 첫 스텁이 이겨서 "지연을 넣었는데 손실 0%" 가 된다 — 실제로
        # 그렇게 한 번 속았다. 이제 start_dma 가 중복을 에러로 막는다.
        dma = start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT,
                        wr_beat_bytes=WR_BEAT,
                        rd_latency=lat if side == "rd" else 0,
                        wr_latency=lat if side == "wr" else 0)
        for k in range(Kt):
            mem.write_bytes(0x1000 + N * k,
                            bytes((int(A[i, k]) & 0xFF) for i in range(N)))
            mem.write_bytes(0x40000 + N * k,
                            bytes((int(B[k, j]) & 0xFF) for j in range(N)))

        per_op = await measure(dut, mem, N, RD_BEAT, Kt, 12, exp, ws=ws)

        raw = mem.read_bytes(0x100000, N * N * 4)
        got = np.array([
            int.from_bytes(raw[(i * N + j) * 4:(i * N + j) * 4 + 4],
                           "little", signed=True)
            for i in range(N) for j in range(N)]).reshape(N, N)
        assert (got == exp).all(), \
            f"{side} 지연 {lat} (ws={ws}): 결과 불일치"

        dma.stop()
        if base is None:
            base = per_op
            rows.append((lat, per_op, None))
        else:
            rows.append((lat, per_op, (per_op / base - 1) * 100))
    return rows


@cocotb.test()
async def test_pipeline_absorbs_dram_latency(dut):
    """읽기 지연을 늘려가며 처리량 손실을 기록한다.

    두 구성을 비교한다.
      WS ON   로드가 활성값만 읽어 여유가 있다 → 흡수해야 한다
      WS off  로드가 활성값+가중치를 읽어 묶는 쪽이다 → 손실이 보여야 한다
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    N, RD_BEAT, WR_BEAT = probe_config(dut)
    # 쓰기 비트 수에 맞춰 Kt 를 고르되 64 보다 아래로는 내리지 않는다.
    # 쓰기 폭이 넓으면 OUT_BEATS 가 작아지는데, Kt 까지 같이 줄이면
    # 쓰기 채널의 여유가 사라져 이중 버퍼의 효과를 볼 수 없다.
    Kt = min(128, max(64, (N * N * 4) // WR_BEAT))
    rng = np.random.default_rng(13579)
    A = rng.integers(-128, 128, size=(N, Kt), dtype=np.int8)
    B = rng.integers(-128, 128, size=(Kt, N), dtype=np.int8)
    exp = A.astype(np.int32) @ B.astype(np.int32)

    out_beats = (N * N * 4) // WR_BEAT
    out_bufs  = int(dut.OUT_BUFS.value)

    dut._log.info("")
    dut._log.info(f"  TILE_N={N} Kt={Kt} RD={RD_BEAT*8}비트 WR={WR_BEAT*8}비트"
                  f" OUT_BUFS={out_bufs}")
    results = {}
    for label, ws in (("WS ON (로드 여유 있음)", True),
                      ("WS off (로드가 병목)", False)):
        rows = await _sweep(dut, N, RD_BEAT, WR_BEAT, Kt, A, B, exp, ws)
        results[label] = rows
        dut._log.info("")
        dut._log.info(f"  {label}")
        dut._log.info(f"  {'읽기 지연':>10}{'사이클/op':>12}{'손실':>10}")
        dut._log.info("  " + "-" * 34)
        for lat, per_op, loss in rows:
            tag = "기준" if loss is None else f"{loss:.1f}%"
            dut._log.info(f"  {lat:>10}{per_op:>12.1f}{tag:>10}")

    # ── 쓰기 지연 ────────────────────────────────────────────────
    # 쓰기 지연의 흡수 여부를 가르는 것은 **쓰기 채널의 점유율 하나**다.
    # 출력 래치를 이중으로 올려도(OUT_BUFS=2) 숫자가 바뀌지 않는다 —
    # 측정해 확인했고, 이유는 gemm_core.sv 의 OUT_BUFS 주석에 있다.
    # 요점: 디스크립터마다 쓰기가 한 번이고 채널이 하나이므로, 처리량은
    # (지연 + OUT_BEATS) 가 주기에 들어가느냐로만 정해진다.
    #
    # 그래서 이 테스트는 OUT_BUFS 를 읽어 두되 기대는 점유율로만 세운다.
    # OUT_BUFS 를 바꿔 돌려도 같은 단언이 통과해야 한다 — 그게 위
    # 결론의 회귀 보호다.
    base_cycles = results["WS ON (로드 여유 있음)"][0][1]
    wr_occupancy = (out_beats + 2) / base_cycles
    # 흡수 조건은 채널 여유 하나다 (OUT_BUFS 와 무관).
    wr_absorbable = wr_occupancy < 0.5

    wr_rows = await _sweep(dut, N, RD_BEAT, WR_BEAT, Kt, A, B, exp,
                           ws=True, side="wr")
    dut._log.info("")
    dut._log.info("  쓰기 지연 (WS ON)")
    dut._log.info(f"  {'쓰기 지연':>10}{'사이클/op':>12}{'손실':>10}")
    dut._log.info("  " + "-" * 34)
    for lat, per_op, loss in wr_rows:
        tag = "기준" if loss is None else f"{loss:.1f}%"
        dut._log.info(f"  {lat:>10}{per_op:>12.1f}{tag:>10}")

    dut._log.info("")
    dut._log.info("  읽기 지연은 로드 쪽에만 얹힌다. 로드에 여유가 있으면")
    dut._log.info("  (WS ON) 파이프라인이 전부 흡수한다. 로드가 병목이면")
    dut._log.info("  (WS off) 지연이 그대로 주기에 더해진다.")
    dut._log.info(f"  쓰기: OUT_BEATS={out_beats} / 주기 {base_cycles:.0f} 사이클"
                  f" = 채널 {wr_occupancy:.0%} 바쁨, OUT_BUFS={out_bufs}")
    dut._log.info(f"  → 흡수 기대: {'있음 (채널 여유)' if wr_absorbable else '없음 (채널 포화)'}"
                  f" — OUT_BUFS 와 무관하다")

    # 측정된 동작을 못박는다. 나빠지면 회귀로 잡힌다.
    ws = {lat: loss for lat, _, loss in results["WS ON (로드 여유 있음)"]}
    off = {lat: loss for lat, _, loss in results["WS off (로드가 병목)"]}

    # WS ON: 24 사이클까지 완전 흡수
    for lat in (8, 24):
        assert ws[lat] is not None and ws[lat] < 1.0, \
            f"WS ON, 지연 {lat}: 손실 {ws[lat]:.1f}% — 흡수하던 것을 못 한다"
    # 48 은 흡수 한계를 넘는다. 측정 17.1% 에서 크게 나빠지면 잡는다.
    assert ws[48] is not None and ws[48] < 25.0, \
        f"WS ON, 지연 48: 손실 {ws[48]:.1f}% (측정 기준 17.1%)"

    # WS off: 로드가 병목이라 지연이 그대로 얹힌다. WS 의 값어치가
    # 트래픽 절감만이 아니라 **지연 내성**에도 있다는 증거다.
    assert off[8] > 10.0, "WS off 인데 지연 8 을 흡수한다 — 측정이 변했다"
    assert off[48] > ws[48] * 3, \
        f"WS off 손실 {off[48]:.1f}% vs WS ON {ws[48]:.1f}% — 차이가 사라졌다"

    # 쓰기 지연.
    wr = {lat: loss for lat, _, loss in wr_rows}
    assert wr[8] is not None
    for lat in LATENCIES[1:]:
        assert wr[lat] is not None and wr[lat] >= -1.0, \
            f"쓰기 지연 {lat} 에서 더 빨라졌다 ({wr[lat]:.1f}%) — 측정 오류"

    if wr_absorbable:
        # 채널에 여유가 있으면 지연 48 까지 전부 흡수한다. 단일 버퍼로도
        # 그렇다 — 측정값이다 (WR=512 에서 0/0/0/0%).
        for lat in LATENCIES[1:]:
            assert wr[lat] < 2.0, (
                f"쓰기 채널이 {wr_occupancy:.0%} 바쁠 뿐인데 지연 {lat} 에서 "
                f"{wr[lat]:.1f}% 를 잃는다 (OUT_BUFS={out_bufs})")
    else:
        # 채널이 포화면 지연이 그대로 얹힌다. 이중 버퍼로도 안 된다.
        assert wr[24] > 10.0, (
            f"쓰기 채널이 {wr_occupancy:.0%} 바쁜데 지연 24 를 흡수한다 "
            f"({wr[24]:.1f}%) — 측정이 변했다")
        assert wr[48] > wr[24], "지연이 커졌는데 손실이 줄었다 — 측정 오류"
