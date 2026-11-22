"""test_mxu_deep_layer.py — BF16 경로가 실제 LLM 레이어 깊이를 견디는가.

INT8 쪽은 `test_real_layer_depth.py` 가 이것을 확인한다. BF16 쪽은
README 에 "실제 레이어 깊이 (K=11008) | 있음 | **미측정**" 으로 남아
있었다. 그 칸을 채운다.

BF16 은 원소가 2바이트라 같은 K 에서 INT8 의 두 배를 읽는다. 그래서
요청 쪼개기가 더 일찍 필요하다.

  K-step 하나      = DIM × 2 바이트   (INT8 은 DIM 바이트)
  한 요청 상한     = 65,504 바이트    (16비트 길이 필드, 비트 정렬)
  DIM=16 이면      K-step 32B → 한 요청에 2,047 K-step
  DIM=32 이면      K-step 64B → 한 요청에 1,023 K-step

따라서 K=4096 은 DIM=16 에서 **요청 3개**, DIM=32 에서 **5개**로
쪼개져야 한다. 쪼개기가 깨지면 조용히 0 바이트를 요청하거나(행업)
길이가 잘려(틀린 값) 끝난다 — INT8 쪽에서 둘 다 당했다.

여기서 재는 것은 처리량이 아니라 **깊은 K 에서 맞는 값이 나오는가**와
**가중치 재사용이 그대로 살아 있는가** 다. Kt 가 크면 시뮬이 느리므로
디스크립터 수를 줄인다.

허용 오차는 `test_mxu_core` 와 같다 — 고정소수점 누산기 LSB 기준
절대 오차 `Kt × 2^-ACC_F`. K 가 깊으면 한계도 그만큼 커진다.
"""
import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from dma_stub import SimpleMem, start_dma
from test_mxu_core import (ACC_F, ACC_LSB, bf16_to_f, boot, make_operands,
                           probe_config, read_result, run_one, to_bf16,
                           write_operands)

# 16비트 길이 필드의 상한 (비트 크기로 내림).
REQ_MAX = 65504


@cocotb.test()
async def test_deep_layer_splits_and_matches(dut):
    """MAX_KT 를 실제 레이어 깊이로 올려도 값이 맞고 요청이 쪼개진다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await boot(dut)
    DIM, RD_BEAT, WR_BEAT, KSTEP_B, MAX_KT = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(777)

    Kt = MAX_KT
    kstep_per_req = (REQ_MAX // RD_BEAT) * RD_BEAT // KSTEP_B
    exp_reqs_per_phase = -(-Kt // kstep_per_req)   # 올림

    dut._log.info("")
    dut._log.info(f"  DIM={DIM}  Kt={Kt}  K-step={KSTEP_B}B  "
                  f"단계당 {Kt*KSTEP_B:,}B")
    dut._log.info(f"  한 요청 최대 {kstep_per_req:,} K-step "
                  f"→ 단계마다 요청 {exp_reqs_per_phase}개 기대")

    A_bf, B_bf, exp = make_operands(rng, DIM, Kt)
    act, wgt, out = 0x1000, 0x400000, 0x800000
    write_operands(mem, act, wgt, A_bf, B_bf, Kt, DIM)

    rd, wr, reqs = await run_one(dut, act, wgt, out, Kt, budget=20_000_000)
    got = read_result(mem, out, DIM)

    # 요청이 쪼개졌는가. 활성값 + 가중치 두 단계분이다.
    assert len(reqs) == 2 * exp_reqs_per_phase, (
        f"읽기 요청 {len(reqs)}개, 기대 {2*exp_reqs_per_phase}개 "
        f"(활성값 {exp_reqs_per_phase} + 가중치 {exp_reqs_per_phase}). "
        f"쪼개기가 깨졌다")
    for addr, length in reqs:
        assert 0 < length <= REQ_MAX, \
            f"요청 길이 {length} — 0 이거나 16비트 한계를 넘는다"

    # 각 단계의 요청이 주소 연속이고 합이 맞는가.
    for phase, base in (("활성값", act), ("가중치", wgt)):
        pr = [(a, l) for a, l in reqs if base <= a < base + Kt * KSTEP_B]
        assert len(pr) == exp_reqs_per_phase, f"{phase} 요청 {len(pr)}개"
        assert sum(l for _, l in pr) == Kt * KSTEP_B, \
            f"{phase} 합계 {sum(l for _, l in pr)}B, 기대 {Kt*KSTEP_B}B"
        nxt = base
        for a, l in pr:
            assert a == nxt, f"{phase} 주소 끊김: {a:#x}, 기대 {nxt:#x}"
            nxt += l
        dut._log.info(f"  {phase} 요청 {len(pr)}개, 합계 "
                      f"{sum(l for _, l in pr):,}B, 주소 연속")

    bound = Kt * ACC_LSB
    err = np.abs(got - exp).max()
    dut._log.info(f"  최악 절대 오차 {err:.3e} "
                  f"(한계 {bound:.3e} = {Kt} LSB)")
    assert err <= bound, (
        f"깊은 K 에서 값이 틀렸다: 오차 {err:.3e} > 한계 {bound:.3e}. "
        f"전부 틀리면 쪼갠 전송의 적재 인덱스, 일부면 누산 경로를 보라")


@cocotb.test()
async def test_weight_reuse_survives_deep_k(dut):
    """깊은 K 에서도 weight-stationary 가 읽기를 절반으로 만든다.

    쪼개기와 재사용이 서로를 깨뜨리기 쉽다 — 재사용 디스크립터가 가중치
    단계를 건너뛸 때 적재 인덱스를 잘못 리셋하면 활성값이 가중치
    자리에 들어간다. INT8 쪽에서 `phase_begin` 으로 고친 함정이다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await boot(dut)
    DIM, RD_BEAT, WR_BEAT, KSTEP_B, MAX_KT = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(778)

    Kt = MAX_KT
    A_bf, B_bf, exp = make_operands(rng, DIM, Kt)
    act, wgt, out = 0x1000, 0x400000, 0x800000
    write_operands(mem, act, wgt, A_bf, B_bf, Kt, DIM)

    bound = Kt * ACC_LSB
    reads = []
    for d in range(3):
        rd, wr, reqs = await run_one(dut, act, wgt, out + 0x10000 * d, Kt,
                                     wgt_keep=(1 if d else 0),
                                     budget=20_000_000)
        got = read_result(mem, out + 0x10000 * d, DIM)
        err = np.abs(got - exp).max()
        assert err <= bound, (
            f"#{d} ({'재사용' if d else '적재'}): 오차 {err:.3e} > {bound:.3e}")
        reads.append(rd)
        dut._log.info(f"  #{d} {'재사용' if d else '적재  '}  "
                      f"읽기 {rd:>5} 비트  오차 {err:.3e}  일치")

    assert reads[1] * 2 == reads[0], (
        f"재사용이 읽기를 절반으로 만들지 않는다: 적재 {reads[0]}, "
        f"재사용 {reads[1]}")
    assert reads[1] == reads[2], "재사용 비트 수가 디스크립터마다 다르다"
    dut._log.info(f"  → 깊은 K(={Kt}) 에서도 재사용이 읽기를 "
                  f"{reads[0]} → {reads[1]} 비트로 줄인다")


@cocotb.test()
async def test_deep_throughput(dut):
    """깊은 K 에서의 정상상태 처리량.

    `test_mxu_core` 의 처리량 측정은 Kt 를 균형점(DIM²·4/WR_BEAT) 으로
    잡는다. 거기서는 쓰기가 연산과 맞먹으므로 쓰기 쪽 설계 선택이
    그대로 주기에 드러난다.

    실제 LLM 레이어는 그 반대다 — K 가 수천이고 출력 타일은 그대로이니
    **연산이 쓰기를 압도한다.** 그 조건에서의 숫자를 따로 가진다.
    `STREAM_RD` 처럼 쓰기 쪽을 바꾸는 선택의 값어치가 여기서 달라진다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await boot(dut)
    DIM, RD_BEAT, WR_BEAT, KSTEP_B, MAX_KT = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(779)

    Kt = MAX_KT
    out_beats = (DIM * DIM * 4) // WR_BEAT
    A_bf, B_bf, exp = make_operands(rng, DIM, Kt)
    act, wgt = 0x1000, 0x400000
    out0 = 0x800000
    write_operands(mem, act, wgt, A_bf, B_bf, Kt, DIM)

    n_ops = 6
    issued = done_seen = total = 0
    started = False
    done_at = []

    for _ in range(40_000_000):
        want = issued < n_ops
        dut.act_addr.value = act
        dut.wgt_addr.value = wgt
        dut.out_addr.value = out0 + 0x10000 * issued
        dut.Kt.value = Kt
        dut.wgt_keep.value = 1 if issued > 0 else 0
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

    bound = Kt * ACC_LSB
    for n in (0, n_ops - 1):
        got = read_result(mem, out0 + 0x10000 * n, DIM)
        err = np.abs(got - exp).max()
        assert err <= bound, f"스트림 #{n}: 오차 {err:.3e} > {bound:.3e}"

    gaps = [b - a for a, b in zip(done_at, done_at[1:])]
    steady = gaps[1:] if len(gaps) > 2 else gaps
    per_op = sum(steady) / len(steady)
    macs = DIM * DIM * Kt

    dut._log.info("")
    dut._log.info(f"  DIM={DIM} Kt={Kt}  MAC/디스크립터 {macs:,}  "
                  f"출력 {out_beats} 비트")
    dut._log.info(f"  정상상태 {min(steady)}~{max(steady)}, "
                  f"평균 {per_op:.1f} 사이클/op")
    dut._log.info(f"  → {macs/per_op:,.0f} MAC/사이클")
    dut._log.info(f"  → 배열 가동률 {macs/per_op/(DIM*DIM)*100:.1f}%")
    dut._log.info(f"  연산 {Kt} 사이클분 vs 쓰기 {out_beats} 비트 "
                  f"→ 연산이 {Kt/out_beats:.0f}배 크다")

    # 연산이 쓰기를 압도하는 구성이므로 가동률이 높아야 한다.
    assert macs / per_op / (DIM * DIM) > 0.90, \
        f"가동률 {macs/per_op/(DIM*DIM):.1%} — 깊은 K 에서는 90% 를 넘어야 한다"
