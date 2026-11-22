"""test_mxu_cluster.py — BF16 텐서 병렬 클러스터.

`gemm_cluster` (INT8) 가 증명한 것을 BF16 쪽에서 다시 확인한다.
베끼기 테스트가 아니라, INT8 쪽에서 **조용히 틀렸던 두 가지**를
BF16 쪽에서도 잡히게 두는 것이 목적이다.

  1) 코어마다 다른 가중치 조각을 쓰고 다른 출력 조각에 쓴다.
     `cmd_valid` 게이트가 없으면 먼저 준비된 코어가 같은 디스크립터를
     두 번 받아 **출력 주소가 어긋난 채 조용히 돌아간다.** INT8 쪽에서
     디스크립터 16개를 흘렸는데 쓰기 주소가 4개뿐이었다. 완료 펄스는
     16개가 와서 겉보기로는 정상이었다. 그래서 주소를 센다.
  2) 완료는 코어마다 따로 세야 한다. 하나의 카운터로 모으면 쓰기가
     직렬화된 틈에 한 코어가 자기 완료 두 개로 카운터를 채운다.
     그래서 **디스크립터마다** 모든 코어의 결과를 검사한다.

그리고 클러스터를 두는 이유 자체를 숫자로 본다: **활성값은 한 번만
읽는다.** 코어가 CORES 개여도 활성값 읽기 비트 수가 늘지 않아야 한다.
늘면 텐서 병렬의 근거가 사라진다.
"""
import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from dma_stub import SimpleMem, start_dma
from test_mxu_core import (ACC_F, ACC_LSB, bits_to_f, check, make_operands,
                           read_result, write_operands)


def probe_config(dut):
    """구성을 DUT 에서 읽는다 (환경변수 불일치 방지)."""
    cores = len(dut.c_busy)
    accs = len(dut.CORE[0].u_core.acc_flat)
    dim = int(round(accs ** 0.5))
    assert dim * dim == accs, f"누산기 {accs} 가 제곱수가 아니다"
    rd_beat = len(dut.rd_data) // 8
    wr_beat = len(dut.wr_data) // 8
    kstep_bytes = dim * 2
    assert rd_beat % kstep_bytes == 0, \
        f"읽기 비트 {rd_beat}B 가 K-step {kstep_bytes}B 의 배수가 아니다"
    return cores, dim, rd_beat, wr_beat, kstep_bytes


async def boot(dut):
    dut.rst_n.value = 0
    dut.cmd_valid.value = 0
    dut.act_addr.value = 0
    dut.wgt_addr.value = 0
    dut.out_addr.value = 0
    dut.Kt.value = 0
    dut.wgt_keep.value = 0
    dut.wgt_stride.value = 0
    dut.out_stride.value = 0
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


async def issue(dut, act, wgt, out, Kt, wgt_stride, out_stride, wgt_keep=0):
    """디스크립터 하나를 발행한다 (완료를 기다리지 않는다)."""
    for _ in range(20_000):
        await Timer(1, unit="ns")
        if int(dut.cmd_ready.value) == 1:
            break
        await RisingEdge(dut.clk)
    else:
        raise TimeoutError("cmd_ready 안 섬")
    dut.act_addr.value = act
    dut.wgt_addr.value = wgt
    dut.out_addr.value = out
    dut.Kt.value = Kt
    dut.wgt_keep.value = wgt_keep
    dut.wgt_stride.value = wgt_stride
    dut.out_stride.value = out_stride
    dut.cmd_valid.value = 1
    await RisingEdge(dut.clk)
    dut.cmd_valid.value = 0


async def wait_done(dut, budget=2_000_000):
    """완료 펄스를 기다리며 읽기/쓰기 비트와 주소를 센다."""
    rd = wr = 0
    rd_reqs, wr_reqs = [], []
    for _ in range(budget):
        await Timer(1, unit="ns")
        if int(dut.rd_req_valid.value) and int(dut.rd_req_ready.value):
            rd_reqs.append((int(dut.rd_req_addr.value),
                            int(dut.rd_req_len_bytes.value)))
        if int(dut.wr_req_valid.value) and int(dut.wr_req_ready.value):
            wr_reqs.append(int(dut.wr_req_addr.value))
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.rd_data_valid.value) and int(dut.rd_data_ready.value):
            rd += 1
        if int(dut.wr_data_valid.value) and int(dut.wr_data_ready.value):
            wr += 1
        if int(dut.done.value) == 1:
            return rd, wr, rd_reqs, wr_reqs
    raise TimeoutError("done 없음")


def layout(dim, cores):
    """메모리 배치. 조각끼리 겹치지 않게 넉넉히 띄운다."""
    act = 0x1000
    wgt = 0x40000
    out = 0x200000
    wgt_stride = 0x10000          # 코어당 가중치 조각 간격
    out_stride = dim * dim * 4    # 출력 타일 하나 (연속으로 둔다)
    return act, wgt, out, wgt_stride, out_stride


@cocotb.test()
async def test_each_core_computes_its_own_slice(dut):
    """코어 k 가 가중치 조각 k 로 계산해 출력 조각 k 에 쓴다.

    조각마다 **다른** 가중치를 넣는다. 코어가 남의 조각을 읽거나 같은
    디스크립터를 두 번 받으면 기준값과 어긋난다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await boot(dut)
    CORES, DIM, RD_BEAT, WR_BEAT, KSTEP_B = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(20250)
    act, wgt, out, wgt_stride, out_stride = layout(DIM, CORES)

    dut._log.info("")
    dut._log.info(f"  CORES={CORES}  DIM={DIM}  MAC={CORES*DIM*DIM}  "
                  f"K-step={KSTEP_B}B  읽기비트={RD_BEAT}B  쓰기비트={WR_BEAT}B")
    dut._log.info(f"  고정소수점 ACC_F={ACC_F} → LSB {ACC_LSB:.3e}")

    for Kt in (1, 3, 17, 64):
        # 활성값은 공유, 가중치는 코어마다 다르다.
        A_bf, B0, _ = make_operands(rng, DIM, Kt)
        exps = []
        for k in range(CORES):
            _, Bk, _ = make_operands(rng, DIM, Kt)
            write_operands(mem, act, wgt + wgt_stride * k, A_bf, Bk, Kt, DIM)
            # 기준값: 양자화된 A 와 양자화된 Bk 의 내적
            from test_mxu_core import bf16_to_f
            A_q = np.vectorize(bf16_to_f)(A_bf)
            B_q = np.vectorize(bf16_to_f)(Bk)
            exps.append(A_q @ B_q)

        await issue(dut, act, wgt, out, Kt, wgt_stride, out_stride)
        rd, wr, rd_reqs, wr_reqs = await wait_done(dut)

        for k in range(CORES):
            got = read_result(mem, out + out_stride * k, DIM)
            check(got, exps[k], Kt, f"Kt={Kt} 코어{k}")

        exp_wr = (DIM * DIM * 4) // WR_BEAT * CORES
        assert wr == exp_wr, f"Kt={Kt}: 쓰기 비트 {wr}, 기대 {exp_wr}"
        # 쓰기 요청 주소가 코어 수만큼 **서로 달라야** 한다.
        assert len(set(wr_reqs)) == CORES, (
            f"Kt={Kt}: 쓰기 주소 {len(set(wr_reqs))}개, 기대 {CORES}개 "
            f"— 코어들이 같은 자리에 쓴다 (디스크립터 중복 수락?)")
        dut._log.info(f"  Kt={Kt:>3}  읽기 {rd:>4} / 쓰기 {wr:>3} 비트  "
                      f"쓰기 주소 {len(set(wr_reqs))}개  {CORES}개 코어 전부 일치")


@cocotb.test()
async def test_activation_is_read_once(dut):
    """활성값은 브로드캐스트된다 — 코어 수만큼 늘지 않는다.

    이것이 텐서 병렬의 근거다. 연산은 CORES 배가 되는데 활성값
    트래픽은 그대로여야 한다. 늘면 읽기 대역폭이 같이 늘어 아무것도
    벌지 못한다.

    weight-stationary 를 켜고 재사용 디스크립터를 보내면 읽기가
    **활성값뿐**이다. 그 비트 수를 코어 수와 비교한다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await boot(dut)
    CORES, DIM, RD_BEAT, WR_BEAT, KSTEP_B = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(31313)
    act, wgt, out, wgt_stride, out_stride = layout(DIM, CORES)

    Kt = min(32, 64)
    A_bf, _, _ = make_operands(rng, DIM, Kt)
    for k in range(CORES):
        _, Bk, _ = make_operands(rng, DIM, Kt)
        write_operands(mem, act, wgt + wgt_stride * k, A_bf, Bk, Kt, DIM)

    # #0: 가중치 적재 + 활성값
    await issue(dut, act, wgt, out, Kt, wgt_stride, out_stride, wgt_keep=0)
    rd_load, _, _, _ = await wait_done(dut)

    # #1..: 재사용 — 활성값만 읽는다
    reuse = []
    for d in range(3):
        await issue(dut, act, wgt, out, Kt, wgt_stride, out_stride, wgt_keep=1)
        rd, _, _, _ = await wait_done(dut)
        reuse.append(rd)

    act_beats = (Kt * KSTEP_B) // RD_BEAT
    dut._log.info("")
    dut._log.info(f"  CORES={CORES}  Kt={Kt}  활성값 한 벌 = {act_beats} 비트")
    dut._log.info(f"  #0 적재   읽기 {rd_load:>4} 비트")
    for i, r in enumerate(reuse):
        dut._log.info(f"  #{i+1} 재사용 읽기 {r:>4} 비트")

    # 재사용 디스크립터의 읽기는 활성값 한 벌이다 — 코어 수와 무관.
    for i, r in enumerate(reuse):
        assert r == act_beats, (
            f"재사용 #{i+1}: 읽기 {r} 비트, 기대 {act_beats} "
            f"(활성값 한 벌). {r // act_beats}배라면 브로드캐스트가 "
            f"깨져 코어마다 따로 읽고 있다")

    # 적재 디스크립터는 활성값 1벌 + 가중치 CORES벌이다.
    exp_load = act_beats * (1 + CORES)
    assert rd_load == exp_load, \
        f"적재: 읽기 {rd_load} 비트, 기대 {exp_load} (활성값 1 + 가중치 {CORES})"
    dut._log.info(f"  → 활성값 트래픽은 코어 수와 무관하다 "
                  f"(연산 {CORES}배, 활성값 1배)")


@cocotb.test()
async def test_streamed_descriptors(dut):
    """디스크립터를 연속으로 밀어 넣어도 코어마다 결과가 맞는다.

    완료 집계가 틀리면 여기서 드러난다. 하나의 카운터로 c_done 을
    모으면, 쓰기가 직렬화된 틈에 한 코어가 자기 완료 두 개로 카운터를
    채워 **아직 안 끝난 코어의 결과를 읽게 된다.** INT8 쪽에서
    CORES=2 는 8번째, CORES=4 는 첫 디스크립터에서 깨졌다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await boot(dut)
    CORES, DIM, RD_BEAT, WR_BEAT, KSTEP_B = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(44444)
    act, wgt, out, wgt_stride, out_stride = layout(DIM, CORES)

    from test_mxu_core import bf16_to_f
    Kt = 16
    NDESC = 8
    # 디스크립터마다 다른 활성값, 다른 출력 자리.
    #   코어 k 디스크립터 d → out + (d*CORES + k) * out_stride
    descs = []
    for d in range(NDESC):
        A_bf, _, _ = make_operands(rng, DIM, Kt)
        A_q = np.vectorize(bf16_to_f)(A_bf)
        descs.append((act + 0x8000 * d, A_bf, A_q))

    # 가중치는 한 번만 적재하고 재사용한다 (weight-stationary).
    Bqs = []
    for k in range(CORES):
        _, Bk, _ = make_operands(rng, DIM, Kt)
        Bqs.append(np.vectorize(bf16_to_f)(Bk))
        for d in range(NDESC):
            write_operands(mem, descs[d][0], wgt + wgt_stride * k,
                           descs[d][1], Bk, Kt, DIM)

    dones = 0
    for d in range(NDESC):
        a_addr, _, _ = descs[d]
        await issue(dut, a_addr, wgt, out + out_stride * CORES * d,
                    Kt, wgt_stride, out_stride, wgt_keep=(1 if d else 0))
        await wait_done(dut)
        dones += 1

    assert dones == NDESC, f"완료 {dones}개, 디스크립터 {NDESC}개"

    # 모든 (디스크립터, 코어) 조합을 검사한다.
    bad = 0
    for d in range(NDESC):
        for k in range(CORES):
            got = read_result(mem, out + out_stride * (CORES * d + k), DIM)
            exp = descs[d][2] @ Bqs[k]
            err = np.abs(got - exp).max()
            if err > Kt * ACC_LSB:
                bad += 1
                dut._log.error(f"  #{d} 코어{k}: 최악 오차 {err:.3e} "
                               f"(한계 {Kt * ACC_LSB:.3e})")
    dut._log.info("")
    dut._log.info(f"  디스크립터 {NDESC}개 × 코어 {CORES}개 = "
                  f"{NDESC * CORES}개 출력 타일 전부 검사")
    assert bad == 0, f"{bad}/{NDESC * CORES}개 타일이 틀렸다"
    dut._log.info(f"  → 전부 일치. 완료 집계가 코어별로 맞다.")


@cocotb.test()
async def test_throughput_scales(dut):
    """코어를 늘리면 처리량이 따라 오는가 — 디스크립터를 흘려 보내며 잰다.

    **디스크립터를 하나씩 기다려서는 안 된다.** 기다리면 적재·연산·쓰기가
    겹치지 못해 주기가 세 단계의 합이 된다. 처음 이 테스트를
    "발행 → 완료 대기 → 발행" 으로 썼더니 CORES=1 에서도 156 사이클/op
    이 나왔다 (단일 코어 직접 측정은 71.0 이다). 클러스터 탓이 아니라
    테스트가 파이프라인을 막은 것이었다.

    그래서 `cmd_valid` 를 계속 세워 두고 완료 펄스 사이 간격을 잰다 —
    `test_mxu_core` 의 처리량 측정과 같은 방식이다.

    균형점: 쓰기 비트가 코어당 DIM²/WPB 이고 코어 수만큼 직렬화되므로
    Kt = CORES · DIM²·4/WR_BEAT 에서 연산과 쓰기가 맞는다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await boot(dut)
    CORES, DIM, RD_BEAT, WR_BEAT, KSTEP_B = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(55555)
    act, wgt, out, wgt_stride, out_stride = layout(DIM, CORES)

    from test_mxu_core import bf16_to_f
    Kt = CORES * (DIM * DIM * 4) // WR_BEAT
    n_ops = 12

    A_bf, _, _ = make_operands(rng, DIM, Kt)
    A_q = np.vectorize(bf16_to_f)(A_bf)
    Bqs = []
    for k in range(CORES):
        _, Bk, _ = make_operands(rng, DIM, Kt)
        Bqs.append(np.vectorize(bf16_to_f)(Bk))
        write_operands(mem, act, wgt + wgt_stride * k, A_bf, Bk, Kt, DIM)

    # 디스크립터마다 출력 자리를 띄운다 (코어 CORES 개 분량씩).
    out_span = out_stride * CORES
    issued = done_seen = total = 0
    started = False
    done_at = []

    for _ in range(8_000_000):
        want = issued < n_ops
        dut.act_addr.value = act
        dut.wgt_addr.value = wgt
        dut.out_addr.value = out + out_span * issued
        dut.Kt.value = Kt
        dut.wgt_keep.value = 1 if issued > 0 else 0
        dut.wgt_stride.value = wgt_stride
        dut.out_stride.value = out_stride
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

    # 흘려 보낸 뒤에도 결과가 맞아야 한다 (코어별로 본다).
    for n in (0, n_ops // 2, n_ops - 1):
        for k in range(CORES):
            got = read_result(mem, out + out_span * n + out_stride * k, DIM)
            check(got, A_q @ Bqs[k], Kt, f"스트림 #{n} 코어{k}")

    gaps = [b - a for a, b in zip(done_at, done_at[1:])]
    steady = gaps[2:-2] if len(gaps) > 6 else gaps
    per_op = sum(steady) / len(steady)
    macs = CORES * DIM * DIM * Kt
    mac_cyc = macs / per_op
    util = mac_cyc / (CORES * DIM * DIM)

    dut._log.info("")
    dut._log.info(f"  CORES={CORES} DIM={DIM} Kt={Kt}  "
                  f"MAC/디스크립터 {macs:,}  (쓰기 {WR_BEAT*8}비트)")
    dut._log.info(f"  정상상태 {min(steady)}~{max(steady)}, "
                  f"평균 {per_op:.1f} 사이클/op")
    dut._log.info(f"  → {mac_cyc:,.0f} MAC/사이클 "
                  f"({2*mac_cyc:,.0f} BF16 FLOP/사이클)")
    dut._log.info(f"  → 배열 가동률 {util:.1%}  (MAC {CORES*DIM*DIM}개)")

    assert max(steady) - min(steady) <= 2, f"간격이 흔들린다: {steady}"
    # 가동률은 코어 수와 무관하게 유지돼야 한다. 떨어지면 중재가
    # 병목이 된 것이고, 코어를 늘린 만큼 벌지 못한다.
    assert util > 0.80, (
        f"가동률 {util:.1%} — 코어 {CORES}개에서 중재가 병목이다")
