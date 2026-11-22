"""test_mxu_core.py — BF16 MXU 가 스스로 데이터를 가져와 계산하는가.

`mxu_bf16_128x128` 은 연산 블록이고 DMA 가 없었다. 피연산자를 포트
배열로 받아 테스트벤치가 먹여 줘야 했다. `mxu_core` 가 그것에
`gemm_core` (INT8) 와 같은 메모리 경로를 붙인다.

여기서 단언하는 것:

  1) DMA → SRAM → MXU → DMA 전 경로가 BF16 기준값과 맞는다.
  2) **피연산자 타이밍.** MXU 안에서 `tile_en` 이 레지스터라 타일은
     `data_valid` 보다 한 사이클 뒤에 피연산자를 먹는다. 동기 SRAM 이
     마침 그 타이밍이지만, 어긋나면 앞쪽 K-step 을 놓치고 마지막 것을
     중복 누산한다 — INT8 코어에서 정확히 그렇게 깨졌다. K>1 을 여러
     값으로 돌려 잡는다.
  3) weight-stationary 가 동작한다 (읽기 비트가 절반).
  4) 요청 쪼개기가 동작한다 (16비트 길이 필드).

## 허용 오차 — 고정소수점 누산기의 분해능

PE 는 FP32 덧셈이 아니라 **고정소수점(Kulisch 방식) 누산**을 쓴다
(`spec/pe-fixed-point-redesign.md`). `bf16_pkg.sv` 의 `ACC_F` 가 소수부
비트 수이고, 기본 16 이면 누산기 LSB 가 2⁻¹⁶ ≈ 1.53e-5 다.

그래서 **상대 오차로 비교하면 안 된다.** 결과가 작을수록 상대 오차가
커진다 — 1e-3 짜리 원소의 1 LSB 오차는 상대 1.5% 다. 처음 이 테스트를
상대 오차 1e-5 로 썼더니 네 테스트 전부가 1~4 LSB 차이로 실패했고,
그건 버그가 아니라 설계 특성이었다.

올바른 기준은 **절대 오차 ≤ Kt × 2^-ACC_F** 다. K-step 마다 곱을
고정소수점 격자에 맞추므로 오차가 그만큼까지 쌓일 수 있다.

> 이것은 교환이다. 고정소수점 누산은 큰 값에서 파괴적 상쇄가 없고
> (FP32 덧셈의 문제) PE 당 2.21배 작지만, **1.5e-5 보다 작은 값은
> 정확하게 표현하지 못한다.** LLM 추론은 정규화 뒤라 보통 괜찮지만
> 실제 제약이므로 적어 둔다. 더 작은 값이 필요하면 `ACC_F` 를 키워야
> 하고 그만큼 `ACC_W` 도 커진다.
"""
import struct

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from dma_stub import SimpleMem, start_dma


def to_bf16(f: float) -> int:
    """FP32 → BF16, round-to-nearest-even (RTL 과 같은 방식)."""
    bits = struct.unpack(">I", struct.pack(">f", float(f)))[0]
    lsb = (bits >> 16) & 1
    bits += 0x7FFF + lsb
    return (bits >> 16) & 0xFFFF


def bf16_to_f(b: int) -> float:
    return struct.unpack(">f", struct.pack(">I", (b & 0xFFFF) << 16))[0]


def bits_to_f(bits: int) -> float:
    return struct.unpack(">f", struct.pack(">I", bits & 0xFFFFFFFF))[0]


def read_acc_f() -> int:
    """rtl/bf16_pkg.sv 에서 ACC_F 를 읽는다 — 하드코딩하면 설계와 어긋난다."""
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "rtl" / "bf16_pkg.sv").read_text()
    m = re.search(r"localparam int ACC_F\s*=\s*(\d+);", src)
    assert m, "bf16_pkg.sv 에서 ACC_F 를 못 찾았다"
    return int(m.group(1))


ACC_F = read_acc_f()
ACC_LSB = 2.0 ** -ACC_F


def probe_config(dut):
    """구성을 DUT 에서 읽는다 (환경변수 불일치 방지)."""
    # acc_flat 을 센다. acc_lat 은 STREAM_RD=1 에서 1 개로 줄어든다
    # (래치를 두면 공유 변환기로 아낀 것을 플롭으로 되돌려주므로 둔다).
    accs = len(dut.acc_flat)
    dim = int(round(accs ** 0.5))
    assert dim * dim == accs, f"누산기 {accs} 가 제곱수가 아니다"
    rd_beat = len(dut.rd_data) // 8
    wr_beat = len(dut.wr_data) // 8
    kstep_bytes = dim * 2                   # BF16 = 2바이트
    assert rd_beat % kstep_bytes == 0, \
        f"읽기 비트 {rd_beat}B 가 K-step {kstep_bytes}B 의 배수가 아니다"
    max_kt = len(dut.u_act.mem) * (rd_beat // kstep_bytes) // 2
    return dim, rd_beat, wr_beat, kstep_bytes, max_kt


async def boot(dut):
    dut.rst_n.value = 0
    dut.cmd_valid.value = 0
    dut.act_addr.value = 0
    dut.wgt_addr.value = 0
    dut.out_addr.value = 0
    dut.Kt.value = 0
    dut.wgt_keep.value = 0
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


async def run_one(dut, act, wgt, out, Kt, wgt_keep=0, budget=500_000):
    """디스크립터 하나. (읽기 비트, 쓰기 비트, 요청 목록) 을 돌려준다."""
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
    dut.cmd_valid.value = 1
    await RisingEdge(dut.clk)
    dut.cmd_valid.value = 0

    rd = wr = 0
    reqs = []
    for _ in range(budget):
        await Timer(1, unit="ns")
        if int(dut.rd_req_valid.value) and int(dut.rd_req_ready.value):
            reqs.append((int(dut.rd_req_addr.value),
                         int(dut.rd_req_len_bytes.value)))
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.rd_data_valid.value) and int(dut.rd_data_ready.value):
            rd += 1
        if int(dut.wr_data_valid.value) and int(dut.wr_data_ready.value):
            wr += 1
        if int(dut.done.value) == 1:
            return rd, wr, reqs
    raise TimeoutError("done 없음")


def write_operands(mem, act_base, wgt_base, A_bf, B_bf, Kt, DIM):
    """A(DIM×Kt) 는 K-step 당 DIM 개 BF16 열, B(Kt×DIM) 는 행."""
    for k in range(Kt):
        mem.write_bytes(act_base + DIM * 2 * k,
                        b"".join(int(A_bf[i, k]).to_bytes(2, "little")
                                 for i in range(DIM)))
        mem.write_bytes(wgt_base + DIM * 2 * k,
                        b"".join(int(B_bf[k, j]).to_bytes(2, "little")
                                 for j in range(DIM)))


def read_result(mem, base, DIM):
    raw = mem.read_bytes(base, DIM * DIM * 4)
    out = np.zeros((DIM, DIM), dtype=np.float64)
    for i in range(DIM):
        for j in range(DIM):
            o = (i * DIM + j) * 4
            out[i, j] = bits_to_f(int.from_bytes(raw[o:o + 4], "little"))
    return out


def make_operands(rng, DIM, Kt):
    """BF16 로 양자화된 피연산자와 그 기준값."""
    A = rng.uniform(-2.0, 2.0, size=(DIM, Kt))
    B = rng.uniform(-2.0, 2.0, size=(Kt, DIM))
    A_bf = np.vectorize(to_bf16)(A).astype(np.uint16)
    B_bf = np.vectorize(to_bf16)(B).astype(np.uint16)
    # 기준값은 **양자화된 값**으로 계산한다
    A_q = np.vectorize(bf16_to_f)(A_bf)
    B_q = np.vectorize(bf16_to_f)(B_bf)
    return A_bf, B_bf, A_q @ B_q


def check(got, exp, Kt, label):
    """고정소수점 누산기의 분해능을 기준으로 본다.

    절대 오차 한계는 Kt × 2^-ACC_F 다 (K-step 마다 1 LSB 까지 반올림).
    큰 값에서는 BF16 피연산자 양자화가 더 크므로 상대 항도 더한다.
    """
    bound = Kt * ACC_LSB + 1e-6 * np.maximum(np.abs(exp), 1.0)
    err = np.abs(got - exp)
    bad = err > bound
    if bad.any():
        worst = np.unravel_index(int(np.argmax(err / bound)), err.shape)
        raise AssertionError(
            f"{label}: {bad.sum()}/{err.size}개가 한계를 넘었다.\n"
            f"  최악 {worst}: got={got[worst]:.9g} exp={exp[worst]:.9g}\n"
            f"  오차 {err[worst]:.3e} = {err[worst]/ACC_LSB:.1f} LSB "
            f"(한계 {bound[worst]:.3e} = {Kt} LSB)\n"
            f"  전부 틀리면 피연산자 타이밍(앞쪽 K-step 누락), "
            f"일부만이면 누산 경로를 보라.")


@cocotb.test()
async def test_bf16_matches_reference(dut):
    """여러 K 에서 전 경로가 BF16 기준값과 맞는다.

    K>1 을 여러 값으로 돈다 — 피연산자 타이밍이 어긋나면 앞쪽 K-step 을
    놓치므로 K=1 에서는 우연히 맞고 K>1 에서만 틀린다.
    """
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await boot(dut)
    DIM, RD_BEAT, WR_BEAT, KSTEP_B, MAX_KT = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(16161)

    dut._log.info("")
    dut._log.info(f"  DIM={DIM}  MAC={DIM*DIM}  K-step={KSTEP_B}B  "
                  f"읽기비트={RD_BEAT}B  쓰기비트={WR_BEAT}B  MAX_KT={MAX_KT}")
    dut._log.info(f"  고정소수점 누산기 ACC_F={ACC_F} → LSB {ACC_LSB:.3e}  "
                  f"(허용 오차 = Kt × LSB)")

    act, wgt, out = 0x1000, 0x40000, 0x80000
    for Kt in (1, 2, 3, 8, 17, 64):
        if Kt > MAX_KT:
            continue
        A_bf, B_bf, exp = make_operands(rng, DIM, Kt)
        write_operands(mem, act, wgt, A_bf, B_bf, Kt, DIM)
        rd, wr, reqs = await run_one(dut, act, wgt, out, Kt)
        got = read_result(mem, out, DIM)
        check(got, exp, Kt, f"Kt={Kt}")
        exp_wr = (DIM * DIM * 4) // WR_BEAT
        assert wr == exp_wr, f"Kt={Kt}: 쓰기 비트 {wr}, 기대 {exp_wr}"
        dut._log.info(f"  Kt={Kt:>3}  읽기 {rd:>4} / 쓰기 {wr:>3} 비트  "
                      f"MAC {DIM*DIM*Kt:>7,}  BF16 기준값 일치")


@cocotb.test()
async def test_weight_stationary(dut):
    """가중치를 한 번 적재하고 재사용하면 읽기가 절반이 된다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await boot(dut)
    DIM, RD_BEAT, WR_BEAT, KSTEP_B, MAX_KT = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(32323)

    Kt = min(32, MAX_KT)
    wgt = 0x40000
    # 가중치는 한 번만 쓴다. 활성값은 디스크립터마다 다르게.
    B = rng.uniform(-2.0, 2.0, size=(Kt, DIM))
    B_bf = np.vectorize(to_bf16)(B).astype(np.uint16)
    B_q = np.vectorize(bf16_to_f)(B_bf)
    for k in range(Kt):
        mem.write_bytes(wgt + DIM * 2 * k,
                        b"".join(int(B_bf[k, j]).to_bytes(2, "little")
                                 for j in range(DIM)))

    beats = []
    for n in range(4):
        A = rng.uniform(-2.0, 2.0, size=(DIM, Kt))
        A_bf = np.vectorize(to_bf16)(A).astype(np.uint16)
        A_q = np.vectorize(bf16_to_f)(A_bf)
        act = 0x1000 + 0x4000 * n
        for k in range(Kt):
            mem.write_bytes(act + DIM * 2 * k,
                            b"".join(int(A_bf[i, k]).to_bytes(2, "little")
                                     for i in range(DIM)))
        out = 0x80000 + 0x10000 * n
        rd, wr, _ = await run_one(dut, act, wgt, out, Kt,
                                  wgt_keep=1 if n else 0)
        beats.append(rd)
        check(read_result(mem, out, DIM), A_q @ B_q, Kt, f"WS #{n}")
        dut._log.info(f"  #{n} {'재사용' if n else '적재  '}  "
                      f"읽기 {rd:>4} 비트  일치")

    assert beats[0] == 2 * beats[1], \
        f"재사용이 절반이 아니다: {beats}"
    assert len(set(beats[1:])) == 1, f"재사용 비트가 흔들린다: {beats}"
    assert int(dut.perf_wgt_load.value) == 1
    assert int(dut.perf_wgt_reuse.value) == 3
    dut._log.info(f"  가중치 적재 1회 / 재사용 3회, 읽기 {beats[0]} → "
                  f"{beats[1]} 비트")


@cocotb.test()
async def test_request_splitting(dut):
    """긴 전송이 16비트 길이 한계 안에서 여러 요청으로 쪼개진다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await boot(dut)
    DIM, RD_BEAT, WR_BEAT, KSTEP_B, MAX_KT = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(64646)

    Kt = MAX_KT
    total = Kt * KSTEP_B
    REQ_MAX = (65535 // RD_BEAT) * RD_BEAT

    A_bf, B_bf, exp = make_operands(rng, DIM, Kt)
    act, wgt, out = 0x100000, 0x400000, 0x800000
    write_operands(mem, act, wgt, A_bf, B_bf, Kt, DIM)
    rd, wr, reqs = await run_one(dut, act, wgt, out, Kt, budget=2_000_000)

    dut._log.info("")
    dut._log.info(f"  Kt={Kt}, 단계당 {total:,}B, 요청 최대 {REQ_MAX:,}B")
    dut._log.info(f"  읽기 요청 {len(reqs)}개")
    for i, (a, ln) in enumerate(reqs):
        assert ln <= REQ_MAX, f"요청 {i} 길이 {ln} > {REQ_MAX}"
        assert ln % RD_BEAT == 0, f"요청 {i} 길이 {ln} 가 비트 배수가 아니다"
    for label, base in (("활성값", act), ("가중치", wgt)):
        group = [r for r in reqs if base <= r[0] < base + total]
        assert sum(ln for _, ln in group) == total, \
            f"{label} 합계 {sum(ln for _, ln in group):,}B, 기대 {total:,}B"
        addr = base
        for a, ln in group:
            assert a == addr, f"{label} 주소 끊김: {a:#x}, 기대 {addr:#x}"
            addr += ln
        dut._log.info(f"    {label} {len(group)}개, 합계 {total:,}B, 주소 연속")

    check(read_result(mem, out, DIM), exp, Kt, f"Kt={Kt} (쪼갠 전송)")
    dut._log.info(f"  BF16 기준값 일치")


@cocotb.test()
async def test_streamed_throughput(dut):
    """연속 처리량. 3단 파이프라인이 겹치는지 숫자로 본다."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    await boot(dut)
    DIM, RD_BEAT, WR_BEAT, KSTEP_B, MAX_KT = probe_config(dut)
    mem = SimpleMem()
    start_dma(dut, mem, log=False, rd_beat_bytes=RD_BEAT, wr_beat_bytes=WR_BEAT)
    rng = np.random.default_rng(12345)

    # 쓰기 비트 = DIM²/WPB. 균형점은 Kt = DIM²·4/WR_BEAT.
    Kt = min(MAX_KT, (DIM * DIM * 4) // WR_BEAT)
    n_ops = 12
    A_bf, B_bf, exp = make_operands(rng, DIM, Kt)
    act, wgt = 0x1000, 0x40000
    write_operands(mem, act, wgt, A_bf, B_bf, Kt, DIM)

    out0 = 0x100000
    issued = done_seen = total = 0
    started = False
    done_at = []

    for _ in range(4_000_000):
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

    for n in (0, n_ops // 2, n_ops - 1):
        check(read_result(mem, out0 + 0x10000 * n, DIM), exp, Kt,
              f"스트림 #{n}")

    gaps = [b - a for a, b in zip(done_at, done_at[1:])]
    steady = gaps[2:-2] if len(gaps) > 6 else gaps
    per_op = sum(steady) / len(steady)
    macs = DIM * DIM * Kt
    dut._log.info("")
    dut._log.info(f"  DIM={DIM} Kt={Kt}  MAC/디스크립터 {macs:,}")
    dut._log.info(f"  정상상태 {min(steady)}~{max(steady)}, 평균 "
                  f"{per_op:.1f} 사이클/op")
    dut._log.info(f"  → {macs/per_op:,.0f} MAC/사이클 "
                  f"({macs/per_op*2:,.0f} BF16 FLOP/사이클)")
    dut._log.info(f"  → 배열 가동률 {macs/per_op/(DIM*DIM)*100:.1f}%")
    dut._log.info("")
