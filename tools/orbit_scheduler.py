"""
orbit_scheduler.py — ORBIT-G2 Scheduler

Supports two modes:
  - "protoa": Serial (1 outstanding). Completion = TC0 IDLE poll.
  - "protob": DMA-aware. Completion = TC0 IDLE + DMA DONE.
              Supports up to max_outstanding ops (default 2).

Both modes share: queue routing, timeout, fault policy, OOM detection.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from tools.orbit_device import OrbitDevice
from tools.orbit_session import OrbitSession, OrbitOp, OrbitCompletion, OpStatus
from tools.orbit_mmio_map import Opcode, OOM_PRESSURE, OOM_CRITICAL, OOM_EMERG
from tools.orbit_desc import pack_nop, pack_gemm


class FaultPolicy(Enum):
    STOP   = "stop"     # stop on first fault, collect diagnostics
    SKIP   = "skip"     # skip faulted op, continue
    ABORT  = "abort"    # stop and don't clear fault


@dataclass
class SchedulerResult:
    ops_submitted: int
    ops_completed: int
    ops_faulted: int
    ops_rejected: int
    oom_throttled: bool
    completions: list[OrbitCompletion]


class OrbitScheduler:
    """Serial descriptor scheduler for Proto-A.

    Usage:
        sched = OrbitScheduler(device)
        sched.add_nop(queue=0)
        sched.add_gemm(act=0x1000, wgt=0x2000, out=0x3000, kt=16)
        result = sched.run(fault_policy=FaultPolicy.STOP)
    """

    # Queue class priority (matches RTL arbiter: Q3>Q0>Q1>Q2)
    QUEUE_PRIORITY = [3, 0, 1, 2]

    def __init__(self, device: OrbitDevice, max_polls: int = 10000,
                 mode: str = "protoa", max_outstanding: int = 1):
        """
        Args:
            mode: "protoa" (TC poll only) or "protob" (TC + DMA poll)
            max_outstanding: max concurrent ops (Proto-A=1, Proto-B up to 4)
        """
        if mode not in ("protoa", "protob"):
            raise ValueError(f"Unknown mode: {mode}")
        self._dev = device
        self._max_polls = max_polls
        self._mode = mode
        self._max_outstanding = max_outstanding
        self._ops: list[OrbitOp] = []
        self._dma: "OrbitDma | None" = None
        if mode == "protob":
            from tools.orbit_dma import OrbitDma
            self._dma = OrbitDma(device.backend)

    def add_op(self, desc: bytes, queue: int = 0, tag: str = ""):
        self._ops.append(OrbitOp(desc=desc, queue=queue, tag=tag))

    def add_nop(self, queue: int = 0, tag: str = ""):
        self.add_op(pack_nop(), queue, tag or "NOP")

    def add_gemm(self, act: int = 0, wgt: int = 0, out: int = 0,
                 kt: int = 4, queue: int = 0, tag: str = "",
                 flags: int = 0, wgt_stride: int = 0, out_stride: int = 0):
        """GEMM 하나를 큐에 넣는다.

        flags 로 weight-stationary / INT4 / K 분할을 켠다 (DescFlag).
        모양 단위로 쪼개려면 add_gemm_shape() 를 쓴다.
        """
        self.add_op(pack_gemm(act, wgt, out, kt, flags=flags,
                              wgt_stride=wgt_stride, out_stride=out_stride),
                    queue, tag or "GEMM")

    def add_gemm_shape(self, shape, cfg, act_base: int, wgt_base: int,
                       out_base: int, queue: int = 0, tag: str = ""):
        """(M, K, N) 행렬곱 하나를 디스크립터 수열로 쪼개 큐에 넣는다.

        타일링 순서와 플래그는 tools.gemm_tiler 가 정한다 — weight-stationary
        는 호스트가 그 순서로 내보낼 때만 값어치가 있다. 주소는 MemLayout
        이 계산하고, 같은 코드를 sim/cocotb/test_gemm_e2e.py 가 실제 RTL
        과 대조한다.

        Args:
            shape: gemm_tiler.GemmShape
            cfg:   gemm_tiler.TileConfig (하드웨어 파라미터와 맞아야 한다)
            *_base: 활성값 / 가중치 / 출력 버퍼의 시작 주소.
                    필요한 크기는 MemLayout.total_bytes() 가 알려준다.

        Returns:
            넣은 디스크립터 수.
        """
        from tools.gemm_tiler import MemLayout, tile_gemm

        layout = MemLayout(shape, cfg, act_base, wgt_base, out_base)
        plan = tile_gemm(shape, cfg)
        base_tag = tag or f"GEMM {shape.M}x{shape.K}x{shape.N}"
        for i, item in enumerate(plan.issue(layout)):
            self.add_op(pack_gemm(**item), queue, f"{base_tag} #{i}")
        return len(plan.descs)

    def add_unsupported(self, opcode: int, queue: int = 0, tag: str = ""):
        """Add an op with an unsupported opcode (for testing error path)."""
        from tools.orbit_desc import pack_descriptor
        self.add_op(pack_descriptor(opcode), queue, tag or f"OP_{opcode:#x}")

    def run(self, fault_policy: FaultPolicy = FaultPolicy.STOP,
            oom_abort_level: int = OOM_EMERG) -> SchedulerResult:
        """Execute all queued ops serially."""
        session = OrbitSession(self._dev, max_polls=self._max_polls)
        oom_throttled = False

        completions = []
        remaining = list(self._ops)
        self._ops.clear()

        for op in remaining:
            # Check OOM before submit
            oom = self._dev.read_oom_state()
            if oom >= oom_abort_level:
                oom_throttled = True
                completions.append(OrbitCompletion(
                    op=op, status=OpStatus.REJECTED,
                    error_msg=f"OOM level {oom} >= abort threshold {oom_abort_level}"
                ))
                if fault_policy == FaultPolicy.STOP:
                    # Reject remaining
                    for r in remaining[remaining.index(op)+1:]:
                        completions.append(OrbitCompletion(
                            op=r, status=OpStatus.REJECTED,
                            error_msg="Aborted: OOM"
                        ))
                    break
                continue

            session.submit(op)
            results = session.run_all(stop_on_fault=(fault_policy == FaultPolicy.STOP))

            # Proto-B: additionally check DMA completion for GEMM ops
            if self._mode == "protob" and self._dma and results:
                last_r = results[-1]
                if last_r.status == OpStatus.COMPLETED and last_r.op.opcode == Opcode.GEMM:
                    dma_comp = self._dma.wait_done(max_polls=self._max_polls)
                    if not dma_comp.ok:
                        last_r = OrbitCompletion(
                            op=last_r.op, status=OpStatus.FAULTED,
                            error_msg=f"DMA failed: err={dma_comp.error_code} timeout={dma_comp.timed_out}",
                        )
                        results[-1] = last_r

            completions.extend(results)
            session.clear()

            # Check fault policy
            last = results[-1] if results else None
            if last and last.status == OpStatus.FAULTED:
                if fault_policy == FaultPolicy.STOP:
                    for r in remaining[remaining.index(op)+1:]:
                        completions.append(OrbitCompletion(
                            op=r, status=OpStatus.REJECTED,
                            error_msg="Aborted: prior fault"
                        ))
                    break
                elif fault_policy == FaultPolicy.ABORT:
                    break
                # SKIP: continue to next

        submitted = sum(1 for c in completions if c.status != OpStatus.REJECTED)
        completed = sum(1 for c in completions if c.status == OpStatus.COMPLETED)
        faulted   = sum(1 for c in completions if c.status == OpStatus.FAULTED)
        rejected  = sum(1 for c in completions if c.status == OpStatus.REJECTED)

        return SchedulerResult(
            ops_submitted=submitted,
            ops_completed=completed,
            ops_faulted=faulted,
            ops_rejected=rejected,
            oom_throttled=oom_throttled,
            completions=completions,
        )
