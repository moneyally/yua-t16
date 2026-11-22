#!/usr/bin/env python3
"""gemm_tiler.py — 실제 (M, K, N) 행렬곱을 디스크립터 수열로 쪼갠다.

RTL 의 weight-stationary 는 **호스트가 그 순서로 내보낼 때만** 값어치가
있다. 가중치 밴드를 고정한 채 출력 행 타일을 돌면 가중치를 한 번만
읽는다. 순서를 거꾸로 하면 같은 하드웨어로 트래픽이 두 배가 된다.

이 모듈은 그 순서를 만들고, 절감량을 숫자로 낸다.

    from tools.gemm_tiler import tile_gemm, GemmShape, TileConfig
    plan = tile_gemm(GemmShape(M=64, K=512, N=128),
                     TileConfig(tile_n=32, cores=2, max_kt=512))
    plan.report()

## 설계 제약 — 왜 전략이 두 개인가

누산기는 T×T 한 벌이다. 그래서 두 가지를 동시에 쓸 수 없다.

* **K ≤ max_kt** — 한 디스크립터가 축소 차원 전체를 처리한다.
  루프를 `for jb: for ib:` 로 두면 가중치 밴드가 고정되므로 행 타일
  전체에 걸쳐 가중치를 재사용한다. **이쪽이 기본 경로다.**

* **K > max_kt** — 축소를 쪼개 누산해야 하고, 한 (ib, jb) 의 부분합이
  누산기에 살아 있어야 하므로 `kb` 가 가장 안쪽이어야 한다. 그러면
  `ib` 를 안쪽에 둘 수 없어 가중치 재사용이 사라진다.
  (DRAM 누산(read-modify-write)이 있으면 둘 다 가능하지만 RTL 에 없다.)

실제 레이어는 `max_kt` 를 K 이상으로 잡으면 첫 경로로 간다. K=4096,
T=32, 2코어면 SRAM 1MB 로 VC1902 온칩 20.5MB 안에 충분히 들어간다.

## M=1 (단일 토큰 디코드) 의 구조적 한계

A 는 T 행이다. M=1 이면 1/T 만 쓰고 나머지는 패딩이다. 디코드가
메모리에 묶이는 이유가 여기 있다 — 활성값에 재사용이 없다. 배치를
키워 M 을 T 이상으로 만들어야 배열이 찬다. `report()` 가 그 낭비를
같이 보여준다.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

# 디스크립터 flags. SSOT 는 tools/orbit_mmio_map.py 의 DescFlag 이고
# 그것이 rtl/ctrl_fsm.sv 의 FLAG_* 와 대조된다. 여기서 다시 정의하면
# 세 곳이 어긋날 수 있으므로 가져다 쓴다.
from tools.orbit_mmio_map import DescFlag

FLAG_WGT_KEEP = int(DescFlag.WGT_KEEP)
FLAG_WGT_INT4 = int(DescFlag.WGT_INT4)
FLAG_ACC_KEEP = int(DescFlag.ACC_KEEP)
FLAG_NO_WB    = int(DescFlag.NO_WB)


@dataclass(frozen=True)
class GemmShape:
    """C[M,N] = A[M,K] · B[K,N]"""
    M: int
    K: int
    N: int

    def __post_init__(self):
        for name in ("M", "K", "N"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")


@dataclass(frozen=True)
class TileConfig:
    tile_n: int = 32        # MAC 타일 한 변 (TILE_N)
    cores: int = 1          # 텐서 병렬 코어 수 (CORES)
    max_kt: int = 256       # 디스크립터당 축소 차원 상한 (MAX_KT)
    wgt_bits: int = 8       # 가중치 비트 (8 또는 4)
    weight_stationary: bool = True

    @property
    def cols_per_desc(self) -> int:
        """디스크립터 하나가 만드는 출력 열 수 (코어마다 T 열)."""
        return self.tile_n * self.cores


@dataclass(frozen=True)
class Desc:
    """디스크립터 하나. 주소는 타일 좌표에서 호스트가 채운다."""
    ib: int          # 출력 행 타일 인덱스
    jb: int          # 출력 열 밴드 인덱스
    kb: int          # 축소 조각 인덱스
    kt: int          # 이 조각의 축소 길이
    flags: int

    @property
    def loads_weights(self) -> bool:
        return not (self.flags & FLAG_WGT_KEEP)

    @property
    def writes_back(self) -> bool:
        return not (self.flags & FLAG_NO_WB)


@dataclass
class Plan:
    shape: GemmShape
    cfg: TileConfig
    descs: list[Desc] = field(default_factory=list)
    strategy: str = ""

    # ── 트래픽 ────────────────────────────────────────────────────
    @property
    def act_bytes(self) -> int:
        """활성값 읽기 바이트. 클러스터는 브로드캐스트라 코어 수와 무관."""
        return sum(d.kt * self.cfg.tile_n for d in self.descs)

    @property
    def wgt_bytes(self) -> int:
        """가중치 읽기 바이트. 적재하는 디스크립터만 센다."""
        per_core = self.cfg.wgt_bits / 8
        return int(sum(d.kt * self.cfg.tile_n * self.cfg.cores * per_core
                       for d in self.descs if d.loads_weights))

    @property
    def out_bytes(self) -> int:
        t = self.cfg.tile_n
        return sum(t * t * self.cfg.cores * 4
                   for d in self.descs if d.writes_back)

    @property
    def dram_bytes(self) -> int:
        return self.act_bytes + self.wgt_bytes + self.out_bytes

    # ── 연산 ──────────────────────────────────────────────────────
    @property
    def macs_issued(self) -> int:
        """하드웨어가 실제로 돌리는 MAC (패딩 포함)."""
        t = self.cfg.tile_n
        return sum(t * t * self.cfg.cores * d.kt for d in self.descs)

    @property
    def macs_useful(self) -> int:
        """수학적으로 필요한 MAC."""
        return self.shape.M * self.shape.K * self.shape.N

    @property
    def pad_efficiency(self) -> float:
        return self.macs_useful / self.macs_issued if self.macs_issued else 0.0

    @property
    def weight_loads(self) -> int:
        return sum(1 for d in self.descs if d.loads_weights)

    def issue(self, layout: MemLayout) -> list[dict]:
        """구체적인 디스크립터 인자 목록. 호스트와 시뮬레이션이 공유한다.

        각 항목을 그대로 tools.orbit_desc.pack_gemm(**item) 에 넘길 수 있다.
        """
        out = []
        k0 = {}      # (ib, jb) 별 누적 K 오프셋
        for d in self.descs:
            key = (d.ib, d.jb)
            off = k0.get(key, 0)
            out.append({
                "act_addr": layout.act_addr(d.ib, off),
                "wgt_addr": layout.wgt_addr(d.jb, off),
                "out_addr": layout.out_addr(d.ib, d.jb),
                "kt": d.kt,
                "flags": d.flags,
                "wgt_stride": layout.wgt_slice_bytes,
                "out_stride": layout.out_core_bytes,
            })
            k0[key] = off + d.kt
        return out

    def report(self) -> None:
        s, c = self.shape, self.cfg
        print(f"GEMM  M={s.M} K={s.K} N={s.N}   "
              f"타일 {c.tile_n}×{c.tile_n} × {c.cores}코어, "
              f"max_kt={c.max_kt}, 가중치 INT{c.wgt_bits}")
        print(f"전략: {self.strategy}")
        print(f"디스크립터 {len(self.descs):,}개 "
              f"(가중치 적재 {self.weight_loads:,}개)")
        print(f"DRAM 읽기/쓰기  활성값 {self.act_bytes/1e6:8.2f} MB"
              f"  가중치 {self.wgt_bytes/1e6:8.2f} MB"
              f"  출력 {self.out_bytes/1e6:8.2f} MB"
              f"  = {self.dram_bytes/1e6:8.2f} MB")
        print(f"MAC  필요 {self.macs_useful:,}  발행 {self.macs_issued:,}"
              f"  패딩 효율 {self.pad_efficiency*100:.1f}%")
        if self.pad_efficiency < 0.9:
            waste = 1 / self.pad_efficiency
            print(f"  ⚠ 패딩으로 {waste:.1f}배를 낭비한다. "
                  f"M={s.M} 이 타일 {c.tile_n} 보다 작으면 구조적으로 그렇다 "
                  f"— 배치를 키워야 한다.")


@dataclass(frozen=True)
class MemLayout:
    """디스크립터 주소를 계산한다. 호스트와 시뮬레이션이 **같은 코드**를 쓴다.

    이 레이아웃이 RTL 의 기대와 어긋나면 결과가 조용히 0 이 되거나 엉뚱한
    값이 나온다. `sim/cocotb/test_gemm_e2e.py` 가 이 클래스로 주소를 만들어
    실제 RTL 과 대조하므로, 거기서 깨지면 여기가 틀린 것이다.

    배치 규칙 (T = tile_n, C = cores):

        활성값   행 타일 ib 마다 A[ib*T:(ib+1)*T, :] 를 K-step 당 T 바이트.
                 타일 하나가 K*T 바이트. 조각 kb 는 그 안에서 k0*T 오프셋.

        가중치   열 밴드 jb 마다 코어 C 개의 조각이 연달아.
                 조각 하나가 K*T_w 바이트 (T_w = INT4 면 T/2).
                 밴드 하나가 C*K*T_w 바이트. 코어 간격이 wgt_stride.

        출력     (ib, jb) 마다 T × T*C, 즉 T*T*4*C 바이트.
                 코어 간격이 out_stride = T*T*4.
                 같은 (ib,jb) 의 모든 K 조각이 **같은 주소**에 쓴다
                 (마지막 조각만 실제로 쓴다).
    """
    shape: GemmShape
    cfg: TileConfig
    act_base: int
    wgt_base: int
    out_base: int

    @property
    def _t(self) -> int:
        return self.cfg.tile_n

    @property
    def _wgt_row_bytes(self) -> int:
        """가중치 K-step 하나의 바이트. INT4 면 절반이다."""
        return self._t * self.cfg.wgt_bits // 8

    @property
    def act_tile_bytes(self) -> int:
        return self.shape.K * self._t

    @property
    def wgt_slice_bytes(self) -> int:
        """코어 하나가 맡는 조각. 이것이 wgt_stride 다."""
        return self.shape.K * self._wgt_row_bytes

    @property
    def wgt_band_bytes(self) -> int:
        return self.wgt_slice_bytes * self.cfg.cores

    @property
    def out_core_bytes(self) -> int:
        """코어 하나의 출력. 이것이 out_stride 다."""
        return self._t * self._t * 4

    @property
    def out_desc_bytes(self) -> int:
        return self.out_core_bytes * self.cfg.cores

    def n_row_tiles(self) -> int:
        return -(-self.shape.M // self._t)

    def n_bands(self) -> int:
        return -(-self.shape.N // self.cfg.cols_per_desc)

    def act_addr(self, ib: int, k0: int) -> int:
        return self.act_base + self.act_tile_bytes * ib + k0 * self._t

    def wgt_addr(self, jb: int, k0: int) -> int:
        return (self.wgt_base + self.wgt_band_bytes * jb
                + k0 * self._wgt_row_bytes)

    def out_addr(self, ib: int, jb: int) -> int:
        return self.out_base + self.out_desc_bytes * (ib * self.n_bands() + jb)

    def total_bytes(self) -> dict:
        """호스트가 잡아야 할 버퍼 크기."""
        return {
            "act": self.act_tile_bytes * self.n_row_tiles(),
            "wgt": self.wgt_band_bytes * self.n_bands(),
            "out": self.out_desc_bytes * self.n_row_tiles() * self.n_bands(),
        }


def _k_chunks(K: int, max_kt: int) -> list[int]:
    out = []
    left = K
    while left > 0:
        n = min(max_kt, left)
        out.append(n)
        left -= n
    return out


def tile_gemm(shape: GemmShape, cfg: TileConfig) -> Plan:
    """디스크립터 수열을 만든다.

    K <= max_kt 면 가중치 밴드를 고정한 채 행 타일을 돌아 가중치를
    재사용한다. K > max_kt 면 축소를 쪼개 누산해야 하므로 kb 가 가장
    안쪽이 되고, 가중치 재사용은 포기한다 (모듈 설명 참조).
    """
    plan = Plan(shape=shape, cfg=cfg)
    n_ib = -(-shape.M // cfg.tile_n)
    n_jb = -(-shape.N // cfg.cols_per_desc)
    chunks = _k_chunks(shape.K, cfg.max_kt)

    if len(chunks) == 1:
        plan.strategy = ("K <= max_kt — 가중치 밴드 고정, 행 타일 순회 "
                         "(weight-stationary)")
        for jb in range(n_jb):
            for ib in range(n_ib):
                flags = 0
                # 밴드 안에서 두 번째부터는 상주 가중치를 쓴다
                if cfg.weight_stationary and ib > 0:
                    flags |= FLAG_WGT_KEEP
                if cfg.wgt_bits == 4:
                    flags |= FLAG_WGT_INT4
                plan.descs.append(Desc(ib, jb, 0, chunks[0], flags))
    else:
        plan.strategy = (f"K > max_kt — 축소를 {len(chunks)}조각으로 쪼개 "
                         f"누산 (가중치 재사용 불가: 누산기가 한 벌)")
        for ib in range(n_ib):
            for jb in range(n_jb):
                for kb, kt in enumerate(chunks):
                    flags = 0
                    if kb > 0:
                        flags |= FLAG_ACC_KEEP
                    if kb < len(chunks) - 1:
                        flags |= FLAG_NO_WB
                    if cfg.wgt_bits == 4:
                        flags |= FLAG_WGT_INT4
                    plan.descs.append(Desc(ib, jb, kb, kt, flags))
    return plan


def compare_orderings(shape: GemmShape, cfg: TileConfig) -> dict:
    """weight-stationary 순서와 그렇지 않은 순서의 트래픽을 비교한다."""
    ws = tile_gemm(shape, cfg)
    naive = tile_gemm(shape, TileConfig(**{**cfg.__dict__,
                                           "weight_stationary": False}))
    return {
        "ws_bytes": ws.dram_bytes,
        "naive_bytes": naive.dram_bytes,
        "saving": naive.dram_bytes / ws.dram_bytes if ws.dram_bytes else 1.0,
        "ws_weight_loads": ws.weight_loads,
        "naive_weight_loads": naive.weight_loads,
    }


# 실제 레이어 모양들. Llama-7B 계열 기준.
REAL_LAYERS = [
    ("7B attn qkv  (4096→12288)", 4096, 12288),
    ("7B attn out  (4096→4096)",  4096, 4096),
    ("7B ffn up    (4096→11008)", 4096, 11008),
    ("7B ffn down  (11008→4096)", 11008, 4096),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tile-n", type=int, default=32)
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--max-kt", type=int, default=4096)
    ap.add_argument("--wgt-bits", type=int, default=4, choices=(4, 8))
    ap.add_argument("--batch", type=int, nargs="*", default=[1, 32, 256])
    a = ap.parse_args()

    print("=" * 78)
    print(" GEMM 타일링 — 실제 레이어, weight-stationary 순서의 값어치")
    print("=" * 78)

    for M in a.batch:
        print(f"\n── 배치(M) = {M} " + "─" * 50)
        for name, K, N in REAL_LAYERS:
            cfg = TileConfig(tile_n=a.tile_n, cores=a.cores,
                             max_kt=a.max_kt, wgt_bits=a.wgt_bits)
            shape = GemmShape(M=M, K=K, N=N)
            cmp = compare_orderings(shape, cfg)
            plan = tile_gemm(shape, cfg)
            print(f"  {name:<28} "
                  f"{plan.dram_bytes/1e6:8.2f} MB  "
                  f"vs 순서 무시 {cmp['naive_bytes']/1e6:8.2f} MB  "
                  f"→ {cmp['saving']:.2f}×  "
                  f"패딩효율 {plan.pad_efficiency*100:5.1f}%")

    print(f"\n{'=' * 78}")
    print(" 자세히 — 7B ffn up, 배치 32")
    print("=" * 78)
    cfg = TileConfig(tile_n=a.tile_n, cores=a.cores, max_kt=a.max_kt,
                     wgt_bits=a.wgt_bits)
    tile_gemm(GemmShape(M=32, K=4096, N=11008), cfg).report()

    print(f"\n{'=' * 78}")
    print(" max_kt 가 K 보다 작으면 — 전략이 바뀌고 가중치 재사용이 사라진다")
    print("=" * 78)
    small = TileConfig(tile_n=a.tile_n, cores=a.cores, max_kt=256,
                       wgt_bits=a.wgt_bits)
    tile_gemm(GemmShape(M=32, K=4096, N=11008), small).report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
