"""test_mxu_core_lanes.py — 공유 변환기 + 레인을 쓰기 폭에 맞춘 구성.

`test_mxu_core_stream` 은 `RD_LANES=1` (타일당 변환기 하나) 을 돈다.
면적은 가장 작지만 비트 하나를 채우는 데 WPB 사이클이 걸린다.

여기서는 `RD_LANES=0` — "쓰기 비트 폭에 맞춘다" 는 뜻이고 WPB 개가
된다. 그러면 비트가 한 사이클에 완성되므로 비트당 사이클 손실이
0 이다. 남는 비용은 겹치기 상실뿐이다 (`rd_hold`, 디스크립터당
OUT_BEATS 사이클).

측정 (DIM=16):

    Kt=64   (균형점)      71.0 →  137.0 사이클/op   1.93배
    Kt=4096 (실제 레이어) 4103.0 → 4169.0 사이클/op  +1.6%

면적은 타일당 677,987 → 500,281 셀 (26.2% 절감, WR=128 의 4 레인).

두 레인 수를 다 회귀에 두는 이유: 레인 경로의 인덱싱이 다르다.
`RD_LANES=1` 은 열 전체를 먹스로 고르고, WPB 레인은 열 **그룹**을
고른 뒤 레인마다 고정 오프셋을 더한다. 한쪽만 돌면 다른 쪽 인덱싱이
틀려도 모른다.
"""
from test_mxu_core import (  # noqa: F401
    test_bf16_matches_reference,
    test_weight_stationary,
    test_request_splitting,
    test_streamed_throughput,
)
