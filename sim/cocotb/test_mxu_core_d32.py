"""test_mxu_core_d32.py — test_mxu_core.py 를 DIM=32 구성에서 돈다.

run_all.py 의 VARIANTS 는 모듈 이름당 구성 하나만 받는다. DIM 은
K-step 폭(DIM*16)과 타일 수(DIM/16)를 동시에 바꾸므로 두 구성 다
회귀에 있어야 한다. 테스트 본문은 DUT 에서 구성을 읽는다.

DIM=16 은 타일 1개 (MXU 의 최소 구성), DIM=32 는 타일 4개로
`PAR`/`TILES` 분기를 지난다.
"""
from test_mxu_core import (  # noqa: F401
    test_bf16_matches_reference,
    test_weight_stationary,
    test_request_splitting,
    test_streamed_throughput,
)
