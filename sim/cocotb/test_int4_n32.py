"""test_int4_n32.py — test_int4_weights.py 를 32×32 / 256비트 구성에서 돈다.

run_all.py 의 VARIANTS 는 모듈 이름당 구성 하나만 받는다. INT4 니블
슬라이싱은 N 과 읽기 폭에 모두 의존하므로(NIB_W = N*4, W4PW = RD/NIB_W)
두 구성 다 회귀에 있어야 한다. 테스트 본문은 DUT 에서 구성을 읽는다.
"""
from test_int4_weights import (  # noqa: F401
    test_int4_matches_reference,
    test_int4_halves_weight_traffic,
    test_int4_resident_across_descriptors,
)
