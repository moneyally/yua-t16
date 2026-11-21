"""test_tile16.py — test_tile32.py 를 기본 구성(N=16)으로 한 번 더 돈다.

run_all.py 의 VARIANTS 는 테스트 모듈 이름당 구성 하나만 받는다.
test_tile32 는 N=32/256비트로 등록되어 있으므로, 파라미터화가 기존
N=16/128비트 동작을 바꾸지 않았는지 확인할 자리가 따로 필요하다.
테스트 본문은 DUT 에서 구성을 직접 읽으므로 재사용으로 충분하다.
"""
from test_tile32 import (  # noqa: F401
    test_result_matches_reference,
    test_streamed_throughput,
)
