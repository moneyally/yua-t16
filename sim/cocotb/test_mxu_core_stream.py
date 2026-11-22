"""test_mxu_core_stream.py — 공유 변환기 순차 읽기를 쓰는 mxu_core.

`STREAM_RD=1` 은 재 보고 **기본값으로 쓰지 않기로 한** 경로다
(DIM=16 에서 71.0 → 392.0 사이클/op, 자세한 근거는
`rtl/mxu_core.sv` 의 STREAM_RD 주석과
`spec/memory-bandwidth-analysis.md` 16단계).

그래도 회귀에 두는 이유는 두 가지다.

  1) 안 쓰는 경로는 썩는다. 파라미터를 남겨 뒀으니 elaborate 되고
     결과가 맞는 상태로 남아 있어야 한다. 대안 설계점(타일당 변환기
     WPB 개)의 기준선이 이 경로다.
  2) 느릴 뿐 **결과는 같아야 한다.** 읽기 방식을 바꿨을 뿐이므로
     기준값 비교가 그대로 통과해야 한다. 통과하지 않으면 순차 읽기
     인덱스 매핑이 깨진 것이다.

`test_mxu_core` 의 테스트를 그대로 돌린다 — 같은 단언, 다른 빌드.
"""
from test_mxu_core import (  # noqa: F401
    test_bf16_matches_reference,
    test_weight_stationary,
    test_request_splitting,
    test_streamed_throughput,
)
