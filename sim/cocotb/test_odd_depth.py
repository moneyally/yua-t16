"""test_odd_depth.py — MAX_KT 가 2의 거듭제곱이 아닐 때.

뱅크 주소를 `{bank, idx}` 로 이어 붙이면 BANK_DEPTH 가 2의 거듭제곱이어야
한다. 실제 LLM 레이어 깊이는 그렇지 않다 (K=11008).

    BANK_DEPTH=11008 → BADDR_W=14 → 뱅크1 = 16,384..27,391
    SRAM_DEPTH=22,016 → 범위 초과

**첫 디스크립터는 뱅크 0 만 쓰므로 통과하고, 두 번째부터 조용히 틀린다.**
K=11008 로 처음 돌렸을 때 정확히 그렇게 깨졌다 (#0 통과, #1 실패).

MAX_KT=11008 은 시뮬레이션이 느리다 (디스크립터당 3만 사이클). 여기서는
같은 성질을 가진 작은 값으로 빠르게 지킨다: MAX_KT=300 이면 BADDR_W=9,
2^9=512 > 300 이므로 이어 붙이기가 똑같이 깨진다.

test_real_layer_depth.py 가 실제 깊이(4096)를 맡고, 이 파일은 2의
거듭제곱이 아닌 성질만 빠르게 본다.
"""
from test_tile32 import (  # noqa: F401
    test_result_matches_reference,
    test_streamed_throughput,
)
