"""test_mxu_cluster_c4.py — BF16 클러스터, 코어 4개.

코어 수를 바꾸면 깨지는 것들이 따로 있다. INT8 쪽에서 완료 집계
버그는 CORES=2 에서 8번째 디스크립터에, CORES=4 에서는 **첫
디스크립터**에서 드러났다. 그래서 두 코어 수를 다 돈다.

`test_mxu_cluster` 의 테스트를 그대로 돌린다 — 같은 단언, 다른 빌드.
"""
from test_mxu_cluster import (  # noqa: F401
    test_each_core_computes_its_own_slice,
    test_activation_is_read_once,
    test_streamed_descriptors,
    test_throughput_scales,
)
