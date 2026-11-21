"""test_tile64.py — 64×64 타일. ASIC 급 설계점의 확장성 확인.

4,096 MAC 은 VC1902 의 DSP58 1,968 개를 2.1배 넘어 단일 FPGA 에 들어가지
않는다. 그래도 회귀에 두는 이유는 두 가지다.

  1) 파라미터화가 어디까지 성립하는지 숫자로 남긴다 (N=16/32/64).
  2) ASIC 설계점의 처리량·가동률을 측정값으로 가진다.

K-step 이 64바이트라 읽기 폭 512비트를 요구하고, 출력이 4,096개
(16,384바이트) 라 쓰기도 512비트여야 균형이 맞는다 (Kt=256).
"""
from test_tile32 import (  # noqa: F401
    test_result_matches_reference,
    test_streamed_throughput,
)
