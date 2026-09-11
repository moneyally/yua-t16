"""orbit_pack.py — Q1.15 벡터/상태의 평탄화 규칙 **단일 출처**

SSOT: `spec/deltarule.md` 3.5절.

규칙은 하나다:

    원소 i  =  bits[i*W +: W]        (i=0 이 최하위)

RTL 의 `q_flat` / `k_flat` / `v_flat` / `rd_data`, cocotb 테스트벤치,
`CocotbDut` 이 **모두 이 함수만** 쓴다. 평탄화 규칙이 두 곳에 있으면
한쪽을 고칠 때 다른 쪽이 조용히 틀린다 — `docs/BUGS.md` BUG-006 과 같은 종류다.

Q1.15 는 **부호 있는** 16비트다. 평탄화할 때는 2의 보수 비트패턴으로 넣고,
풀 때 다시 부호를 복원한다.
"""

from __future__ import annotations

import numpy as np

W_DEFAULT = 16


def to_bits(value: int, w: int = W_DEFAULT) -> int:
    """부호 있는 정수 → w비트 2의 보수 비트패턴."""
    return int(value) & ((1 << w) - 1)


def from_bits(raw: int, w: int = W_DEFAULT) -> int:
    """w비트 2의 보수 비트패턴 → 부호 있는 정수."""
    raw &= (1 << w) - 1
    return raw - (1 << w) if raw >> (w - 1) else raw


def pack_vec(vec, w: int = W_DEFAULT) -> int:
    """Q1.15 벡터 → 평탄화된 정수. 원소 i 가 bits[i*w +: w] 에 놓인다."""
    out = 0
    for i, x in enumerate(np.asarray(vec).reshape(-1)):
        out |= to_bits(int(x), w) << (i * w)
    return out


def unpack_vec(flat: int, d: int, w: int = W_DEFAULT):
    """평탄화된 정수 → (d,) Q1.15 int64 배열."""
    flat = int(flat)
    mask = (1 << w) - 1
    return np.array(
        [from_bits((flat >> (i * w)) & mask, w) for i in range(d)],
        dtype=np.int64,
    )


def pack_state(S, w: int = W_DEFAULT) -> list[int]:
    """상태 S (d,d) → **행 우선** 평탄화 정수 리스트 (행마다 하나).

    행 r 의 값 = pack_vec(S[r]). `spec/deltarule.md` 3.5절의 DELTA_DUMP 순서다.
    """
    S = np.asarray(S)
    return [pack_vec(S[r], w) for r in range(S.shape[0])]


def unpack_state(rows, d: int, w: int = W_DEFAULT):
    """행 우선 평탄화 정수 리스트 → (d,d) Q1.15 int64 배열."""
    assert len(rows) == d, f"행 개수가 {d} 가 아니다: {len(rows)}"
    return np.stack([unpack_vec(r, d, w) for r in rows])


def bytes_of_vec(vec, w: int = W_DEFAULT) -> bytes:
    """Q1.15 벡터 → 리틀엔디언 바이트 (스크래치 SRAM 적재용).

    평탄화 규칙(원소 i 가 낮은 비트)과 리틀엔디언이 일치하므로,
    바이트 순서도 원소 순서와 같다.
    """
    assert w % 8 == 0, "w 는 8의 배수여야 바이트로 떨어진다"
    nb = w // 8
    out = bytearray()
    for x in np.asarray(vec).reshape(-1):
        out += to_bits(int(x), w).to_bytes(nb, "little")
    return bytes(out)


def vec_of_bytes(buf: bytes, d: int, w: int = W_DEFAULT):
    """바이트 → Q1.15 벡터. `bytes_of_vec` 의 역."""
    nb = w // 8
    assert len(buf) == d * nb, f"길이가 {d * nb} 바이트여야 한다: {len(buf)}"
    return np.array(
        [from_bits(int.from_bytes(buf[i * nb:(i + 1) * nb], "little"), w) for i in range(d)],
        dtype=np.int64,
    )
