"""
orbit_desc.py — ORBIT-G2 Descriptor Packer

Packs descriptors for desc_fsm_v2 / ctrl_fsm consumption.
Format derived from RTL: desc_fsm_v2.sv, ctrl_fsm.sv.
"""
from __future__ import annotations
import struct
from tools.orbit_mmio_map import (
    DESC_SIZE, DESC_OPCODE_OFF, DESC_ACT_ADDR_OFF, DESC_WGT_ADDR_OFF,
    DESC_OUT_ADDR_OFF, DESC_KT_OFF, DESC_CRC_OFF, Opcode,
    DESC_STAGE_BASE, BASE, DESC_STAGE_WORDS,
    DR1_SLOT_OFF, DR1_Q_ADDR_OFF, DR1_K_ADDR_OFF, DR1_O_ADDR_OFF,
    DR1_V_ADDR_OFF, DR1_ALPHA_OFF, DR1_BETA_OFF,
    DR1_NUM_SLOTS, DR1_ADDR_ALIGN, UQ15_ONE,
)


def crc8(data: bytes | list[int]) -> int:
    """CRC-8 with polynomial 0x07 (matches RTL crc8_byte)."""
    crc = 0x00
    for byte in data:
        crc ^= byte & 0xFF
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def pack_descriptor(
    opcode: int,
    act_addr: int = 0,
    wgt_addr: int = 0,
    out_addr: int = 0,
    kt: int = 0,
) -> bytes:
    """Pack a DESC_SIZE-byte descriptor with CRC.

    Byte layout (RTL-derived):
      [0]      opcode
      [1:15]   reserved (zero)
      [16:23]  act_addr  (u64 LE)
      [24:31]  wgt_addr  (u64 LE)
      [32:39]  out_addr  (u64 LE)
      [40:43]  Kt        (u32 LE)
      [44:62]  reserved (zero)
      [63]     CRC-8 over bytes [0:62]
    """
    buf = bytearray(DESC_SIZE)
    buf[DESC_OPCODE_OFF] = opcode & 0xFF
    struct.pack_into("<Q", buf, DESC_ACT_ADDR_OFF, act_addr)
    struct.pack_into("<Q", buf, DESC_WGT_ADDR_OFF, wgt_addr)
    struct.pack_into("<Q", buf, DESC_OUT_ADDR_OFF, out_addr)
    struct.pack_into("<I", buf, DESC_KT_OFF, kt)
    buf[DESC_CRC_OFF] = crc8(buf[:DESC_CRC_OFF])
    return bytes(buf)


def pack_nop() -> bytes:
    """Pack a NOP descriptor."""
    return pack_descriptor(Opcode.NOP)


def pack_gemm(act_addr: int, wgt_addr: int, out_addr: int, kt: int) -> bytes:
    """Pack a GEMM descriptor."""
    return pack_descriptor(Opcode.GEMM, act_addr, wgt_addr, out_addr, kt)


def desc_to_words(desc: bytes) -> list[int]:
    """Convert DESC_SIZE bytes to DESC_STAGE_WORDS 32-bit words (LE)."""
    assert len(desc) == DESC_SIZE
    words = []
    for i in range(0, DESC_SIZE, 4):
        words.append(struct.unpack_from("<I", desc, i)[0])
    return words


def stage_and_doorbell(backend, desc: bytes, queue: int = 0):
    """Write descriptor to staging registers and kick doorbell.

    Args:
        backend: object with .write(addr, data) method
        desc: packed descriptor bytes
        queue: queue index (0-3)
    """
    words = desc_to_words(desc)
    stage_base_off = DESC_STAGE_BASE - BASE
    for i, w in enumerate(words):
        backend.write(stage_base_off + i * 4, w)
    doorbell_off = 0x0_2000 + queue * 4
    backend.write(doorbell_off, 0x0000_0001)


# ═══════════════════════════════════════════════════════════════════
# ORBIT-DR1 디스크립터 — SSOT: spec/deltarule.md 3절
#
# 기존 필드 위치를 재사용한다 (q=act@16, k=wgt@24, o=out@32). 새로 쓰는 것은
# slot@1, v_addr@44, alpha@52, beta@54 뿐이다. CRC 는 그대로 [63].
# ═══════════════════════════════════════════════════════════════════


class Dr1FieldError(ValueError):
    """DR1 디스크립터 필드가 spec/deltarule.md 4절의 실패 조건에 걸린다.

    호스트 쪽에서 먼저 잡으라고 두는 것이다. 하드웨어는 같은 조건을
    fault_code 0x05(DR1_BAD_SLOT) / 0x06(DR1_UNALIGNED) 로 보고한다.
    """


def _check_slot(slot: int) -> int:
    if not (0 <= slot < DR1_NUM_SLOTS):
        raise Dr1FieldError(
            f"slot={slot} 범위 초과 (0 <= slot < {DR1_NUM_SLOTS}). "
            f"하드웨어는 fault_code 0x05 DR1_BAD_SLOT 을 낸다."
        )
    return slot


def _check_align(name: str, addr: int) -> int:
    if addr % DR1_ADDR_ALIGN != 0:
        raise Dr1FieldError(
            f"{name}=0x{addr:X} 가 {DR1_ADDR_ALIGN}바이트 정렬이 아니다. "
            f"스크래치 SRAM 데이터 폭이 128비트라서 필요하다. "
            f"하드웨어는 fault_code 0x06 DR1_UNALIGNED 를 낸다."
        )
    return addr


def _check_uq15(name: str, value: int) -> int:
    """alpha/beta 는 UQ1.15 이고 1.0(0x8000) 초과는 미정의다.

    spec/deltarule.md 1절: 골든은 ValueError, RTL 은 0x8000 으로 클램프.
    호스트 패커도 골든과 같은 편에 선다 — 조용히 클램프하지 않는다.
    """
    if not (0 <= value <= UQ15_ONE):
        raise Dr1FieldError(
            f"{name}=0x{value:04X} 는 UQ1.15 의 정의된 범위 [0, 0x8000] 밖이다. "
            f"1.0 초과는 미정의 — RTL 은 0x8000 으로 클램프하고 CLAMP_EVENT 를 남긴다."
        )
    return value


def pack_delta_init(slot: int = 0) -> bytes:
    """DELTA_INIT (0x50): 상태 슬롯을 0 으로."""
    buf = bytearray(DESC_SIZE)
    buf[DESC_OPCODE_OFF] = int(Opcode.DELTA_INIT)
    buf[DR1_SLOT_OFF] = _check_slot(slot)
    buf[DESC_CRC_OFF] = crc8(buf[:DESC_CRC_OFF])
    return bytes(buf)


def pack_delta_step(
    q_addr: int,
    k_addr: int,
    v_addr: int,
    o_addr: int,
    alpha_uq15: int,
    beta_uq15: int,
    slot: int = 0,
) -> bytes:
    """DELTA_STEP (0x51): 토큰 1개.

    alpha_uq15 / beta_uq15 는 **UQ1.15** 다. 0x8000 = 1.0 (정확).
    """
    buf = bytearray(DESC_SIZE)
    buf[DESC_OPCODE_OFF] = int(Opcode.DELTA_STEP)
    buf[DR1_SLOT_OFF] = _check_slot(slot)
    struct.pack_into("<Q", buf, DR1_Q_ADDR_OFF, _check_align("q_addr", q_addr))
    struct.pack_into("<Q", buf, DR1_K_ADDR_OFF, _check_align("k_addr", k_addr))
    struct.pack_into("<Q", buf, DR1_O_ADDR_OFF, _check_align("o_addr", o_addr))
    struct.pack_into("<Q", buf, DR1_V_ADDR_OFF, _check_align("v_addr", v_addr))
    struct.pack_into("<H", buf, DR1_ALPHA_OFF, _check_uq15("alpha_uq15", alpha_uq15))
    struct.pack_into("<H", buf, DR1_BETA_OFF, _check_uq15("beta_uq15", beta_uq15))
    buf[DESC_CRC_OFF] = crc8(buf[:DESC_CRC_OFF])
    return bytes(buf)


def pack_delta_dump(dst_addr: int, slot: int = 0) -> bytes:
    """DELTA_DUMP (0x52): 상태 S 전체를 dst_addr 로."""
    buf = bytearray(DESC_SIZE)
    buf[DESC_OPCODE_OFF] = int(Opcode.DELTA_DUMP)
    buf[DR1_SLOT_OFF] = _check_slot(slot)
    struct.pack_into("<Q", buf, DR1_O_ADDR_OFF, _check_align("dst_addr", dst_addr))
    buf[DESC_CRC_OFF] = crc8(buf[:DESC_CRC_OFF])
    return bytes(buf)


def unpack_delta_step(desc: bytes) -> dict:
    """DELTA_STEP 디스크립터를 되읽는다. 패킹이 스펙대로인지 확인하는 용도."""
    assert len(desc) == DESC_SIZE
    return {
        "opcode": desc[DESC_OPCODE_OFF],
        "slot": desc[DR1_SLOT_OFF],
        "q_addr": struct.unpack_from("<Q", desc, DR1_Q_ADDR_OFF)[0],
        "k_addr": struct.unpack_from("<Q", desc, DR1_K_ADDR_OFF)[0],
        "o_addr": struct.unpack_from("<Q", desc, DR1_O_ADDR_OFF)[0],
        "v_addr": struct.unpack_from("<Q", desc, DR1_V_ADDR_OFF)[0],
        "alpha_uq15": struct.unpack_from("<H", desc, DR1_ALPHA_OFF)[0],
        "beta_uq15": struct.unpack_from("<H", desc, DR1_BETA_OFF)[0],
        "crc8": desc[DESC_CRC_OFF],
    }
