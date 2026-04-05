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
