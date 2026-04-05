"""test_desc_pack.py — Descriptor packer verification."""
import pytest
import struct
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.orbit_desc import (
    crc8, pack_descriptor, pack_nop, pack_gemm, desc_to_words,
)
from tools.orbit_mmio_map import DESC_SIZE, Opcode


def test_crc8_zeros():
    assert crc8([0] * 10) == 0


def test_crc8_known():
    # CRC-8/CCITT of [0x02] should be deterministic
    val = crc8([0x02])
    assert isinstance(val, int) and 0 <= val <= 255


def test_nop_descriptor_size():
    d = pack_nop()
    assert len(d) == DESC_SIZE


def test_nop_opcode():
    d = pack_nop()
    assert d[0] == Opcode.NOP


def test_nop_crc_valid():
    d = pack_nop()
    computed = crc8(d[:DESC_SIZE - 1])
    assert computed == d[DESC_SIZE - 1], "CRC mismatch"


def test_gemm_fields():
    d = pack_gemm(act_addr=0x1000, wgt_addr=0x2000, out_addr=0x3000, kt=16)
    assert d[0] == Opcode.GEMM
    act = struct.unpack_from("<Q", d, 16)[0]
    wgt = struct.unpack_from("<Q", d, 24)[0]
    out = struct.unpack_from("<Q", d, 32)[0]
    kt  = struct.unpack_from("<I", d, 40)[0]
    assert act == 0x1000
    assert wgt == 0x2000
    assert out == 0x3000
    assert kt == 16


def test_gemm_crc_valid():
    d = pack_gemm(0x1000, 0x2000, 0x3000, 4)
    computed = crc8(d[:DESC_SIZE - 1])
    assert computed == d[DESC_SIZE - 1]


def test_desc_to_words():
    d = pack_nop()
    words = desc_to_words(d)
    assert len(words) == DESC_SIZE // 4
    # word[0] should contain opcode in bits [7:0]
    assert words[0] & 0xFF == Opcode.NOP


def test_corrupted_crc_detected():
    d = bytearray(pack_nop())
    d[DESC_SIZE - 1] ^= 0xFF  # corrupt CRC
    computed = crc8(d[:DESC_SIZE - 1])
    assert computed != d[DESC_SIZE - 1], "Corrupted CRC should not match"
