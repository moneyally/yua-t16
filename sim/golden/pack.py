import struct

# Descriptor type IDs
DESC_DMA_2D    = 0x01
DESC_GEMM_INT8 = 0x02
DESC_BARRIER   = 0x07

# Common header layout (little-endian)
# uint8  type
# uint8  flags
# uint16 reserved0
# uint32 length
# uint64 next_desc
HEADER_FMT = "<BBHIQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 16 bytes

DESC_SIZE = 64


def pack_header(desc_type, flags=0, length=0, next_desc=0):
    return struct.pack(
        HEADER_FMT,
        desc_type & 0xFF,
        flags & 0xFF,
        0,                 # reserved0
        length & 0xFFFFFFFF,
        next_desc & 0xFFFFFFFFFFFFFFFF
    )


def pack_gemm_int8(act_addr, wgt_addr, out_addr, Kt,
                   m_tiles=1, n_tiles=1,
                   scale_a=0, scale_b=0,
                   epilogue=0):
    """
    Packs a GEMM_INT8 descriptor into exactly 64 bytes.
    """
    header = pack_header(DESC_GEMM_INT8)

    body = struct.pack(
        "<QQQ I H H I I I I",
        act_addr & 0xFFFFFFFFFFFFFFFF,
        wgt_addr & 0xFFFFFFFFFFFFFFFF,
        out_addr & 0xFFFFFFFFFFFFFFFF,
        Kt & 0xFFFFFFFF,
        m_tiles & 0xFFFF,
        n_tiles & 0xFFFF,
        scale_a & 0xFFFFFFFF,
        scale_b & 0xFFFFFFFF,
        epilogue & 0xFFFFFFFF,
        0  # reserved
    )

    blob = header + body
    assert len(blob) <= DESC_SIZE, f"Descriptor overflow: {len(blob)} bytes"

    return blob.ljust(DESC_SIZE, b"\x00")


def unpack_header(blob: bytes):
    assert len(blob) >= HEADER_SIZE
    return struct.unpack(HEADER_FMT, blob[:HEADER_SIZE])


def unpack_gemm_int8(blob: bytes):
    """
    Unpacks GEMM_INT8 fields from a 64B descriptor blob.
    """
    fields = struct.unpack(
        "<QQQ I H H I I I I",
        blob[HEADER_SIZE:HEADER_SIZE + struct.calcsize("<QQQ I H H I I I I")]
    )

    return {
        "act_addr": fields[0],
        "wgt_addr": fields[1],
        "out_addr": fields[2],
        "Kt": fields[3],
        "m_tiles": fields[4],
        "n_tiles": fields[5],
        "scale_a": fields[6],
        "scale_b": fields[7],
        "epilogue": fields[8],
    }
