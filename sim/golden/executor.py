import struct
import numpy as np

from .memory import Memory
from .ops import gemm_int8, dma_2d
from .pack import unpack_header, unpack_gemm_int8, DESC_GEMM_INT8

DESC_DMA_2D   = 0x01
DESC_GEMM_INT8 = 0x02
DESC_BARRIER  = 0x07


class Executor:
    def __init__(self):
        self.mem = Memory()

    def run(self, desc: dict):
        dtype = desc["type"]

        if dtype == DESC_DMA_2D:
            self._run_dma(desc)

        elif dtype == DESC_GEMM_INT8:
            self._run_gemm(desc)

        elif dtype == DESC_BARRIER:
            pass  # explicit no-op in golden

        else:
            raise RuntimeError(f"Unknown descriptor type {dtype}")

    def _run_dma(self, d):
        dma_2d(
            self.mem,
            d["src_addr"],
            d["dst_addr"],
            d["width_bytes"],
            d["height"],
            d["src_stride"],
            d["dst_stride"],
        )

    def _run_gemm(self, d):
        Kt = d["Kt"]

        act_raw = self.mem.read(d["act_addr"], 16 * Kt)
        wgt_raw = self.mem.read(d["wgt_addr"], Kt * 16)

        act = np.frombuffer(act_raw, dtype=np.int8).reshape(16, Kt)
        wgt = np.frombuffer(wgt_raw, dtype=np.int8).reshape(Kt, 16)

        out = gemm_int8(act, wgt)

        self.mem.write(d["out_addr"], out.astype(np.int32).tobytes())

    def run_blob(self, blob: bytes):
        desc_type, flags, _, length, next_desc = unpack_header(blob)

        if desc_type == DESC_GEMM_INT8:
            fields = unpack_gemm_int8(blob)
            self._run_gemm(fields)

        else:
            raise RuntimeError(f"Unsupported descriptor type {desc_type}")
