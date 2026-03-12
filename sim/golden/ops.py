import numpy as np

def gemm_int8(act: np.ndarray, wgt: np.ndarray) -> np.ndarray:
    """
    act: [16, Kt] int8
    wgt: [Kt, 16] int8
    returns: [16, 16] int32
    """
    return act.astype(np.int32) @ wgt.astype(np.int32)


def dma_2d(mem, src, dst, width, height, src_stride, dst_stride):
    for y in range(height):
        line = mem.read(src + y * src_stride, width)
        mem.write(dst + y * dst_stride, line)
