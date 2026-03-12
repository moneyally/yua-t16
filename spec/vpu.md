# VPU Spec v1.1
## Vector Processing Unit
## ✅ 구현 완료 — RTL + cocotb 테스트 8/8 PASS (2026-03-13)

## 0. Purpose and Identity

The VPU (Vector Processing Unit) is a fixed-function SIMD processing unit
designed for elementwise and reduction operations required in LLM inference.

It handles all non-GEMM operations in the GPT-OSS-20B forward pass:

- RMSNorm (pre-attention and pre-FFN layer normalization)
- SiLU (sigmoid linear unit, FFN activation)
- RoPE (rotary positional embedding)
- Softmax (attention probability normalization)
- Residual Add (skip connection accumulation)
- Scale (attention score scaling by 1/sqrt(d_head))

The VPU is a peer unit to YUA-T16 inside an ORBIT-G1 compute cluster.
It is not a CPU and does not execute general-purpose code.

VPU prioritizes:
- Deterministic, cycle-exact execution
- Numerical fidelity for FP16/BF16 inference
- Low-latency elementwise operations (pipeline depth <= 8 cycles)
- Simple descriptor-driven control, compatible with ORBIT-G1 descriptor system

---

## 1. Scope Definition

### 1.1 In Scope

- 256-wide SIMD execution (256 elements per clock cycle)
- FP16 and BF16 arithmetic
- Operations: RMSNorm, SiLU, RoPE, Softmax, Residual Add, Scale
- LUT-based sigmoid and exp (for SiLU and Softmax)
- VECTOR_OP_EX descriptor (type 0x0D)
- On-chip scratchpad for intermediate reduction values
- GDDR6 memory interface (read and write)

### 1.2 Out of Scope

- INT8 elementwise operations (handled by v1 VECTOR_OP, type 0x03)
- GEMM or matrix-matrix multiply (handled by YUA-T16)
- KV-cache management (handled by KVC)
- MoE routing (handled by MOE_ROUTE descriptor, separate unit)
- General-purpose vector ISA
- Floating-point transcendental functions beyond sigmoid/exp

---

## 2. Data Types

The VPU operates on FP16 and BF16 data. Internal reduction accumulators
use FP32 to prevent catastrophic cancellation in sum operations.

| Format | Exponent | Mantissa | Total | Notes                      |
|--------|----------|----------|-------|----------------------------|
| FP16   | 5 bits   | 10 bits  | 16    | Default inference format   |
| BF16   | 8 bits   | 7 bits   | 16    | Broader dynamic range      |
| FP32   | 8 bits   | 23 bits  | 32    | Internal accumulator only  |

Data type is specified per-descriptor in the data_type field.

---

## 3. Architectural Overview

The VPU contains the following functional blocks:

```
  ┌────────────────────────────────────────────────────────────────────┐
  │                          VPU                                        │
  │                                                                      │
  │  ┌─────────────┐   ┌──────────────┐   ┌───────────────────────────┐│
  │  │ DESC DECODE │──▶│ VECTOR FETCH │──▶│     SIMD LANE ARRAY       ││
  │  │ (op_type,   │   │ (GDDR6 read  │   │   256 lanes × FP16/BF16   ││
  │  │  data_type, │   │  256 elem/   │   │                           ││
  │  │  vec_len,   │   │  cycle burst)│   │  ┌──────┐  ┌──────────┐  ││
  │  │  params)    │   └──────────────┘   │  │ ADD  │  │  MUL     │  ││
  │  └─────────────┘                      │  │ SUB  │  │  DIV     │  ││
  │                                        │  │ MAX  │  │  RSQRT   │  ││
  │  ┌─────────────┐                      │  └──────┘  └──────────┘  ││
  │  │ REDUCTION   │◀──────────────────── │  ┌──────────────────────┐ ││
  │  │ SCRATCHPAD  │                      │  │  LUT: sigmoid, exp   │ ││
  │  │ (FP32 acc)  │                      │  │  (256-entry, 16-bit) │ ││
  │  │ 256×FP32    │                      │  └──────────────────────┘ ││
  │  └──────┬──────┘                      └───────────────────────────┘│
  │         │                                           │               │
  │         ▼                              ┌────────────▼─────────────┐│
  │  ┌─────────────┐                      │    VECTOR WRITEBACK       ││
  │  │ BROADCAST   │                      │    (GDDR6 write,          ││
  │  │ UNIT        │                      │     256 elem/cycle burst) ││
  │  │ (rsqrt,     │                      └──────────────────────────┘│
  │  │  scale out) │                                                    │
  │  └─────────────┘                                                    │
  └────────────────────────────────────────────────────────────────────┘
```

All data paths are 256 elements wide. Vectors longer than 256 elements
are processed in 256-element chunks. Partial last chunks are zero-padded.

---

## 4. Supported Operations

### 4.1 Operation Table

| op_type | Mnemonic   | Description                       | Passes |
|---------|------------|-----------------------------------|--------|
| 0x00    | ELEM_ADD   | C[i] = A[i] + B[i]               | 1      |
| 0x01    | ELEM_MUL   | C[i] = A[i] * B[i]               | 1      |
| 0x02    | SCALE      | C[i] = A[i] * imm_scale           | 1      |
| 0x03    | RESIDUAL   | C[i] = A[i] + B[i]  (alias 0x00) | 1      |
| 0x04    | RMSNORM    | C[i] = A[i] * w[i] * rsqrt(mean_sq + eps) | 3 |
| 0x05    | SILU       | C[i] = A[i] * sigmoid(A[i])       | 2      |
| 0x06    | ROPE       | In-place RoPE on Q or K tensor    | 2      |
| 0x07    | SOFTMAX    | C[i] = exp(A[i]-max) / sum(exp)   | 3      |
| 0x08    | CLAMP      | C[i] = clamp(A[i], lo, hi)        | 1      |
| 0x09    | GELU_APPROX| C[i] ≈ A[i] * sigmoid(1.702*A[i]) | 2      |

"Passes" indicates the number of memory passes over the input vector
required to compute the operation.

### 4.2 RMSNorm Detail

RMSNorm computes: C[i] = A[i] * weight[i] / sqrt( mean(A^2) + eps )

Pass 1: Read A, compute A[i]^2, sum across all elements → sum_sq (FP32 accumulator)
Pass 2: Compute rsqrt(sum_sq / N + eps) → scalar r
Pass 3: Read A and weight vector, write C[i] = A[i] * weight[i] * r

Inputs:
- src_addr: A vector (FP16/BF16, length vec_len)
- aux_addr: weight vector w (FP16/BF16, length vec_len)
- dst_addr: output C (FP16/BF16, length vec_len)
- imm_fp16[0]: epsilon (eps), encoded as FP16 immediate

### 4.3 SiLU Detail

SiLU computes: C[i] = A[i] * sigmoid(A[i])

The sigmoid is implemented using a 256-entry LUT indexed by the top 8 bits
of the FP16 input value. LUT entries are FP16. Linear interpolation between
adjacent LUT entries is performed using the lower 2 bits of the mantissa.

Pass 1: Read A, apply LUT sigmoid → sigmoid_A (stored in lane registers)
Pass 2: Multiply A * sigmoid_A, write C

The LUT covers the range [-8.0, 8.0]. Values outside this range are clamped
to 0.0 (negative saturation) or 1.0 (positive saturation) before LUT lookup.

### 4.4 RoPE Detail

RoPE applies rotary positional embeddings in-place to a head dimension slice.

For head dimension d_head, positions 2i and 2i+1:

```
  out[2i]   = x[2i]   * cos(theta[i]) - x[2i+1] * sin(theta[i])
  out[2i+1] = x[2i+1] * cos(theta[i]) + x[2i]   * sin(theta[i])
```

The cos and sin tables are precomputed by the host and passed via aux_addr.

Pass 1: Read x[even] and x[odd] interleaved, read cos/sin from aux_addr,
        apply rotation, write output.

Due to the paired-element structure, RoPE processes pairs (2 elements per
lane pair). Effective throughput is 128 rotations per clock cycle.

Inputs:
- src_addr: input tensor slice (shape: [num_heads * d_head] or subset)
- aux_addr: precomputed cos/sin table (FP16, shape: [d_head/2 * 2])
- dst_addr: output (same shape as src)
- vec_len: total elements (must be even)

### 4.5 Softmax Detail

Softmax computes: C[i] = exp(A[i] - max_val) / sum(exp(A - max_val))

Three passes for numerical stability (online softmax):

Pass 1: Read A, compute max_val (reduction over all elements)
Pass 2: Read A, compute exp(A[i] - max_val) using LUT, accumulate sum_exp
Pass 3: Read exp results (buffered in scratchpad or re-computed),
        divide by sum_exp, write C

The exp LUT is 256 entries, FP16, covering the range [-8.0, 0.0]
(post-subtraction input is always <= 0). Values < -8.0 produce exp ≈ 0
(LUT clamps to minimum representable FP16 positive).

Scratchpad: max_val and sum_exp are stored in the 256-element FP32
reduction scratchpad as scalars (indices 0 and 1 respectively).

### 4.6 Scale Detail

Scale multiplies a vector by a scalar immediate:

```
  C[i] = A[i] * scale
```

Used for attention score scaling by 1/sqrt(d_head). The scale value is
encoded in the VECTOR_OP_EX descriptor's imm_fp16[0] field.

---

## 5. LUT Architecture

### 5.1 LUT Tables

Two LUT tables are instantiated on-chip:

| Table      | Function  | Range        | Entries | Width | Total    |
|------------|-----------|--------------|---------|-------|----------|
| LUT_SIGMOID| sigmoid   | [-8.0, +8.0] | 256     | FP16  | 512 B    |
| LUT_EXP    | exp       | [-8.0, 0.0]  | 256     | FP16  | 512 B    |

Each table is implemented as a synchronous SRAM with 1-cycle read latency.

### 5.2 LUT Indexing

For LUT_SIGMOID:

```
  Input x (FP16): clamp to [-8.0, 8.0]
  Normalized:     n = (x + 8.0) / 16.0            → [0.0, 1.0]
  Index:          idx = floor(n * 255)             → [0, 255]
  Frac:           frac = n * 255 - idx             → [0.0, 1.0)
  Output:         sigmoid(x) ≈ LUT[idx] + frac * (LUT[idx+1] - LUT[idx])
```

For LUT_EXP:

```
  Input x (FP16): clamp to [-8.0, 0.0]
  Normalized:     n = (x + 8.0) / 8.0             → [0.0, 1.0]
  Index:          idx = floor(n * 255)             → [0, 255]
  Frac:           frac = n * 255 - idx             → [0.0, 1.0)
  Output:         exp(x) ≈ LUT[idx] + frac * (LUT[idx+1] - LUT[idx])
```

Linear interpolation uses FP16 multiply-add in each SIMD lane.

### 5.3 LUT Initialization

LUT tables are loaded from GDDR6 at power-on via the VPU_LUT_LOAD
configuration sequence (MMIO write to VPU_CTRL.LUT_LOAD bit, with
LUT_SIGMOID_ADDR and LUT_EXP_ADDR registers pointing to precomputed tables
in GDDR6). LUT content is constant during inference; no runtime updates.

---

## 6. VECTOR_OP_EX Descriptor (Type 0x0D)

The VECTOR_OP_EX descriptor extends the v1 VECTOR_OP (type 0x03) for
FP16/BF16 LLM-specific operations. It is 64 bytes.

```c
struct orbit_desc_vector_op_ex {
  orbit_desc_header h;     // type=0x0D, flags, reserved0, length=vec_len,
                           //   next_desc

  uint64_t src_addr;       // Source vector A (FP16/BF16, vec_len elements)
  uint64_t dst_addr;       // Destination vector C (FP16/BF16, vec_len elems)

  uint64_t aux_addr;       // Auxiliary input (weight vec for RMSNorm,
                           //   cos/sin table for RoPE, second operand for
                           //   ELEM_ADD/MUL; 0 if not used)

  uint32_t vec_len;        // Number of elements to process
  uint8_t  op_type;        // Operation (see table in Section 4.1)
  uint8_t  data_type;      // 0x00=FP16, 0x01=BF16
  uint16_t flags;          // [0]=in_place (dst=src), [15:1]=reserved

  uint16_t imm_fp16[4];    // FP16 immediate values (op-specific use):
                           //   RMSNORM: imm_fp16[0] = epsilon
                           //   SCALE:   imm_fp16[0] = scale factor
                           //   CLAMP:   imm_fp16[0] = lo, imm_fp16[1] = hi
                           //   others:  reserved
};
```

Total: 8 + 8 + 8 + 8 + 4 + 1 + 1 + 2 + 8 = 48 bytes.
Remaining 16 bytes are reserved (must be zero), filling to 64 bytes.

### 6.1 Descriptor Header Fields

- type: 0x0D
- flags: [0] = debug_trace enable; [1] = perf_count enable
- length: same as vec_len (redundant, must match)
- next_desc: chained descriptor pointer (0 = end of chain)

### 6.2 Backward Compatibility

VECTOR_OP (type 0x03) continues to handle INT8/INT16 elementwise ops as
defined in descriptor.md. VECTOR_OP_EX (type 0x0D) is the FP16/BF16 path.
Hardware dispatches to the appropriate execution unit based on type.

---

## 7. Register Map

### 7.1 VPU Configuration Registers (MMIO)

| Offset | Name             | Width | Description                         |
|--------|------------------|-------|-------------------------------------|
| 0x000  | VPU_CTRL         | 32    | [0]=start, [1]=busy, [2]=done,      |
|        |                  |       | [3]=lut_load, [4]=error             |
| 0x004  | VPU_STATUS       | 32    | [7:0]=op_type, [8]=pass_num,        |
|        |                  |       | [15:9]=reserved, [31:16]=err_code   |
| 0x008  | VPU_VEC_LEN      | 32    | Current vector length               |
| 0x00C  | VPU_OP_TYPE      | 8     | Current operation                   |
| 0x010  | VPU_LUT_SIG_ADDR | 64    | GDDR6 address for sigmoid LUT       |
| 0x018  | VPU_LUT_EXP_ADDR | 64    | GDDR6 address for exp LUT           |
| 0x020  | VPU_PERF_CYC     | 32    | Cycle counter (per op)              |
| 0x024  | VPU_PERF_PASSES  | 32    | Memory pass counter                 |
| 0x028  | VPU_PERF_ELEMS   | 32    | Elements processed counter          |
| 0x02C  | VPU_SCRATCH_CTRL | 32    | Reduction scratchpad control        |

---

## 8. Performance Model

### 8.1 Throughput

VPU processes 256 FP16/BF16 elements per clock cycle per pass.

For a vector of length L:

```
  chunks = ceil(L / 256)
  cycles_per_pass ≈ chunks + pipeline_fill_cycles
  pipeline_fill_cycles = 4 (fetch + compute + writeback stages)
```

| Operation   | Passes | Approx cycles (L=4096)      |
|-------------|--------|-----------------------------|
| ELEM_ADD    | 1      | 16 + 4  ≈ 20                |
| RESIDUAL    | 1      | 16 + 4  ≈ 20                |
| SCALE       | 1      | 16 + 4  ≈ 20                |
| SILU        | 2      | 32 + 8  ≈ 40                |
| ROPE        | 2      | 32 + 8  ≈ 40                |
| RMSNORM     | 3      | 48 + 12 ≈ 60                |
| SOFTMAX     | 3      | 48 + 12 ≈ 60                |

Memory bandwidth assumption: GDDR6 at 512 GB/s (ORBIT-G1 target).
Each FP16 element is 2 bytes. Bandwidth per pass for L=4096:
  8192 bytes read + 8192 bytes write ≈ 16 KB per pass.

### 8.2 GDDR6 Interface Requirement

VPU issues 256-element (512-byte) burst reads and writes aligned to 64-byte
boundaries. Maximum outstanding requests: 4 (to hide GDDR6 latency).

---

## 9. Timing Diagram

### 9.1 Single-Pass Operation (e.g., ELEM_ADD, L=512)

```
Cycle:  1   2   3   4   5   6   7   8   9
        |   |   |   |   |   |   |   |   |
FETCH:  [0:255] [256:511]  .   .
EXEC:    .  [0:255] [256:511]  .
WB:      .   .  [0:255] [256:511]  .
DONE:    .   .   .   .   .  _/‾‾‾
```

### 9.2 Three-Pass Operation (e.g., SOFTMAX, L=512)

```
Phase   Pass 1 (max)    Pass 2 (exp+sum)  Pass 3 (normalize)
        ─────────────   ─────────────     ──────────────
FETCH:  [0:255][256:]   [0:255][256:]     [0:255][256:]
REDUCE: → max_val       → sum_exp         → divide+write
        (FP32 acc)      (FP32 acc)
```

Pass boundary: hardware stalls 1 cycle to broadcast reduction result
(max_val or sum_exp) from scratchpad to all 256 SIMD lanes before Pass 2/3.

---

## 10. Design Constraints

- Target FPGA clock: 150 MHz (safe), 200 MHz (stretch)
- Target ASIC clock: 50–100 MHz
- LUT SRAM: 2 x 512 bytes, 1-cycle read latency
- Reduction scratchpad: 256 FP32 registers (1 KB) in flip-flop array
- SIMD lane FP16 multiply-add: single-cycle latency with full pipelining
- Maximum vector length: 2^20 elements (1M) per descriptor
- Minimum vector length: 1 element (no constraint)
- vec_len must be a multiple of the SIMD width (256) for optimal throughput;
  non-multiples are handled by masking the final chunk

---

## 11. Verification Strategy

- Unit test each op_type independently (golden model in Python/NumPy)
- LUT accuracy test: compare sigmoid/exp LUT output vs math.h reference
  over full input range; require max absolute error < 2^-7 (FP16 ULP * 4)
- RMSNorm: test with known pathological inputs (zero vector, uniform vector)
- Softmax: verify numerical stability with large logit differences (>100)
- RoPE: compare rotation output against PyTorch reference implementation
- Multi-chunk test: vectors at 255, 256, 257, 512, 4096 element lengths
- Back-to-back ops: interleave RMSNORM and SILU without barrier; verify
  correctness (descriptors are serialized within VPU queue)
- BF16 mode: repeat all tests with data_type=BF16
- Performance: measure cycles-per-element for each op at L=4096;
  require within 10% of model in Section 8.1

---

## 12. Non-Goals

The VPU does not:

- Execute GEMM (use YUA-T16)
- Manage KV-cache (use KVC)
- Perform MoE routing (use MOE_ROUTE descriptor)
- Support integer arithmetic in VECTOR_OP_EX (use VECTOR_OP type 0x03)
- Expose a general-purpose vector ISA
- Support variable-width SIMD (always 256-wide)
- Perform training operations (no gradient computation)

---

## 13. Role in System

The VPU is the elementwise complement to YUA-T16 within each ORBIT-G1
compute cluster. A complete Transformer layer forward pass requires
alternating GEMM_INT8/INT4 descriptors (YUA-T16) with VECTOR_OP_EX
descriptors (VPU) for normalization, activation, and positional encoding.

The VPU consumes and produces data in GDDR6 memory, using the same physical
address space and memory controller as YUA-T16 and KVC.
