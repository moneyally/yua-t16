# YUA-T16 Spec v2

## 0. Purpose and Identity

YUA-T16 v2 is an extension of YUA-T16 v1.

It retains the complete v1 interface and behavioral contract (INT8 x INT8,
16x16 output tile, output-stationary dataflow, deterministic execution) and
adds the following capabilities:

- INT4 weight decompression pre-stage (INT4 → INT8 before MAC array)
- 32x32 tile execution mode (4x v1 tile throughput)
- AWQ (Activated Weight Quantization) dequantization via per-group scale
  registers
- Wider GEMM_INT4 descriptor (type 0x0E)

All v1 signals, register offsets, and descriptor formats remain valid.
v2-only features are gated by a MODE register.

YUA-T16 v2 is not a general-purpose processor.

---

## 1. Scope Definition

### 1.1 Additions to v1 Scope

- INT4 x INT8 signed multiplication (weight INT4, activation INT8)
- INT4 weight decompression: nibble unpack + per-group scale multiply
- Optional 32x32 tile mode (output-stationary, INT32 accumulation)
- AWQ dequantization: scale_a and scale_b per-group FP16 registers
- GEMM_INT4 descriptor (type 0x0E)
- Extended PERF counters: decompression stall cycles

### 1.2 Retained v1 Scope (unchanged)

- INT8 x INT8 signed multiplication
- INT32 accumulation
- 16x16 tile mode (default)
- Output-stationary dataflow
- MMIO-controlled execution
- Deterministic cycle behavior

### 1.3 Still Out of Scope

- Attention, softmax, KV-cache
- Activation functions (handled by VPU)
- Training or backpropagation
- INT2 or binary precision
- PCIe, cache coherence, virtual memory
- OS or framework integration

---

## 2. Mathematical Contract

### 2.1 INT8 Mode (v1-compatible)

Unchanged from v1:

```
C[i,j] = sum_{k=0..Kt-1} ( A[i,k] * B[k,j] )
```

Where A and B are INT8. Accumulation is INT32.

### 2.2 INT4 Weight Mode

INT4 weights are stored packed (two weights per byte, little-endian nibble
order). Before entering the MAC array, each INT4 weight is:

1. Unpacked from the packed byte
2. Sign-extended to INT8
3. Multiplied by the per-group scale to produce a dequantized INT8 value
   (scale applied as fixed-point multiply, result clamped to [-128, 127])

The effective computation is:

```
B_dq[k,j] = clamp( round( B_int4[k,j] * scale_b[group(k)] ), -128, 127 )
C[i,j]    = sum_{k=0..Kt-1} ( A[i,k] * B_dq[k,j] )
```

Where group(k) = floor(k / GROUP_SIZE), and GROUP_SIZE is configurable
(power of 2, default 128).

### 2.3 AWQ Scale Registers

AWQ introduces per-channel activation scaling applied before quantization.
In YUA-T16 v2, this is approximated as a pre-scale on the A tile:

```
A_scaled[i,k] = clamp( round( A[i,k] * scale_a[channel(i)] ), -128, 127 )
```

scale_a[] is stored as FP16 in a dedicated register bank (SCALE_A_BASE).
scale_b[] is stored as FP16 in a dedicated register bank (SCALE_B_BASE).

Scaling is bypassed in INT8 mode (scale registers are ignored).

### 2.4 32x32 Tile Mode

In 32x32 mode, the tile dimensions double on both axes:

```
A_tile: INT8 [32 x Kt]
B_tile: INT8 or INT4-packed [Kt x 32]
C_tile: INT32 [32 x 32]
```

The 32x32 tile is computed as four 16x16 sub-tiles executed sequentially
by the same MAC array, with accumulator state preserved between sub-tiles
within a single tile execution.

Sub-tile execution order:

```
  sub-tile (0,0): A[0:16, :] x B[:, 0:16]   → C[0:16, 0:16]
  sub-tile (0,1): A[0:16, :] x B[:, 16:32]  → C[0:16, 16:32]
  sub-tile (1,0): A[16:32,:] x B[:, 0:16]   → C[16:32, 0:16]
  sub-tile (1,1): A[16:32,:] x B[:, 16:32]  → C[16:32, 16:32]
```

Accumulator is cleared once per 32x32 tile, not per sub-tile.

### 2.5 Correctness Requirement

INT8 mode: bit-exact equivalence with v1 NumPy reference.

INT4 mode: bit-exact equivalence with the following NumPy reference:

```python
# INT4 dequantization reference
def dequant_int4(W_packed, scale_b, group_size=128):
    # Unpack nibbles (little-endian)
    W_lo = (W_packed & 0x0F).astype(np.int8)
    W_hi = ((W_packed >> 4) & 0x0F).astype(np.int8)
    # Sign extend 4-bit to 8-bit
    W_lo = np.where(W_lo >= 8, W_lo - 16, W_lo).astype(np.int8)
    W_hi = np.where(W_hi >= 8, W_hi - 16, W_hi).astype(np.int8)
    W_int8 = np.empty(W_packed.shape[0] * 2, dtype=np.int8)
    W_int8[0::2] = W_lo
    W_int8[1::2] = W_hi
    # Per-group scale
    groups = len(W_int8) // group_size
    for g in range(groups):
        s = float(scale_b[g])  # FP16 converted to float
        sl = slice(g * group_size, (g + 1) * group_size)
        W_int8[sl] = np.clip(np.round(W_int8[sl] * s), -128, 127)
    return W_int8

C_ref = A_tile.astype(np.int32) @ dequant_int4(B_packed, scale_b).reshape(Kt, 16).astype(np.int32)
```

---

## 3. Architectural Overview

YUA-T16 v2 adds the following blocks to the v1 architecture:

```
                    ┌─────────────────────────────────────────────────┐
                    │                  YUA-T16 v2                      │
                    │                                                   │
  ┌──────────┐      │  ┌──────────────┐    ┌──────────────────────┐   │
  │ ACT_SRAM │──────┼─▶│ SCALE_A unit │───▶│                      │   │
  │ (INT8)   │      │  │ (FP16→INT8   │    │   16×16 MAC ARRAY    │   │
  └──────────┘      │  │  optional)   │    │   (256 mac_pe units) │   │
                    │  └──────────────┘    │                      │   │
  ┌──────────┐      │  ┌──────────────┐    │   INT32 accumulators │   │
  │ WGT_SRAM │──────┼─▶│ INT4 DECOMP  │───▶│                      │   │
  │(INT8 or  │      │  │ STAGE        │    └──────────┬───────────┘   │
  │ INT4     │      │  │ nibble unpack│               │               │
  │ packed)  │      │  │ + scale_b    │               ▼               │
  └──────────┘      │  └──────────────┘      ┌──────────────┐        │
                    │                         │  OUT_SRAM /  │        │
  ┌──────────────┐  │  ┌──────────────┐       │  DMA WB      │        │
  │ SCALE_A_BASE │──┼─▶│ Scale Reg    │       │ (INT32 tile) │        │
  │ (FP16 bank)  │  │  │ Banks        │       └──────────────┘        │
  └──────────────┘  │  └──────────────┘                               │
  ┌──────────────┐  │  ┌──────────────┐                               │
  │ SCALE_B_BASE │──┼─▶│ MODE / CTRL  │                               │
  │ (FP16 bank)  │  │  │ FSM          │                               │
  └──────────────┘  │  └──────────────┘                               │
                    └─────────────────────────────────────────────────┘
```

The INT4 decompression stage is a pure combinational pipeline that feeds
the MAC array's b_col inputs. It adds 2 pipeline stages (unpack + scale)
and introduces at most 2 cycles of latency before the first MAC cycle
in INT4 mode.

The SCALE_A unit is bypassed in INT8 mode (direct feed from ACT_SRAM).

---

## 4. Execution Model

### 4.1 Tile Lifecycle (INT8 mode, v1-compatible)

Identical to v1:

1. Load A_tile into ACT_SRAM
2. Load B_tile into WGT_SRAM
3. Clear accumulators
4. Execute Kt cycles of MAC operations
5. Write back C_tile
6. Signal completion

### 4.2 Tile Lifecycle (INT4 weight mode)

1. Load A_tile (INT8) into ACT_SRAM
2. Load B_tile (INT4 packed) into WGT_SRAM
   - B_tile size in bytes = (Kt * tile_N) / 2  (half of INT8)
3. Load scale_b groups into SCALE_B_BASE registers
4. Optionally load scale_a per-channel into SCALE_A_BASE registers
5. Clear accumulators
6. Execute Kt cycles of MAC operations
   - Each cycle: INT4 nibbles are unpacked and scaled to INT8 before MAC
7. Write back C_tile (INT32)
8. Signal completion

### 4.3 Tile Lifecycle (32x32 mode)

1. Load A_tile (INT8, 32 x Kt) into ACT_SRAM
2. Load B_tile into WGT_SRAM
3. Load scale registers if INT4 mode
4. Clear accumulators (once for the entire 32x32 tile)
5. Execute 4 sub-tile passes (see Section 2.4)
   - Each sub-tile takes Kt MAC cycles
   - Total compute cycles = 4 * Kt
6. Write back C_tile (INT32, 32x32 = 1024 elements = 4096 bytes)
7. Signal completion

### 4.4 INT4 Decompression Pipeline

The decompression pipeline operates as follows each clock cycle:

```
Cycle N:   WGT_SRAM read — fetch 8 packed bytes (16 INT4 weights for b_col)
Cycle N+1: Nibble unpack — produce 16 INT8 values (sign-extended)
Cycle N+2: Scale multiply — apply scale_b[group], clamp, feed to MAC array
```

The pipeline is flushed at the start of each tile. During flush (cycles 1-2),
the MAC enable signal is held low.

### 4.5 Timing Assumptions

All v1 timing assumptions hold. Additional INT4-specific rules:

- INT4 decomp adds 2 pipeline stages; effective MAC compute begins at cycle 3
- scale_b registers must be loaded before CTRL.start is asserted
- GROUP_SIZE must divide Kt evenly; violation results in undefined behavior

---

## 5. Configuration Registers

### 5.1 v1-Compatible Registers (preserved at same offsets)

| Offset | Name      | Width | Description                          |
|--------|-----------|-------|--------------------------------------|
| 0x00   | CTRL      | 32    | [0]=start, [1]=busy, [2]=done        |
| 0x04   | MODE      | 32    | [1:0]=TILE_SIZE, [3:2]=WEIGHT_FMT    |
| 0x08   | Kt        | 32    | Accumulation depth                   |
| 0x10   | ACT_BASE  | 64    | A tile base address                  |
| 0x18   | WGT_BASE  | 64    | B tile base address (packed if INT4) |
| 0x20   | OUT_BASE  | 64    | C tile output address                |
| 0x28   | PERF_CYC  | 32    | Cycle counter                        |
| 0x2C   | PERF_BYTES| 32    | Byte transfer counter                |

### 5.2 v2-Only Registers

| Offset | Name          | Width | Description                            |
|--------|---------------|-------|----------------------------------------|
| 0x30   | GROUP_SIZE    | 32    | AWQ group size (default 128)           |
| 0x34   | PERF_DECOMP   | 32    | INT4 decompression stall cycles        |
| 0x40   | SCALE_A_BASE  | 64    | FP16 scale_a array base address        |
| 0x48   | SCALE_B_BASE  | 64    | FP16 scale_b array base address        |

### 5.3 MODE Register Bit Fields

```
MODE[1:0] — TILE_SIZE
  00 = 16x16 (default, v1-compatible)
  01 = 32x32
  10 = reserved
  11 = reserved

MODE[3:2] — WEIGHT_FMT
  00 = INT8 (default, v1-compatible)
  01 = INT4 packed
  10 = INT4 packed + scale_a activation scaling (full AWQ)
  11 = reserved

MODE[4] — SCALE_A_EN
  0 = scale_a bypass (default)
  1 = apply scale_a to activations before MAC

MODE[31:5] — reserved, must be zero
```

---

## 6. Descriptor: GEMM_INT4 (Type 0x0E)

The GEMM_INT4 descriptor extends GEMM_INT8 with INT4 weight addressing and
AWQ scale pointers. It is 64 bytes, conformant with the ORBIT descriptor
header format.

```c
struct orbit_desc_gemm_int4 {
  orbit_desc_header h;   // type=0x0E, flags, reserved0, length, next_desc

  uint64_t act_addr;     // A tile base address (INT8, row-major)
  uint64_t wgt_addr;     // B tile base address (INT4 packed, row-major)
  uint64_t out_addr;     // C tile output address (INT32, row-major)

  uint32_t Kt;           // K dimension (accumulation depth)
  uint16_t tile_mode;    // 0=16x16, 1=32x32
  uint16_t group_size;   // AWQ group size (must be power of 2)

  uint64_t scale_a_addr; // FP16 scale_a array (per activation channel)
                         //   size: tile_M elements (16 or 32)
                         //   set to 0 if scale_a not used

  uint64_t scale_b_addr; // FP16 scale_b array (per weight group)
                         //   size: ceil(Kt / group_size) elements
};
```

Total: 8 (header) + 8 + 8 + 8 + 4 + 2 + 2 + 8 + 8 = 56 bytes.
Remaining 8 bytes are reserved (must be zero), filling to 64 bytes.

### 6.1 Semantics

- act_addr: row-major INT8 tensor, shape [tile_M x Kt]
- wgt_addr: INT4 packed tensor, shape [Kt x tile_N / 2] bytes
  - Byte layout: byte[k][j/2] holds weight[k][j] in bits[3:0] and
    weight[k][j+1] in bits[7:4] (little-endian nibble order)
- out_addr: row-major INT32 tensor, shape [tile_M x tile_N]
- scale_a_addr = 0: activation scaling bypassed
- scale_b_addr must be non-zero in INT4 mode

### 6.2 GEMM_INT8 Descriptor in v2 Context

GEMM_INT8 (type 0x02) operates identically to v1. The scale_a and scale_b
fields that were reserved in v1 remain reserved when type 0x02 is used.
Hardware MUST NOT apply INT4 dequantization when processing type 0x02.

---

## 7. Memory Layout

### 7.1 INT4 Packed Weight Layout (WGT_SRAM)

For a tile of shape [Kt x tile_N]:

```
Byte address:   0          1          2
               [7:4][3:0] [7:4][3:0] [7:4][3:0] ...
Weight index:   [1] [0]   [3] [2]   [5] [4]  ...

B_int4[k, j]  stored at:
  byte_offset = k * (tile_N / 2) + j / 2
  nibble      = j % 2   (0 = low nibble [3:0], 1 = high nibble [7:4])
```

For a 16x16 tile at INT4: WGT_SRAM occupies Kt * 8 bytes (vs Kt * 16 for INT8).
For a 32x32 tile at INT4: WGT_SRAM occupies Kt * 16 bytes (vs Kt * 32 for INT8).

### 7.2 Scale Array Layout (GDDR6 → on-chip scale register bank)

scale_b array (FP16):

```
Address offset: 0       2       4       ...
Group index:    [0]    [1]     [2]    ...
Size (bytes): 2 * ceil(Kt / group_size)
```

scale_a array (FP16):

```
Address offset: 0       2       4       ...
Channel index:  [0]    [1]     [2]    ...
Size (bytes): 2 * tile_M  (32 bytes for 16x16, 64 bytes for 32x32)
```

Scale arrays are DMA-transferred from GDDR6 into on-chip registers by the
control FSM before execution begins.

### 7.3 Output Tile Layout (C_tile)

Identical to v1 for 16x16 mode: 16 x 16 x 4 bytes = 1024 bytes, row-major.

For 32x32 mode: 32 x 32 x 4 bytes = 4096 bytes, row-major.

---

## 8. Timing Diagram

### 8.1 INT4 Decompression Pipeline

```
         Clk: ___/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\___
  SRAM read k:   [k=0]  [k=1]  [k=2]  [k=3]  [k=4]
  Unpack k  :          [k=0]  [k=1]  [k=2]  [k=3]
  Scale k   :                 [k=0]  [k=1]  [k=2]
  MAC en    :          0      0      1      1      1
  acc_clr   :  ‾‾‾‾‾\________________________________
```

MAC enable rises on cycle 3 (after 2-stage decomp latency).
Effective compute cycles = Kt (not Kt - 2); the pipeline drains
the first 2 output slots without accumulation (MAC en = 0).

### 8.2 32x32 Sub-tile Sequencing

```
  Sub-tile  (0,0)        (0,1)        (1,0)        (1,1)
           |<-- Kt -->|<-- Kt -->|<-- Kt -->|<-- Kt -->|
  acc_clr   ‾‾\__________________________________________
  done           _                                      _/‾‾
```

Total cycles for one 32x32 tile = 4 * Kt (plus pipeline fill/drain).

---

## 9. Design Constraints

All v1 constraints apply:

- Target FPGA clock: 150 MHz (safe), 200 MHz (stretch)
- Target ASIC clock (MPW): 50–100 MHz
- Deterministic behavior prioritized over peak throughput

Additional v2 constraints:

- INT4 decompression logic must not extend the critical path beyond 200 MHz
  on target FPGA; use registered pipeline stages
- Scale register banks (scale_a, scale_b): implemented as synchronous SRAM
  or flip-flop arrays; size <= 256 FP16 values per bank (512 bytes)
- GROUP_SIZE must be a power of 2 in [32, 64, 128, 256]; hardware may assert
  an error flag if GROUP_SIZE is out of range

---

## 10. Verification Strategy

All v1 tests must pass unchanged in INT8 mode.

Additional v2 tests:

- INT4 nibble unpack correctness (all 16 signed patterns, -8 to +7)
- INT4 tile accumulation against NumPy dequantization reference
- AWQ scale application: verify clamp behavior at boundary values
- 32x32 tile: compare four 16x16 sub-tile results vs single 32x32 NumPy ref
- Mode transitions: INT8 → INT4 → INT8 without reset; verify no stale state
- Group boundary test: Kt = 128, group_size = 128 (single group)
- Multi-group test: Kt = 512, group_size = 128 (4 groups, different scales)
- Throughput test: 32x32 INT4, back-to-back tile execution, utilization
  measurement

---

## 11. Non-Goals

All v1 non-goals apply.

Additionally, YUA-T16 v2 does not:

- Perform FP16 native MAC operations (decomp output is clamped INT8)
- Support mixed-precision (FP16 A x INT4 B) directly in hardware;
  AWQ approximation via INT8 is the supported method
- Implement runtime GROUP_SIZE changes mid-tile

---

## 12. Role in System

YUA-T16 v2 replaces v1 as the GEMM compute primitive in ORBIT-G1 clusters.

It enables ORBIT-G1 to execute AWQ-quantized LLM weights (e.g., GPT-OSS-20B
in INT4 AWQ format) with ~2x memory efficiency compared to INT8, while
preserving the v1 deterministic execution model and verification approach.

The 32x32 tile mode reduces dispatch overhead for large GEMM workloads by
halving the number of GEMM_INT4 descriptors required for a given matrix.
