# KVC Spec v1
## KV-Cache Controller

## 0. Purpose and Identity

The KVC (KV-Cache Controller) is a memory management unit dedicated to
storing and retrieving Key and Value tensors for autoregressive LLM
inference (the KV-cache).

It handles the memory layout, address translation, and DMA transfer for
K/V tensors across all Transformer layers, allowing the attention mechanism
to reuse computed keys and values from previously generated tokens without
recomputation.

The KVC implements PagedAttention-style block management, directly analogous
to the vLLM approach, enabling non-contiguous physical memory allocation
for logically contiguous sequence slots.

KVC is a peer unit to YUA-T16 and VPU within ORBIT-G1 clusters.

KVC prioritizes:
- Correct K/V tensor read/write with deterministic timing
- Efficient GDDR6 bandwidth utilization via burst transfers
- PagedAttention block management for long-context and multi-sequence support
- Minimal host software intervention during decode

---

## 1. Scope Definition

### 1.1 In Scope

- K and V tensor storage in GDDR6 for up to 32K sequence tokens
- GPT-OSS-20B dimensions: 32 layers, 32 attention heads, head_dim=128
- PagedAttention physical block management (vLLM-style)
- Prefill mode: sequential K/V write for all input tokens
- Decode mode: append new K/V at current position, read full history
- KVC_READ (type 0x0A) and KVC_WRITE (type 0x0B) descriptors
- Block table (page table) resident in GDDR6, cached on-chip
- Sequence ID multiplexing (up to 64 concurrent sequences per KVC instance)

### 1.2 Out of Scope

- Attention score computation (handled by YUA-T16 GEMM)
- KV compression or quantization (host responsibility)
- Eviction policy (host allocates/frees blocks via driver API)
- Cross-device KV migration
- Speculative decoding KV management
- Virtual memory or OS paging

---

## 2. System Parameters

### 2.1 GPT-OSS-20B Attention Dimensions

| Parameter    | Value  | Description                         |
|--------------|--------|-------------------------------------|
| num_layers   | 32     | Transformer layers (KV per layer)   |
| num_heads    | 32     | Attention heads per layer           |
| head_dim     | 128    | Dimension per head                  |
| kv_dtype     | FP16   | K and V tensor element type         |
| bytes_per_kv | 2      | FP16 = 2 bytes per element          |

K and V tensor size per token per layer:

```
  bytes_per_token_per_layer = 2 * num_heads * head_dim * bytes_per_kv
                            = 2 * 32 * 128 * 2
                            = 16,384 bytes = 16 KB
```

(Factor of 2 for both K and V.)

### 2.2 Sequence Length and Memory

| Sequence Length | KV Size (1 layer) | KV Size (all 32 layers) |
|-----------------|-------------------|--------------------------|
| 1K tokens       | 16 MB             | 512 MB                   |
| 4K tokens       | 64 MB             | 2 GB                     |
| 8K tokens       | 128 MB            | 4 GB                     |
| 32K tokens      | 512 MB            | 16 GB                    |

For ORBIT-G1 with 32 GB GDDR6:
- Weight storage (GPT-OSS-20B INT8): ~20 GB
- KV-cache budget: ~12 GB remaining
- Maximum context at 32K: 0.75 sequences at full context
- Maximum context at 8K: ~3 sequences
- Maximum context at 4K: ~6 sequences

KV pool size is configured at runtime by the host driver based on
model weight memory consumption.

---

## 3. PagedAttention Architecture

### 3.1 Block Abstraction

KVC divides the K/V memory pool into fixed-size physical blocks.

```
  BLOCK_SIZE: 16 tokens per block (configurable: 8, 16, 32)
  Block content: K[BLOCK_SIZE, num_heads, head_dim] +
                 V[BLOCK_SIZE, num_heads, head_dim]
  Block size bytes = BLOCK_SIZE * num_heads * head_dim * 2 * 2
                   = 16 * 32 * 128 * 2 * 2 = 262,144 bytes = 256 KB

  KV pool (12 GB) / 256 KB per block = 48,000 blocks available
```

Physical blocks are addressed by block_id (uint16_t, max 65535 blocks).

### 3.2 Block Table

Each sequence has a block table: a list of physical block IDs mapping
logical block slots to physical blocks.

```
  Sequence seq_id:
    block_table[seq_id][logical_block] = physical_block_id

  Logical token position t maps to:
    logical_block  = t / BLOCK_SIZE
    block_offset   = t % BLOCK_SIZE
```

Block tables are stored in a dedicated GDDR6 region (KVC_BLOCK_TABLE_BASE)
and cached in an on-chip SRAM (block table cache, see Section 5).

### 3.3 Physical Address Calculation

For reading/writing K (or V) at a specific position:

```
  Given: layer_id, head_id, seq_id, token_pos, K_or_V (0 or 1)

  Step 1: logical_block = token_pos / BLOCK_SIZE
          block_offset  = token_pos % BLOCK_SIZE

  Step 2: phys_block = block_table[seq_id][logical_block]

  Step 3: block_base_addr = KVC_POOL_BASE + phys_block * BLOCK_BYTES

  Step 4: Within-block layout (see Section 4):
          kv_offset  = K_or_V * (BLOCK_SIZE * num_heads * head_dim * 2)
          layer_addr = block_base_addr + layer_id * 2 * BLOCK_SIZE * num_heads * head_dim * 2
          head_addr  = layer_addr + kv_offset + head_id * BLOCK_SIZE * head_dim * 2
          elem_addr  = head_addr + block_offset * head_dim * 2

  Final address: elem_addr
```

Note: See Section 4 for the precise within-block memory layout, which
interleaves K and V for cache-line efficiency.

---

## 4. GDDR6 Memory Layout

### 4.1 KVC Memory Map

```
GDDR6 Address Space (ORBIT-G1)
┌────────────────────────────────────────────────────┐  0x0000_0000_0000
│  Model Weights                                      │
│  (YUA-T16 reads via GEMM descriptors)               │
│  ~20 GB for GPT-OSS-20B INT8                       │
├────────────────────────────────────────────────────┤  offset: model_end
│  KVC_BLOCK_TABLE region                             │
│  64 sequences * 2048 max_blocks * 2 bytes           │
│  = 256 KB (rounded to next 1 MB boundary)          │
├────────────────────────────────────────────────────┤  KVC_POOL_BASE
│  KV-Cache Pool                                      │
│  N_BLOCKS * BLOCK_BYTES (up to 12 GB)              │
│                                                     │
│  Block 0:  [K, V] for all layers, BLOCK_SIZE tokens│
│  Block 1:  [K, V] for all layers, BLOCK_SIZE tokens│
│  ...                                                │
│  Block N:  [K, V] for all layers, BLOCK_SIZE tokens│
└────────────────────────────────────────────────────┘
```

KVC_POOL_BASE and KVC_BLOCK_TABLE_BASE are configured by the host driver
at initialization via KVC MMIO registers.

### 4.2 Physical Block Internal Layout

Each physical block contains K and V for all layers and all heads for
BLOCK_SIZE tokens. Layout is layer-major, then K/V-major, then head-major:

```
Physical Block (256 KB, BLOCK_SIZE=16, 32 layers, 32 heads, head_dim=128):

Offset 0x00000:  Layer 0, K, head 0, tokens [0..15], head_dim=128 × FP16
                 128 * 16 * 2 = 4096 bytes
Offset 0x01000:  Layer 0, K, head 1, tokens [0..15]
                 4096 bytes
...
Offset 0x1F000:  Layer 0, K, head 31, tokens [0..15]
                 4096 bytes
Offset 0x20000:  Layer 0, V, head 0, tokens [0..15]
                 4096 bytes
...
Offset 0x3F000:  Layer 0, V, head 31, tokens [0..15]
                 4096 bytes
Offset 0x40000:  Layer 1, K, head 0, tokens [0..15]
...
Offset 0xFF000:  Layer 31, V, head 31, tokens [0..15]  (last)
                 End: 0xFF000 + 0x1000 = 0x100000 = 256 KB ✓
```

Formula for element address within a block:

```
  layer_stride = 2 * num_heads * BLOCK_SIZE * head_dim * sizeof(FP16)
               = 2 * 32 * 16 * 128 * 2 = 262144 / 32 = 8192 bytes per KV set

  Wait — full derivation:
    per_head_tokens = BLOCK_SIZE * head_dim * 2 = 16 * 128 * 2 = 4096 bytes
    per_layer_k = num_heads * per_head_tokens    = 32 * 4096 = 131072 bytes
    per_layer_v = num_heads * per_head_tokens    = 131072 bytes
    per_layer   = per_layer_k + per_layer_v      = 262144 bytes = 256 KB / 32 layers ≈ 8192 bytes

  NOTE: 256 KB / 32 layers = 8192 bytes per layer. Check:
    per_layer = 2 * 32 * 16 * 128 * 2 = 262,144 bytes
    Total = 32 * 262,144 = 8,388,608 bytes = 8 MB per block?

  CORRECTION: BLOCK_SIZE is 16 tokens, num_layers=32, num_heads=32, head_dim=128.
    Per token, per layer: 2 * 32 * 128 * 2 = 16,384 bytes (K+V)
    Per block (16 tokens), per layer: 16 * 16,384 = 262,144 bytes
    Per block (16 tokens), all 32 layers: 32 * 262,144 = 8,388,608 bytes = 8 MB

  BLOCK_BYTES = 8 MB for BLOCK_SIZE=16, all layers stored per block.
  Revised pool: 12 GB / 8 MB = 1,536 blocks.
```

**Design decision**: Store all layers within one block to minimize
address arithmetic during decode. The block table maps (seq_id,
logical_block) → one physical block containing all layers.

Revised constants:

| Parameter    | Value     |
|--------------|-----------|
| BLOCK_SIZE   | 16 tokens |
| BLOCK_BYTES  | 8 MB      |
| Pool (12 GB) | 1,536 blocks |
| Max seqs     | 64        |
| Max logical  | 32,768 tokens / 16 = 2,048 blocks per sequence |

### 4.3 Within-Block Address Formula (final)

```
  block_base    = KVC_POOL_BASE + phys_block * BLOCK_BYTES
  kv_select     = K_or_V  (0 = K, 1 = V)
  layer_offset  = layer_id  * (2 * num_heads * BLOCK_SIZE * head_dim * 2)
  kv_offset     = kv_select * (    num_heads * BLOCK_SIZE * head_dim * 2)
  head_offset   = head_id   * (               BLOCK_SIZE * head_dim * 2)
  token_offset  = block_token_idx * (head_dim * 2)
                  where block_token_idx = token_pos % BLOCK_SIZE

  elem_addr = block_base + layer_offset + kv_offset + head_offset + token_offset
```

All offsets are in bytes. All addresses must be 64-byte aligned for GDDR6
burst efficiency; the formula above produces 256-byte aligned head start
addresses (128 FP16 elements = 256 bytes).

---

## 5. On-Chip Block Table Cache

Accessing the block table from GDDR6 on every KVC operation would add
prohibitive latency. KVC maintains an on-chip block table cache:

- Capacity: 64 sequences × 2048 logical blocks = 131,072 entries
- Entry width: 16 bits (block_id, uint16_t)
- Total size: 262,144 bytes = 256 KB
- Implementation: synchronous SRAM, 1-cycle read latency
- Organization: (seq_id, logical_block) → block_id, direct-mapped

The block table cache is fully resident (no eviction). The host driver
must update the on-chip cache whenever blocks are allocated or freed,
using the KVC_BT_UPDATE descriptor or MMIO write path.

Cache coherence: Software responsibility. Hardware does not snoop GDDR6
block table region after initialization.

---

## 6. Operating Modes

### 6.1 Prefill Mode

In prefill, all input tokens are processed in a single forward pass.
The KVC sequentially writes K and V vectors for tokens 0..N-1.

```
  For each token t in [0, seq_len):
    KVC_WRITE descriptor:
      layer_id   = current layer
      head_id    = all heads (burst write for all heads at once)
      seq_id     = sequence ID
      token_pos  = t
      src_addr   = address of K_t and V_t vectors in GDDR6 scratch
```

KVC_WRITE in prefill mode accepts a bulk write:
- src_addr points to K[0..seq_len-1, all_heads, head_dim] concatenated,
  followed by V[0..seq_len-1, all_heads, head_dim].
- write_mode = PREFILL_BULK.
- KVC performs address scatter: places each token into the correct block
  based on the block table.

### 6.2 Decode Mode

In decode, one new token is generated per step. The KVC:
1. Writes K and V for the new token (position = current_seq_len).
2. Reads K and V for positions 0..current_seq_len (including new token).

```
  Decode step:
    ① KVC_WRITE (mode=DECODE): write K_new, V_new at token_pos=current_len
    ② KVC_READ: read K[0..current_len], V[0..current_len] to scratch
    ③ YUA-T16 GEMM: Q @ K^T (attention)
    ④ VPU VECTOR_OP_EX: scale + softmax
    ⑤ YUA-T16 GEMM: attn_scores @ V
```

KVC_READ in decode mode performs address gather: assembles contiguous
K[0..seq_len-1] and V[0..seq_len-1] tensors in GDDR6 scratch from
potentially non-contiguous physical blocks.

---

## 7. Descriptors

### 7.1 KVC_READ Descriptor (Type 0x0A)

Reads K and V tensors for a range of token positions into a contiguous
scratch buffer for subsequent attention GEMM.

```c
struct orbit_desc_kvc_read {
  orbit_desc_header h;    // type=0x0A, flags, reserved0,
                          //   length=seq_len_to_read, next_desc

  uint32_t seq_id;        // Sequence ID (0..63)
  uint32_t layer_id;      // Transformer layer (0..31)
  uint32_t seq_start;     // Start token position (inclusive)
  uint32_t seq_len;       // Number of tokens to read

  uint64_t k_dst_addr;    // GDDR6 scratch address for output K tensor
                          //   Shape: [seq_len, num_heads, head_dim] FP16
                          //   Size:  seq_len * 32 * 128 * 2 bytes

  uint64_t v_dst_addr;    // GDDR6 scratch address for output V tensor
                          //   Same shape and size as K

  uint8_t  read_mode;     // 0=ALL_HEADS, 1=SINGLE_HEAD (head_id field used)
  uint8_t  head_id;       // Head index (only if read_mode=SINGLE_HEAD)
  uint16_t reserved;
};
```

Total: 8 + 4+4+4+4 + 8 + 8 + 1+1+2 = 44 bytes.
Remaining 20 bytes reserved (must be zero), filling to 64 bytes.

Output layout at k_dst_addr:

```
k_dst_addr + 0:
  K[seq_start, head=0, 0..127]   (256 bytes, 128×FP16)
  K[seq_start, head=1, 0..127]   (256 bytes)
  ...
  K[seq_start, head=31, 0..127]  (256 bytes)
  K[seq_start+1, head=0, 0..127] (256 bytes)
  ...
  K[seq_start+seq_len-1, head=31, 0..127]

Total K size: seq_len * 32 * 256 = seq_len * 8192 bytes
Total V size: same

k_dst_addr and v_dst_addr must be 256-byte aligned.
```

### 7.2 KVC_WRITE Descriptor (Type 0x0B)

Writes K and V tensors from a source buffer into the KV-cache for one
or more token positions.

```c
struct orbit_desc_kvc_write {
  orbit_desc_header h;    // type=0x0B, flags, reserved0,
                          //   length=tokens_to_write, next_desc

  uint32_t seq_id;        // Sequence ID (0..63)
  uint32_t layer_id;      // Transformer layer (0..31)
  uint32_t token_pos;     // Start token position for write
  uint32_t write_count;   // Number of tokens to write (1 for decode)

  uint64_t k_src_addr;    // GDDR6 source address for K tensor
                          //   Shape: [write_count, num_heads, head_dim] FP16

  uint64_t v_src_addr;    // GDDR6 source address for V tensor
                          //   Same shape as K

  uint8_t  write_mode;    // 0=DECODE (single token), 1=PREFILL_BULK
  uint8_t  alloc_blocks;  // If 1, KVC auto-allocates new physical blocks
                          //   as needed from free pool.
                          //   If 0, blocks must already be allocated.
  uint16_t reserved;
};
```

Total: 8 + 4+4+4+4 + 8 + 8 + 1+1+2 = 44 bytes.
Remaining 20 bytes reserved, filling to 64 bytes.

### 7.3 Block Allocation Model

When alloc_blocks=1 in KVC_WRITE, KVC allocates new physical blocks
from its internal free list. The free list is a FIFO queue of available
block IDs, initialized by the host driver at startup.

If no blocks are available, KVC stalls and asserts an error flag in
KVC_STATUS (OOM condition). The host driver must free blocks before
issuing further writes.

Block allocation updates the on-chip block table cache immediately.
The GDDR6 block table region is updated asynchronously by firmware
(not required to be coherent before next KVC operation).

---

## 8. Register Map

### 8.1 KVC MMIO Registers

| Offset | Name                | Width | Description                              |
|--------|---------------------|-------|------------------------------------------|
| 0x000  | KVC_CTRL            | 32    | [0]=busy, [1]=error, [2]=oom             |
| 0x004  | KVC_STATUS          | 32    | [15:0]=active_seqs, [31:16]=free_blocks  |
| 0x008  | KVC_CFG_BLOCK_SIZE  | 8     | Block size: 8, 16, or 32 tokens          |
| 0x00C  | KVC_CFG_NUM_HEADS   | 8     | Attention heads (default 32)             |
| 0x00D  | KVC_CFG_HEAD_DIM    | 16    | Head dimension (default 128)             |
| 0x010  | KVC_POOL_BASE       | 64    | GDDR6 base address of KV-cache pool      |
| 0x018  | KVC_POOL_BYTES      | 64    | KV-cache pool size in bytes              |
| 0x020  | KVC_BT_BASE         | 64    | GDDR6 base address of block table        |
| 0x028  | KVC_BT_ONCHIP_BASE  | 64    | On-chip BT SRAM base (internal mapping)  |
| 0x030  | KVC_FREE_LIST_HEAD  | 16    | Head pointer into free block list        |
| 0x032  | KVC_FREE_LIST_TAIL  | 16    | Tail pointer into free block list        |
| 0x034  | KVC_FREE_COUNT      | 16    | Number of free blocks available          |
| 0x038  | KVC_PERF_READS      | 32    | Total KVC_READ operations completed      |
| 0x03C  | KVC_PERF_WRITES     | 32    | Total KVC_WRITE operations completed     |
| 0x040  | KVC_PERF_BW_BYTES   | 64    | Total GDDR6 bytes transferred            |
| 0x048  | KVC_PERF_STALL_CYC  | 32    | Cycles stalled waiting for GDDR6         |

### 8.2 Block Table Update (Direct MMIO Path)

For single-block updates during decode (avoid descriptor overhead):

| Offset | Name               | Width | Description                        |
|--------|--------------------|-------|------------------------------------|
| 0x100  | KVC_BT_UPDATE_SEQ  | 32    | seq_id for block table write       |
| 0x104  | KVC_BT_UPDATE_LBN  | 32    | logical_block_number               |
| 0x108  | KVC_BT_UPDATE_PBN  | 16    | physical_block_number (new value)  |
| 0x10A  | KVC_BT_UPDATE_GO   | 8     | Write 1 to commit update           |

Hardware updates on-chip BT SRAM immediately upon KVC_BT_UPDATE_GO write.

---

## 9. Prefill vs Decode Timing

### 9.1 Prefill: Bulk Write (seq_len = 512 tokens, 1 layer)

```
Prefill KVC_WRITE timeline:

  Tokens: 0────────────────────────────────────511
          |                                       |
  Blocks: B0[0..15] B1[16..31] ... B31[496..511]   (32 blocks allocated)
          ↑          ↑               ↑
          GDDR6 scatter write (block address computed per 16-token chunk)

  Cycles: 512 tokens * 8192 bytes/token / (512 GB/s GDDR6 / 1.6 GB cycle)
          ≈ 512 * 8192 / 320 ≈ 13,107 cycles @ 150 MHz
```

The KVC address scatter engine issues independent burst writes to each
physical block. Up to 4 concurrent GDDR6 write requests are outstanding.

### 9.2 Decode: Append + Read (current_seq_len = 512 tokens)

```
Decode step timeline:

  Cycle 0:       KVC_WRITE issued (token_pos=512, 1 new K/V token)
  Cycle 1-10:    Block table lookup (on-chip SRAM, 1 cycle) + GDDR6 write burst
                 1 token * 8192 bytes → ~26 cycles @ 512 GB/s
  Cycle 11:      KVC_WRITE done

  Cycle 12:      KVC_READ issued (seq_start=0, seq_len=513)
  Cycle 13-...:  For each of 33 blocks (ceil(513/16)):
                   Block table lookup → phys_block → burst read
                   16 tokens * 8192 bytes = 131,072 bytes per block
                   per block: ~410 cycles @ 512 GB/s
                 Total read: 33 * ~410 ≈ 13,530 cycles

  Cycle ~13542:  K and V tensors assembled in scratch
                 YUA-T16 GEMM can begin
```

Prefetch: The KVC can overlap GDDR6 reads for block N+1 while block N
is being written to scratch (double-buffered gather).

---

## 10. Sequence Lifecycle

```
  ALLOCATE  →  PREFILL  →  DECODE × T  →  FREE
  ─────────────────────────────────────────────
  1. Host allocates seq_id (MMIO write to KVC_ALLOC_SEQ)
  2. Host allocates initial physical blocks (free list pop)
  3. Host writes block table via KVC_BT_UPDATE
  4. KVC_WRITE PREFILL_BULK: store K/V for all input tokens
  5. KVC_WRITE DECODE: store K/V for new token (each decode step)
  6. KVC_READ: retrieve K/V for attention (each decode step)
     Repeat 5-6 until EOS or max_seq_len
  7. Host frees seq_id; returns physical blocks to free list
```

### 10.1 Maximum Concurrent Sequences

| GDDR6 Pool | Tokens/Seq | Blocks/Seq | Max Seqs (1536 blocks) |
|------------|------------|------------|------------------------|
| 12 GB      | 512        | 32         | 48                     |
| 12 GB      | 1024       | 64         | 24                     |
| 12 GB      | 4096       | 256        | 6                      |
| 12 GB      | 8192       | 512        | 3                      |

---

## 11. Design Constraints

- Target FPGA clock: 150 MHz
- Target ASIC clock: 50–100 MHz
- On-chip block table cache SRAM: 256 KB (64 seq × 2048 entries × 2 bytes)
- Free list SRAM: 2 × 16 KB (1,536 entries × 2 bytes, dual-port)
- Block table lookup latency: 1 cycle (on-chip SRAM)
- GDDR6 burst size: 256 bytes (minimum), 8 KB (preferred for throughput)
- Burst alignment: all GDDR6 accesses must be 256-byte aligned
- Outstanding GDDR6 requests: up to 4 (read or write)
- KVC_READ must assemble contiguous K and V tensors at dst addresses;
  physical blocks may be non-contiguous in GDDR6
- KVC_WRITE DECODE must complete in < 100 cycles to avoid decode latency

---

## 12. Verification Strategy

- Block table: unit test address formula for all (layer, head, token_pos)
  combinations in a 2-layer, 2-head, head_dim=4 minimodel
- Prefill write: verify correct GDDR6 scatter pattern with stub memory model
- Decode write: back-to-back 512 appends; verify block_id transitions at
  BLOCK_SIZE boundaries
- Decode read: verify gather assembles contiguous output from
  non-contiguous physical blocks (golden: reference address scatter/gather)
- OOM: fill all blocks, verify KVC_STATUS.oom is set, verify stall
- Sequence free and reallocate: reuse freed blocks in new sequence without
  data contamination
- Multi-sequence: 4 concurrent sequences with interleaved decode steps;
  verify no cross-sequence data corruption
- Bandwidth: measure GDDR6 utilization for KVC_READ at seq_len=4096;
  require > 80% of peak bandwidth

---

## 13. Non-Goals

The KVC does not:

- Perform attention computation (use YUA-T16 GEMM)
- Compress or quantize K/V tensors
- Implement eviction policy (host driver responsibility)
- Provide virtual addressing or OS integration
- Support cross-chip KV sharing
- Manage model weight storage (separate GDDR6 region, separate DMA path)

---

## 14. Role in System

The KVC is required to enable autoregressive generation in ORBIT-G1.
Without KVC, the system would need to recompute K and V for all previous
tokens on each decode step (O(T^2) compute), making generation impractical
for sequences longer than a few dozen tokens.

With KVC, decode attention becomes an O(T) memory read + O(d_head) compute
per head, at the cost of GDDR6 bandwidth. At 512 GB/s GDDR6 and 8K context,
the KV read occupies approximately 25% of available GDDR6 bandwidth per
decode step, leaving bandwidth headroom for model weight loads.

KVC, YUA-T16, and VPU together constitute the complete compute path for
GPT-OSS-20B inference on ORBIT-G1.
