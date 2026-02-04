# YUA-T16 Spec v1

## 0. Purpose and Identity

YUA-T16 is a fixed-function INT8 GEMM tile accelerator designed for
LLM feed-forward network (FFN) inference workloads.

The accelerator computes a single 16x16 output tile with accumulation
over the K dimension and is intended to be instantiated as a basic
compute unit inside a larger accelerator system (e.g., ORBIT-G1).

YUA-T16 prioritizes:
- Deterministic execution
- Bit-exact numerical correctness
- Simple control and verification
- FPGA-first, MPW-ready implementation

It is not intended to be a general-purpose processor.

---

## 1. Scope Definition

### 1.1 In Scope

- INT8 x INT8 signed multiplication
- INT32 accumulation
- Fixed output tile size: 16 x 16
- Accumulation over configurable Kt dimension
- Output-stationary dataflow
- On-chip SRAM for A and B tiles
- MMIO-controlled execution
- Deterministic cycle behavior

### 1.2 Out of Scope

- Attention, softmax, KV-cache
- Activation functions (GELU, SiLU, etc.)
- Bias addition or requantization
- Training or backpropagation
- INT4 or sparsity
- PCIe, cache coherence, virtual memory
- OS or framework integration

All out-of-scope functionality must be handled externally.

---

## 2. Mathematical Contract

### 2.1 Input and Output Tensors

- A_tile: INT8 matrix of shape [16 x Kt]
- B_tile: INT8 matrix of shape [Kt x 16]
- C_tile: INT32 matrix of shape [16 x 16]

### 2.2 Computation Definition

For each output element:

C[i,j] = sum_{k=0..Kt-1} ( A[i,k] * B[k,j] )

- Accumulation is performed in INT32
- No saturation or rounding is applied
- Accumulator is cleared exactly once per tile execution
- Kt must be chosen such that INT32 overflow does not occur

### 2.3 Correctness Requirement

The output C_tile must be bit-exact equivalent to:

NumPy reference:
C_ref = A_tile.astype(int32) @ B_tile.astype(int32)

Any deviation is considered a functional error.

---

## 3. Architectural Overview

YUA-T16 consists of:
- A 16x16 MAC array
- Local accumulator registers (INT32)
- Separate SRAMs for activation (A) and weight (B)
- A simple control FSM
- Optional output SRAM or direct DMA writeback

The design follows an output-stationary dataflow model.

---

## 4. Execution Model

### 4.1 Tile Lifecycle

1. Load A_tile into ACT_SRAM
2. Load B_tile into WGT_SRAM
3. Clear accumulators
4. Execute Kt cycles of MAC operations
5. Write back C_tile
6. Signal completion

### 4.2 Timing Assumptions

- SRAM read latency is 1 cycle
- MAC operations occur only when valid data is present
- Compute begins only after both A and B loads complete

---

## 5. Configuration Registers

The following registers are required:

- CTRL: start, busy, done
- Kt: accumulation length
- ACT_BASE: base address of A_tile
- WGT_BASE: base address of B_tile
- OUT_BASE: base address of C_tile
- PERF_CYC: cycle counter
- PERF_BYTES: byte counter

The exact register map is defined in the integration-level specification.

---

## 6. Design Constraints

- Target FPGA clock: 150 MHz (safe), 200 MHz (stretch)
- Target ASIC clock (MPW): 50–100 MHz
- Deterministic behavior is prioritized over peak throughput
- Verification simplicity is a first-class requirement

---

## 7. Verification Strategy

YUA-T16 must be verifiable via:
- Unit-level MAC testing
- Array-level accumulation testing
- End-to-end tile execution tests
- Randomized Kt testing
- Repeated execution without reset

All tests must compare against a software golden model.

---

## 8. Non-Goals

YUA-T16 explicitly does not attempt to:
- Replace a GPU
- Execute full LLM graphs
- Perform dynamic scheduling
- Support speculative execution

It is a single-purpose, verifiable GEMM tile engine.

---

## 9. Role in System

YUA-T16 is intended to serve as:
- A building block for larger tensor accelerators
- A draft-model compute unit for speculative decoding
- A deterministic compute core for bring-up and verification

System-level concerns are handled by the surrounding architecture.
