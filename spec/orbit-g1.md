# ORBIT-G1 Spec v1

## 0. Product Identity

ORBIT-G1 is a PCIe-attached processing accelerator card designed to
augment general-purpose computers with deterministic compute and
utility offloading capabilities.

It is not a GPU and does not perform display or graphics rendering.
Instead, ORBIT-G1 operates as a sidecar accelerator for:
- LLM inference (INT8 GEMM / FFN)
- Data movement and format conversion
- Streaming, capture, and telemetry workloads
- Bring-up, diagnostics, and reliability monitoring

ORBIT-G1 is installed in a standard PCIe slot and operates alongside
existing CPU and GPU devices.

---

## 1. System-Level Role

ORBIT-G1 provides three primary capabilities:

1. Compute Acceleration
   - INT8 GEMM via YUA-T16 tiles
   - Deterministic execution for inference workloads

2. Utility Offload (SUP: Sidecar Utility Processor)
   - 2D copy with stride and tiling
   - Data format conversion (e.g., RGB↔YUV, FP16↔INT8)
   - Lightweight resize, clamp, and pack/unpack
   - Frame fingerprinting and integrity checks

3. Diagnostics and Assurance (Optional)
   - Fault detection via sampling and hashing
   - Trace capture for bring-up and debugging
   - Manufacturing and field diagnostics

Utility functionality is considered a first-class feature.
Assurance functionality is optional and mode-controlled.

---

## 2. Physical Integration

### 2.1 Form Factor

- PCIe add-in card
- PCIe Gen4 x16 electrical interface
- Single- or dual-slot mechanical profile (implementation-dependent)
- No display outputs (HDMI/DP)

### 2.2 Power

- PCIe slot power (up to 75W)
- Optional auxiliary power connector for higher-TDP SKUs

### 2.3 Host Visibility

From the host OS perspective, ORBIT-G1 appears as:
- A PCIe processing accelerator device
- Not a display controller
- Not a network device

A dedicated kernel driver is required.

---

## 3. High-Level Architecture

ORBIT-G1 consists of the following major blocks:

- PCIe subsystem (DMA, doorbell, interrupts)
- Command processor and descriptor dispatcher
- Compute clusters (YUA-T16-based)
- Global memory subsystem (e.g., GDDR6)
- SUP (Sidecar Utility Processor)
- Telemetry, tracing, and debug infrastructure

All blocks communicate via an on-chip interconnect.

---

## 4. Compute Subsystem

### 4.1 Clusters

- Multiple compute clusters per device
- Each cluster contains:
  - Multiple YUA-T16 GEMM tiles
  - Local scratchpad SRAM
  - Vector/SIMD unit for elementwise operations

### 4.2 Compute Scope

The compute subsystem is optimized for:
- LLM feed-forward network (FFN) layers
- Deterministic, tile-based execution
- High utilization via data reuse

Full graph execution and scheduling are handled externally.

---

## 5. SUP: Sidecar Utility Processor

### 5.1 Purpose

SUP exists to make ORBIT-G1 valuable outside pure AI workloads.
It handles data-centric tasks that would otherwise burden
the CPU or interfere with GPU rendering.

### 5.2 SUP Functional Blocks

- Copy + Format Engine (CFE)
  - 2D copy with stride
  - Crop and simple resize
  - Data packing and unpacking

- Format Conversion
  - RGB ↔ YUV
  - FP16/BF16 ↔ INT8
  - Clamp and dither operations

- Stream Hashing and Fingerprinting
  - CRC32/CRC64
  - Lightweight hash functions
  - Sample-based integrity checks

- Diagnostics and Trace
  - Ring buffer trace capture
  - Watchdog and hang detection
  - Fault injection (debug mode)

### 5.3 Operating Modes

SUP supports the following modes:

- OFF: fully gated, no activity
- UTILITY: copy/format/fingerprint enabled
- ASSURE: sampling and verification enabled
- DEBUG: fault injection and extended tracing

Modes are controlled via firmware and driver configuration.

---

## 6. Command and Queue Model

ORBIT-G1 uses a descriptor-based execution model.

### 6.1 Queues

At minimum, the following queues are supported:

- Compute Queue: GEMM and vector operations
- Utility Queue: SUP copy/format tasks
- Telemetry Queue: diagnostics and trace operations

Queues operate independently and may execute concurrently.

### 6.2 Descriptors

Descriptors are fixed-size and describe high-level operations.
The following descriptor classes are defined:

- DMA_2D
- GEMM_INT8
- VECTOR_OP
- COPY_2D_PLUS
- FORMAT_CONVERT
- FRAME_FINGERPRINT
- DIAG_RUN
- BARRIER
- EVENT
- PERF_SNAPSHOT

Exact binary formats are defined in a separate descriptor specification.

---

## 7. Software Stack

### 7.1 Driver

- Linux kernel driver
- PCIe enumeration and BAR mapping
- DMA buffer management
- Interrupt and event handling

### 7.2 Runtime

- User-space runtime library
- Command buffer construction and submission
- Memory allocation and synchronization
- Error and trace retrieval

### 7.3 Integration

Initial integration targets:
- Standalone test applications
- Python/C++ host utilities
- Incremental integration with ML frameworks

---

## 8. Reliability and Diagnostics

ORBIT-G1 is designed to support:
- Deterministic execution
- Detectable failure modes
- Post-mortem trace analysis

Diagnostics are intended to reduce bring-up time,
manufacturing cost, and field failure rates.

---

## 9. Non-Goals

ORBIT-G1 explicitly does not aim to:
- Replace a GPU
- Perform graphics rendering
- Execute complete ML graphs autonomously
- Provide OS-level scheduling

It is a cooperative accelerator designed to work
alongside existing system components.

---

## 10. Design Philosophy

ORBIT-G1 prioritizes:
- Verifiability over peak theoretical performance
- Utility and product value beyond AI benchmarks
- Simple, deterministic hardware behavior
- Clear separation of concerns between hardware and software

This philosophy guides all design decisions.
