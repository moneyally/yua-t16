 # ORBIT Descriptor Specification v1

This document defines the binary descriptor format used to control
execution on the ORBIT-G1 accelerator.

Descriptors are fixed-size, host-generated command structures
consumed by the ORBIT-G1 command processor.

This specification is the single source of truth (SSOT)
for hardware, firmware, driver, runtime, and verification.

---

## 0. Design Principles

- No general-purpose ISA
- No speculative execution
- No implicit ordering
- Deterministic, descriptor-driven execution
- Explicit synchronization via barriers
- Fixed-size descriptors for simple parsing

All descriptors are exactly 64 bytes in size.

---

## 1. Descriptor Header Layout (Common)

All descriptors begin with a common header.

```c
struct orbit_desc_header {
  uint8_t  type;        // Descriptor type ID
  uint8_t  flags;       // Control flags
  uint16_t reserved0;   // Must be zero
  uint32_t length;      // Operation length (bytes, elements, or tiles)
  uint64_t next_desc;   // Optional pointer to next descriptor (0 = end)
};
Header Rules
type selects the execution unit

flags may control debug, priority, or mode

length semantics depend on descriptor type

next_desc allows linked-list execution

Hardware may ignore next_desc in v1

2. Descriptor Type Enumeration
Type ID	Name	Description
0x01	DMA_2D	2D memory copy
0x02	GEMM_INT8	INT8 GEMM via YUA-T16
0x03	VECTOR_OP	Elementwise or reduction operation
0x04	COPY_2D_PLUS	SUP enhanced copy
0x05	FORMAT_CONVERT	SUP format conversion
0x06	FRAME_FINGERPRINT	SUP hash / integrity check
0x07	BARRIER	Explicit synchronization
0x08	EVENT	Interrupt or signal
0x09	PERF_SNAPSHOT	Performance counter capture
Descriptors with unknown type IDs are ignored or may raise an error.

3. DMA_2D Descriptor (Type 0x01)
struct orbit_desc_dma_2d {
  orbit_desc_header h;

  uint64_t src_addr;
  uint64_t dst_addr;

  uint32_t width_bytes;
  uint32_t height;

  uint32_t src_stride;
  uint32_t dst_stride;

  uint32_t reserved1;
  uint32_t reserved2;
};
Semantics
Copies a 2D region line by line

width_bytes per line

height number of lines

Strides are in bytes

Source and destination must not overlap unless explicitly supported

4. GEMM_INT8 Descriptor (Type 0x02)
struct orbit_desc_gemm_int8 {
  orbit_desc_header h;

  uint64_t act_addr;   // A tile base address
  uint64_t wgt_addr;   // B tile base address
  uint64_t out_addr;   // C tile output address

  uint32_t Kt;         // K dimension (accumulation length)
  uint16_t m_tiles;    // Fixed to 1 in v1
  uint16_t n_tiles;    // Fixed to 1 in v1

  uint32_t scale_a;    // Reserved (ignored in v1)
  uint32_t scale_b;    // Reserved (ignored in v1)

  uint32_t epilogue;   // Epilogue flags (reserved)
  uint32_t reserved;
};
Semantics
Executes one 16x16 INT8 GEMM tile

Accumulation is INT32

No bias, activation, or scaling in v1

Epilogue field is reserved for future fusion

Global M/N tiling is handled by the host

5. VECTOR_OP Descriptor (Type 0x03)
struct orbit_desc_vector_op {
  orbit_desc_header h;

  uint64_t src_addr;
  uint64_t dst_addr;

  uint32_t element_count;
  uint16_t op_type;     // ADD, MUL, MAX, CLAMP, etc.
  uint16_t data_type;   // INT8, INT16, FP16

  uint32_t imm;         // Immediate or scalar
  uint32_t reserved;
};
Semantics
Performs elementwise or reduction operations

Exact supported ops are implementation-defined

Used for lightweight post-processing

6. COPY_2D_PLUS Descriptor (Type 0x04)
struct orbit_desc_copy_2d_plus {
  orbit_desc_header h;

  uint64_t src_addr;
  uint64_t dst_addr;

  uint32_t width_bytes;
  uint32_t height;

  uint32_t src_stride;
  uint32_t dst_stride;

  uint32_t options;     // Crop, mirror, pad
  uint32_t reserved;
};
Used by SUP for enhanced data movement.

7. FORMAT_CONVERT Descriptor (Type 0x05)
struct orbit_desc_format_convert {
  orbit_desc_header h;

  uint64_t src_addr;
  uint64_t dst_addr;

  uint32_t element_count;
  uint16_t src_format;
  uint16_t dst_format;

  uint32_t options;
  uint32_t reserved;
};
Examples
RGB → YUV

FP16 → INT8

INT8 → FP16

8. FRAME_FINGERPRINT Descriptor (Type 0x06)
struct orbit_desc_frame_fingerprint {
  orbit_desc_header h;

  uint64_t src_addr;
  uint64_t result_addr;

  uint32_t byte_count;
  uint32_t hash_type;   // CRC32, CRC64, etc.

  uint64_t reserved;
};
Used for integrity checking, diagnostics, and assurance.

9. BARRIER Descriptor (Type 0x07)
struct orbit_desc_barrier {
  orbit_desc_header h;

  uint64_t reserved1;
  uint64_t reserved2;
  uint64_t reserved3;
};
Semantics
All previous descriptors must complete before subsequent ones execute

Required for explicit ordering

No implicit synchronization exists without barriers

10. EVENT Descriptor (Type 0x08)
struct orbit_desc_event {
  orbit_desc_header h;

  uint32_t event_id;
  uint32_t options;

  uint64_t reserved1;
  uint64_t reserved2;
};
Used to signal host interrupts or notifications.

11. PERF_SNAPSHOT Descriptor (Type 0x09)
struct orbit_desc_perf_snapshot {
  orbit_desc_header h;

  uint64_t dst_addr;   // Where to write counters

  uint64_t reserved1;
  uint64_t reserved2;
};
Captures performance counters for profiling and debugging.

12. Execution and Ordering Rules
Descriptors execute in submission order

No implicit dependency tracking

Barriers define ordering boundaries

Errors are reported via status registers or events

Partial completion is not visible to software

13. Error Handling
Invalid descriptor types may be ignored or faulted

Address alignment violations may fault

Debug flags may enable stricter checking

Exact error reporting is implementation-defined.

14. Versioning
This document defines Descriptor Specification v1.

Any incompatible changes require a new major version.



