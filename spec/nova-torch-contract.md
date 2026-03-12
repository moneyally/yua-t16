# nova_torch — PyTorch Backend for ORBIT-G1
## Contract Specification v0.1 (SSOT)

**Status:** Pre-1.0 — ABI not frozen except where noted
**Hardware target:** ORBIT-G1 (ORBIT-G1 v2 architecture, Descriptor Spec v1 + v2 extensions)
**nova_core C API:** v0.1 (`nova.h` / `nova_types.h`)
**Last updated:** 2026-03-13

This document is the single source of truth for implementing `nova_torch`.
A developer reading only this document should be able to produce a complete,
correct implementation without consulting any other file.

---

## Table of Contents

1. Stack Overview
2. Device Registration Contract
3. Allocator Interface Contract
4. ATen → ORBIT Kernel Mapping Table
5. ORBIT Kernel ABI Specification
6. Descriptor Emission Contract
7. Version and Compatibility Matrix
8. Implementation Order

---

## Section 1: Stack Overview

```
PyTorch user code
  (torch.tensor(..., device="orbit:0"), model.to("orbit"), torch.mm, F.silu, ...)
  ↓
nova_torch  (Python extension module, PrivateUse1 backend)
  ├── OrbitGuardImpl       (c10::impl::DeviceGuardImplInterface)
  ├── OrbitAllocator       (at::Allocator)
  └── ATen kernel dispatch (TORCH_LIBRARY_IMPL)
  ↓  ATen dispatch (PrivateUse1 key)
orbit kernel library
  ├── orbitBLAS            (orbit_gemm, orbit_gemm_batched, orbit_gemm_bias)
  ├── orbitNN              (orbit_layernorm, orbit_rmsnorm, orbit_silu, orbit_gelu,
  │                         orbit_relu, orbit_add, orbit_mul, orbit_softmax)
  ├── orbitAttention       (orbit_attention_fwd)
  └── orbitMoE             (orbit_moe_route)
  ↓  calls nova C API
nova_core runtime  (libnova_core.so)
  ├── novaLaunchDescriptor()  — single descriptor enqueue
  ├── novaLaunchBatch()       — atomic multi-descriptor enqueue
  ├── novaMalloc / novaFree
  ├── novaMallocHost / novaFreeHost
  ├── novaMemcpy / novaMemcpyAsync
  └── novaStreamCreate / novaStreamSynchronize
  ↓  descriptor queue (nova_descriptor_t / nova_stream_t)
liborbit.so  (kernel driver userspace interface)
  ↓  ioctl ORBIT_IOC_SUBMIT_DESC
orbit_g1.ko  (Linux kernel module)
  ↓  PCIe descriptor DMA (Gen4 x16)
ORBIT-G1 hardware
  ├── Command Processor (Descriptor Queue × 4)
  ├── Compute Clusters  (N × YUA-T16 tiles, INT8 + INT4 GEMM)
  ├── VPU               (256-wide SIMD: RMSNorm, SiLU, RoPE, Softmax, Residual)
  ├── KVC Controller    (KV-Cache in GDDR6, PagedAttention-style)
  ├── MoE Router        (top-k selection, gather/scatter)
  └── Global Memory     (GDDR6 16 GB / 32 GB — weights + KV-cache)
```

### Simulation Mode

When `NOVA_SIMULATION=1` is set or no ORBIT-G1 hardware is present, `novaInit`
enters simulation mode. All descriptor launches execute synchronously on the CPU.
`nova_device_info_t.is_simulation == 1`. `nova_torch` requires no code changes;
the same dispatch path is exercised.

---

## Section 2: Device Registration Contract

### 2.1 Backend Name and Registration

| Property | Value |
|----------|-------|
| Backend name | `"orbit"` |
| PyTorch device type | `c10::DeviceType::PrivateUse1` |
| Registration call | `c10::register_privateuse1_backend("orbit")` |
| Python rename call | `torch.utils.rename_privateuse1_backend("orbit")` |
| Module name | `nova_torch` (Python import triggers registration) |

### 2.2 OrbitGuardImpl — Full C++ Class Contract

`OrbitGuardImpl` is registered as the `DeviceGuardImplInterface` for
`c10::DeviceType::PrivateUse1`. It delegates all state to the nova_core
device management API (`novaSetDevice`, `novaGetDevice`).

```cpp
// File: nova_torch/csrc/orbit_guard_impl.h

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/Stream.h>
#include <nova/nova.h>

// Thread-local default stream per device.
// Index: device index (0-based). Size: ORBIT_MAX_DEVICES (8).
extern thread_local nova_stream_t g_orbit_default_stream[8];

struct OrbitGuardImpl final : public c10::impl::DeviceGuardImplInterface {

  // Returns DeviceType::PrivateUse1.
  c10::DeviceType type() const override {
    return c10::DeviceType::PrivateUse1;
  }

  // Returns the currently active orbit device as c10::Device.
  // Calls novaGetDevice(&id) and wraps as Device(PrivateUse1, id).
  c10::Device getDevice() const override {
    int id = 0;
    novaGetDevice(&id);
    return c10::Device(c10::DeviceType::PrivateUse1, static_cast<c10::DeviceIndex>(id));
  }

  // Sets the active orbit device.
  // Calls novaSetDevice(d.index()).
  // Throws std::runtime_error if novaSetDevice returns non-NOVA_SUCCESS.
  void setDevice(c10::Device d) const override {
    nova_status_t s = novaSetDevice(static_cast<int>(d.index()));
    TORCH_CHECK(s == NOVA_SUCCESS,
      "novaSetDevice(", d.index(), ") failed: ", novaGetErrorString(s));
  }

  // Identical to setDevice. Required by interface.
  void uncheckedSetDevice(c10::Device d) const noexcept override {
    novaSetDevice(static_cast<int>(d.index()));
  }

  // Returns the default stream for the given device.
  // Stream ID is encoded as: (device_index << 16) | stream_slot.
  // Default stream slot = 0.
  c10::Stream getStream(c10::Device d) const noexcept override {
    return c10::Stream(
      c10::Stream::UNSAFE,
      d,
      static_cast<c10::StreamId>(d.index() << 16 | 0)
    );
  }

  // Returns a new non-default stream for the device.
  // Calls novaStreamCreate and wraps in c10::Stream.
  // StreamId encodes the nova_stream_t pointer cast to int64_t.
  c10::Stream getNewStream(c10::Device d, int priority = 0) const override {
    nova_stream_t s = nullptr;
    nova_status_t st = novaStreamCreate(&s);
    TORCH_CHECK(st == NOVA_SUCCESS, "novaStreamCreate failed: ", novaGetErrorString(st));
    return c10::Stream(
      c10::Stream::UNSAFE,
      d,
      static_cast<c10::StreamId>(reinterpret_cast<intptr_t>(s))
    );
  }

  // Swaps in `requested_stream` as the current stream on its device.
  // Returns the previous current stream.
  // Thread-local current stream per device is tracked in g_orbit_current_stream[].
  c10::Stream exchangeStream(c10::Stream requested_stream) const noexcept override;

  // Returns the number of orbit devices visible to the process.
  c10::DeviceIndex deviceCount() const noexcept override {
    int count = 0;
    novaGetDeviceCount(&count);
    return static_cast<c10::DeviceIndex>(count);
  }

  // Calls novaStreamSynchronize on the stream associated with the event.
  // In nova_torch v0.1 events are not fully implemented; this is a no-op.
  void record(void** event, const c10::Stream& stream,
               const c10::DeviceIndex device_index,
               const c10::EventFlag flag) const override { /* TODO v0.2 */ }

  void block(void* event, const c10::Stream& stream) const override { /* TODO v0.2 */ }

  bool queryEvent(void* event) const override { return true; }

  void destroyEvent(void* event, const c10::DeviceIndex device_index)
    const noexcept override { /* TODO v0.2 */ }
};
```

### 2.3 Module Initialization Sequence

```cpp
// File: nova_torch/csrc/module.cpp
// Called when Python executes: import nova_torch

PYBIND11_MODULE(nova_torch, m) {
  // 1. Initialize nova runtime (idempotent after first call).
  nova_status_t s = novaInit(0);
  TORCH_CHECK(s == NOVA_SUCCESS || s == NOVA_ERROR_UNINITIALIZED == false,
    "novaInit failed: ", novaGetErrorString(s));

  // 2. Register PrivateUse1 backend name.
  c10::register_privateuse1_backend("orbit");

  // 3. Register OrbitGuardImpl.
  c10::impl::register_privateuse1_device_guard_impl(
    c10::DeviceType::PrivateUse1,
    c10::impl::make_static_device_guard_impl<OrbitGuardImpl>()
  );

  // 4. Register OrbitAllocator.
  at::RegisterAllocator(c10::DeviceType::PrivateUse1, &g_orbit_allocator, 100);

  // 5. Expose Python rename so user can write: torch.device("orbit:0")
  //    Called once: torch.utils.rename_privateuse1_backend("orbit")
  //    (nova_torch does this automatically; user need not call it again.)
}
```

Python usage after `import nova_torch`:

```python
import torch
import nova_torch           # triggers registration above

x = torch.empty(4, 4, dtype=torch.float16, device="orbit:0")
y = torch.mm(x, x)
z = y.cpu()
```

---

## Section 3: Allocator Interface Contract

### 3.1 OrbitAllocator C++ Class

```cpp
// File: nova_torch/csrc/orbit_allocator.h

#include <ATen/Allocator.h>
#include <nova/nova.h>

class OrbitAllocator final : public at::Allocator {
public:
  // Allocate `nbytes` bytes in ORBIT-G1 GDDR6 device memory.
  // Returns at::DataPtr with device context set to PrivateUse1.
  // Throws c10::Error (TORCH_CHECK) on OOM.
  at::DataPtr allocate(size_t nbytes) override;

  // Returns the raw delete function pointer used to release device memory.
  // The returned function calls novaFree(ptr).
  at::DeleterFnPtr raw_deleter() const override;

  // Copies `count` bytes from `src` to `dest`, both on the orbit device.
  // Uses novaMemcpy with NOVA_MEMCPY_D2D.
  void copy_data(void* dest, const void* src, std::size_t count) const override;
};

// Singleton. Registered in module init.
extern OrbitAllocator g_orbit_allocator;
```

### 3.2 Allocator Function Table

Every function below is defined in `nova.h` / `nova_types.h` (nova_core v0.1).

| Function | Signature | Behavior | Error handling |
|----------|-----------|----------|----------------|
| `novaMalloc` | `nova_status_t novaMalloc(nova_ptr_t* ptr, size_t bytes)` | Allocates `bytes` from GDDR6 device memory via kernel buddy allocator. Sets `*ptr` to device-side address. | Returns `NOVA_ERROR_OUT_OF_MEMORY` if pool exhausted. In sim mode, uses `aligned_alloc(64, bytes)`. |
| `novaFree` | `nova_status_t novaFree(nova_ptr_t ptr)` | Returns allocation to buddy allocator. | Returns `NOVA_ERROR_INVALID_ARGS` if `ptr` is not a live allocation. Must not double-free. |
| `novaMemcpy` (H2D) | `nova_status_t novaMemcpy(nova_ptr_t dst, const void* src, size_t bytes, NOVA_MEMCPY_H2D)` | Emits `DMA_2D` descriptor: `width_bytes=bytes, height=1, src_stride=0, dst_stride=0`. Synchronous — blocks until DMA completes on the default stream. | Returns `NOVA_ERROR_DEVICE` on PCIe DMA fault. |
| `novaMemcpy` (D2H) | `nova_status_t novaMemcpy(void* dst, nova_ptr_t src, size_t bytes, NOVA_MEMCPY_D2H)` | Reverse `DMA_2D`. Synchronous. | Same as H2D. |
| `novaMemcpy` (D2D) | `nova_status_t novaMemcpy(nova_ptr_t dst, nova_ptr_t src, size_t bytes, NOVA_MEMCPY_D2D)` | In-device copy via `DMA_2D` descriptor. Same device only. | Returns `NOVA_ERROR_INVALID_ARGS` if addresses are on different devices. |
| `novaMemcpyAsync` | `nova_status_t novaMemcpyAsync(nova_ptr_t dst, const void* src, size_t bytes, nova_memcpy_kind_t kind, nova_stream_t stream)` | Non-blocking. Enqueues `DMA_2D` on `stream`. Returns immediately. | Same fault codes as sync variant. |
| `novaMallocHost` | `nova_status_t novaMallocHost(void** ptr, size_t bytes)` | Allocates pinned (page-locked) host memory via `dma_alloc_coherent` (kernel ORBIT_IOC_ALLOC_MEM with PINNED flag). PCIe-accessible. | Returns `NOVA_ERROR_OUT_OF_MEMORY` if system locked memory exhausted. |
| `novaFreeHost` | `nova_status_t novaFreeHost(void* ptr)` | Releases pinned allocation. Must match pointer returned by `novaMallocHost`. | Returns `NOVA_ERROR_INVALID_ARGS` if ptr is not a pinned allocation. |
| `novaMemset` | `nova_status_t novaMemset(nova_ptr_t ptr, int value, size_t bytes)` | Fills `bytes` bytes at device address `ptr` with `(uint8_t)value`. Emits `NOVA_OP_MEMSET` descriptor. | Returns `NOVA_ERROR_INVALID_ARGS` on null ptr. |

### 3.3 OrbitAllocator::allocate Implementation Contract

```
1. Call novaMalloc(&raw_ptr, nbytes).
2. If status == NOVA_ERROR_OUT_OF_MEMORY:
     throw c10::OutOfMemoryError("OrbitAllocator: OOM requesting " + nbytes + " bytes");
3. If status != NOVA_SUCCESS:
     throw c10::Error("novaMalloc failed: " + novaGetErrorString(status));
4. Wrap raw_ptr in at::DataPtr:
     - data:    reinterpret_cast<void*>(raw_ptr)
     - ctx:     reinterpret_cast<void*>(raw_ptr)  // stored for deletion
     - deleter: [](void* ctx) { novaFree(reinterpret_cast<nova_ptr_t>(ctx)); }
     - device:  c10::Device(PrivateUse1, current_device_index)
5. Return DataPtr.
```

---

## Section 4: ATen → ORBIT Kernel Mapping Table

Dispatch key: `c10::DispatchKey::PrivateUse1`
Registration macro: `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)`

| ATen op | ORBIT kernel / nova API | dtype support | Shape contract | Status | Fallback |
|---------|------------------------|---------------|----------------|--------|----------|
| `aten::mm` | `orbit_gemm` | fp16, int8 | `[M,K] x [K,N] → [M,N]`; M,N,K multiples of 16 (pad internally) | IMPLEMENTED | None — raises on wrong device |
| `aten::bmm` | `orbit_gemm_batched` | fp16, int8 | `[B,M,K] x [B,K,N] → [B,M,N]` | IMPLEMENTED | None |
| `aten::addmm` | `orbit_gemm_bias` | fp16 | `bias[N] + [M,K]x[K,N] → [M,N]` | IMPLEMENTED | None |
| `aten::linear` | `orbit_gemm_bias` | fp16 | input `[*, in]`, weight `[out, in]`, optional bias `[out]` | IMPLEMENTED | None |
| `aten::matmul` | `orbit_gemm` (shape-normalized) | fp16, int8 | Handles 1D/2D/3D/4D via reshape to `[M,K]x[K,N]` then reshape back | IMPLEMENTED | None |
| `aten::scaled_dot_product_attention` | `orbit_attention_fwd` | fp16 | Q/K/V: `[B,H,S,d]`; causal mask supported | IMPLEMENTED | None |
| `aten::_scaled_dot_product_flash_attention` | `orbit_attention_fwd` | fp16 | Same as above; flash variant selects tiled softmax path | IMPLEMENTED | None |
| `aten::layer_norm` | CPU fallback | fp16 | Any shape; last `normalized_shape` dims | UNIMPLEMENTED — see §5.4 | CPU fallback (D2H → layer_norm → H2D); logs SLOW_FALLBACK |
| `aten::native_layer_norm` | CPU fallback | fp16 | Same as layer_norm | UNIMPLEMENTED | Same |
| `aten::rms_norm` | `orbit_rmsnorm` | fp16 | `[*, d_model]`, weight `[d_model]` | IMPLEMENTED | None |
| `aten::silu` | `orbit_silu` (via `novaSiLU`) | fp16 | Any contiguous shape; n = total elements | IMPLEMENTED | None |
| `aten::silu_` | `orbit_silu` in-place | fp16 | Same | IMPLEMENTED | None |
| `aten::gelu` | `orbit_gelu` | fp16 | Any contiguous | IMPLEMENTED (VPU LUT-based) | None |
| `aten::relu` | `orbit_relu` | fp16, int8 | Any contiguous | CPU fallback acceptable (low priority) | CPU fallback |
| `aten::add.Tensor` | `orbit_add` (via `NOVA_EW_ADD`) | fp16 | Same shape or broadcast (1D broadcast only in v0.1) | IMPLEMENTED (no-broadcast path) | CPU fallback for broadcast shapes |
| `aten::mul.Tensor` | `orbit_mul` (via `NOVA_EW_MUL`) | fp16 | Same shape | IMPLEMENTED | CPU fallback for broadcast |
| `aten::softmax.int` | `orbit_softmax` (via `novaSoftmax`) | fp16 | `[*, N]`, last dim softmax; N ≤ 65535 | IMPLEMENTED | Raises RuntimeError if N > 65535 |
| `aten::_softmax` | `orbit_softmax` | fp16 | Same | IMPLEMENTED | Same |
| `aten::copy_` | `novaMemcpy` (H2D / D2H / D2D) | all | Contiguous tensors only in v0.1 | IMPLEMENTED | None |
| `aten::empty.memory_format` | `novaMalloc` via `OrbitAllocator` | all | Any shape | IMPLEMENTED | None |
| `aten::zeros.names` | `novaMalloc` + `novaMemset(0)` | all | Any shape | IMPLEMENTED | None |
| `aten::ones` | `novaMalloc` + fill kernel | fp16 | Any shape | IMPLEMENTED (VPU fill) | CPU fallback |
| `aten::to.device` | `novaMemcpy` H2D or D2H | all | Contiguous | IMPLEMENTED | None |
| `aten::_to_copy` | `novaMemcpy` H2D or D2H | all | Contiguous | IMPLEMENTED | None |

### Fallback Policy

**Registered but unimplemented ops (CPU fallback):**
1. Detect tensor is on `orbit` device.
2. Copy all input tensors D2H: `novaMemcpy(..., NOVA_MEMCPY_D2H)`.
3. Call the equivalent CPU ATen kernel.
4. Copy output tensor H2D: `novaMemcpy(..., NOVA_MEMCPY_H2D)`.
5. Log: `[SLOW_FALLBACK] aten::<op_name> fell back to CPU`.

**Hard-error ops (no fallback):**
- In-place ops on an orbit tensor where the fallback would silently produce
  wrong aliasing (e.g., `aten::resize_` that changes byte size of a device
  allocation that is aliased elsewhere).
- Action: `throw std::runtime_error("aten::<op> on orbit device: not supported, cannot safely fall back")`.

**Unregistered ops:**
- PyTorch will dispatch to CPU via the existing fallback mechanism only if the
  tensor can be implicitly moved. If implicit movement is not possible, PyTorch
  raises `RuntimeError: Could not run '<op>' with arguments from the 'PrivateUse1'
  backend`. This is acceptable for v0.1.

---

## Section 5: ORBIT Kernel ABI Specification

All kernels below are C functions exported from `libnova_torch_kernels.so`.
They emit descriptors via `novaLaunchDescriptor` / `novaLaunchBatch`.
All memory pointers are `nova_ptr_t` (device-side 64-bit addresses).
All layouts are row-major unless stated otherwise.
All dtypes are `nova_dtype_t` values as defined in `nova_types.h`.

---

### 5.0 Common Execution Contract (applies to ALL kernels)

#### Descriptor Emission Flow

Every kernel call follows this exact pipeline:

```
orbit_kernel(stream, args...)
  │
  ├─ 1. Validate inputs (shape, dtype, alignment)
  │      → returns NOVA_ERROR_INVALID_ARGUMENT on failure
  │      → NO descriptor emitted on error
  │
  ├─ 2. Compute tiling (tile count = ceil(M/16) * ceil(N/16) etc.)
  │
  ├─ 3. For each tile / descriptor:
  │       nova_descriptor_t d = {0};
  │       d.type   = NOVA_OP_GEMM_INT8;       // descriptor type ID
  │       d.flags  = NOVA_FLAG_NONE;
  │       d.fields = { .act_addr, .wgt_addr, .out_addr, .Kt, ... };
  │       novaLaunchDescriptor(stream, &d);    // push to stream FIFO
  │
  ├─ 4. novaLaunchDescriptor (inside nova_core):
  │       orbit_stream_push(stream, &d);       // append to stream's pending list
  │       // does NOT submit to kernel yet — batched at novaStreamFlush
  │
  ├─ 5. novaStreamFlush(stream) OR novaStreamSynchronize(stream):
  │       ioctl(fd, ORBIT_IOC_SUBMIT_DESC, batch)  // kernel driver
  │       orbit_g1_queue_enqueue(q, descs, n)       // ring buffer push
  │       writel(doorbell, q->doorbell_reg)          // PCIe doorbell
  │       ORBIT-G1 command processor picks up desc
  │       hardware executes
  │
  └─ 6. Completion:
         MSI-X interrupt fires → orbit_queue_complete() → done_cookie updated
         novaStreamSynchronize(): blocks until done_cookie >= submitted cookie
```

#### Synchronization Behavior

| Call | Blocks caller? | When hardware finishes |
|------|---------------|----------------------|
| `orbit_kernel(stream, ...)` | **No** — async enqueue | Descriptor in stream FIFO, not yet submitted |
| `novaStreamFlush(stream)` | No | Descriptors submitted to hardware ring |
| `novaStreamSynchronize(stream)` | **Yes** — blocks until all descriptors on stream complete | After MSI-X interrupt, done_cookie ≥ cookie |
| `novaDeviceSynchronize()` | Yes — blocks all streams | All queues drained |

**Default stream behavior:** If `stream = NOVA_STREAM_DEFAULT`, kernel is submitted and flushed synchronously (blocks). Use explicit streams for async overlap.

#### Memory Ownership Contract

| Buffer role | Who allocates | Who frees | Lifetime |
|-------------|--------------|-----------|---------|
| Input tensor (A, src, Q, ...) | Caller | Caller | Must remain valid until `novaStreamSynchronize(stream)` returns |
| Output tensor (C, dst, out, ...) | Caller | Caller | Written by hardware; valid after `novaStreamSynchronize(stream)` |
| Workspace | Caller provides ptr | Caller | Must remain valid for duration of kernel execution |
| Internal scratch (KVC cache pages) | KVCacheManager | KVCacheManager | Managed by session; freed on session destroy |
| Descriptor memory | nova_core (DMA coherent) | nova_core | Freed after hardware acknowledges completion |

**Rules:**
- Caller MUST NOT read output buffers before `novaStreamSynchronize(stream)`.
- Caller MUST NOT free input buffers before `novaStreamSynchronize(stream)`.
- In-place operations (`src == dst`) are allowed ONLY if the kernel explicitly states so.
- Overlapping input/output regions (aliasing) produce undefined behavior unless kernel states otherwise.

---

### 5.1 orbit_gemm

Computes `C = A @ B` (no bias, no transpose by default).

```c
// File: orbit_kernels/orbitBLAS.h

nova_status_t orbit_gemm(
  nova_stream_t   stream,
  nova_ptr_t      A,         // [M, K] row-major
  nova_ptr_t      B,         // [K, N] row-major
  nova_ptr_t      C,         // [M, N] row-major, output
  int             M,
  int             K,
  int             N,
  nova_dtype_t    dtype_A,   // NOVA_DTYPE_FLOAT16 | NOVA_DTYPE_INT8
  nova_dtype_t    dtype_B,   // NOVA_DTYPE_FLOAT16 | NOVA_DTYPE_INT8 | NOVA_DTYPE_UINT8 (INT4 packed)
  nova_dtype_t    dtype_C    // NOVA_DTYPE_FLOAT16 | NOVA_DTYPE_INT32
);
```

| Property | Value |
|----------|-------|
| Valid dtype combos | fp16×fp16→fp16; int8×int8→int32; int8×int8→fp16 (with INT32 accumulate + cvt) |
| Internal accumulate | Always INT32 (GEMM_INT8 descriptor), converted to dtype_C on output |
| Hardware tile | 16×16 (YUA-T16 v1). M and N must be multiples of 16. |
| Padding policy | If M or N not multiple of 16: kernel zero-pads internally, crops output. Caller MAY pre-pad to avoid overhead. |
| K constraint | No alignment constraint on K in v0.1 (software accumulation loop) |
| Workspace | None |
| Descriptor emitted | `NOVA_OP_GEMM_INT8` (0x02) per tile. Total tiles = `ceil(M/16) * ceil(N/16) * ceil(K/16)` |
| Synchronization | Async. Returns immediately after descriptor enqueue. Output C valid only after `novaStreamSynchronize(stream)`. |
| Memory ownership | A, B: read-only, caller owns, must stay valid until stream sync. C: write-only output, caller owns, valid after sync. |
| In-place | NOT allowed (A==C or B==C → NOVA_ERROR_INVALID_ARGUMENT) |

### 5.2 orbit_gemm_batched

Computes `C[b] = A[b] @ B[b]` for each batch index `b`.

```c
nova_status_t orbit_gemm_batched(
  nova_stream_t   stream,
  nova_ptr_t      A,         // [B, M, K] row-major, contiguous
  nova_ptr_t      B,         // [B, K, N] row-major, contiguous
  nova_ptr_t      C,         // [B, M, N] row-major, output
  int             batch,
  int             M,
  int             K,
  int             N,
  nova_dtype_t    dtype_A,
  nova_dtype_t    dtype_B,
  nova_dtype_t    dtype_C
);
```

Implementation: loop over batch dimension, call `orbit_gemm` for each slice.
Uses `novaLaunchBatch` to submit all batch descriptors atomically.

### 5.3 orbit_gemm_bias

Computes `C = A @ B + bias` where `bias` is broadcast across M rows.

```c
nova_status_t orbit_gemm_bias(
  nova_stream_t   stream,
  nova_ptr_t      A,         // [M, K]
  nova_ptr_t      B,         // [K, N]
  nova_ptr_t      bias,      // [N] or NULL (no bias)
  nova_ptr_t      C,         // [M, N] output
  int             M,
  int             K,
  int             N,
  float           alpha,     // scalar multiplier on A@B (pass 1.0 for standard linear)
  nova_dtype_t    dtype
);
```

Implementation:
1. Emit GEMM descriptors for `A @ B → C` (same as orbit_gemm).
2. If `bias != NULL`: emit `NOVA_OP_ELEMENTWISE` (NOVA_EW_ADD) to broadcast-add bias[N] across C[M,N].
   This requires a row-wise VPU loop: M iterations of `VECTOR_OP` on N elements.
   Alternatively, if hardware supports epilogue fusion in future: set `epilogue` field.

### 5.4 orbit_softmax

```c
nova_status_t orbit_softmax(
  nova_stream_t   stream,
  nova_ptr_t      src,       // [outer, N] FP16, last dim is softmax dim
  nova_ptr_t      dst,       // [outer, N] FP16 output
  int             outer,     // product of all leading dims (batch * heads * seq)
  int             N,         // softmax dimension size
  nova_dtype_t    dtype      // NOVA_DTYPE_FLOAT16 only
);
```

| Property | Value |
|----------|-------|
| Valid dtype | NOVA_DTYPE_FLOAT16 only |
| N constraint | N ≤ 65535 (`vec_len` field is uint16 in SOFTMAX descriptor) |
| Numerically stable | max-shift implemented in VPU (max → subtract → exp → sum → div) |
| Descriptor emitted | `NOVA_OP_SOFTMAX` (0x0F), one per row (outer iterations via `novaLaunchBatch`) |
| Workspace | None |
| Synchronization | Async. src readable until stream sync. dst valid after stream sync. |
| Memory ownership | src: read-only, caller owns. dst: write-only output, caller owns. |
| In-place | Allowed (src == dst). VPU reads full row before writing. |

Wraps: `novaSoftmax(stream, src, dst, outer, N, /*dim=*/1)`

### 5.5 orbit_rmsnorm

```c
nova_status_t orbit_rmsnorm(
  nova_stream_t   stream,
  nova_ptr_t      src,       // [outer, d_model] FP16
  nova_ptr_t      weight,    // [d_model] FP16 scale weights
  nova_ptr_t      dst,       // [outer, d_model] FP16
  int             outer,     // batch * seq
  int             d_model,
  float           eps,       // typically 1e-5
  nova_dtype_t    dtype      // NOVA_DTYPE_FLOAT16 only
);
```

| Property | Value |
|----------|-------|
| Algorithm | RMSNorm: `out = x / sqrt(mean(x²) + eps) * weight` (no mean subtraction) |
| VPU passes | Pass 1: `sum_sq = sum(x²)` over d_model. Pass 2: `out = x * weight / sqrt(sum_sq/d_model + eps)` |
| Descriptor emitted | `VECTOR_OP_EX` (0x0D, subtype=RMSNORM), one per row via `novaLaunchBatch` |
| Valid dtype | NOVA_DTYPE_FLOAT16 only |
| Workspace | None (VPU accumulates sum_sq in scalar register) |
| Synchronization | Async. src/weight readable until stream sync. dst valid after stream sync. |
| Memory ownership | src: read-only, caller owns. weight: read-only, caller owns. dst: write-only, caller owns. |
| In-place | Allowed (src == dst). |

Wraps: `novaRMSNorm(stream, src, weight, dst, d_model, eps)` for single-row.
For batched: loop `outer` times or emit `outer` descriptors via `novaLaunchBatch`.

### 5.6 orbit_layernorm

```c
nova_status_t orbit_layernorm(
  nova_stream_t   stream,
  nova_ptr_t      src,       // [outer, d_model] FP16
  nova_ptr_t      weight,    // [d_model] FP16 or NULL
  nova_ptr_t      bias,      // [d_model] FP16 or NULL
  nova_ptr_t      dst,       // [outer, d_model] FP16
  int             outer,
  int             d_model,
  float           eps        // typically 1e-5
);
```

| Property | Value |
|----------|-------|
| Algorithm | `out = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias` |
| VPU requirement | 3 passes: (1) compute mean, (2) compute variance, (3) normalize + scale + shift |
| Hardware status | **UNIMPLEMENTED** — `vpu_core.sv` implements 2-pass RMSNorm but not 3-pass LayerNorm |
| Descriptor | TBD — needs new `VECTOR_OP_EX` subtype (LAYERNORM) or 3-descriptor sequence |
| Fallback | CPU fallback: D2H copy → `at::layer_norm` CPU → H2D copy. Logs `[SLOW_FALLBACK] orbit_layernorm`. |
| When to implement | After VPU 3-pass RTL is added; coordinate with hardware team |

### 5.7 orbit_silu

```c
nova_status_t orbit_silu(
  nova_stream_t   stream,
  nova_ptr_t      src,       // [n] FP16, contiguous
  nova_ptr_t      dst,       // [n] FP16 output
  int             n,         // total element count
  nova_dtype_t    dtype      // NOVA_DTYPE_FLOAT16 only
);
```

| Property | Value |
|----------|-------|
| Algorithm | `out[i] = src[i] * sigmoid(src[i])` = `src[i] / (1 + exp(-src[i]))` |
| VPU impl | LUT-based sigmoid (256-entry FP16 LUT) + elementwise multiply |
| Descriptor emitted | `NOVA_OP_SILU` (type 0x04 per nova_types.h) |
| Workspace | None |

Wraps: `novaSiLU(stream, src, dst, n)`.

### 5.8 orbit_gelu

```c
nova_status_t orbit_gelu(
  nova_stream_t   stream,
  nova_ptr_t      src,       // [n] FP16
  nova_ptr_t      dst,       // [n] FP16
  int             n,
  nova_dtype_t    dtype      // NOVA_DTYPE_FLOAT16 only
);
```

| Property | Value |
|----------|-------|
| Algorithm | `out[i] = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x³)))` (tanh-GELU) |
| VPU impl | LUT-based tanh + poly approximation via `VECTOR_OP_EX` |
| Descriptor | `VECTOR_OP_EX` with `op_subtype=GELU` |
| Fallback | CPU fallback if op_subtype not yet in VPU firmware |

### 5.9 orbit_add / orbit_mul

```c
nova_status_t orbit_elementwise(
  nova_stream_t          stream,
  nova_ptr_t             A,     // [n] FP16
  nova_ptr_t             B,     // [n] FP16
  nova_ptr_t             out,   // [n] FP16
  int                    n,
  nova_elementwise_op_t  op,    // NOVA_EW_ADD | NOVA_EW_MUL | NOVA_EW_SUB | NOVA_EW_DIV
  nova_dtype_t           dtype  // NOVA_DTYPE_FLOAT16 | NOVA_DTYPE_INT8
);
```

| Property | Value |
|----------|-------|
| Shape constraint | A and B must be same shape and contiguous. No broadcast in v0.1. |
| Descriptor emitted | `NOVA_OP_ELEMENTWISE` (type 0x0B) |
| Workspace | None |

### 5.10 orbit_attention_fwd

Fused multi-head attention. Encapsulates RoPE, KVC, GEMM, softmax as a single multi-descriptor batch.

```c
nova_status_t orbit_attention_fwd(
  nova_stream_t   stream,
  nova_ptr_t      Q,           // [B, H, seq_q, d_head] FP16
  nova_ptr_t      K,           // [B, H, seq_k, d_head] FP16
  nova_ptr_t      V,           // [B, H, seq_k, d_head] FP16
  nova_ptr_t      out,         // [B, H, seq_q, d_head] FP16 output
  nova_ptr_t      kv_cache,    // KVC base device address, or NULL (no cache)
  nova_ptr_t      cos_sin,     // [seq_q, d_head] FP16 cos/sin interleaved, or NULL (no RoPE)
  nova_ptr_t      workspace,   // scratch buffer, size = B*H*seq_q*seq_k*2 bytes
  int             B,           // batch size
  int             H,           // number of attention heads
  int             seq_q,       // query sequence length
  int             seq_k,       // key/value sequence length (incl. KVC)
  int             d_head,      // head dimension (must be multiple of 16)
  float           scale,       // attention scale = 1/sqrt(d_head)
  bool            causal,      // apply causal (lower-triangular) mask
  nova_dtype_t    dtype        // NOVA_DTYPE_FLOAT16 only
);
```

#### Descriptor Sequence Emitted (in order, via novaLaunchBatch)

| Step | Descriptor type | Operation | Notes |
|------|----------------|-----------|-------|
| 1 | `NOVA_OP_ROPE` (0x05) | RoPE on Q: rotate Q using cos_sin | Skipped if `cos_sin == NULL` |
| 2 | `NOVA_OP_ROPE` (0x05) | RoPE on K: rotate K using cos_sin | Skipped if `cos_sin == NULL` |
| 3 | `NOVA_OP_KVC_APPEND` (0x08) | KVC_WRITE: store new K,V tokens to cache at current seq_pos | Skipped if `kv_cache == NULL` |
| 4 | `NOVA_OP_KVC_LOOKUP` (0x09) | KVC_READ: read full seq_k of K,V from cache into temp buffers | Skipped if `kv_cache == NULL` |
| 5 | `NOVA_OP_GEMM_INT8` (0x02) | `scores = Q @ K^T` → `[B, H, seq_q, seq_k]` stored in workspace | Uses orbit_gemm_batched internally |
| 6 | `NOVA_OP_ELEMENTWISE` (0x0B) | Scale scores by `scale` (multiply by scalar) | NOVA_EW_MUL with scalar broadcast |
| 7 | `NOVA_OP_SOFTMAX` (0x0F) | Softmax over last dim `[B*H*seq_q, seq_k]` | Causal mask applied before softmax if `causal=true` |
| 8 | `NOVA_OP_GEMM_INT8` (0x02) | `out = softmax_scores @ V` → `[B, H, seq_q, d_head]` | Uses orbit_gemm_batched internally |

Workspace size formula: `workspace_bytes = B * H * seq_q * seq_k * 2`  (FP16 = 2 bytes per element)

Causal mask implementation: Before step 7, emit a `VECTOR_OP` that sets upper-triangular elements
of scores to `-65504.0` (FP16 max negative). This causes them to become 0 after softmax.

#### KVC Descriptor Field Mapping

For KVC_WRITE (per head, per layer):
```c
nova_kvc_args_t kvc_write = {
  .cache    = kv_cache,       // base address of KVC buffer
  .data     = K_ptr,          // pointer to new K token(s)
  .layer    = layer_id,       // caller must pass layer_id to orbit_attention_fwd
  .head     = h,              // loop variable over H
  .seq_pos  = current_seq_pos,
  .head_dim = d_head
};
```

For KVC_READ:
```c
nova_kvc_args_t kvc_read = {
  .cache    = kv_cache,
  .data     = temp_kv_buf,    // workspace region for full K,V sequence
  .layer    = layer_id,
  .head     = h,
  .seq_pos  = 0,              // read from position 0
  .head_dim = d_head
  // seq_len encoded in descriptor header.length = seq_k
};
```

### 5.11 orbit_moe_route

```c
nova_status_t orbit_moe_route(
  nova_stream_t   stream,
  nova_ptr_t      logits,      // [num_tokens, num_experts] FP16
  nova_ptr_t      indices,     // [num_tokens, top_k] INT32 output
  nova_ptr_t      scores,      // [num_tokens, top_k] FP16 output (sum to 1.0 per token)
  int             num_tokens,
  int             num_experts,
  int             top_k
);
```

| Property | Value |
|----------|-------|
| Algorithm | (1) softmax(logits) → probs, (2) top-k selection per token, (3) normalize selected scores to sum=1.0 |
| Output ordering | scores descending per token |
| Descriptor emitted | `NOVA_OP_MOE_GATE` (0x07 per nova_types.h) = MOE_ROUTE (0x0C per descriptor.md v2) |
| num_experts constraint | Must be ≤ 256 in v0.1 (hardware top-k unit width) |
| Workspace | None |

---

## Section 6: Descriptor Emission Contract

This table maps each kernel to the exact `nova_descriptor_t` fields that must be populated.
All descriptors are 64 bytes (fixed, per descriptor.md §0).

| Kernel | nova_op_type | Key fields populated | Computed/derived fields |
|--------|-------------|---------------------|------------------------|
| `orbit_gemm` | `NOVA_OP_GEMM_INT8` (1) | `gemm.A`, `gemm.B`, `gemm.C`, `gemm.M`, `gemm.K`, `gemm.N`, `gemm.dtype`, `gemm.alpha=1.0`, `gemm.beta=0.0`, `gemm.trans_A=0`, `gemm.trans_B=0` | Tiled: emit one descriptor per 16×16 tile; offset addresses by tile row/col |
| `orbit_gemm` (INT4) | `NOVA_OP_GEMM_INT4` (GEMM_INT4 0x0E) | Same as INT8 plus `B_scale` pointer at `gemm.B+offset` convention (packed scale at end of B buffer) | Tile size 32×32 for INT4 path |
| `orbit_gemm_bias` | `NOVA_OP_GEMM_INT8` + `NOVA_OP_ELEMENTWISE` | GEMM fields as above; then elementwise: `elementwise.A=C`, `elementwise.B=bias_broadcast`, `elementwise.out=C`, `elementwise.n=M*N`, `elementwise.op=NOVA_EW_ADD` | Bias broadcast: emit M VECTOR_OP descriptors of size N, or single if firmware supports stride-0 |
| `orbit_softmax` | `NOVA_OP_SOFTMAX` | via `nova_softmax_args_t`: `softmax.x=src`, `softmax.out=dst`, `softmax.rows=outer`, `softmax.cols=N`, `softmax.dim=1` | `header.length = N` |
| `orbit_rmsnorm` | `NOVA_OP_RMSNORM` | via `nova_rmsnorm_args_t`: `rmsnorm.x=src`, `rmsnorm.weight=weight`, `rmsnorm.out=dst`, `rmsnorm.N=d_model`, `rmsnorm.eps=eps` | Emit `outer` descriptors, each pointing to one row; addresses offset by `d_model * 2` bytes per row |
| `orbit_silu` | `NOVA_OP_SILU` | via `nova_silu_args_t`: `silu.x=src`, `silu.out=dst`, `silu.n=n` | `header.length = n` |
| `orbit_attention_fwd` (RoPE Q) | `NOVA_OP_ROPE` | via `nova_rope_args_t`: `rope.x=Q`, `rope.cos_sin=cos_sin`, `rope.out=Q_rope`, `rope.seq=seq_q`, `rope.heads=H`, `rope.head_dim=d_head` | In-place: `rope.out=Q` is acceptable if hardware supports |
| `orbit_attention_fwd` (RoPE K) | `NOVA_OP_ROPE` | Same as above with `rope.x=K` | |
| `orbit_attention_fwd` (KVC write) | `NOVA_OP_KVC_APPEND` | `nova_kvc_args_t` fields as shown in §5.10 | Loop H heads: H descriptors |
| `orbit_attention_fwd` (KVC read) | `NOVA_OP_KVC_LOOKUP` | `nova_kvc_args_t` with `seq_pos=0`, `header.length=seq_k` | Loop H heads: H descriptors |
| `orbit_attention_fwd` (QK scores) | `NOVA_OP_GEMM_INT8` | `gemm.A=Q_rope`, `gemm.B=K_rope`, `gemm.trans_B=1`, `gemm.C=workspace`, `gemm.M=seq_q`, `gemm.K=d_head`, `gemm.N=seq_k` | B×H instances via batch loop |
| `orbit_attention_fwd` (scale) | `NOVA_OP_ELEMENTWISE` | `elementwise.A=workspace`, `elementwise.B=NULL` (scalar via imm), `elementwise.op=NOVA_EW_MUL`, `elementwise.n=B*H*seq_q*seq_k` | `imm` field carries `*(uint32_t*)&scale` (FP16 bits) |
| `orbit_attention_fwd` (softmax) | `NOVA_OP_SOFTMAX` | `softmax.x=workspace`, `softmax.out=workspace`, `softmax.rows=B*H*seq_q`, `softmax.cols=seq_k` | In-place softmax |
| `orbit_attention_fwd` (scores@V) | `NOVA_OP_GEMM_INT8` | `gemm.A=workspace`, `gemm.B=V`, `gemm.C=out`, `gemm.M=seq_q`, `gemm.K=seq_k`, `gemm.N=d_head` | B×H instances |
| `orbit_moe_route` | `NOVA_OP_MOE_GATE` | via descriptor v2 `MOE_ROUTE` (0x0C): `logits_addr`, `indices_addr`, `scores_addr`, `num_tokens`, `num_experts`, `top_k` | Struct layout from yua-llm-hw-design §2.3 |
| `aten::copy_` (H2D) | `NOVA_OP_MEMSET` ... DMA_2D | `dma: src_addr=host_ptr`, `dst_addr=device_ptr`, `width_bytes=nbytes`, `height=1`, `src_stride=nbytes`, `dst_stride=nbytes` | Use `novaMemcpy(dst, src, n, NOVA_MEMCPY_H2D)` which emits DMA_2D |
| `aten::zeros` | `NOVA_OP_MEMSET` | `memset.ptr=device_ptr`, `memset.value=0`, `memset.count=nbytes` | Via `novaMemset` |

---

## Section 7: Version and Compatibility Matrix

| Component | Version | ABI frozen? | Notes |
|-----------|---------|-------------|-------|
| nova_core C API (`nova.h` / `nova_types.h`) | 0.1 | No — pre-1.0 | Function signatures may change; recompile nova_torch after any nova_core update |
| nova_torch Python extension | 0.1 | No | |
| Descriptor Spec v1 (`descriptor.md`) | 1.0 | **Yes (hardware)** | Types 0x01–0x09. Do not redefine. |
| Descriptor Spec v2 extensions (`yua-llm-hw-design.md §6`) | 2.0 | **Yes (hardware RTL frozen)** | Types 0x0A–0x0F. VPU/KVC/MoE RTL passes cocotb. |
| nova_descriptor_t union layout | 0.1 | No | Fields added as new ops are defined |
| GEMM tile size (16×16) | 1.0 | **Yes (YUA-T16 v1 RTL)** | Tile is 16×16 for INT8. INT4 path uses 32×32 (v2). |
| KVC layout: `[layer][head][seq_pos]` | 1.0 | **Yes (KVC RTL frozen)** | |
| Python device string | — | **Yes** | Always `"orbit"` or `"orbit:N"` |

### Descriptor Type Enumeration (Consolidated)

| Type ID | Name | Spec version | Emitted by |
|---------|------|-------------|------------|
| 0x01 | DMA_2D | v1 | novaMemcpy |
| 0x02 | GEMM_INT8 | v1 | orbit_gemm, orbit_attention_fwd |
| 0x03 | VECTOR_OP | v1 | elementwise, softmax (legacy) |
| 0x04 | COPY_2D_PLUS | v1 | SUP only |
| 0x05 | FORMAT_CONVERT | v1 | SUP only |
| 0x06 | FRAME_FINGERPRINT | v1 | SUP only |
| 0x07 | BARRIER | v1 | novaStreamSynchronize path |
| 0x08 | EVENT | v1 | stream completion callbacks |
| 0x09 | PERF_SNAPSHOT | v1 | profiling |
| 0x0A | KVC_READ | v2 | orbit_attention_fwd |
| 0x0B | KVC_WRITE | v2 | orbit_attention_fwd |
| 0x0C | MOE_ROUTE | v2 | orbit_moe_route |
| 0x0D | VECTOR_OP_EX | v2 | orbit_rmsnorm, orbit_silu, orbit_gelu, orbit_rope |
| 0x0E | GEMM_INT4 | v2 | orbit_gemm (INT4 dtype path) |
| 0x0F | SOFTMAX | v2 | orbit_softmax, orbit_attention_fwd |

Note: `nova_op_type_t` in `nova_types.h` currently only enumerates through `NOVA_OP_ELEMENTWISE = 11`.
The v2 types (0x0A–0x0F) must be added to `nova_op_type_t` before the corresponding descriptor paths
in the dispatch switch will function. This is a required nova_core change for Phase 3+.

---

## Section 8: Implementation Order

### Phase 1 — Device Registration + Allocator

**Files to create:**
- `nova_torch/csrc/orbit_guard_impl.h` + `.cpp`
- `nova_torch/csrc/orbit_allocator.h` + `.cpp`
- `nova_torch/csrc/module.cpp`
- `nova_torch/CMakeLists.txt`
- `nova_torch/__init__.py` (Python module entry; imports the `.so`)

**ATen ops to register:**
- `aten::empty.memory_format` → `OrbitAllocator::allocate`
- `aten::copy_` → `novaMemcpy`
- `aten::_to_copy` → `novaMemcpy` H2D / D2H
- `aten::zeros` → `novaMalloc` + `novaMemset(0)`

**Acceptance test:**
```python
import torch, nova_torch
x = torch.empty(4, 4, dtype=torch.float16, device="orbit:0")
assert x.device.type == "orbit"
y = x.cpu()                          # D2H
z = y.to("orbit:0")                  # H2D
assert z.device.type == "orbit"
w = torch.zeros(4, 4, dtype=torch.float16, device="orbit:0")
assert w.cpu().sum() == 0.0
```

### Phase 2 — GEMM

**Files to create:**
- `nova_torch/kernels/orbitBLAS.h` + `.cpp`  (orbit_gemm, orbit_gemm_batched, orbit_gemm_bias)

**ATen ops to register:**
- `aten::mm` → `orbit_gemm`
- `aten::bmm` → `orbit_gemm_batched`
- `aten::addmm` → `orbit_gemm_bias`
- `aten::linear` → `orbit_gemm_bias`
- `aten::matmul` → shape-normalize then `orbit_gemm`

**Acceptance test:**
```python
import torch, nova_torch
A = torch.randn(64, 128, dtype=torch.float16).to("orbit")
B = torch.randn(128, 64, dtype=torch.float16).to("orbit")
C = torch.mm(A, B)                   # runs on ORBIT-G1
assert C.shape == (64, 64)
C_cpu = C.cpu()
A_cpu = A.cpu(); B_cpu = B.cpu()
ref = torch.mm(A_cpu, B_cpu)
assert torch.allclose(C_cpu, ref, atol=1e-2)   # FP16 tolerance
```

### Phase 3 — Elementwise + Norm + Activation

**Files to create:**
- `nova_torch/kernels/orbitNN.h` + `.cpp`

**ATen ops to register:**
- `aten::silu` / `aten::silu_` → `orbit_silu`
- `aten::gelu` → `orbit_gelu`
- `aten::relu` → CPU fallback (SLOW_FALLBACK log)
- `aten::add.Tensor` → `orbit_elementwise` (NOVA_EW_ADD) or CPU fallback for broadcast
- `aten::mul.Tensor` → `orbit_elementwise` (NOVA_EW_MUL) or CPU fallback
- `aten::softmax.int` / `aten::_softmax` → `orbit_softmax`
- `aten::rms_norm` → `orbit_rmsnorm`
- `aten::layer_norm` / `aten::native_layer_norm` → CPU fallback (log SLOW_FALLBACK)
- `aten::zeros` → use Phase 1 allocator + `novaMemset`

**Acceptance tests:**
```python
x = torch.randn(512, dtype=torch.float16).to("orbit")
out = torch.nn.functional.silu(x)
assert torch.allclose(out.cpu(), torch.nn.functional.silu(x.cpu()), atol=1e-2)

x2 = torch.randn(8, 512, dtype=torch.float16).to("orbit")
out2 = torch.nn.functional.softmax(x2, dim=-1)
assert torch.allclose(out2.cpu().sum(dim=-1), torch.ones(8), atol=1e-3)

x3 = torch.randn(4, 64, dtype=torch.float16).to("orbit")
w3 = torch.ones(64, dtype=torch.float16).to("orbit")
out3 = torch.nn.functional.rms_norm(x3, (64,), w3)
assert out3.shape == (4, 64)
```

### Phase 4 — Attention

**Files to create:**
- `nova_torch/kernels/orbitAttention.h` + `.cpp`

**ATen ops to register:**
- `aten::scaled_dot_product_attention` → `orbit_attention_fwd`
- `aten::_scaled_dot_product_flash_attention` → `orbit_attention_fwd`

**Acceptance test:**
```python
import torch, torch.nn.functional as F, nova_torch
B, H, S, D = 2, 8, 128, 64
Q = torch.randn(B, H, S, D, dtype=torch.float16).to("orbit")
K = torch.randn(B, H, S, D, dtype=torch.float16).to("orbit")
V = torch.randn(B, H, S, D, dtype=torch.float16).to("orbit")
out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
assert out.shape == (B, H, S, D)
# Compare to CPU reference
out_ref = F.scaled_dot_product_attention(Q.cpu(), K.cpu(), V.cpu(), is_causal=True)
assert torch.allclose(out.cpu(), out_ref, atol=2e-2)
```

### Phase 5 — Full Model End-to-End

**Goal:** Run a GPT-class model on `device="orbit"` end-to-end.

**Test A — GPT-2 (HuggingFace, no MoE):**
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch, nova_torch

model = GPT2LMHeadModel.from_pretrained("gpt2").to(dtype=torch.float16)
model = model.to("orbit")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
inputs = tokenizer("Hello, ORBIT-G1!", return_tensors="pt")
input_ids = inputs["input_ids"].to("orbit")
with torch.no_grad():
    out = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(out[0].cpu()))
```

Expected: generation completes without error; output is coherent text.
LayerNorm fallback to CPU is acceptable at this phase (SLOW_FALLBACK logged).

**Test B — GPT-2 performance baseline:**
```python
import time
# 100 forward passes; measure tokens/sec
# Baseline: must exceed CPU-only speed
```

**Test C — MoE model (after orbit_moe_route implemented):**
```python
# Requires model with MoE layers configured for "orbit"
# orbit_moe_route dispatches expert routing via NOVA_OP_MOE_GATE
```

---

## Appendix A: Error Handling Policy

| Condition | Response |
|-----------|----------|
| `novaMalloc` returns `NOVA_ERROR_OUT_OF_MEMORY` | Raise `torch.cuda.OutOfMemoryError` (or `c10::OutOfMemoryError`) with bytes requested |
| Any nova API returns non-NOVA_SUCCESS | Raise `RuntimeError("nova_torch: <function_name> failed: " + novaGetErrorString(status))` |
| ATen op called with unsupported dtype | Raise `RuntimeError("nova_torch: <op> does not support dtype " + dtype)` |
| Softmax with N > 65535 | Raise `RuntimeError("nova_torch: softmax dim size " + N + " exceeds hardware max 65535")` |
| Non-contiguous tensor passed to kernel | Raise `RuntimeError("nova_torch: <op> requires contiguous tensor; call .contiguous() first")` |
| CPU fallback triggered | Log via `TORCH_WARN` (once per process per op): `[nova_torch SLOW_FALLBACK] <op> not implemented on orbit device; executing on CPU` |

---

## Appendix B: Build System Contract

```cmake
# nova_torch/CMakeLists.txt (minimum required fields)

cmake_minimum_required(VERSION 3.18)
project(nova_torch)

find_package(Torch REQUIRED)           # via torch.utils.cmake_prefix_path
find_package(nova_core REQUIRED)       # installs to /usr/local or cmake find path

# C++ extension
pybind11_add_module(nova_torch
  csrc/module.cpp
  csrc/orbit_guard_impl.cpp
  csrc/orbit_allocator.cpp
  kernels/orbitBLAS.cpp
  kernels/orbitNN.cpp
  kernels/orbitAttention.cpp
  kernels/orbitMoE.cpp
)

target_include_directories(nova_torch PRIVATE
  ${TORCH_INCLUDE_DIRS}
  ${nova_core_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(nova_torch PRIVATE
  ${TORCH_LIBRARIES}
  nova_core             # libnova_core.so
)

set_property(TARGET nova_torch PROPERTY CXX_STANDARD 17)
```

Python install: `pip install -e .` or `python setup.py develop`.

---

## Appendix C: Thread Safety Contract

| Resource | Thread safety |
|----------|--------------|
| `novaInit` | Call once from main thread before any use |
| `novaSetDevice` / `novaGetDevice` | Thread-local device index (same as CUDA semantics) |
| `novaMalloc` / `novaFree` | Thread-safe (internal mutex in buddy allocator) |
| `novaStreamCreate` | Thread-safe |
| `novaLaunchDescriptor` | Per-stream ordering is guaranteed; cross-stream requires explicit BARRIER |
| `g_orbit_default_stream` | Thread-local array; no locking needed |
| `OrbitAllocator::allocate` | Thread-safe (delegates to `novaMalloc`) |

---

---

## Appendix D: Descriptor Count Per Kernel

The number of descriptors emitted depends on the kernel and input shape.
Queue capacity = 256 per queue. Kernels emitting more than 256 must flush mid-batch.

| Kernel | Descriptor count formula | Worst-case example |
|--------|--------------------------|--------------------|
| `orbit_gemm` | `ceil(M/16) × ceil(N/16)` tiles, each = 1 GEMM_INT8 desc | M=4096, N=4096 → 65536 tiles (must batch-flush) |
| `orbit_gemm` (INT4) | `ceil(M/32) × ceil(N/32)` tiles | M=4096, N=4096 → 16384 tiles |
| `orbit_gemm_batched` | `batch × ceil(M/16) × ceil(N/16)` | B=8, M=128, N=128 → 64 descriptors |
| `orbit_gemm_bias` | GEMM tiles + M×VECTOR_OP for bias | M=64, N=256 → 16 GEMM + 64 VECTOR_OP = 80 |
| `orbit_softmax` | `outer` (one SOFTMAX desc per row) | outer=512 → 512 descriptors |
| `orbit_rmsnorm` | `outer` (one VECTOR_OP_EX per row) | outer=512 → 512 descriptors |
| `orbit_silu` | 1 descriptor | Always 1 |
| `orbit_rope` | 1 descriptor (processes full tensor) | Always 1 |
| `orbit_attention_fwd` | 2 (RoPE Q+K) + 2H (KVC write) + 2H (KVC read) + B×H (QK^T GEMM) + 1 (scale) + B×H×seq_q (softmax) + B×H (scores@V GEMM) | B=1,H=32,seq=512 → 2+64+64+32+1+16384+32 = 16579 |
| `orbit_moe_route` | 1 (MOE_ROUTE desc) | Always 1 |
| `orbit_layernorm` | CPU fallback → 0 hardware descriptors | — |

**Flush rule:** If total descriptors for a kernel exceeds 200 (safety margin below 256),
`novaLaunchBatch` is called in chunks of 200 with intermediate `novaStreamFlush`.
The kernel is responsible for chunking — caller sees one logical async operation.

---

## Appendix E: Error Behavior — Full Layer Stack

Every error has a defined detection layer. Errors propagate upward and are never silently ignored.

### E.1 Error Detection Layers

```
Layer 0: nova_torch (Python/C++ ATen dispatch)
  — Shape validation (M,N,K alignment, batch dims)
  — dtype validation (unsupported combos)
  — Contiguity check
  — N ≤ 65535 for softmax
  → Raises Python RuntimeError immediately, NO descriptor emitted

Layer 1: orbit kernel library (orbitBLAS.cpp, orbitNN.cpp, ...)
  — Null pointer check on device addresses
  — Tile count overflow (> queue capacity without flush)
  — Workspace size insufficient
  → Returns NOVA_ERROR_INVALID_ARGUMENT, propagates to Layer 0

Layer 2: nova_core runtime (nova_core/src/*.c)
  — novaLaunchDescriptor: checks stream is valid
  — novaStreamSynchronize: checks done_cookie ≥ submitted
  — novaMalloc: checks pool not exhausted
  → Returns nova_status_t code, propagates to Layer 1

Layer 3: kernel driver (orbit_g1.ko)
  — ORBIT_IOC_SUBMIT_DESC: validates descriptor count ≤ ring capacity
  — copy_from_user: validates user pointer
  — buddy allocator: checks free list not empty
  → Returns errno (ENOMEM, EINVAL, EBUSY), propagates to nova_core

Layer 4: ORBIT-G1 hardware
  — Command processor: unknown descriptor type → ERROR interrupt
  — DMA: out-of-range address → bus error interrupt
  — VPU: div-by-zero in RMSNorm (eps=0) → output clamped, no fault
  → MSI-X error interrupt → orbit_g1_error_irq() → sets error_status register
     nova_core polls error_status after novaStreamSynchronize()
     → Returns NOVA_ERROR_HARDWARE_FAULT
```

### E.2 Error Table

| Error condition | Detection layer | Response |
|----------------|----------------|---------|
| Invalid shape (M%16 ≠ 0, no padding) | Layer 0 | `RuntimeError`: shape must be multiple of 16, or use auto-pad |
| Unsupported dtype | Layer 0 | `RuntimeError`: orbit does not support dtype X for op Y |
| Non-contiguous tensor | Layer 0 | `RuntimeError`: call .contiguous() first |
| Softmax N > 65535 | Layer 0 | `RuntimeError`: softmax dim exceeds hardware max 65535 |
| Device memory OOM | Layer 2 | `torch.OutOfMemoryError`: requested N bytes, N available |
| Descriptor queue full | Layer 2/3 | Kernel auto-flushes in chunks; if ring still full → `RuntimeError`: descriptor queue deadlock |
| Null device address | Layer 1 | `RuntimeError`: null device pointer passed to orbit_gemm |
| Unsupported BF16 | Layer 0 | CPU fallback (SLOW_FALLBACK) — BF16 not in hardware dtype |
| Hardware fault (bad address) | Layer 4 → Layer 2 | `RuntimeError`: ORBIT-G1 hardware fault (error_status=0xXX) |
| Driver ioctl fail (EINVAL) | Layer 3 → Layer 2 | `RuntimeError`: orbit_g1 driver rejected descriptor batch |
| Kernel module not loaded | Layer 2 | `RuntimeError`: cannot open /dev/orbit_g1_0: No such file or directory |
| Simulation mode (no hardware) | Layer 2 | Silently uses CPU simulator — no error, but perf is CPU speed |

### E.3 BF16 Policy

BF16 is NOT supported in hardware (YUA-T16 v1 is FP16/INT8 only).

```
BF16 tensor on "orbit" device:
  → nova_torch intercepts at ATen dispatch
  → TORCH_WARN("[nova_torch] BF16 not supported on orbit hardware; falling back to CPU")
  → D2H copy (if tensor is on device) → CPU op → H2D copy back
  → Result tensor dtype = BF16 on orbit device (storage is actually CPU-backed via copy)
```

Future: When YUA-T16 v3 adds BF16 MACs, remove fallback and register hardware kernel.

---

## Appendix F: Alignment Rules

### F.1 Why Alignment Matters

ORBIT-G1 uses 256-byte DMA burst transfers. Misaligned accesses cause:
- DMA padding/masking overhead (performance loss)
- Hardware address fault if sub-burst (< 64B) misalignment on certain BARs

### F.2 Tensor Pointer Alignment

| Dtype | Minimum alignment | Recommended alignment | Reason |
|-------|------------------|----------------------|--------|
| FP16 | 2 bytes (hardware minimum) | **64 bytes** | Cache line for PCIe DMA |
| INT8 | 1 byte (hardware minimum) | **64 bytes** | Cache line for PCIe DMA |
| INT4 (packed) | 1 byte | **64 bytes** | MAC array input alignment |
| INT32 | 4 bytes | **64 bytes** | Output SRAM alignment |

**Contract:** `novaMalloc` always returns 64-byte aligned addresses (buddy allocator minimum block = 4 KB, naturally aligned). Callers MUST NOT manually offset into allocated regions to create misaligned sub-tensors.

**Violation:** If `nova_ptr_t % 64 != 0`, the kernel driver logs a warning and rounds down to nearest 64-byte boundary. This may corrupt neighboring tensors — treat as undefined behavior.

### F.3 Descriptor Alignment

```
nova_descriptor_t = 64 bytes (fixed)
Descriptor ring buffer base: 64-byte aligned (dma_alloc_coherent guaranteed)
Each descriptor in ring: naturally 64-byte aligned (ring_base + index * 64)
→ No padding needed between descriptors.
→ Hardware command processor reads 64-byte aligned bursts.
```

**Rule:** Never directly `memcpy` a partial descriptor. Always write the full 64-byte struct atomically via `novaLaunchDescriptor`.

### F.4 DMA Alignment

| DMA operation | Source alignment | Destination alignment | Burst size |
|---------------|-----------------|----------------------|------------|
| H2D (host→device) | 64 bytes (pinned host ptr) | 64 bytes (device ptr) | 256 bytes |
| D2H (device→host) | 64 bytes (device ptr) | 64 bytes (host ptr) | 256 bytes |
| D2D (device→device) | 64 bytes | 64 bytes | 256 bytes |

**DMA_2D descriptor `width_bytes` constraint:**
- Must be a multiple of 64 bytes for full burst efficiency.
- If `width_bytes % 64 != 0`: hardware pads the last burst with zeros on write, masks on read. Safe but slower.
- If `width_bytes < 64`: hardware issues a sub-burst. Supported but generates a `PERF_WARN` in driver.

**`src_stride` and `dst_stride` (DMA_2D):**
- Must be ≥ `width_bytes`.
- Must be aligned to 64 bytes if hardware pipelining of rows is desired.
- Stride = 0 is invalid (reserved = broadcast mode, not implemented in v1).

### F.5 SRAM Alignment (YUA-T16 internal)

The YUA-T16 tile engine uses internal `act_sram` and `wgt_sram`:
- Activation SRAM: loaded from GDDR6 as `Kt × 16` INT8 rows. Row width = 16 bytes → 16-byte alignment.
- Weight SRAM: loaded as `Kt × 16` INT8. Same 16-byte alignment.
- Output SRAM: `16 × 16` INT32 = 1024 bytes. Written to GDDR6 after computation.

These are internal to the hardware — not user-visible. Mentioned for driver developers verifying address generation.

### F.6 Summary: Required Alignment Checklist

Before submitting any descriptor batch, verify:

```
[ ] All nova_ptr_t tensor addresses are 64-byte aligned
[ ] Descriptor ring buffer base is 64-byte aligned (guaranteed by dma_alloc_coherent)
[ ] DMA_2D width_bytes is multiple of 64 (or accept PERF_WARN)
[ ] DMA_2D src_stride and dst_stride ≥ width_bytes
[ ] Pinned host memory (novaHostAlloc) is 64-byte aligned (guaranteed)
[ ] No sub-tensor views with non-zero storage_offset on "orbit" device
```

---

*End of nova-torch-contract.md v0.1*
