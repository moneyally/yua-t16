# ORBIT-G1 Linux Kernel Driver + Userspace Runtime — Design Specification v1.0

> **Status:** Design only. This document is the SSOT for three parallel implementation agents.
> **Hardware base:** ORBIT-G1 v2 (descriptor spec v1 + v2 extensions: KVC_READ/WRITE, MOE_ROUTE, VECTOR_OP_EX, GEMM_INT4, SOFTMAX)
> **Target workload:** GPT-OSS-20B (MoE Transformer, 32 experts), 21-step decode descriptor sequence

---

## Table of Contents

1. [Hardware Assumptions](#1-hardware-assumptions)
2. [Linux Kernel Driver — orbit_g1.ko](#2-linux-kernel-driver--orbit_g1ko)
3. [Userspace Runtime Library — liborbit.so](#3-userspace-runtime-library--liborbitso)
4. [OpenAI-compatible Inference Server — orbit_server](#4-openai-compatible-inference-server--orbit_server)
5. [Directory Structure](#5-directory-structure)
6. [Key Header Definitions](#6-key-header-definitions)
7. [Data Flow Diagrams](#7-data-flow-diagrams)
8. [Error Handling Strategy](#8-error-handling-strategy)
9. [Agent Implementation Assignments](#9-agent-implementation-assignments)

---

## 1. Hardware Assumptions

These assumptions are derived from the three spec files and must be respected throughout all implementation work.

### 1.1 PCIe Configuration

| Parameter | Value |
|-----------|-------|
| PCIe generation | Gen4 x16 |
| Vendor ID | TBD (placeholder: 0x1EDA) |
| Device ID | TBD (placeholder: 0x0001) |
| BAR0 | MMIO — descriptor queue registers, status, doorbell |
| BAR1 | MMIO — GDDR6 window (weight upload, DMA base) |
| Interrupt model | MSI-X, 1 vector per queue (minimum 4 vectors) |

### 1.2 Descriptor System

- All descriptors are exactly **64 bytes**, fixed size.
- Types 0x01–0x09 from descriptor.md v1 (DMA_2D through PERF_SNAPSHOT).
- Types 0x0A–0x0F from yua-llm-hw-design.md v2 (KVC_READ, KVC_WRITE, MOE_ROUTE, VECTOR_OP_EX, GEMM_INT4, SOFTMAX).
- Execution is in submission order; no implicit ordering; BARRIER descriptor (0x07) required for cross-queue ordering.
- EVENT descriptor (0x08) triggers host interrupt.

### 1.3 Queue Model

- **4 queues**, each with 256 ring entries:
  - Queue 0: Compute (GEMM, VECTOR_OP, KVC, MoE)
  - Queue 1: Utility / SUP (DMA_2D, COPY_2D_PLUS, FORMAT_CONVERT, FINGERPRINT)
  - Queue 2: Telemetry / Diagnostics (PERF_SNAPSHOT, DIAG_RUN)
  - Queue 3: Reserved / High-priority compute

- Each queue has independent head/tail registers in BAR0 MMIO space.

### 1.4 Memory

- GDDR6 global memory: 16 GB or 32 GB (runtime-detected).
- Host-visible GDDR6 window mapped via BAR1 (up to BAR1 size; may be partial).
- Device-side DMA: addresses are GDDR6-physical, not host-physical.
- For host→device weight upload: mmap BAR1 or use DMA_2D descriptor with host pinned buffer.

### 1.5 Descriptor Address Space

All `src_addr` / `dst_addr` / `act_addr` / `wgt_addr` fields in descriptors refer to **GDDR6 device addresses** (64-bit, device-physical). The runtime is responsible for translating logical allocation handles to device addresses before building descriptors.

---

## 2. Linux Kernel Driver — orbit_g1.ko

### 2.1 Source File Layout (within driver/)

```
driver/
  orbit_g1_main.c        — pci_driver probe/remove, module init/exit
  orbit_g1_mmio.c        — BAR0/BAR1 map/unmap, register accessors
  orbit_g1_queue.c       — descriptor ring buffer, doorbell
  orbit_g1_dma.c         — dma_alloc_coherent management
  orbit_g1_irq.c         — MSI-X setup, ISR, event completion
  orbit_g1_cdev.c        — /dev/orbit_g1_N chardev, file_operations
  orbit_g1_ioctl.c       — ioctl dispatch, all ORBIT_IOC_* handlers
  orbit_g1_mmap.c        — mmap for BAR1 zero-copy weight window
  orbit_g1_mem.c         — GDDR6 region allocator (buddy system)
  orbit_g1_debug.c       — debugfs entries, trace ring
  orbit_g1.h             — internal shared structs (NOT exported to userspace)
  orbit_g1_uapi.h        — exported ioctl structs (copied to include/uapi/)
  Makefile
  Kconfig
```

### 2.2 PCIe Device Probe / Remove

**File:** `orbit_g1_main.c`

The driver registers a `pci_driver` with the kernel's PCI subsystem.

```
orbit_g1_probe(pci_dev, pci_device_id):
  1. pci_enable_device()
  2. pci_set_master()                        — enable bus mastering for DMA
  3. pci_request_regions()                   — reserve BAR0, BAR1
  4. orbit_g1_mmio_init()                    — map BAR0, BAR1 (ioremap_wc for BAR1)
  5. orbit_g1_dma_init()                     — allocate descriptor ring DMA memory
  6. orbit_g1_queue_init()                   — initialize 4 × 256 ring buffers
  7. orbit_g1_irq_init()                     — MSI-X setup (4 vectors minimum)
  8. orbit_g1_mem_init()                     — GDDR6 buddy allocator init
  9. orbit_g1_cdev_create()                  — register /dev/orbit_g1_N
  10. orbit_g1_debugfs_create()              — /sys/kernel/debug/orbit_g1/
  11. pci_set_drvdata(pdev, priv)

orbit_g1_remove(pci_dev):
  reverse order: debugfs → cdev → mem → irq → queues → dma → mmio → pci_regions → pci_disable
```

**Internal device state struct:**

```c
struct orbit_g1_device {
    struct pci_dev          *pdev;
    void __iomem            *bar0;          /* descriptor queue registers */
    void __iomem            *bar1;          /* GDDR6 window */
    resource_size_t          bar0_len;
    resource_size_t          bar1_len;

    struct orbit_queue       queues[ORBIT_NUM_QUEUES];   /* 4 queues */
    struct orbit_mem_pool    gddr_pool;     /* GDDR6 buddy allocator */

    int                      msix_nvec;
    struct msix_entry        msix_entries[ORBIT_NUM_QUEUES];

    struct cdev              cdev;
    struct device           *dev;
    dev_t                    devt;

    spinlock_t               lock;          /* protect queue head/tail */
    atomic_t                 open_count;

    u64                      gddr_size_bytes;
    u32                      fw_version;
    u32                      hw_revision;
};
```

### 2.3 BAR0 MMIO — Descriptor Queue Registers

**File:** `orbit_g1_mmio.c`

BAR0 layout (all offsets relative to BAR0 base, 32-bit registers unless noted):

```
Offset    Register                  Access    Description
------    --------                  ------    -----------
0x0000    ORBIT_REG_ID              RO        Magic + version (0x4F524231 = "ORB1")
0x0004    ORBIT_REG_HW_REV          RO        Hardware revision
0x0008    ORBIT_REG_FW_VER          RO        Firmware version
0x000C    ORBIT_REG_STATUS          RO        Global status (ready, fault, etc.)
0x0010    ORBIT_REG_GDDR_SIZE_LO    RO        GDDR6 size bytes [31:0]
0x0014    ORBIT_REG_GDDR_SIZE_HI    RO        GDDR6 size bytes [63:32]
0x0018    ORBIT_REG_GLOBAL_CTRL     RW        Soft reset (bit 0), SUP mode (bits 3:2)
0x001C    ORBIT_REG_INTR_STATUS     RW1C      Per-queue interrupt status (bits 3:0)
0x0020    ORBIT_REG_INTR_MASK       RW        Per-queue interrupt mask

Per-queue block (stride = 0x40 per queue, queues 0–3):
Base = 0x0100 + (queue_id * 0x40)

0x00    Q_RING_BASE_LO    RW    Descriptor ring DMA address [31:0]
0x04    Q_RING_BASE_HI    RW    Descriptor ring DMA address [63:32]
0x08    Q_RING_SIZE       RW    Ring capacity (entries, must be power-of-2, max 256)
0x0C    Q_HEAD            RO    Hardware consumer head pointer
0x10    Q_TAIL            RW    Software producer tail pointer (doorbell)
0x14    Q_STATUS          RO    Queue status (active, error, idle)
0x18    Q_ERROR_CODE      RO    Last error code
0x1C    Q_COMPLETE_CNT    RO    Completed descriptor count (wrapping 32-bit)
0x20    Q_INTR_VECTOR     RW    MSI-X vector assignment
0x24    Q_CONFIG          RW    Queue config flags
0x28    (reserved)
```

**Register accessor macros:**

```c
#define orbit_read32(dev, off)         ioread32((dev)->bar0 + (off))
#define orbit_write32(dev, off, val)   iowrite32((val), (dev)->bar0 + (off))
#define orbit_read64(dev, off)         ioread64((dev)->bar0 + (off))   /* if supported */

#define ORBIT_Q_BASE(q)    (0x0100 + (q) * 0x40)
#define ORBIT_Q_TAIL(q)    (ORBIT_Q_BASE(q) + 0x10)
#define ORBIT_Q_HEAD(q)    (ORBIT_Q_BASE(q) + 0x0C)
```

### 2.4 BAR1 MMIO — GDDR6 Window

**File:** `orbit_g1_mmio.c`

BAR1 maps a contiguous window of GDDR6 device memory directly into host virtual address space. The window size is hardware-defined (e.g., 256 MB aperture for a 16 GB device).

- Mapped with `ioremap_wc()` (write-combining, uncached reads) to maximize PCIe write throughput.
- Used by `mmap()` handler to enable zero-copy weight upload from userspace.
- The window base in device address space is reported in a BAR1 metadata register (or hardcoded at GDDR6 address 0x0 if spec leaves it fixed).

### 2.5 Descriptor Ring Buffer

**File:** `orbit_g1_queue.c`

Each queue maintains a ring of 256 descriptor slots. Each slot is 64 bytes (one descriptor).

```c
struct orbit_queue {
    void           *ring_cpu;      /* kernel virtual address (DMA coherent) */
    dma_addr_t      ring_dma;      /* bus address, written to Q_RING_BASE */
    u32             ring_size;     /* 256 entries */
    u32             tail;          /* next slot to write (software-owned) */
    u32             head_cached;   /* cached snapshot of hardware head */
    spinlock_t      lock;          /* protect tail update */
    wait_queue_head_t wq;          /* wait_event for completion */
    atomic_t        inflight;      /* submitted but not yet completed */
    u32             queue_id;
    struct orbit_g1_device *dev;
};
```

**Producer flow (submit path):**

```
orbit_queue_submit(queue, descs[], count):
  spin_lock(&queue->lock)
  for each desc:
    avail = ring_size - (tail - head_cached) — read fresh head if avail==0
    if avail == 0: spin wait (or return -ENOSPC with NOWAIT flag)
    copy 64-byte desc to ring_cpu[tail % ring_size]
    tail++
  wmb()                           — memory barrier before doorbell
  orbit_write32(dev, ORBIT_Q_TAIL(qid), tail)   — doorbell
  atomic_add(count, &queue->inflight)
  spin_unlock(&queue->lock)
```

**Consumer (completion) flow — interrupt driven:**

```
orbit_irq_handler(queue_id):
  new_head = orbit_read32(dev, ORBIT_Q_HEAD(qid))
  completed = new_head - queue->head_cached
  queue->head_cached = new_head
  atomic_sub(completed, &queue->inflight)
  wake_up(&queue->wq)
  return IRQ_HANDLED
```

### 2.6 DMA Memory Allocation

**File:** `orbit_g1_dma.c`

The descriptor rings are allocated once at probe time using `dma_alloc_coherent()`:

```
Per queue:
  ring_cpu = dma_alloc_coherent(&pdev->dev,
                                ring_size * ORBIT_DESC_SIZE,  /* 256 * 64 = 16 KB */
                                &ring_dma, GFP_KERNEL)
```

This ensures:
- CPU writes are visible to the device without explicit cache flush.
- Device reads are coherent.
- PCIe IOMMU is respected via the kernel DMA API.

Additionally, a small coherent buffer is allocated per queue for completion status writes by the hardware (if the hardware uses write-back completion rather than pure register polling).

### 2.7 Interrupt Handling — MSI-X

**File:** `orbit_g1_irq.c`

```
orbit_g1_irq_init(dev):
  nvec = pci_alloc_irq_vectors(pdev, ORBIT_NUM_QUEUES, ORBIT_NUM_QUEUES, PCI_IRQ_MSIX)
  for q in 0..3:
    msix_entries[q].entry = q
    request_irq(pci_irq_vector(pdev, q), orbit_irq_handler_q[q], 0, "orbit_g1_qN", dev)
    orbit_write32(dev, ORBIT_Q_INTR_VECTOR(q), q)
```

One ISR per queue:

```c
static irqreturn_t orbit_irq_q0(int irq, void *data) {
    return orbit_irq_common(data, 0);
}
/* repeated for q1, q2, q3 */

static irqreturn_t orbit_irq_common(struct orbit_g1_device *dev, int qid) {
    /* clear interrupt */
    orbit_write32(dev, ORBIT_REG_INTR_STATUS, BIT(qid));
    /* advance completion, wake waiters */
    orbit_queue_complete(dev, qid);
    return IRQ_HANDLED;
}
```

### 2.8 Character Device Interface

**File:** `orbit_g1_cdev.c`

- Device node: `/dev/orbit_g1_0` (minor 0 for first card, minor 1 for second, etc.)
- Major number: dynamically allocated via `alloc_chrdev_region()`.
- Class: `orbit_g1` (visible as `/sys/class/orbit_g1/orbit_g1_0`).

```c
static const struct file_operations orbit_g1_fops = {
    .owner          = THIS_MODULE,
    .open           = orbit_g1_open,
    .release        = orbit_g1_release,
    .unlocked_ioctl = orbit_g1_ioctl,
    .mmap           = orbit_g1_mmap,
    .poll           = orbit_g1_poll,        /* for non-blocking wait */
};
```

**open:** increments `open_count`, initializes per-fd state (`orbit_fd_ctx`).
**release:** decrements `open_count`, frees any per-fd allocations, cleans up incomplete submissions.

```c
struct orbit_fd_ctx {
    struct orbit_g1_device  *dev;
    spinlock_t               alloc_lock;
    struct list_head         alloc_list;    /* GDDR6 regions owned by this fd */
    u32                      session_id;    /* unique per open() */
};
```

### 2.9 ioctl Interface

**File:** `orbit_g1_ioctl.c`

All ioctl numbers use the `_IOWR` / `_IOW` / `_IOR` macros with magic `'O'` (0x4F).

#### ORBIT_IOC_SUBMIT_DESC

Submit a batch of descriptors to a specific queue.

```c
#define ORBIT_IOC_SUBMIT_DESC  _IOWR('O', 0x01, struct orbit_desc_submit)
```

Semantics:
1. Copy `count` × 64-byte descriptors from `descs_ptr` (userspace) into a temporary kernel buffer.
2. Validate each descriptor: type in known range, address alignment, no reserved fields set.
3. Call `orbit_queue_submit()` for the target queue.
4. On success, return a `submit_cookie` (monotonically increasing u64) that can be used with `WAIT_DONE`.
5. If `flags & ORBIT_SUBMIT_WAIT`, block until all submitted descriptors complete before returning.

#### ORBIT_IOC_WAIT_DONE

Block until a previously submitted batch completes.

```c
#define ORBIT_IOC_WAIT_DONE    _IOWR('O', 0x02, struct orbit_wait_done)
```

Semantics:
- Sleeps on `queue->wq` using `wait_event_timeout()`.
- Returns 0 on completion, `-ETIMEDOUT` on timeout, `-ERESTARTSYS` on signal.
- The `submit_cookie` identifies which batch to wait for (driver tracks the completion head at submit time).

#### ORBIT_IOC_ALLOC_MEM

Allocate a region from the GDDR6 memory pool.

```c
#define ORBIT_IOC_ALLOC_MEM    _IOWR('O', 0x03, struct orbit_mem_alloc)
```

Semantics:
1. Call `orbit_mem_alloc(&dev->gddr_pool, size_bytes, align)`.
2. On success, return `device_addr` (GDDR6-physical) and `handle` (opaque 64-bit token for later free).
3. Track allocation in the fd's `alloc_list` for cleanup on `release()`.

#### ORBIT_IOC_FREE_MEM

Free a previously allocated GDDR6 region.

```c
#define ORBIT_IOC_FREE_MEM     _IOW('O', 0x04, struct orbit_mem_free)
```

Semantics:
1. Look up `handle` in fd's `alloc_list`.
2. Call `orbit_mem_free(&dev->gddr_pool, handle)`.
3. Remove from `alloc_list`.

#### ORBIT_IOC_GET_INFO

Query device capabilities.

```c
#define ORBIT_IOC_GET_INFO     _IOR('O', 0x05, struct orbit_device_info)
```

Returns: gddr_size, queue_depth, num_queues, bar1_size (mmap window), fw_version, hw_revision, descriptor_spec_version, supported_desc_types (bitmask).

#### ORBIT_IOC_RESET_QUEUE

Drain and reset a specific queue (for error recovery).

```c
#define ORBIT_IOC_RESET_QUEUE  _IOW('O', 0x06, struct orbit_queue_reset)
```

### 2.10 mmap Support — Zero-Copy Weight Upload

**File:** `orbit_g1_mmap.c`

Allows userspace to map the BAR1 GDDR6 window directly into its virtual address space, bypassing the kernel copy path.

```
orbit_g1_mmap(file, vma):
  offset = vma->vm_pgoff << PAGE_SHIFT
  size   = vma->vm_end - vma->vm_start

  if offset + size > bar1_len: return -EINVAL
  if vma->vm_flags & VM_WRITE and not privileged: check policy

  vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot)
  return io_remap_pfn_range(vma, vma->vm_start,
                             (pci_resource_start(pdev, 1) + offset) >> PAGE_SHIFT,
                             size, vma->vm_page_prot)
```

Usage from userspace:
- Open `/dev/orbit_g1_0`.
- `ORBIT_IOC_ALLOC_MEM` → get `device_addr` (GDDR6-physical) and offset within BAR1 window.
- `mmap(fd, ..., MAP_SHARED, offset_within_bar1)` → get host virtual pointer.
- `memcpy(ptr, weight_data, weight_size)` → direct GDDR6 write over PCIe.
- Issue BARRIER descriptor before executing GEMM on those weights.

### 2.11 GDDR6 Buddy Allocator (Kernel Side)

**File:** `orbit_g1_mem.c`

A simple power-of-two buddy allocator managing the GDDR6 address space visible to the driver. Not all GDDR6 may be mappable via BAR1 (BAR1 window may be 256 MB for a 16 GB device), but the allocator tracks the full GDDR6 space — allocation returns device addresses which are used in descriptor fields regardless of BAR1 visibility.

```c
struct orbit_mem_pool {
    u64          base;          /* GDDR6 base address (usually 0) */
    u64          size;          /* total GDDR6 size in bytes */
    spinlock_t   lock;
    struct list_head free_lists[ORBIT_MEM_MAX_ORDER]; /* order 0..max */
    /* buddy bitmap */
    unsigned long *bitmap;
    u32           min_order;    /* minimum allocation: 2^min_order bytes (e.g., 4KB = order 12) */
    u32           max_order;    /* maximum allocation order */
};
```

Minimum allocation granularity: 4 KB (page-aligned, matching BAR1 mmap granularity).

### 2.12 debugfs Interface

**File:** `orbit_g1_debug.c`

Mounted at `/sys/kernel/debug/orbit_g1/orbit_g1_0/`:

```
status          — global status register dump
queue_0/        — per-queue head, tail, inflight, error_code
queue_1/
queue_2/
queue_3/
mem_pool        — buddy allocator free/used summary
gddr_size       — total GDDR6 size
fw_version      — firmware version string
reset           — write 1 to trigger soft reset
```

---

## 3. Userspace Runtime Library — liborbit.so

**Language:** C++17
**Build:** CMake, produces `liborbit.so` (shared) and `liborbit.a` (static)
**Public API header:** `include/orbit.h`

### 3.1 Module Overview

```
runtime/src/
  device.cpp            — Device class
  memory_pool.cpp       — MemoryPool (GDDR6 buddy client)
  descriptor_queue.cpp  — DescriptorQueue builder + submit + wait
  weight_loader.cpp     — WeightLoader (INT4/INT8 quantized models)
  kv_cache_manager.cpp  — KVCacheManager (PagedAttention)
  inference_session.cpp — InferenceSession (per-request state)
  executor.cpp          — Executor (21-step decode orchestration)
  util/
    desc_builder.cpp    — low-level 64-byte descriptor construction
    quantize.cpp        — host-side INT4/INT8 quantization helpers
    error.cpp           — error code mapping
```

### 3.2 Device Class

**File:** `runtime/src/device.cpp`
**Header:** `include/orbit/device.h`

The `Device` class is the entry point. It opens `/dev/orbit_g1_N`, queries device info, and provides ioctl wrappers.

```cpp
class Device {
public:
    // Factory: open /dev/orbit_g1_N (default N=0)
    static std::unique_ptr<Device> open(int card_index = 0);
    ~Device();

    // Device info
    const OrbitDeviceInfo& info() const;

    // Raw ioctl wrappers (used by higher-level modules)
    int submit_desc(const OrbitDescSubmit& req);
    int wait_done(const OrbitWaitDone& req);
    int alloc_gddr(size_t size_bytes, size_t align, OrbitMemAlloc* out);
    int free_gddr(uint64_t handle);
    int reset_queue(uint32_t queue_id);

    // mmap BAR1 window
    void* mmap_bar1(uint64_t bar1_offset, size_t length);
    void  munmap_bar1(void* ptr, size_t length);

    // File descriptor (for poll/epoll integration)
    int fd() const { return fd_; }

private:
    int             fd_;
    OrbitDeviceInfo info_;

    Device(int fd, const OrbitDeviceInfo& info);
};
```

**Design notes:**
- One `Device` instance per open fd. Multiple instances are allowed (multi-process sharing).
- Thread-safety: individual ioctl calls are protected by the kernel driver. The `Device` class itself is thread-safe for concurrent ioctl calls.
- Error handling: all methods return negative errno on failure (Linux convention) or throw `OrbitException` depending on the build configuration (`ORBIT_THROW_ON_ERROR`).

### 3.3 MemoryPool

**File:** `runtime/src/memory_pool.cpp`
**Header:** `include/orbit/memory_pool.h`

Client-side GDDR6 memory manager. Wraps `ORBIT_IOC_ALLOC_MEM` / `ORBIT_IOC_FREE_MEM` with a slab-style front-end to reduce ioctl overhead for small allocations.

```cpp
class MemoryPool {
public:
    explicit MemoryPool(Device& dev);
    ~MemoryPool();  // frees all remaining allocations

    // Allocate a GDDR6 region. Returns an opaque handle.
    MemHandle alloc(size_t size_bytes, size_t align = 4096);

    // Free a previously allocated region.
    void free(MemHandle handle);

    // Get the device address (GDDR6-physical) of a handle.
    uint64_t device_addr(MemHandle handle) const;

    // Get a host-mapped pointer via BAR1 mmap (if region is within BAR1 window).
    // Returns nullptr if region is outside BAR1 aperture.
    void* host_ptr(MemHandle handle);

    // Total and free GDDR6 bytes (approximate, from driver).
    uint64_t total_bytes() const;
    uint64_t free_bytes() const;

private:
    Device&     dev_;
    struct AllocEntry {
        uint64_t device_addr;
        uint64_t kernel_handle;
        size_t   size;
        void*    host_ptr;      /* nullptr if not mmap'd */
        bool     bar1_mapped;
    };
    std::unordered_map<MemHandle, AllocEntry> allocs_;
    mutable std::mutex mutex_;

    MemHandle next_handle_{1};

    // Slab cache for small allocations (< 1 MB) to batch ioctl calls.
    // Large allocations (>= 1 MB) go directly to ioctl.
    struct Slab {
        MemHandle parent_handle;
        uint64_t  base_device_addr;
        size_t    slab_size;        /* e.g., 64 MB */
        /* bump pointer sub-allocator within slab */
        size_t    used;
    };
    std::vector<Slab> slabs_;
};
```

**MemHandle** is `uint64_t` (typedef). Value 0 is always invalid.

**Design notes:**
- Slabs are allocated in 64 MB chunks from the driver. Sub-allocations within a slab avoid ioctl round-trips.
- `host_ptr()` calls `mmap_bar1()` lazily on first access, caching the result.
- Destructor calls `free()` on all outstanding allocations — suitable for RAII weight management.

### 3.4 DescriptorQueue

**File:** `runtime/src/descriptor_queue.cpp`
**Header:** `include/orbit/descriptor_queue.h`

Batch builder and submission interface for one logical queue.

```cpp
class DescriptorQueue {
public:
    DescriptorQueue(Device& dev, uint32_t queue_id, size_t max_batch = 64);

    // Descriptor builder methods — each appends one 64-byte descriptor to the batch.
    void add_dma_2d(uint64_t src, uint64_t dst,
                    uint32_t width_bytes, uint32_t height,
                    uint32_t src_stride, uint32_t dst_stride);

    void add_gemm_int8(uint64_t act_addr, uint64_t wgt_addr, uint64_t out_addr,
                       uint32_t Kt, uint16_t m_tiles = 1, uint16_t n_tiles = 1);

    void add_gemm_int4(uint64_t act_addr, uint64_t wgt_addr, uint64_t out_addr,
                       uint32_t Kt);

    void add_vector_op(uint64_t src, uint64_t dst, uint32_t element_count,
                       VectorOpType op, DataType dtype, uint32_t imm = 0);

    void add_vector_op_ex(uint64_t src, uint64_t dst, uint32_t element_count,
                          VectorOpExType op, DataType dtype);

    void add_kvc_read(uint32_t layer_id, uint32_t head_id,
                      uint32_t seq_start, uint32_t seq_len, uint64_t dst_addr);

    void add_kvc_write(uint32_t layer_id, uint32_t head_id,
                       uint32_t seq_pos, uint64_t src_addr);

    void add_moe_route(uint64_t logits_addr, uint64_t indices_addr,
                       uint64_t scores_addr, uint32_t num_tokens,
                       uint32_t num_experts, uint32_t top_k);

    void add_format_convert(uint64_t src, uint64_t dst, uint32_t element_count,
                             DataFormat src_fmt, DataFormat dst_fmt, uint32_t options = 0);

    void add_copy_2d_plus(uint64_t src, uint64_t dst,
                          uint32_t width_bytes, uint32_t height,
                          uint32_t src_stride, uint32_t dst_stride, uint32_t options = 0);

    void add_frame_fingerprint(uint64_t src, uint64_t result_addr,
                                uint32_t byte_count, HashType hash_type);

    void add_barrier();

    void add_event(uint32_t event_id, uint32_t options = 0);

    void add_perf_snapshot(uint64_t dst_addr);

    void add_softmax(uint64_t src, uint64_t dst, uint32_t element_count);

    // Submit the current batch.
    // Returns a SubmitCookie for use with wait().
    SubmitCookie submit();

    // Submit and block until completion.
    void submit_and_wait(uint32_t timeout_ms = 5000);

    // Wait for a previously submitted cookie.
    void wait(SubmitCookie cookie, uint32_t timeout_ms = 5000);

    // Reset/clear the pending batch without submitting.
    void reset_batch();

    size_t batch_size() const;

private:
    Device&     dev_;
    uint32_t    queue_id_;
    size_t      max_batch_;

    // In-memory batch, grows up to max_batch descriptors.
    std::vector<std::array<uint8_t, 64>> batch_;

    template<typename T>
    void append_desc(const T& desc_struct);
};
```

**Design notes:**
- `DescriptorQueue` is NOT thread-safe. Each thread should use its own instance, or use external locking.
- The `batch_` vector holds raw 64-byte descriptor bytes, which are bulk-copied into the kernel via `ORBIT_IOC_SUBMIT_DESC`.
- `add_barrier()` inserts a BARRIER descriptor (type 0x07) — this is required between operations on different hardware units that must be ordered.
- The `SubmitCookie` is the `cookie` value returned from `ORBIT_IOC_SUBMIT_DESC`.

### 3.5 WeightLoader

**File:** `runtime/src/weight_loader.cpp`
**Header:** `include/orbit/weight_loader.h`

Loads quantized model weights from host filesystem into GDDR6 memory.

```cpp
class WeightLoader {
public:
    WeightLoader(Device& dev, MemoryPool& pool, DescriptorQueue& util_queue);

    // Load a weight tensor from a file.
    // format: INT8 or INT4 (AWQ).
    // Returns a MemHandle pointing to the loaded GDDR6 region.
    MemHandle load_weight_file(const std::string& path, QuantFormat format);

    // Load from a host memory buffer (already quantized).
    MemHandle load_weight_buffer(const void* data, size_t size_bytes, QuantFormat format);

    // Async variant: returns immediately, completion signaled via callback.
    using LoadCallback = std::function<void(MemHandle, int err)>;
    void load_weight_async(const std::string& path, QuantFormat format, LoadCallback cb);

    // Get weight tensor layout info.
    struct WeightInfo {
        MemHandle  handle;
        uint64_t   device_addr;
        size_t     size_bytes;
        QuantFormat format;
        std::vector<uint32_t> shape;
    };
    const WeightInfo& weight_info(MemHandle handle) const;

    // Unload (free GDDR6 region).
    void unload(MemHandle handle);

private:
    Device&          dev_;
    MemoryPool&      pool_;
    DescriptorQueue& util_queue_;

    std::unordered_map<MemHandle, WeightInfo> loaded_;
    mutable std::mutex mutex_;

    // Upload via BAR1 mmap (fast path, if region fits in BAR1 window).
    MemHandle upload_via_mmap(const void* data, size_t size_bytes);

    // Upload via DMA_2D descriptor (fallback for large weights outside BAR1 window).
    // Requires a temporary pinned host buffer.
    MemHandle upload_via_dma(const void* data, size_t size_bytes);
};
```

**Upload strategy:**
1. Allocate GDDR6 region via `MemoryPool::alloc()`.
2. If `host_ptr()` returns non-null (within BAR1 window): `memcpy()` directly — zero kernel overhead.
3. If outside BAR1 window: allocate a DMA-coherent pinned buffer via a helper ioctl or `posix_memalign`, copy weight data there, then issue a `DMA_2D` descriptor to move it within GDDR6.
4. Issue `BARRIER` before returning handle.

**INT4 (AWQ) handling:**
- AWQ weights are stored as packed INT4 with per-group FP16 scales and zeros.
- `WeightLoader` uploads the packed INT4 bytes as-is; the GEMM_INT4 descriptor handles dequantization on-chip.
- Scale/zero tensors are uploaded separately to a companion GDDR6 region and the handle stored in `WeightInfo`.

### 3.6 KVCacheManager

**File:** `runtime/src/kv_cache_manager.cpp`
**Header:** `include/orbit/kv_cache_manager.h`

PagedAttention-style KV-cache allocator. Divides GDDR6 KV space into fixed-size pages and maps logical sequence positions to physical pages.

```cpp
class KVCacheManager {
public:
    struct Config {
        uint32_t num_layers;       /* e.g., 32 for GPT-OSS-20B */
        uint32_t num_heads;        /* e.g., 32 */
        uint32_t head_dim;         /* e.g., 128 */
        uint32_t page_size_tokens; /* tokens per page, e.g., 16 */
        uint32_t max_pages;        /* total pages available */
        DataType dtype;            /* FP16 typically */
    };

    KVCacheManager(Device& dev, MemoryPool& pool, const Config& cfg);
    ~KVCacheManager();

    // Allocate KV pages for a new sequence. Returns a sequence handle.
    using SeqHandle = uint32_t;
    SeqHandle alloc_sequence(uint32_t initial_capacity_tokens = 0);

    // Extend an existing sequence (add more pages as generation proceeds).
    // Returns true on success, false if OOM.
    bool extend_sequence(SeqHandle seq, uint32_t additional_tokens);

    // Free all pages for a sequence (called when generation completes).
    void free_sequence(SeqHandle seq);

    // Get the GDDR6 device address for a specific token's KV slot.
    // Used by the Executor to build KVC_READ/KVC_WRITE descriptors.
    uint64_t kv_addr(SeqHandle seq, uint32_t layer_id, uint32_t head_id,
                     uint32_t token_pos, bool is_key) const;

    // Get total pages and free pages.
    uint32_t total_pages() const;
    uint32_t free_pages() const;

    // Page table for a sequence (for batch KVC_READ operations).
    struct PageTable {
        std::vector<uint64_t> page_addrs; /* GDDR6 addr of each page */
        uint32_t              num_tokens;
    };
    const PageTable& page_table(SeqHandle seq) const;

private:
    Device&      dev_;
    MemoryPool&  pool_;
    Config       cfg_;

    // Physical page pool
    struct KVPage {
        uint64_t device_addr;   /* GDDR6 base address of this page */
        bool     in_use;
    };
    std::vector<KVPage> pages_;
    std::queue<uint32_t> free_page_ids_;  /* indices into pages_ */
    mutable std::mutex mutex_;

    // Per-sequence page table
    struct SeqState {
        std::vector<uint32_t> page_ids;   /* physical page ids */
        uint32_t              num_tokens;
    };
    std::unordered_map<SeqHandle, SeqState> seqs_;
    SeqHandle next_seq_handle_{1};

    // Layout: one page covers page_size_tokens token positions
    //   within a page, layout is: [layer][head][token_within_page][K|V][head_dim]
    //   K at offset 0, V at offset head_dim * sizeof(dtype)
    size_t page_bytes() const;
    uint64_t token_offset_within_page(uint32_t layer_id, uint32_t head_id,
                                       uint32_t token_within_page, bool is_key) const;
};
```

**Memory layout per page:**

```
page_bytes = num_layers * num_heads * page_size_tokens * head_dim * 2 * sizeof(dtype)
             (factor 2: K and V)

For GPT-OSS-20B (FP16): 32 layers × 32 heads × 16 tokens × 128 dim × 2 × 2B = 8 MB per page
```

**Design notes:**
- Pages are pre-allocated at construction time to avoid allocation latency during inference.
- `kv_addr()` computes: `page_base + layer_offset + head_offset + token_offset + k_or_v_offset`.
- The `Executor` is responsible for mapping PagedAttention page lookups to actual KVC_READ/KVC_WRITE descriptor generation.

### 3.7 InferenceSession

**File:** `runtime/src/inference_session.cpp`
**Header:** `include/orbit/inference_session.h`

Per-request state container.

```cpp
class InferenceSession {
public:
    struct Config {
        uint32_t  max_new_tokens;
        uint32_t  temperature_fp16;   /* FP16 bits */
        uint32_t  top_p_fp16;
        bool      do_sample;
        uint32_t  seed;
    };

    InferenceSession(KVCacheManager& kvc_mgr, const Config& cfg);
    ~InferenceSession();  /* frees KV pages */

    // Session lifecycle
    void prefill(const std::vector<int32_t>& input_token_ids);
    bool step();    /* decode one token; returns false when EOS or max_tokens reached */
    void cancel();

    // Current state
    uint32_t                    seq_id() const;
    uint32_t                    context_length() const;   /* prompt + generated so far */
    const std::vector<int32_t>& generated_tokens() const;
    int32_t                     last_token() const;
    bool                        is_done() const;

    // KV-cache handle (used by Executor to build descriptors).
    KVCacheManager::SeqHandle   kv_handle() const;

    // Scratch buffer addresses in GDDR6 (allocated by Executor during session init).
    struct ScratchBuffers {
        uint64_t hidden_state;      /* [1 × d_model] FP16 */
        uint64_t qkv_proj;          /* [3 × d_model] FP16 */
        uint64_t attention_scores;  /* [num_heads × seq_len] FP16 */
        uint64_t attention_out;     /* [d_model] FP16 */
        uint64_t ffn_gate;          /* [d_ffn] FP16 per expert */
        uint64_t ffn_up;            /* [d_ffn] FP16 per expert */
        uint64_t moe_logits;        /* [num_experts] FP16 */
        uint64_t moe_indices;       /* [top_k] INT32 */
        uint64_t moe_scores;        /* [top_k] FP16 */
        uint64_t lm_head_out;       /* [vocab_size] FP16 */
    };
    const ScratchBuffers& scratch() const;

private:
    KVCacheManager&              kvc_mgr_;
    KVCacheManager::SeqHandle    kv_handle_;
    Config                       cfg_;
    uint32_t                     seq_id_;
    uint32_t                     context_len_;
    std::vector<int32_t>         generated_tokens_;
    bool                         done_;
    ScratchBuffers               scratch_;

    static std::atomic<uint32_t> next_seq_id_;
};
```

### 3.8 Executor

**File:** `runtime/src/executor.cpp`
**Header:** `include/orbit/executor.h`

Orchestrates the 21-step descriptor sequence for one token decode of GPT-OSS-20B. This is the highest-level module in liborbit.

```cpp
class Executor {
public:
    struct ModelConfig {
        uint32_t  num_layers;       /* 32 */
        uint32_t  d_model;          /* e.g., 4096 */
        uint32_t  num_heads;        /* 32 */
        uint32_t  head_dim;         /* 128 */
        uint32_t  num_experts;      /* 32 */
        uint32_t  top_k_experts;    /* 2 */
        uint32_t  d_ffn;            /* FFN hidden dim per expert */
        uint32_t  vocab_size;
        QuantFormat weight_format;  /* INT8 or INT4 */
    };

    Executor(Device& dev, MemoryPool& pool, KVCacheManager& kvc_mgr,
             const ModelConfig& model_cfg);
    ~Executor();

    // Load all model weights. Must be called once before run_prefill/run_decode.
    void load_weights(const std::string& model_dir);

    // Prefill: process prompt tokens.
    void run_prefill(InferenceSession& session);

    // Decode: generate one token. Updates session state.
    // Returns the generated token id.
    int32_t run_decode_step(InferenceSession& session);

private:
    Device&          dev_;
    MemoryPool&      pool_;
    KVCacheManager&  kvc_mgr_;
    ModelConfig      model_cfg_;
    WeightLoader     weight_loader_;

    // Compute queue for GEMM/VPU/KVC operations.
    DescriptorQueue  compute_q_;    /* queue 0 */
    // Utility queue for DMA weight prefetch.
    DescriptorQueue  util_q_;       /* queue 1 */

    // Loaded weight handles (per layer).
    struct LayerWeights {
        MemHandle q_proj, k_proj, v_proj, o_proj;
        MemHandle router;
        std::vector<MemHandle> expert_gate;   /* num_experts */
        std::vector<MemHandle> expert_up;
        std::vector<MemHandle> expert_down;
        MemHandle rms_norm_weight;
    };
    std::vector<LayerWeights> layer_weights_;  /* num_layers */
    MemHandle embedding_weight_;
    MemHandle lm_head_weight_;

    // Build the descriptor sequence for one decode step of one transformer layer.
    // All descriptors are appended to compute_q_.
    void build_layer_decode(InferenceSession& session, uint32_t layer_id);

    // Step-by-step descriptor builders (maps to the 21-step sequence):
    void step_01_dma_embedding(InferenceSession& session);
    void step_02_qkv_projection(InferenceSession& session, uint32_t layer_id);
    void step_03_rope(InferenceSession& session);
    void step_04_kvc_write(InferenceSession& session, uint32_t layer_id);
    void step_05_kvc_read(InferenceSession& session, uint32_t layer_id);
    void step_06_attn_qkt(InferenceSession& session);
    void step_07_scale_softmax(InferenceSession& session);
    void step_08_attn_v(InferenceSession& session);
    void step_09_output_proj(InferenceSession& session, uint32_t layer_id);
    void step_10_residual_add(InferenceSession& session);
    void step_11_rmsnorm(InferenceSession& session, uint32_t layer_id);
    void step_12_moe_router_gemm(InferenceSession& session, uint32_t layer_id);
    void step_13_moe_route(InferenceSession& session);
    void step_14_expert_gate(InferenceSession& session, uint32_t layer_id);
    void step_15_silu(InferenceSession& session);
    void step_16_expert_up(InferenceSession& session, uint32_t layer_id);
    void step_17_element_mul(InferenceSession& session);
    void step_18_expert_down(InferenceSession& session, uint32_t layer_id);
    void step_19_barrier_residual();
    // After all layers:
    void step_20_lm_head(InferenceSession& session);
    void step_21_softmax_argmax_event(InferenceSession& session);
};
```

**21-Step Decode Descriptor Sequence (per token, all layers):**

```
For each layer (0 to num_layers-1):
  Step  1: DMA_2D         — input embedding → scratch.hidden_state
  Step  2: GEMM_INT8/4    — [Q, K, V] = hidden @ [Wq, Wk, Wv]  (3 GEMMs, submittable as 1 batch)
  Step  3: VECTOR_OP_EX   — RoPE on Q and K
  Step  4: KVC_WRITE      — store new K, V to KV-Cache
  Step  5: KVC_READ       — load all K, V (seq_len entries) from KV-Cache
  Step  6: GEMM_INT8/4    — attention_scores = Q @ K^T
  Step  7: SOFTMAX        — scale(1/√head_dim) + softmax(attention_scores)
  Step  8: GEMM_INT8/4    — attention_out = attention_scores @ V
  Step  9: GEMM_INT8/4    — hidden = attention_out @ Wo
  Step 10: VECTOR_OP      — residual add: hidden += original_hidden
  Step 11: VECTOR_OP_EX   — RMSNorm(hidden)
  Step 12: GEMM_INT8/4    — moe_logits = hidden @ router_weight
  Step 13: MOE_ROUTE      — top-2 expert indices + scores
  Step 14: GEMM_INT8/4    — gate_proj (for each selected expert, sequentially)
  Step 15: VECTOR_OP_EX   — SiLU(gate_proj)
  Step 16: GEMM_INT8/4    — up_proj (for each selected expert)
  Step 17: VECTOR_OP      — gated_ffn = SiLU(gate) * up
  Step 18: GEMM_INT8/4    — down_proj (for each selected expert)
  Step 19: BARRIER + VECTOR_OP — expert_out combine + residual add
  (BARRIER between layers to ensure ordering)

After all layers:
  Step 20: GEMM_INT8/4    — lm_head = hidden @ Wlm
  Step 21: VECTOR_OP + EVENT — softmax(lm_head) → argmax → next_token_id
                              — EVENT notifies host with token id
```

**MoE execution note:** For GPT-OSS-20B top-2 routing, steps 14–18 are executed once per selected expert (2 iterations). The `MOE_ROUTE` result (indices) is read back by the host (via a small GDDR6→host DMA after step 13) to determine which expert weight handles to reference in steps 14–18.

---

## 4. OpenAI-compatible Inference Server — orbit_server

**Language:** C++ (with Python bindings optional via pybind11)
**Build:** CMake, produces `orbit_server` binary
**Protocol:** HTTP/1.1 via `httplib.h` (cpp-httplib, header-only) or libmicrohttpd

### 4.1 Architecture Overview

```
                   ┌─────────────────────────────────────────┐
HTTP client ──────►│           orbit_server                   │
                   │                                          │
                   │  ┌──────────────────────────────────┐   │
                   │  │  HTTP Layer (cpp-httplib)         │   │
                   │  │  POST /v1/chat/completions        │   │
                   │  │  GET  /v1/models                  │   │
                   │  │  GET  /health                     │   │
                   │  └─────────────┬────────────────────┘   │
                   │                │                         │
                   │  ┌─────────────▼────────────────────┐   │
                   │  │  RequestQueue                     │   │
                   │  │  (thread-safe priority queue)     │   │
                   │  └─────────────┬────────────────────┘   │
                   │                │                         │
                   │  ┌─────────────▼────────────────────┐   │
                   │  │  SessionManager                   │   │
                   │  │  (creates InferenceSession,       │   │
                   │  │   manages lifecycle)              │   │
                   │  └─────────────┬────────────────────┘   │
                   │                │                         │
                   │  ┌─────────────▼────────────────────┐   │
                   │  │  Executor                         │   │
                   │  │  (liborbit — single instance)     │   │
                   │  └─────────────────────────────────-─┘   │
                   └─────────────────────────────────────────┘
```

### 4.2 Modules

**File layout:**

```
runtime/server/
  main.cpp                — arg parsing, server startup
  http_handler.cpp        — HTTP request parsing, response formatting
  request_queue.cpp       — thread-safe FIFO queue with cancellation
  session_manager.cpp     — session lifecycle, streaming state
  tokenizer.cpp           — BPE tokenizer (tiktoken-compatible)
  sse_writer.cpp          — SSE chunked response writer
  openai_types.h          — ChatCompletion request/response structs
  config.h                — server config (port, model path, max concurrency)
```

### 4.3 POST /v1/chat/completions

**Request parsing:**

```cpp
struct ChatCompletionRequest {
    std::string              model;
    std::vector<ChatMessage> messages;
    float                    temperature;   /* default 1.0 */
    float                    top_p;         /* default 1.0 */
    int32_t                  max_tokens;    /* default 512 */
    bool                     stream;        /* true for SSE */
    std::optional<int32_t>   seed;
};
```

**Handler flow:**

```
1. Parse JSON body → ChatCompletionRequest
2. Tokenize messages (apply chat template → token ids)
3. Enqueue request in RequestQueue (with cancel token)
4. If stream=true:
     Set response headers: Content-Type: text/event-stream
     Start chunked response
     Worker loop:
       session.prefill(token_ids)
       while not session.is_done():
         token = executor.run_decode_step(session)
         text = tokenizer.decode(token)
         sse_writer.write_token_chunk(response, text, session.seq_id())
       sse_writer.write_done(response)
   If stream=false:
     Collect all tokens, return full JSON response
```

### 4.4 SSE Streaming Format

SSE chunks follow the OpenAI streaming format:

```
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"index":0}]}

data: [DONE]
```

### 4.5 Request Queue + Session Management

**RequestQueue:** Fixed-capacity priority queue (default capacity: 64 requests).
- FIFO ordering (priority field reserved for future use).
- `push()` blocks if full (or returns error with 503 if configured non-blocking).
- Worker thread pool: 1 worker thread (ORBIT-G1 has one compute queue; parallelism is within a single request via descriptor batching, not across requests).

**SessionManager:**
- Creates `InferenceSession` per request.
- Enforces max concurrent sessions (limited by `KVCacheManager` total pages).
- On cancellation (client disconnect): calls `session.cancel()` which stops decode loop.
- Cleanup: frees KV pages immediately on session end.

### 4.6 Tokenizer

- Implements BPE decoding compatible with GPT-based models.
- Loads a `tokenizer.json` (HuggingFace format) at server startup.
- Applies a configurable chat template (e.g., Llama-3 / Mixtral format).
- Special tokens: EOS token id is model-specific, configurable via `config.json`.

### 4.7 Configuration

```cpp
struct ServerConfig {
    std::string  model_dir;          /* path to model weights */
    std::string  device_node;        /* /dev/orbit_g1_0 */
    uint16_t     port;               /* default 8080 */
    uint32_t     max_batch_tokens;   /* max tokens per request */
    uint32_t     max_kv_pages;       /* limits concurrent sessions */
    bool         enable_int4;        /* use GEMM_INT4 if available */
    std::string  log_level;          /* info, debug, warn */
};
```

---

## 5. Directory Structure

```
/home/dmsal020813/project/yua-t16/
├── spec/
│   ├── orbit-g1.md
│   ├── descriptor.md
│   ├── yua-llm-hw-design.md
│   ├── kvc.md
│   ├── vpu.md
│   └── driver-runtime-design.md        ← THIS FILE
│
├── driver/                              ← Linux kernel module (orbit_g1.ko)
│   ├── Makefile
│   ├── Kconfig
│   ├── orbit_g1_main.c                 — pci_driver probe/remove, module init/exit
│   ├── orbit_g1_mmio.c                 — BAR0/BAR1 map, register accessors
│   ├── orbit_g1_queue.c                — descriptor ring buffer, doorbell
│   ├── orbit_g1_dma.c                  — dma_alloc_coherent management
│   ├── orbit_g1_irq.c                  — MSI-X ISR, completion wake
│   ├── orbit_g1_cdev.c                 — /dev/orbit_g1_N char device
│   ├── orbit_g1_ioctl.c                — ioctl handler dispatch
│   ├── orbit_g1_mmap.c                 — BAR1 zero-copy mmap
│   ├── orbit_g1_mem.c                  — GDDR6 buddy allocator
│   ├── orbit_g1_debug.c                — debugfs entries
│   ├── orbit_g1.h                      — internal driver structs (not userspace)
│   └── include/
│       └── uapi/
│           └── orbit_g1.h              — EXPORTED: ioctl numbers, structs (kernel uapi)
│
├── runtime/                             ← Userspace library + server
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── orbit.h                     — PUBLIC C++ API (all-in-one include)
│   │   └── orbit/
│   │       ├── device.h
│   │       ├── memory_pool.h
│   │       ├── descriptor_queue.h
│   │       ├── weight_loader.h
│   │       ├── kv_cache_manager.h
│   │       ├── inference_session.h
│   │       ├── executor.h
│   │       ├── types.h                 — enums: QuantFormat, DataType, VectorOpType, etc.
│   │       └── error.h                 — OrbitException, error codes
│   ├── src/
│   │   ├── device.cpp
│   │   ├── memory_pool.cpp
│   │   ├── descriptor_queue.cpp
│   │   ├── weight_loader.cpp
│   │   ├── kv_cache_manager.cpp
│   │   ├── inference_session.cpp
│   │   ├── executor.cpp
│   │   └── util/
│   │       ├── desc_builder.cpp        — raw 64-byte descriptor assembly
│   │       ├── quantize.cpp            — host-side INT4/INT8 quant helpers
│   │       └── error.cpp               — errno → OrbitError mapping
│   ├── server/
│   │   ├── main.cpp
│   │   ├── http_handler.cpp
│   │   ├── request_queue.cpp
│   │   ├── session_manager.cpp
│   │   ├── tokenizer.cpp
│   │   ├── sse_writer.cpp
│   │   ├── openai_types.h
│   │   └── config.h
│   └── tests/
│       ├── test_device.cpp
│       ├── test_memory_pool.cpp
│       ├── test_descriptor_queue.cpp
│       ├── test_weight_loader.cpp
│       ├── test_kv_cache.cpp
│       └── test_executor.cpp
│
└── tools/
    ├── orbit_info.c                     — CLI tool: query device info
    ├── orbit_bench.cpp                  — benchmark descriptor throughput
    └── orbit_diag.cpp                   — diagnostic runner (DIAG_RUN descriptors)
```

---

## 6. Key Header Definitions

### 6.1 Kernel UAPI Header — `driver/include/uapi/orbit_g1.h`

This header is copied to `/usr/include/linux/orbit_g1.h` on driver install and included by liborbit.

```c
/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * ORBIT-G1 Kernel UAPI
 * User/kernel interface: ioctl structs and numbers.
 * All structs are ABI-stable. Do not reorder fields.
 */
#ifndef _UAPI_ORBIT_G1_H
#define _UAPI_ORBIT_G1_H

#include <linux/types.h>
#include <linux/ioctl.h>

#define ORBIT_IOC_MAGIC  'O'

/* ---------- Constants ---------- */

#define ORBIT_NUM_QUEUES         4
#define ORBIT_QUEUE_DEPTH        256
#define ORBIT_DESC_SIZE          64    /* bytes, fixed */
#define ORBIT_MAX_SUBMIT_BATCH   256   /* max descriptors per submit ioctl */
#define ORBIT_MIN_ALLOC_BYTES    4096  /* 4KB minimum GDDR6 allocation */

/* Queue IDs */
#define ORBIT_QUEUE_COMPUTE      0
#define ORBIT_QUEUE_UTILITY      1
#define ORBIT_QUEUE_TELEMETRY    2
#define ORBIT_QUEUE_RESERVED     3

/* Descriptor type IDs (v1 + v2) */
#define ORBIT_DESC_DMA_2D        0x01
#define ORBIT_DESC_GEMM_INT8     0x02
#define ORBIT_DESC_VECTOR_OP     0x03
#define ORBIT_DESC_COPY_2D_PLUS  0x04
#define ORBIT_DESC_FORMAT_CONV   0x05
#define ORBIT_DESC_FRAME_FP      0x06
#define ORBIT_DESC_BARRIER       0x07
#define ORBIT_DESC_EVENT         0x08
#define ORBIT_DESC_PERF_SNAP     0x09
#define ORBIT_DESC_KVC_READ      0x0A
#define ORBIT_DESC_KVC_WRITE     0x0B
#define ORBIT_DESC_MOE_ROUTE     0x0C
#define ORBIT_DESC_VECTOR_OP_EX  0x0D
#define ORBIT_DESC_GEMM_INT4     0x0E
#define ORBIT_DESC_SOFTMAX       0x0F

/* Flags for orbit_desc_submit.flags */
#define ORBIT_SUBMIT_WAIT        (1U << 0)  /* block until batch completes */
#define ORBIT_SUBMIT_NOWAIT      (1U << 1)  /* return -ENOSPC if ring full */
#define ORBIT_SUBMIT_NO_INTR     (1U << 2)  /* suppress completion interrupt */

/* ---------- ioctl structs ---------- */

/*
 * ORBIT_IOC_SUBMIT_DESC
 * Submit a batch of descriptors to a queue.
 */
struct orbit_desc_submit {
    __u32  queue_id;          /* 0–3 */
    __u32  count;             /* number of descriptors, 1–ORBIT_MAX_SUBMIT_BATCH */
    __u32  flags;             /* ORBIT_SUBMIT_* flags */
    __u32  _pad;
    __u64  descs_ptr;         /* userspace pointer to count×64 bytes of descriptors */
    __u64  cookie_out;        /* OUTPUT: submit cookie for use with WAIT_DONE */
};

/*
 * ORBIT_IOC_WAIT_DONE
 * Block until a submitted batch completes or timeout expires.
 */
struct orbit_wait_done {
    __u32  queue_id;          /* must match the queue used in submit */
    __u32  timeout_ms;        /* 0 = poll (non-blocking), UINT32_MAX = forever */
    __u64  cookie;            /* cookie from orbit_desc_submit.cookie_out */
    __s32  status_out;        /* OUTPUT: 0=ok, negative errno on error */
    __u32  _pad;
};

/*
 * ORBIT_IOC_ALLOC_MEM
 * Allocate a region from GDDR6 memory.
 */
struct orbit_mem_alloc {
    __u64  size_bytes;        /* requested size (rounded up to min alignment) */
    __u64  align_bytes;       /* alignment (0 = default 4KB, must be power-of-2) */
    __u32  flags;             /* reserved, must be 0 */
    __u32  _pad;
    __u64  device_addr_out;   /* OUTPUT: GDDR6 device address */
    __u64  handle_out;        /* OUTPUT: opaque handle for FREE_MEM / mmap */
    __u64  bar1_offset_out;   /* OUTPUT: byte offset within BAR1 window
                                         (UINT64_MAX if not within BAR1 aperture) */
};

/*
 * ORBIT_IOC_FREE_MEM
 * Free a previously allocated GDDR6 region.
 */
struct orbit_mem_free {
    __u64  handle;            /* from orbit_mem_alloc.handle_out */
};

/*
 * ORBIT_IOC_GET_INFO
 * Query device capabilities and parameters.
 */
struct orbit_device_info {
    __u32  hw_revision;
    __u32  fw_version;
    __u32  desc_spec_version;     /* descriptor spec version (1 or 2) */
    __u32  num_queues;            /* always 4 */
    __u32  queue_depth;           /* always 256 */
    __u32  desc_size;             /* always 64 */
    __u64  gddr_total_bytes;
    __u64  gddr_free_bytes;
    __u64  bar1_window_bytes;     /* size of BAR1 mmap aperture */
    __u32  supported_desc_types;  /* bitmask, bit N = descriptor type N supported */
    __u32  sup_mode;              /* current SUP mode: 0=OFF,1=UTILITY,2=ASSURE,3=DEBUG */
    __u32  max_seq_len;           /* max KV-cache sequence length */
    __u32  _pad[3];
};

/*
 * ORBIT_IOC_RESET_QUEUE
 * Drain and reset a queue (error recovery).
 */
struct orbit_queue_reset {
    __u32  queue_id;
    __u32  flags;    /* reserved, must be 0 */
};

/* ---------- ioctl numbers ---------- */

#define ORBIT_IOC_SUBMIT_DESC  _IOWR(ORBIT_IOC_MAGIC, 0x01, struct orbit_desc_submit)
#define ORBIT_IOC_WAIT_DONE    _IOWR(ORBIT_IOC_MAGIC, 0x02, struct orbit_wait_done)
#define ORBIT_IOC_ALLOC_MEM    _IOWR(ORBIT_IOC_MAGIC, 0x03, struct orbit_mem_alloc)
#define ORBIT_IOC_FREE_MEM     _IOW (ORBIT_IOC_MAGIC, 0x04, struct orbit_mem_free)
#define ORBIT_IOC_GET_INFO     _IOR (ORBIT_IOC_MAGIC, 0x05, struct orbit_device_info)
#define ORBIT_IOC_RESET_QUEUE  _IOW (ORBIT_IOC_MAGIC, 0x06, struct orbit_queue_reset)

#endif /* _UAPI_ORBIT_G1_H */
```

### 6.2 liborbit Public API Header — `runtime/include/orbit.h`

```cpp
/*
 * orbit.h — ORBIT-G1 Userspace Runtime Public API
 * Include this single header to access all liborbit functionality.
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <stdexcept>

/* All public symbols are in the orbit:: namespace */
namespace orbit {

/* ─── Forward declarations ──────────────────────────────────────── */
class Device;
class MemoryPool;
class DescriptorQueue;
class WeightLoader;
class KVCacheManager;
class InferenceSession;
class Executor;

/* ─── Handle types ───────────────────────────────────────────────── */
using MemHandle    = uint64_t;
using SubmitCookie = uint64_t;
static constexpr MemHandle    INVALID_MEM_HANDLE = 0;
static constexpr SubmitCookie INVALID_COOKIE     = 0;

/* ─── Enumerations ───────────────────────────────────────────────── */

enum class QuantFormat : uint32_t {
    INT8    = 0,
    INT4    = 1,   /* AWQ-style packed INT4 */
    FP16    = 2,   /* unquantized, for debug */
};

enum class DataType : uint16_t {
    INT4   = 0,
    INT8   = 1,
    INT16  = 2,
    FP16   = 3,
    BF16   = 4,
    INT32  = 5,
};

enum class DataFormat : uint16_t {
    RGB    = 0,
    YUV    = 1,
    FP16   = 2,
    INT8   = 3,
    BF16   = 4,
};

enum class VectorOpType : uint16_t {
    ADD    = 0,
    MUL    = 1,
    MAX    = 2,
    MIN    = 3,
    CLAMP  = 4,
    SUB    = 5,
};

enum class VectorOpExType : uint16_t {
    RMSNORM  = 0,
    SILU     = 1,
    ROPE     = 2,
    RESIDUAL = 3,
    SCALE    = 4,
    GELU     = 5,
};

enum class HashType : uint32_t {
    CRC32  = 0,
    CRC64  = 1,
};

/* ─── Error handling ─────────────────────────────────────────────── */

class OrbitException : public std::runtime_error {
public:
    explicit OrbitException(int errnum, const std::string& msg)
        : std::runtime_error(msg), errnum_(errnum) {}
    int errnum() const { return errnum_; }
private:
    int errnum_;
};

/* ─── Device ─────────────────────────────────────────────────────── */

struct DeviceInfo {
    uint32_t hw_revision;
    uint32_t fw_version;
    uint32_t desc_spec_version;
    uint64_t gddr_total_bytes;
    uint64_t gddr_free_bytes;
    uint64_t bar1_window_bytes;
    uint32_t supported_desc_types;   /* bitmask */
    uint32_t sup_mode;
    uint32_t max_seq_len;
};

class Device {
public:
    static std::unique_ptr<Device> open(int card_index = 0);
    ~Device();

    const DeviceInfo& info() const;
    int fd() const;

    /* ioctl wrappers — return 0 on success, negative errno on failure */
    int submit_descriptors(uint32_t queue_id,
                           const void* descs,   /* count × 64 bytes */
                           uint32_t count,
                           uint32_t flags,
                           SubmitCookie* cookie_out);

    int wait_done(uint32_t queue_id,
                  SubmitCookie cookie,
                  uint32_t timeout_ms,
                  int* status_out);

    int alloc_gddr(uint64_t size_bytes,
                   uint64_t align_bytes,
                   uint64_t* device_addr_out,
                   uint64_t* handle_out,
                   uint64_t* bar1_offset_out);

    int free_gddr(uint64_t handle);
    int reset_queue(uint32_t queue_id);

    void* mmap_bar1(uint64_t bar1_offset, size_t length);
    void  munmap_bar1(void* ptr, size_t length);

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit Device(std::unique_ptr<Impl> impl);
};

/* ─── MemoryPool ─────────────────────────────────────────────────── */

class MemoryPool {
public:
    explicit MemoryPool(Device& dev);
    ~MemoryPool();

    MemHandle alloc(size_t size_bytes, size_t align = 4096);
    void      free(MemHandle handle);

    uint64_t device_addr(MemHandle handle) const;
    void*    host_ptr(MemHandle handle);   /* nullptr if not BAR1-mappable */

    uint64_t total_bytes() const;
    uint64_t free_bytes() const;

    MemoryPool(const MemoryPool&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/* ─── DescriptorQueue ────────────────────────────────────────────── */

class DescriptorQueue {
public:
    DescriptorQueue(Device& dev, uint32_t queue_id, size_t max_batch = 64);

    /* Descriptor append methods */
    void add_dma_2d(uint64_t src, uint64_t dst,
                    uint32_t width_bytes, uint32_t height,
                    uint32_t src_stride, uint32_t dst_stride);

    void add_gemm_int8(uint64_t act_addr, uint64_t wgt_addr, uint64_t out_addr,
                       uint32_t Kt, uint16_t m_tiles = 1, uint16_t n_tiles = 1);

    void add_gemm_int4(uint64_t act_addr, uint64_t wgt_addr, uint64_t out_addr,
                       uint32_t Kt);

    void add_vector_op(uint64_t src, uint64_t dst, uint32_t element_count,
                       VectorOpType op, DataType dtype, uint32_t imm = 0);

    void add_vector_op_ex(uint64_t src, uint64_t dst, uint32_t element_count,
                          VectorOpExType op, DataType dtype);

    void add_softmax(uint64_t src, uint64_t dst, uint32_t element_count);

    void add_kvc_read(uint32_t layer_id, uint32_t head_id,
                      uint32_t seq_start, uint32_t seq_len, uint64_t dst_addr);

    void add_kvc_write(uint32_t layer_id, uint32_t head_id,
                       uint32_t seq_pos, uint64_t src_addr);

    void add_moe_route(uint64_t logits_addr, uint64_t indices_addr,
                       uint64_t scores_addr, uint32_t num_tokens,
                       uint32_t num_experts, uint32_t top_k);

    void add_format_convert(uint64_t src, uint64_t dst, uint32_t element_count,
                             DataFormat src_fmt, DataFormat dst_fmt,
                             uint32_t options = 0);

    void add_copy_2d_plus(uint64_t src, uint64_t dst,
                          uint32_t width_bytes, uint32_t height,
                          uint32_t src_stride, uint32_t dst_stride,
                          uint32_t options = 0);

    void add_frame_fingerprint(uint64_t src, uint64_t result_addr,
                                uint32_t byte_count, HashType hash_type);

    void add_barrier();
    void add_event(uint32_t event_id, uint32_t options = 0);
    void add_perf_snapshot(uint64_t dst_addr);

    /* Submission */
    SubmitCookie submit();
    void         submit_and_wait(uint32_t timeout_ms = 5000);
    void         wait(SubmitCookie cookie, uint32_t timeout_ms = 5000);
    void         reset_batch();
    size_t       batch_size() const;

    DescriptorQueue(const DescriptorQueue&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/* ─── WeightLoader ───────────────────────────────────────────────── */

struct WeightInfo {
    MemHandle   handle;
    uint64_t    device_addr;
    size_t      size_bytes;
    QuantFormat format;
    std::vector<uint32_t> shape;
};

class WeightLoader {
public:
    WeightLoader(Device& dev, MemoryPool& pool, DescriptorQueue& util_queue);
    ~WeightLoader();

    MemHandle load_file(const std::string& path, QuantFormat format);
    MemHandle load_buffer(const void* data, size_t size_bytes, QuantFormat format);

    using LoadCallback = std::function<void(MemHandle, int err)>;
    void load_async(const std::string& path, QuantFormat format, LoadCallback cb);

    const WeightInfo& get_info(MemHandle handle) const;
    void              unload(MemHandle handle);

    WeightLoader(const WeightLoader&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/* ─── KVCacheManager ─────────────────────────────────────────────── */

class KVCacheManager {
public:
    struct Config {
        uint32_t num_layers;
        uint32_t num_heads;
        uint32_t head_dim;
        uint32_t page_size_tokens;
        uint32_t max_pages;
        DataType dtype;
    };

    using SeqHandle = uint32_t;
    static constexpr SeqHandle INVALID_SEQ = 0;

    KVCacheManager(Device& dev, MemoryPool& pool, const Config& cfg);
    ~KVCacheManager();

    SeqHandle alloc_sequence(uint32_t initial_capacity_tokens = 0);
    bool      extend_sequence(SeqHandle seq, uint32_t additional_tokens);
    void      free_sequence(SeqHandle seq);

    uint64_t  kv_addr(SeqHandle seq, uint32_t layer_id, uint32_t head_id,
                      uint32_t token_pos, bool is_key) const;

    uint32_t  total_pages() const;
    uint32_t  free_pages() const;

    KVCacheManager(const KVCacheManager&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/* ─── InferenceSession ───────────────────────────────────────────── */

class InferenceSession {
public:
    struct Config {
        uint32_t max_new_tokens;
        float    temperature;
        float    top_p;
        bool     do_sample;
        uint32_t seed;
    };

    struct ScratchBuffers {
        uint64_t hidden_state;
        uint64_t qkv_proj;
        uint64_t attention_scores;
        uint64_t attention_out;
        uint64_t ffn_gate;
        uint64_t ffn_up;
        uint64_t moe_logits;
        uint64_t moe_indices;
        uint64_t moe_scores;
        uint64_t lm_head_out;
    };

    InferenceSession(KVCacheManager& kvc_mgr, MemoryPool& pool, const Config& cfg);
    ~InferenceSession();

    void     prefill_tokens(const std::vector<int32_t>& token_ids);
    bool     is_prefilled() const;
    bool     is_done() const;
    void     cancel();

    uint32_t                     seq_id() const;
    uint32_t                     context_length() const;
    const std::vector<int32_t>&  generated_tokens() const;
    int32_t                      last_token() const;

    KVCacheManager::SeqHandle    kv_handle() const;
    const ScratchBuffers&        scratch() const;

    InferenceSession(const InferenceSession&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/* ─── Executor ───────────────────────────────────────────────────── */

class Executor {
public:
    struct ModelConfig {
        uint32_t    num_layers;
        uint32_t    d_model;
        uint32_t    num_heads;
        uint32_t    head_dim;
        uint32_t    num_experts;
        uint32_t    top_k_experts;
        uint32_t    d_ffn;
        uint32_t    vocab_size;
        QuantFormat weight_format;
    };

    Executor(Device& dev, MemoryPool& pool, KVCacheManager& kvc_mgr,
             const ModelConfig& model_cfg);
    ~Executor();

    void    load_weights(const std::string& model_dir);
    void    run_prefill(InferenceSession& session);
    int32_t run_decode_step(InferenceSession& session);

    Executor(const Executor&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} /* namespace orbit */
```

---

## 7. Data Flow Diagrams

### 7.1 Weight Upload Flow

```
Host filesystem
      │  read()
      ▼
   Host RAM (weight_data[])
      │  memcpy() via BAR1 mmap  ── fast path (within BAR1 window)
      │  OR
      │  DMA_2D descriptor        ── fallback (weight outside BAR1 aperture)
      ▼
   GDDR6 device memory
      │  MemHandle → device_addr
      ▼
   GEMM_INT8/4 descriptor wgt_addr field
      │
      ▼
   YUA-T16 compute tile
```

### 7.2 Single Token Decode Flow

```
Host: orbit_server → Executor::run_decode_step(session)
  │
  ├─ Build 21-step descriptor batch (compute_q_)
  │    Steps 1–21 per layer × num_layers
  │    Final: LM head + EVENT
  │
  ├─ compute_q_.submit()
  │    → ORBIT_IOC_SUBMIT_DESC (kernel)
  │        → orbit_queue_submit() — write to DMA ring, doorbell
  │            → ORBIT-G1 hardware executes descriptor sequence
  │                → YUA-T16 tiles, VPU, KVC, MoE Router
  │                → EVENT descriptor fires MSI-X interrupt
  │
  ├─ compute_q_.wait(cookie)
  │    → ORBIT_IOC_WAIT_DONE (kernel)
  │        → wait_event_timeout(&queue->wq) — woken by ISR
  │
  ├─ Read token_id from scratch.lm_head_out (DMA back to host or BAR1)
  │
  └─ session.record_token(token_id) → append to generated_tokens_
```

### 7.3 KV-Cache PagedAttention Flow

```
Session start: alloc_sequence() → assigns N pages from free pool
  Page 0 → tokens 0–15
  Page 1 → tokens 16–31
  ...

At token pos 16: extend_sequence() → allocate Page 1

KVC_WRITE (step 4): writes new K,V to current token's page slot
  kv_addr(seq, layer, head, token_pos, is_key)
  = page_base + layer_offset + head_offset + intra_page_offset

KVC_READ (step 5): reads K,V for all tokens 0..pos
  Split into per-page KVC_READ descriptors (hardware reads physically
  discontiguous pages — the driver must expand PagedAttention page tables
  into descriptor sequences if hardware doesn't support scatter page tables)
  Simplification for v1: if hardware KVC_READ accepts a page table address,
  the Executor uploads the page table to GDDR6 before issuing KVC_READ.
```

---

## 8. Error Handling Strategy

### 8.1 Kernel Driver

| Error condition | Kernel action |
|----------------|---------------|
| Invalid descriptor type | Return `-EINVAL` from ioctl; do not submit |
| Unaligned address in descriptor | Return `-EINVAL` |
| Ring buffer full (NOWAIT flag) | Return `-ENOSPC` |
| Ring buffer full (blocking) | Sleep on `queue->wq`, wake when head advances |
| Timeout in WAIT_DONE | Return `-ETIMEDOUT` |
| Hardware error status (Q_ERROR_CODE) | Wake waiter with error, set `status_out` negative |
| GDDR6 OOM | Return `-ENOMEM` from ALLOC_MEM |
| Device not ready | Return `-ENODEV` |

### 8.2 Userspace Runtime

| Error condition | liborbit action |
|----------------|-----------------|
| Kernel ioctl returns error | `OrbitException(errno, msg)` if `ORBIT_THROW_ON_ERROR`, else return code |
| KV-cache OOM | `extend_sequence()` returns `false`; server returns 503 |
| Weight load failure | `OrbitException` with file path and errno |
| Submit timeout | `OrbitException(ETIMEDOUT, ...)` |
| Session cancelled | `is_done()` returns `true`, `last_token()` returns `EOS` |

### 8.3 Descriptor Validation (kernel)

Before any descriptor batch is submitted to hardware, the kernel driver performs:
1. **Type check**: descriptor type in `[0x01, 0x0F]`.
2. **Size check**: `count ≤ ORBIT_MAX_SUBMIT_BATCH`.
3. **Address alignment**: `src_addr`, `dst_addr` must be 64-byte aligned (descriptor spec requirement for GEMM tile addresses).
4. **Range check**: addresses must be within GDDR6 range `[0, gddr_size)`.
5. **Reserved fields**: all reserved fields must be zero (in strict mode / DEBUG SUP mode).

---

## 9. Agent Implementation Assignments

This design document is intended for three parallel implementation agents. Each agent owns a distinct file set with no overlap.

### Agent A — Kernel Driver (`driver/`)

**Files owned:**
- `orbit_g1_main.c`, `orbit_g1_mmio.c`, `orbit_g1_queue.c`, `orbit_g1_dma.c`
- `orbit_g1_irq.c`, `orbit_g1_cdev.c`, `orbit_g1_ioctl.c`, `orbit_g1_mmap.c`
- `orbit_g1_mem.c`, `orbit_g1_debug.c`, `orbit_g1.h`
- `include/uapi/orbit_g1.h` (the header in section 6.1 above)
- `Makefile`, `Kconfig`

**Key constraints:**
- Must compile against Linux kernel 6.1+ headers.
- All DMA must go through the DMA API (`dma_alloc_coherent`, `dma_map_*`).
- No sleeping in interrupt context. ISRs do minimal work (clear interrupt, advance head, wake waiters).
- `orbit_g1.h` uapi section 6.1 is the ABI contract — do not change struct layouts.

### Agent B — Runtime Library (`runtime/src/`, `runtime/include/`)

**Files owned:**
- `runtime/src/*.cpp` and `runtime/src/util/*.cpp`
- `runtime/include/orbit.h` and `runtime/include/orbit/*.h`
- `runtime/CMakeLists.txt`
- `runtime/tests/*.cpp`

**Key constraints:**
- C++17. No C++20 features (compiler compatibility).
- `orbit.h` public API in section 6.2 is the ABI contract — do not change class interfaces.
- PIMPL idiom (`struct Impl`) is mandatory for all public classes (ABI stability).
- All ioctl calls go through `Device` class — no raw `ioctl()` calls in other modules.
- Thread safety: `Device` ioctl calls are safe to call from multiple threads. `DescriptorQueue` is single-threaded per instance.

### Agent C — Server (`runtime/server/`)

**Files owned:**
- `runtime/server/*.cpp`, `runtime/server/*.h`

**Key constraints:**
- Must not include any kernel headers.
- Depends only on `orbit.h` public API (no direct access to `Device` internals).
- HTTP layer: `cpp-httplib` (header-only, include as `httplib.h`). No Boost.Asio dependency.
- OpenAI streaming format must be byte-for-byte compatible with the OpenAI API spec (for drop-in client compatibility).
- Server must handle client disconnect gracefully: detect broken pipe on SSE write, call `session.cancel()`.

---

## Appendix A: BAR0 Register Map Summary

```
Offset    Register                Width  Access
------    --------                -----  ------
0x0000    ORBIT_ID                32     RO     Magic "ORB1" = 0x4F524231
0x0004    HW_REVISION             32     RO
0x0008    FW_VERSION              32     RO
0x000C    GLOBAL_STATUS           32     RO     bit 0=ready, bit 1=fault, bit 2=busy
0x0010    GDDR_SIZE_LO            32     RO
0x0014    GDDR_SIZE_HI            32     RO
0x0018    GLOBAL_CTRL             32     RW     bit 0=soft_reset, bits 3:2=sup_mode
0x001C    INTR_STATUS             32     RW1C   bits 3:0 per-queue interrupt
0x0020    INTR_MASK               32     RW     bits 3:0 per-queue mask (1=enable)
0x0024–0x00FF  (reserved)

Per-queue registers (base = 0x0100 + queue_id * 0x40):
+0x00    Q_RING_BASE_LO          32     RW     DMA ring bus address [31:0]
+0x04    Q_RING_BASE_HI          32     RW     DMA ring bus address [63:32]
+0x08    Q_RING_SIZE             32     RW     Ring depth (256)
+0x0C    Q_HEAD                  32     RO     Hardware consumer pointer
+0x10    Q_TAIL                  32     RW     Software producer / doorbell
+0x14    Q_STATUS                32     RO     bit 0=active, bit 1=error, bit 2=idle
+0x18    Q_ERROR_CODE            32     RO     Last error (0=none)
+0x1C    Q_COMPLETE_CNT          32     RO     Cumulative completed count
+0x20    Q_INTR_VECTOR           32     RW     MSI-X vector index
+0x24    Q_CONFIG                32     RW     Config flags
+0x28–0x3F  (reserved per queue)
```

---

## Appendix B: GDDR6 Memory Layout Convention

```
GDDR6 address 0x0000_0000_0000_0000
│
├─ [0x0000_0000 .. 0x0000_0FFF]    Reserved / firmware scratch (4 KB)
├─ [0x0000_1000 .. ?             ]  Runtime-allocated (via ORBIT_IOC_ALLOC_MEM)
│   ├─ Embedding weights
│   ├─ Per-layer weights (Wq, Wk, Wv, Wo, router, expert gate/up/down × 32)
│   ├─ LM head weights
│   ├─ KV-cache pages (PagedAttention pool)
│   └─ Per-session scratch buffers (hidden_state, qkv_proj, attention_scores, ...)
│
└─ [top - 64MB .. top]             Reserved for firmware / telemetry ring buffers
```

The driver's buddy allocator manages the entire GDDR6 space minus the reserved regions at both ends. The firmware reserved region sizes are read from BAR0 registers at probe time (not hardcoded).

---

## Appendix C: Descriptor Binary Layouts (v2 additions)

These are the additional descriptor types from `yua-llm-hw-design.md` not in `descriptor.md` v1.

```c
/* KVC_READ (type 0x0A) — 64 bytes */
struct orbit_desc_kvc_read {
    struct orbit_desc_header h;   /* 16 bytes */
    uint32_t layer_id;
    uint32_t head_id;
    uint32_t seq_start;
    uint32_t seq_len;
    uint64_t dst_addr;            /* GDDR6 address to write K,V data */
    uint32_t reserved[2];
};  /* total: 16 + 4+4+4+4 + 8 + 8 = 48 bytes → pad to 64 */

/* KVC_WRITE (type 0x0B) — 64 bytes */
struct orbit_desc_kvc_write {
    struct orbit_desc_header h;
    uint32_t layer_id;
    uint32_t head_id;
    uint32_t seq_pos;
    uint32_t _pad0;
    uint64_t src_addr;            /* GDDR6 address of K,V to store */
    uint32_t reserved[4];
};

/* MOE_ROUTE (type 0x0C) — 64 bytes */
struct orbit_desc_moe_route {
    struct orbit_desc_header h;
    uint64_t logits_addr;
    uint64_t indices_addr;
    uint64_t scores_addr;
    uint32_t num_tokens;
    uint32_t num_experts;
    uint32_t top_k;
    uint32_t _pad;
};

/* VECTOR_OP_EX (type 0x0D) — 64 bytes */
struct orbit_desc_vector_op_ex {
    struct orbit_desc_header h;
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t element_count;
    uint16_t op_type;     /* 0=RMSNORM, 1=SILU, 2=ROPE, 3=RESIDUAL, 4=SCALE, 5=GELU */
    uint16_t data_type;   /* FP16=3, BF16=4 */
    uint64_t aux_addr;    /* auxiliary data: scale weights for RMSNORM, cos/sin for ROPE */
    uint32_t reserved[2];
};

/* GEMM_INT4 (type 0x0E) — 64 bytes */
struct orbit_desc_gemm_int4 {
    struct orbit_desc_header h;
    uint64_t act_addr;    /* activations (INT4 packed) */
    uint64_t wgt_addr;    /* weights (INT4 packed, AWQ format) */
    uint64_t out_addr;    /* output (INT32 or FP16 depending on epilogue) */
    uint32_t Kt;
    uint16_t m_tiles;
    uint16_t n_tiles;
    uint64_t scale_addr;  /* per-group FP16 scales address */
    uint32_t group_size;  /* AWQ group size, typically 128 */
    uint32_t reserved;
};

/* SOFTMAX (type 0x0F) — 64 bytes */
struct orbit_desc_softmax {
    struct orbit_desc_header h;
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t element_count;
    uint32_t options;     /* bit 0: apply scale before softmax; bit 1: argmax only */
    uint32_t scale_fp32;  /* scale factor as IEEE 754 float (e.g., 1/sqrt(head_dim)) */
    uint32_t reserved;
    uint64_t reserved2;
};
```

---

*End of ORBIT-G1 Driver + Runtime Design Specification v1.0*
*Generated: 2026-03-13*
*SSOT for Agent A (kernel), Agent B (runtime), Agent C (server)*
