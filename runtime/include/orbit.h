/**
 * orbit.h — liborbit public C++ API
 *
 * Self-contained public header for the ORBIT-G1 userspace runtime library.
 * No internal headers are leaked. C++17 required.
 *
 * Hardware: ORBIT-G1 v2 (descriptor spec v1 + v2 extensions)
 * Target workload: GPT-OSS-20B (MoE Transformer, 32 experts)
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <mutex>
#include <queue>
#include <array>
#include <atomic>
#include <stdexcept>
#include <optional>

// ============================================================
// Forward declarations
// ============================================================
namespace orbit {

class Device;
class MemoryPool;
class DescriptorQueue;
class WeightLoader;
class KVCacheManager;
class InferenceSession;
class Executor;

// ============================================================
// Handle types
// ============================================================

/** Opaque handle to a GDDR6 memory region managed by MemoryPool. */
using MemHandle = uint64_t;
static constexpr MemHandle INVALID_MEM_HANDLE = 0;

/** Cookie returned by DescriptorQueue::submit() for async wait. */
using SubmitCookie = uint64_t;
static constexpr SubmitCookie INVALID_SUBMIT_COOKIE = 0;

// ============================================================
// Enumerations
// ============================================================

/** Descriptor type IDs — v1 (0x01-0x09) + v2 extensions (0x0A-0x0F) */
enum class DescType : uint8_t {
    DMA_2D          = 0x01,
    GEMM_INT8       = 0x02,
    VECTOR_OP       = 0x03,
    COPY_2D_PLUS    = 0x04,
    FORMAT_CONVERT  = 0x05,
    FRAME_FINGERPRINT = 0x06,
    BARRIER         = 0x07,
    EVENT           = 0x08,
    PERF_SNAPSHOT   = 0x09,
    // v2 extensions
    KVC_READ        = 0x0A,
    KVC_WRITE       = 0x0B,
    MOE_ROUTE       = 0x0C,
    VECTOR_OP_EX    = 0x0D,
    GEMM_INT4       = 0x0E,
    SOFTMAX         = 0x0F,
};

/** Elementwise / reduction operations for VECTOR_OP (type 0x03). */
enum class VectorOpType : uint16_t {
    ADD     = 0x01,
    SUB     = 0x02,
    MUL     = 0x03,
    DIV     = 0x04,
    MAX     = 0x05,
    MIN     = 0x06,
    CLAMP   = 0x07,
    ABS     = 0x08,
    NEG     = 0x09,
    SUM_REDUCE = 0x0A,
    MAX_REDUCE = 0x0B,
};

/** Extended operations for VECTOR_OP_EX (type 0x0D). */
enum class VectorOpExType : uint16_t {
    RMSNORM = 0x01,
    SILU    = 0x02,
    ROPE    = 0x03,
    RESIDUAL_ADD = 0x04,
    GELU    = 0x05,
};

/** Data type for VECTOR_OP operands. */
enum class DataType : uint16_t {
    INT8  = 0x01,
    INT16 = 0x02,
    FP16  = 0x03,
    BF16  = 0x04,
    INT32 = 0x05,
    INT4  = 0x06,
};

/** Format identifiers for FORMAT_CONVERT descriptor. */
enum class DataFormat : uint16_t {
    RGB_U8    = 0x01,
    YUV_U8    = 0x02,
    FP16      = 0x03,
    INT8      = 0x04,
    INT4_PACKED = 0x05,
    BF16      = 0x06,
};

/** Hash algorithm for FRAME_FINGERPRINT descriptor. */
enum class HashType : uint32_t {
    CRC32  = 0x01,
    CRC64  = 0x02,
    SHA256 = 0x03,
};

/** Quantization format for weights. */
enum class QuantFormat : uint32_t {
    INT8     = 0x01,   /**< Symmetric INT8 per-tensor or per-channel */
    INT4_AWQ = 0x02,   /**< AWQ packed INT4 with FP16 group scales/zeros */
};

/** Hardware queue IDs. */
enum class QueueId : uint32_t {
    COMPUTE   = 0,  /**< GEMM, VECTOR_OP, KVC, MoE */
    UTILITY   = 1,  /**< DMA_2D, COPY_2D_PLUS, FORMAT_CONVERT, FINGERPRINT */
    TELEMETRY = 2,  /**< PERF_SNAPSHOT, DIAG_RUN */
    HIPRI     = 3,  /**< Reserved / high-priority compute */
};

// ============================================================
// Error handling
// ============================================================

/** Runtime exception thrown by liborbit on error (when ORBIT_THROW_ON_ERROR is set). */
class OrbitException : public std::runtime_error {
public:
    explicit OrbitException(const std::string& msg, int err = 0)
        : std::runtime_error(msg), errno_(err) {}
    int error_code() const noexcept { return errno_; }
private:
    int errno_;
};

// ============================================================
// Low-level descriptor binary layout structures
// (64 bytes each, matching descriptor.md + yua-llm-hw-design.md SSOT)
// ============================================================

/** Common 16-byte descriptor header. */
struct __attribute__((packed)) OrbitDescHeader {
    uint8_t  type;        ///< Descriptor type ID (DescType)
    uint8_t  flags;       ///< Control flags
    uint16_t reserved0;   ///< Must be zero
    uint32_t length;      ///< Operation length (bytes, elements, or tiles)
    uint64_t next_desc;   ///< Linked-list next pointer (0 = end)
};
static_assert(sizeof(OrbitDescHeader) == 16, "OrbitDescHeader must be 16 bytes");

/** DMA_2D descriptor (type 0x01) — 64 bytes. */
struct __attribute__((packed)) OrbitDescDma2D {
    OrbitDescHeader h;          ///< type=0x01
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t width_bytes;
    uint32_t height;
    uint32_t src_stride;
    uint32_t dst_stride;
    uint32_t reserved1;
    uint32_t reserved2;
    uint8_t  _pad[8];           ///< Pad to 64 bytes (56 -> 64)
};
static_assert(sizeof(OrbitDescDma2D) == 64, "OrbitDescDma2D must be 64 bytes");

/** GEMM_INT8 descriptor (type 0x02) — 64 bytes. */
struct __attribute__((packed)) OrbitDescGemmInt8 {
    OrbitDescHeader h;          ///< type=0x02
    uint64_t act_addr;          ///< A tile base address (activations)
    uint64_t wgt_addr;          ///< B tile base address (weights)
    uint64_t out_addr;          ///< C tile output address
    uint32_t Kt;                ///< K accumulation dimension
    uint16_t m_tiles;           ///< Fixed to 1 in v1
    uint16_t n_tiles;           ///< Fixed to 1 in v1
    uint32_t scale_a;           ///< Reserved (ignored in v1)
    uint32_t scale_b;           ///< Reserved (ignored in v1)
    uint32_t epilogue;          ///< Epilogue flags (reserved)
    uint32_t reserved;
};
static_assert(sizeof(OrbitDescGemmInt8) == 64, "OrbitDescGemmInt8 must be 64 bytes");

/** VECTOR_OP descriptor (type 0x03) — 64 bytes. */
struct __attribute__((packed)) OrbitDescVectorOp {
    OrbitDescHeader h;          ///< type=0x03
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t element_count;
    uint16_t op_type;           ///< VectorOpType
    uint16_t data_type;         ///< DataType
    uint32_t imm;               ///< Immediate scalar value
    uint32_t reserved;
    uint8_t  _pad[16];          ///< Pad to 64 bytes
};
static_assert(sizeof(OrbitDescVectorOp) == 64, "OrbitDescVectorOp must be 64 bytes");

/** COPY_2D_PLUS descriptor (type 0x04) — 64 bytes. */
struct __attribute__((packed)) OrbitDescCopy2DPlus {
    OrbitDescHeader h;          ///< type=0x04
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t width_bytes;
    uint32_t height;
    uint32_t src_stride;
    uint32_t dst_stride;
    uint32_t options;           ///< Crop, mirror, pad flags
    uint32_t reserved;
    uint8_t  _pad[8];           ///< Pad to 64 bytes (56 -> 64)
};
static_assert(sizeof(OrbitDescCopy2DPlus) == 64, "OrbitDescCopy2DPlus must be 64 bytes");

/** FORMAT_CONVERT descriptor (type 0x05) — 64 bytes. */
struct __attribute__((packed)) OrbitDescFormatConvert {
    OrbitDescHeader h;          ///< type=0x05
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t element_count;
    uint16_t src_format;        ///< DataFormat
    uint16_t dst_format;        ///< DataFormat
    uint32_t options;
    uint32_t reserved;
    uint8_t  _pad[16];          ///< Pad to 64 bytes (48 -> 64)
};
static_assert(sizeof(OrbitDescFormatConvert) == 64, "OrbitDescFormatConvert must be 64 bytes");

/** FRAME_FINGERPRINT descriptor (type 0x06) — 64 bytes. */
struct __attribute__((packed)) OrbitDescFrameFingerprint {
    OrbitDescHeader h;          ///< type=0x06
    uint64_t src_addr;
    uint64_t result_addr;
    uint32_t byte_count;
    uint32_t hash_type;         ///< HashType
    uint64_t reserved;
    uint8_t  _pad[16];          ///< Pad to 64 bytes (48 -> 64)
};
static_assert(sizeof(OrbitDescFrameFingerprint) == 64, "OrbitDescFrameFingerprint must be 64 bytes");

/** BARRIER descriptor (type 0x07) — 64 bytes. */
struct __attribute__((packed)) OrbitDescBarrier {
    OrbitDescHeader h;          ///< type=0x07
    uint64_t reserved1;
    uint64_t reserved2;
    uint64_t reserved3;
    uint8_t  _pad[24];          ///< Pad to 64 bytes (40 -> 64)
};
static_assert(sizeof(OrbitDescBarrier) == 64, "OrbitDescBarrier must be 64 bytes");

/** EVENT descriptor (type 0x08) — 64 bytes. */
struct __attribute__((packed)) OrbitDescEvent {
    OrbitDescHeader h;          ///< type=0x08
    uint32_t event_id;
    uint32_t options;
    uint64_t reserved1;
    uint64_t reserved2;
    uint8_t  _pad[24];          ///< Pad to 64 bytes (40 -> 64)
};
static_assert(sizeof(OrbitDescEvent) == 64, "OrbitDescEvent must be 64 bytes");

/** PERF_SNAPSHOT descriptor (type 0x09) — 64 bytes. */
struct __attribute__((packed)) OrbitDescPerfSnapshot {
    OrbitDescHeader h;          ///< type=0x09
    uint64_t dst_addr;          ///< Address to write performance counters
    uint64_t reserved1;
    uint64_t reserved2;
    uint8_t  _pad[24];          ///< Pad to 64 bytes (40 -> 64)
};
static_assert(sizeof(OrbitDescPerfSnapshot) == 64, "OrbitDescPerfSnapshot must be 64 bytes");

// --- v2 extension descriptors ---

/** KVC_READ descriptor (type 0x0A) — 64 bytes. */
struct __attribute__((packed)) OrbitDescKvcRead {
    OrbitDescHeader h;          ///< type=0x0A, length=seq_len_to_read
    uint32_t seq_id;            ///< Sequence ID (0..63)
    uint32_t layer_id;          ///< Transformer layer (0..31)
    uint32_t seq_start;         ///< Start token position (inclusive)
    uint32_t seq_len;           ///< Number of tokens to read
    uint64_t k_dst_addr;        ///< GDDR6 scratch for output K tensor
    uint64_t v_dst_addr;        ///< GDDR6 scratch for output V tensor
    uint8_t  read_mode;         ///< 0=ALL_HEADS, 1=SINGLE_HEAD
    uint8_t  head_id;           ///< Head index (only if SINGLE_HEAD)
    uint16_t reserved;
    uint8_t  _pad[12];          ///< Pad to 64 bytes (52 -> 64)
};
static_assert(sizeof(OrbitDescKvcRead) == 64, "OrbitDescKvcRead must be 64 bytes");

/** KVC_WRITE descriptor (type 0x0B) — 64 bytes. */
struct __attribute__((packed)) OrbitDescKvcWrite {
    OrbitDescHeader h;          ///< type=0x0B, length=tokens_to_write
    uint32_t seq_id;            ///< Sequence ID (0..63)
    uint32_t layer_id;          ///< Transformer layer (0..31)
    uint32_t token_pos;         ///< Start token position for write
    uint32_t write_count;       ///< Number of tokens to write
    uint64_t k_src_addr;        ///< GDDR6 source for K tensor
    uint64_t v_src_addr;        ///< GDDR6 source for V tensor
    uint8_t  write_mode;        ///< 0=DECODE, 1=PREFILL_BULK
    uint8_t  alloc_blocks;      ///< 1 = KVC auto-allocates pages
    uint16_t reserved;
    uint8_t  _pad[12];          ///< Pad to 64 bytes (52 -> 64)
};
static_assert(sizeof(OrbitDescKvcWrite) == 64, "OrbitDescKvcWrite must be 64 bytes");

/** MOE_ROUTE descriptor (type 0x0C) — 64 bytes. */
struct __attribute__((packed)) OrbitDescMoeRoute {
    OrbitDescHeader h;          ///< type=0x0C
    uint64_t logits_addr;       ///< Input: router logits
    uint64_t indices_addr;      ///< Output: selected expert indices [top_k × INT32]
    uint64_t scores_addr;       ///< Output: routing weights [top_k × FP16]
    uint32_t num_tokens;
    uint32_t num_experts;
    uint32_t top_k;
    uint32_t reserved;
    uint8_t  _pad[8];           ///< Pad to 64 bytes (56 -> 64)
};
static_assert(sizeof(OrbitDescMoeRoute) == 64, "OrbitDescMoeRoute must be 64 bytes");

/** VECTOR_OP_EX descriptor (type 0x0D) — 64 bytes. */
struct __attribute__((packed)) OrbitDescVectorOpEx {
    OrbitDescHeader h;          ///< type=0x0D
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t element_count;
    uint16_t op_type;           ///< VectorOpExType
    uint16_t data_type;         ///< DataType
    uint64_t aux_addr;          ///< Auxiliary address (e.g., RoPE cos/sin table)
    uint32_t aux_param;         ///< Op-specific parameter
    uint32_t reserved;
    uint8_t  _pad[8];           ///< Pad to 64 bytes (56 -> 64)
};
static_assert(sizeof(OrbitDescVectorOpEx) == 64, "OrbitDescVectorOpEx must be 64 bytes");

/** GEMM_INT4 descriptor (type 0x0E) — 64 bytes. */
struct __attribute__((packed)) OrbitDescGemmInt4 {
    OrbitDescHeader h;          ///< type=0x0E
    uint64_t act_addr;          ///< A tile base (FP16 activations)
    uint64_t wgt_addr;          ///< B tile base (packed INT4 weights)
    uint64_t out_addr;          ///< C tile output (FP16)
    uint64_t scale_addr;        ///< AWQ group scales/zeros address
    uint32_t Kt;                ///< K accumulation dimension
    uint16_t m_tiles;
    uint16_t n_tiles;
    uint8_t  _pad[8];           ///< Pad to 64 bytes (56 -> 64)
};
static_assert(sizeof(OrbitDescGemmInt4) == 64, "OrbitDescGemmInt4 must be 64 bytes");

/** SOFTMAX descriptor (type 0x0F) — 64 bytes. */
struct __attribute__((packed)) OrbitDescSoftmax {
    OrbitDescHeader h;          ///< type=0x0F
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t element_count;     ///< Number of elements (logits) per row
    uint32_t num_rows;          ///< Number of rows (batch)
    uint32_t scale_fp32;        ///< Pre-softmax scale factor (raw fp32 bits; e.g. 1/sqrt(head_dim))
    uint32_t reserved;
    uint8_t  _pad[16];          ///< Pad to 64 bytes (48 -> 64)
};
static_assert(sizeof(OrbitDescSoftmax) == 64, "OrbitDescSoftmax must be 64 bytes");

// ============================================================
// ioctl structs (mirrors orbit_g1_uapi.h, userspace-native types)
// ============================================================

struct OrbitDescSubmit {
    uint32_t queue_id;
    uint32_t count;
    uint32_t flags;
    uint32_t _pad;
    uint64_t descs_ptr;         ///< pointer to raw descriptor bytes
    uint64_t submit_cookie;     ///< OUT: completion token
};

struct OrbitWaitDone {
    uint32_t queue_id;
    uint32_t timeout_ms;
    uint64_t submit_cookie;
};

struct OrbitMemAlloc {
    uint64_t size_bytes;
    uint64_t align_bytes;
    uint64_t device_addr;       ///< OUT: GDDR6-physical
    uint64_t handle;            ///< OUT: opaque token
};

struct OrbitMemFree {
    uint64_t handle;
};

struct OrbitDeviceInfo {
    uint64_t gddr_size_bytes;
    uint64_t bar1_size_bytes;
    uint32_t num_queues;
    uint32_t queue_depth;
    uint32_t fw_version;
    uint32_t hw_revision;
    uint32_t desc_spec_version;
    uint32_t supported_desc_types;
    char     device_name[32];
    uint8_t  _pad[4];
};

struct OrbitQueueReset {
    uint32_t queue_id;
    uint32_t flags;
};

// ============================================================
// Device — opens /dev/orbit_g1_N, ioctl wrappers, mmap
// ============================================================

class Device {
public:
    /**
     * Factory: open /dev/orbit_g1_N (default N=0).
     * Throws OrbitException on failure.
     */
    static std::unique_ptr<Device> open(int card_index = 0);
    ~Device();

    // Non-copyable, movable
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) noexcept;
    Device& operator=(Device&&) noexcept;

    /** Device capability info. */
    const OrbitDeviceInfo& info() const noexcept;

    // Raw ioctl wrappers — return 0 on success, negative errno on failure.
    int submit_desc(OrbitDescSubmit& req);
    int wait_done(OrbitWaitDone& req);
    int alloc_gddr(size_t size_bytes, size_t align, OrbitMemAlloc* out);
    int free_gddr(uint64_t handle);
    int reset_queue(uint32_t queue_id);

    /**
     * Map a window of BAR1 (GDDR6 aperture) into host virtual address space.
     * @param bar1_offset  byte offset within BAR1 (must be page-aligned)
     * @param length       number of bytes to map
     * @return host virtual pointer, or nullptr on failure
     */
    void* mmap_bar1(uint64_t bar1_offset, size_t length);
    void  munmap_bar1(void* ptr, size_t length);

    /** Underlying file descriptor (for poll/epoll). */
    int fd() const noexcept { return fd_; }

private:
    explicit Device(int fd, const OrbitDeviceInfo& info);

    int             fd_;
    OrbitDeviceInfo info_;
};

// ============================================================
// MemoryPool — slab front-end for GDDR6 memory
// ============================================================

class MemoryPool {
public:
    explicit MemoryPool(Device& dev);
    ~MemoryPool();

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    /**
     * Allocate a GDDR6 region.
     * @param size_bytes  requested size in bytes
     * @param align       alignment in bytes (default: 4096)
     * @return opaque MemHandle; INVALID_MEM_HANDLE on failure
     */
    MemHandle alloc(size_t size_bytes, size_t align = 4096);

    /** Free a previously allocated region. */
    void free(MemHandle handle);

    /**
     * GDDR6-physical device address of a handle.
     * Must be called with a valid handle.
     */
    uint64_t device_addr(MemHandle handle) const;

    /**
     * Host pointer via BAR1 mmap (lazy: mapped on first call).
     * Returns nullptr if the region is outside the BAR1 aperture.
     */
    void* host_ptr(MemHandle handle);

    uint64_t total_bytes() const noexcept;
    uint64_t free_bytes() const noexcept;

private:
    static constexpr size_t SLAB_SIZE = 64 * 1024 * 1024;  ///< 64 MB per slab
    static constexpr size_t LARGE_THRESHOLD = 1 * 1024 * 1024;  ///< 1 MB

    struct AllocEntry {
        uint64_t device_addr;
        uint64_t kernel_handle;
        size_t   size;
        void*    host_ptr;
        bool     bar1_mapped;
    };

    struct Slab {
        MemHandle parent_handle;
        uint64_t  base_device_addr;
        size_t    slab_size;
        size_t    used;
    };

    Device& dev_;
    std::unordered_map<MemHandle, AllocEntry> allocs_;
    mutable std::mutex mutex_;
    MemHandle next_handle_{1};
    std::vector<Slab> slabs_;

    MemHandle alloc_large(size_t size_bytes, size_t align);
    MemHandle alloc_from_slab(size_t size_bytes, size_t align);
    Slab& get_or_create_slab(size_t min_size);
};

// ============================================================
// DescriptorQueue — batch builder + submit + wait
// ============================================================

class DescriptorQueue {
public:
    DescriptorQueue(Device& dev, uint32_t queue_id, size_t max_batch = 256);

    DescriptorQueue(const DescriptorQueue&) = delete;
    DescriptorQueue& operator=(const DescriptorQueue&) = delete;

    // --- add_*() methods: append one 64-byte descriptor to the pending batch ---

    /** DMA_2D (0x01): 2D memory copy. */
    void add_dma_2d(uint64_t src, uint64_t dst,
                    uint32_t width_bytes, uint32_t height,
                    uint32_t src_stride, uint32_t dst_stride);

    /** GEMM_INT8 (0x02): 16×16 INT8 tile GEMM. */
    void add_gemm_int8(uint64_t act_addr, uint64_t wgt_addr, uint64_t out_addr,
                       uint32_t Kt, uint16_t m_tiles = 1, uint16_t n_tiles = 1);

    /** GEMM_INT4 (0x0E): packed INT4 GEMM with AWQ dequant. */
    void add_gemm_int4(uint64_t act_addr, uint64_t wgt_addr, uint64_t out_addr,
                       uint64_t scale_addr, uint32_t Kt,
                       uint16_t m_tiles = 1, uint16_t n_tiles = 1);

    /** VECTOR_OP (0x03): elementwise or reduction. */
    void add_vector_op(uint64_t src, uint64_t dst, uint32_t element_count,
                       VectorOpType op, DataType dtype, uint32_t imm = 0);

    /** VECTOR_OP_EX (0x0D): RMSNorm, SiLU, RoPE, etc. */
    void add_vector_op_ex(uint64_t src, uint64_t dst, uint32_t element_count,
                          VectorOpExType op, DataType dtype,
                          uint64_t aux_addr = 0, uint32_t aux_param = 0);

    /** KVC_READ (0x0A): read K,V tensors from KV-Cache. */
    void add_kvc_read(uint32_t seq_id, uint32_t layer_id,
                      uint32_t seq_start, uint32_t seq_len,
                      uint64_t k_dst_addr, uint64_t v_dst_addr,
                      uint8_t read_mode = 0, uint8_t head_id = 0);

    /** KVC_WRITE (0x0B): write K,V tensors into KV-Cache. */
    void add_kvc_write(uint32_t seq_id, uint32_t layer_id,
                       uint32_t token_pos, uint32_t write_count,
                       uint64_t k_src_addr, uint64_t v_src_addr,
                       uint8_t write_mode = 0, uint8_t alloc_blocks = 0);

    /** MOE_ROUTE (0x0C): MoE top-k expert selection. */
    void add_moe_route(uint64_t logits_addr, uint64_t indices_addr,
                       uint64_t scores_addr, uint32_t num_tokens,
                       uint32_t num_experts, uint32_t top_k);

    /** FORMAT_CONVERT (0x05): format conversion (e.g. FP16 → INT8). */
    void add_format_convert(uint64_t src, uint64_t dst, uint32_t element_count,
                            DataFormat src_fmt, DataFormat dst_fmt,
                            uint32_t options = 0);

    /** COPY_2D_PLUS (0x04): enhanced 2D copy (crop/mirror/pad). */
    void add_copy_2d_plus(uint64_t src, uint64_t dst,
                          uint32_t width_bytes, uint32_t height,
                          uint32_t src_stride, uint32_t dst_stride,
                          uint32_t options = 0);

    /** FRAME_FINGERPRINT (0x06): integrity hash. */
    void add_frame_fingerprint(uint64_t src, uint64_t result_addr,
                               uint32_t byte_count, HashType hash_type);

    /** BARRIER (0x07): explicit ordering fence. */
    void add_barrier();

    /** EVENT (0x08): trigger host interrupt/notification. */
    void add_event(uint32_t event_id, uint32_t options = 0);

    /** PERF_SNAPSHOT (0x09): capture performance counters. */
    void add_perf_snapshot(uint64_t dst_addr);

    /** SOFTMAX (0x0F): attention softmax with optional pre-scale. */
    void add_softmax(uint64_t src, uint64_t dst,
                     uint32_t element_count, uint32_t num_rows = 1,
                     float scale = 1.0f);

    // --- Submission ---

    /**
     * Submit the pending batch to the hardware queue.
     * @return SubmitCookie for use with wait().
     */
    SubmitCookie submit();

    /** Submit and block until all submitted descriptors complete. */
    void submit_and_wait(uint32_t timeout_ms = 5000);

    /** Wait for a previously submitted cookie. */
    void wait(SubmitCookie cookie, uint32_t timeout_ms = 5000);

    /** Discard the pending batch without submitting. */
    void reset_batch();

    /** Number of descriptors in the pending batch. */
    size_t batch_size() const noexcept;

    /** Queue ID this instance is bound to. */
    uint32_t queue_id() const noexcept { return queue_id_; }

private:
    Device&  dev_;
    uint32_t queue_id_;
    size_t   max_batch_;
    std::vector<std::array<uint8_t, 64>> batch_;

    template<typename T>
    void append_desc(const T& desc_struct);
};

// ============================================================
// WeightLoader — INT4/INT8 weight upload to GDDR6
// ============================================================

class WeightLoader {
public:
    WeightLoader(Device& dev, MemoryPool& pool, DescriptorQueue& util_queue);

    WeightLoader(const WeightLoader&) = delete;
    WeightLoader& operator=(const WeightLoader&) = delete;

    /** Load a weight tensor from file. Returns GDDR6 MemHandle. */
    MemHandle load_weight_file(const std::string& path, QuantFormat format);

    /** Load from a host memory buffer (already quantized). */
    MemHandle load_weight_buffer(const void* data, size_t size_bytes, QuantFormat format);

    using LoadCallback = std::function<void(MemHandle, int err)>;

    /** Async variant: returns immediately; callback invoked on completion. */
    void load_weight_async(const std::string& path, QuantFormat format, LoadCallback cb);

    /** Weight tensor metadata. */
    struct WeightInfo {
        MemHandle   handle;
        uint64_t    device_addr;
        size_t      size_bytes;
        QuantFormat format;
        std::vector<uint32_t> shape;
        MemHandle   scale_handle;   ///< AWQ group scales (INT4_AWQ only), INVALID if unused
    };

    /** Query metadata for a loaded weight. */
    const WeightInfo& weight_info(MemHandle handle) const;

    /** Unload (free GDDR6 region). */
    void unload(MemHandle handle);

private:
    Device&          dev_;
    MemoryPool&      pool_;
    DescriptorQueue& util_queue_;
    std::unordered_map<MemHandle, WeightInfo> loaded_;
    mutable std::mutex mutex_;

    MemHandle upload_via_mmap(const void* data, size_t size_bytes);
    MemHandle upload_via_dma(const void* data, size_t size_bytes);
    MemHandle upload_via_mmap_region(const void* data, size_t size_bytes,
                                     MemHandle region_h, void* bar1_ptr);
    MemHandle upload_via_dma_region(const void* data, size_t size_bytes,
                                    MemHandle region_h);
};

// ============================================================
// KVCacheManager — PagedAttention KV-cache allocator
// ============================================================

class KVCacheManager {
public:
    struct Config {
        uint32_t num_layers;         ///< e.g., 32
        uint32_t num_heads;          ///< e.g., 32
        uint32_t head_dim;           ///< e.g., 128
        uint32_t page_size_tokens;   ///< tokens per page, e.g., 16
        uint32_t max_pages;          ///< total physical pages available
        DataType dtype;              ///< FP16 typically
    };

    KVCacheManager(Device& dev, MemoryPool& pool, const Config& cfg);
    ~KVCacheManager();

    KVCacheManager(const KVCacheManager&) = delete;
    KVCacheManager& operator=(const KVCacheManager&) = delete;

    using SeqHandle = uint32_t;
    static constexpr SeqHandle INVALID_SEQ = 0;

    /** Allocate KV pages for a new sequence. */
    SeqHandle alloc_sequence(uint32_t initial_capacity_tokens = 0);

    /** Extend a sequence (allocate more pages). Returns false if OOM. */
    bool extend_sequence(SeqHandle seq, uint32_t additional_tokens);

    /** Free all pages for a sequence. */
    void free_sequence(SeqHandle seq);

    /**
     * Compute GDDR6 address for a specific KV element.
     * Used by Executor to build KVC_READ/KVC_WRITE descriptors.
     *
     * Formula (from kvc.md §4.3):
     *   block_base   = kvc_pool_base + phys_block * block_bytes
     *   layer_offset = layer_id  * (2 * num_heads * page_size * head_dim * dtype_bytes)
     *   kv_offset    = is_key ? 0 : (num_heads * page_size * head_dim * dtype_bytes)
     *   head_offset  = head_id * (page_size * head_dim * dtype_bytes)
     *   token_offset = (token_pos % page_size) * (head_dim * dtype_bytes)
     *   elem_addr    = block_base + layer_offset + kv_offset + head_offset + token_offset
     */
    uint64_t kv_addr(SeqHandle seq, uint32_t layer_id, uint32_t head_id,
                     uint32_t token_pos, bool is_key) const;

    uint32_t total_pages() const noexcept;
    uint32_t free_pages() const noexcept;

    struct PageTable {
        std::vector<uint64_t> page_addrs;  ///< GDDR6 base addr of each page
        uint32_t              num_tokens;
    };
    const PageTable& page_table(SeqHandle seq) const;

    const Config& config() const noexcept { return cfg_; }

private:
    Device&     dev_;
    MemoryPool& pool_;
    Config      cfg_;

    struct KVPage {
        uint64_t device_addr;
        bool     in_use;
    };

    std::vector<KVPage>    pages_;
    std::queue<uint32_t>   free_page_ids_;
    mutable std::mutex     mutex_;

    struct SeqState {
        std::vector<uint32_t> page_ids;
        PageTable             page_table;
        uint32_t              num_tokens;
    };
    std::unordered_map<SeqHandle, SeqState> seqs_;
    SeqHandle next_seq_handle_{1};

    size_t   page_bytes() const noexcept;
    uint32_t dtype_bytes() const noexcept;
    uint64_t token_offset_within_page(uint32_t layer_id, uint32_t head_id,
                                      uint32_t token_within_page, bool is_key) const noexcept;
    uint32_t alloc_page();   ///< Pops from free_page_ids_; throws on OOM
};

// ============================================================
// InferenceSession — per-request state
// ============================================================

class InferenceSession {
public:
    struct Config {
        uint32_t max_new_tokens;
        uint32_t temperature_fp16;  ///< FP16 bit-pattern
        uint32_t top_p_fp16;        ///< FP16 bit-pattern
        bool     do_sample;
        uint32_t seed;
    };

    InferenceSession(KVCacheManager& kvc_mgr, const Config& cfg);
    ~InferenceSession();  ///< Frees KV pages

    InferenceSession(const InferenceSession&) = delete;
    InferenceSession& operator=(const InferenceSession&) = delete;

    // --- Lifecycle ---

    /** Process the prompt; populates KV-cache for all input tokens. */
    void prefill(const std::vector<int32_t>& input_token_ids);

    /**
     * Decode one token.
     * @return false when EOS or max_new_tokens reached.
     */
    bool step();

    /** Cancel an in-progress session. */
    void cancel();

    // --- State accessors ---
    uint32_t                    seq_id() const noexcept;
    uint32_t                    context_length() const noexcept;
    const std::vector<int32_t>& generated_tokens() const noexcept;
    int32_t                     last_token() const noexcept;
    bool                        is_done() const noexcept;
    KVCacheManager::SeqHandle   kv_handle() const noexcept;

    /** GDDR6 scratch buffer addresses allocated by Executor. */
    struct ScratchBuffers {
        uint64_t hidden_state;      ///< [1 × d_model] FP16
        uint64_t hidden_save;       ///< [1 × d_model] FP16 — saved copy of hidden before step 2 (for residual add in step 10)
        uint64_t qkv_proj;          ///< [3 × d_model] FP16
        uint64_t attention_scores;  ///< [num_heads × seq_len] FP16
        uint64_t attention_out;     ///< [d_model] FP16
        uint64_t ffn_gate;          ///< [d_ffn] FP16 per expert
        uint64_t ffn_up;            ///< [d_ffn] FP16 per expert
        uint64_t ffn_out;           ///< [d_model] FP16 — accumulated FFN output for post-FFN residual (step 19)
        uint64_t moe_logits;        ///< [num_experts] FP16
        uint64_t moe_indices;       ///< [top_k] INT32
        uint64_t moe_scores;        ///< [top_k] FP16
        uint64_t lm_head_out;       ///< [vocab_size] FP16
    };
    const ScratchBuffers& scratch() const noexcept;

    /** Scratch buffer setter (used by Executor during session init). */
    void set_scratch(const ScratchBuffers& s);

private:
    KVCacheManager&              kvc_mgr_;
    KVCacheManager::SeqHandle    kv_handle_;
    Config                       cfg_;
    uint32_t                     seq_id_;
    uint32_t                     context_len_;
    std::vector<int32_t>         generated_tokens_;
    bool                         done_;
    bool                         cancelled_;
    ScratchBuffers               scratch_;

    static std::atomic<uint32_t> next_seq_id_;
};

// ============================================================
// Executor — orchestrates the 21-step decode descriptor sequence
// ============================================================

class Executor {
public:
    struct ModelConfig {
        uint32_t    num_layers;         ///< 32 for GPT-OSS-20B
        uint32_t    d_model;            ///< hidden dimension (e.g., 4096)
        uint32_t    num_heads;          ///< 32
        uint32_t    head_dim;           ///< 128
        uint32_t    num_experts;        ///< 32
        uint32_t    top_k_experts;      ///< 2
        uint32_t    d_ffn;              ///< FFN hidden dim per expert
        uint32_t    vocab_size;
        QuantFormat weight_format;      ///< INT8 or INT4_AWQ
    };

    Executor(Device& dev, MemoryPool& pool, KVCacheManager& kvc_mgr,
             const ModelConfig& model_cfg);
    ~Executor();

    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;

    /** Load all model weights from model_dir. Must be called before run_prefill/run_decode. */
    void load_weights(const std::string& model_dir);

    /** Prefill: process prompt tokens for a session. */
    void run_prefill(InferenceSession& session);

    /**
     * Decode: generate one token.
     * Builds the 21-step descriptor sequence, submits, waits, returns token id.
     */
    int32_t run_decode_step(InferenceSession& session);

private:
    Device&          dev_;
    MemoryPool&      pool_;
    KVCacheManager&  kvc_mgr_;
    ModelConfig      model_cfg_;
    DescriptorQueue  compute_q_;   ///< queue 0 — GEMM/VPU/KVC/MoE
    DescriptorQueue  util_q_;      ///< queue 1 — DMA weight prefetch
    // weight_loader_ must be declared AFTER util_q_ because the constructor
    // initialises it with a reference to util_q_.  C++ initialises members in
    // declaration order, so declaring weight_loader_ before util_q_ would
    // pass a reference to an uninitialised object — undefined behaviour.
    WeightLoader     weight_loader_;

    struct LayerWeights {
        MemHandle q_proj, k_proj, v_proj, o_proj;
        MemHandle router;
        std::vector<MemHandle> expert_gate;   ///< [num_experts]
        std::vector<MemHandle> expert_up;
        std::vector<MemHandle> expert_down;
        MemHandle rms_norm_weight;
    };
    std::vector<LayerWeights> layer_weights_;  ///< [num_layers]
    MemHandle embedding_weight_{INVALID_MEM_HANDLE};
    MemHandle lm_head_weight_{INVALID_MEM_HANDLE};

    void alloc_scratch_buffers(InferenceSession& session);

    /** Build all layer descriptors for one decode step. */
    void build_layer_decode(InferenceSession& session, uint32_t layer_id);

    // --- 21-step decode sequence builders ---
    //
    // Each method appends descriptor(s) to compute_q_.
    // Steps follow yua-llm-hw-design.md §4 exactly.

    /** Step 1: DMA_2D — input embedding → scratch.hidden_state */
    void step_01_dma_embedding(InferenceSession& session);

    /** Step 2: GEMM_INT8/4 — [Q,K,V] = hidden @ [Wq,Wk,Wv]  (3 GEMMs) */
    void step_02_qkv_projection(InferenceSession& session, uint32_t layer_id);

    /** Step 3: VECTOR_OP_EX — RoPE on Q and K */
    void step_03_rope(InferenceSession& session);

    /** Step 4: KVC_WRITE — store new K,V to KV-Cache */
    void step_04_kvc_write(InferenceSession& session, uint32_t layer_id);

    /** Step 5: KVC_READ — load all K,V (seq_len entries) from KV-Cache */
    void step_05_kvc_read(InferenceSession& session, uint32_t layer_id);

    /** Step 6: GEMM_INT8/4 — attention_scores = Q @ K^T */
    void step_06_attn_qkt(InferenceSession& session);

    /** Step 7: SOFTMAX — scale(1/√head_dim) + softmax(attention_scores) */
    void step_07_scale_softmax(InferenceSession& session);

    /** Step 8: GEMM_INT8/4 — attention_out = attention_scores @ V */
    void step_08_attn_v(InferenceSession& session);

    /** Step 9: GEMM_INT8/4 — hidden = attention_out @ Wo */
    void step_09_output_proj(InferenceSession& session, uint32_t layer_id);

    /** Step 10: VECTOR_OP — residual add: hidden += original_hidden */
    void step_10_residual_add(InferenceSession& session);

    /** Step 11: VECTOR_OP_EX — RMSNorm(hidden) */
    void step_11_rmsnorm(InferenceSession& session, uint32_t layer_id);

    /** Step 12: GEMM_INT8/4 — moe_logits = hidden @ router_weight */
    void step_12_moe_router_gemm(InferenceSession& session, uint32_t layer_id);

    /** Step 13: MOE_ROUTE — top-2 expert indices + scores */
    void step_13_moe_route(InferenceSession& session);

    /** Step 14: GEMM_INT8/4 — gate_proj per selected expert */
    void step_14_expert_gate(InferenceSession& session, uint32_t layer_id);

    /** Step 15: VECTOR_OP_EX — SiLU(gate_proj) */
    void step_15_silu(InferenceSession& session);

    /** Step 16: GEMM_INT8/4 — up_proj per selected expert */
    void step_16_expert_up(InferenceSession& session, uint32_t layer_id);

    /** Step 17: VECTOR_OP — gated_ffn = SiLU(gate) * up */
    void step_17_element_mul(InferenceSession& session);

    /** Step 18: GEMM_INT8/4 — down_proj per selected expert */
    void step_18_expert_down(InferenceSession& session, uint32_t layer_id);

    /** Step 19: BARRIER + VECTOR_OP — combine expert outputs + residual add */
    void step_19_barrier_residual(InferenceSession& session);

    /** Step 20: GEMM_INT8/4 — lm_head = hidden @ Wlm  (after all layers) */
    void step_20_lm_head(InferenceSession& session);

    /** Step 21: VECTOR_OP + EVENT — softmax(lm_head) → argmax → next_token_id */
    void step_21_softmax_argmax_event(InferenceSession& session);

    /** Choose GEMM descriptor type based on weight_format. */
    void add_gemm(uint64_t act, uint64_t wgt, uint64_t out,
                  uint64_t scale_addr, uint32_t Kt);
};

} // namespace orbit
