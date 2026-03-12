/**
 * descriptor_queue.cpp — DescriptorQueue implementation
 *
 * Batch builder for all 15 descriptor types (v1 + v2).
 * Descriptors are packed to the exact 64-byte binary layout defined in
 * descriptor.md and yua-llm-hw-design.md §6.
 *
 * Thread-safety: NOT thread-safe. External locking required for concurrent use.
 */

#include "orbit.h"

#include <cstring>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <bit>       // std::bit_cast (C++20); use memcpy fallback for C++17

namespace orbit {

// ============================================================
// Helper: float → uint32_t bit-pattern (C++17-safe)
// ============================================================

static inline uint32_t float_to_bits(float f) {
    uint32_t v;
    std::memcpy(&v, &f, sizeof(v));
    return v;
}

// ============================================================
// Constructor
// ============================================================

DescriptorQueue::DescriptorQueue(Device& dev, uint32_t queue_id, size_t max_batch)
    : dev_(dev), queue_id_(queue_id), max_batch_(max_batch) {
    batch_.reserve(max_batch);
}

// ============================================================
// append_desc<T> — copy 64-byte descriptor into batch
// ============================================================

template<typename T>
void DescriptorQueue::append_desc(const T& desc_struct) {
    static_assert(sizeof(T) == 64, "All descriptors must be exactly 64 bytes");

    if (batch_.size() >= max_batch_) {
        throw OrbitException("DescriptorQueue: batch full (max=" +
                             std::to_string(max_batch_) + ")");
    }

    std::array<uint8_t, 64> raw{};
    std::memcpy(raw.data(), &desc_struct, 64);
    batch_.push_back(raw);
}

// ============================================================
// add_dma_2d — type 0x01
// ============================================================

void DescriptorQueue::add_dma_2d(uint64_t src, uint64_t dst,
                                  uint32_t width_bytes, uint32_t height,
                                  uint32_t src_stride, uint32_t dst_stride) {
    OrbitDescDma2D d{};
    d.h.type       = static_cast<uint8_t>(DescType::DMA_2D);
    d.h.flags      = 0;
    d.h.reserved0  = 0;
    d.h.length     = width_bytes * height;
    d.h.next_desc  = 0;
    d.src_addr     = src;
    d.dst_addr     = dst;
    d.width_bytes  = width_bytes;
    d.height       = height;
    d.src_stride   = src_stride;
    d.dst_stride   = dst_stride;
    d.reserved1    = 0;
    d.reserved2    = 0;
    append_desc(d);
}

// ============================================================
// add_gemm_int8 — type 0x02
// ============================================================

void DescriptorQueue::add_gemm_int8(uint64_t act_addr, uint64_t wgt_addr, uint64_t out_addr,
                                     uint32_t Kt, uint16_t m_tiles, uint16_t n_tiles) {
    OrbitDescGemmInt8 d{};
    d.h.type      = static_cast<uint8_t>(DescType::GEMM_INT8);
    d.h.flags     = 0;
    d.h.reserved0 = 0;
    d.h.length    = Kt;
    d.h.next_desc = 0;
    d.act_addr    = act_addr;
    d.wgt_addr    = wgt_addr;
    d.out_addr    = out_addr;
    d.Kt          = Kt;
    d.m_tiles     = m_tiles;
    d.n_tiles     = n_tiles;
    d.scale_a     = 0;
    d.scale_b     = 0;
    d.epilogue    = 0;
    d.reserved    = 0;
    append_desc(d);
}

// ============================================================
// add_gemm_int4 — type 0x0E
// ============================================================

void DescriptorQueue::add_gemm_int4(uint64_t act_addr, uint64_t wgt_addr, uint64_t out_addr,
                                     uint64_t scale_addr, uint32_t Kt,
                                     uint16_t m_tiles, uint16_t n_tiles) {
    OrbitDescGemmInt4 d{};
    d.h.type      = static_cast<uint8_t>(DescType::GEMM_INT4);
    d.h.flags     = 0;
    d.h.reserved0 = 0;
    d.h.length    = Kt;
    d.h.next_desc = 0;
    d.act_addr    = act_addr;
    d.wgt_addr    = wgt_addr;
    d.out_addr    = out_addr;
    d.scale_addr  = scale_addr;
    d.Kt          = Kt;
    d.m_tiles     = m_tiles;
    d.n_tiles     = n_tiles;
    append_desc(d);
}

// ============================================================
// add_vector_op — type 0x03
// ============================================================

void DescriptorQueue::add_vector_op(uint64_t src, uint64_t dst, uint32_t element_count,
                                     VectorOpType op, DataType dtype, uint32_t imm) {
    OrbitDescVectorOp d{};
    d.h.type          = static_cast<uint8_t>(DescType::VECTOR_OP);
    d.h.flags         = 0;
    d.h.reserved0     = 0;
    d.h.length        = element_count;
    d.h.next_desc     = 0;
    d.src_addr        = src;
    d.dst_addr        = dst;
    d.element_count   = element_count;
    d.op_type         = static_cast<uint16_t>(op);
    d.data_type       = static_cast<uint16_t>(dtype);
    d.imm             = imm;
    d.reserved        = 0;
    std::memset(d._pad, 0, sizeof(d._pad));
    append_desc(d);
}

// ============================================================
// add_vector_op_ex — type 0x0D
// ============================================================

void DescriptorQueue::add_vector_op_ex(uint64_t src, uint64_t dst, uint32_t element_count,
                                        VectorOpExType op, DataType dtype,
                                        uint64_t aux_addr, uint32_t aux_param) {
    OrbitDescVectorOpEx d{};
    d.h.type        = static_cast<uint8_t>(DescType::VECTOR_OP_EX);
    d.h.flags       = 0;
    d.h.reserved0   = 0;
    d.h.length      = element_count;
    d.h.next_desc   = 0;
    d.src_addr      = src;
    d.dst_addr      = dst;
    d.element_count = element_count;
    d.op_type       = static_cast<uint16_t>(op);
    d.data_type     = static_cast<uint16_t>(dtype);
    d.aux_addr      = aux_addr;
    d.aux_param     = aux_param;
    d.reserved      = 0;
    append_desc(d);
}

// ============================================================
// add_kvc_read — type 0x0A
// ============================================================

void DescriptorQueue::add_kvc_read(uint32_t seq_id, uint32_t layer_id,
                                    uint32_t seq_start, uint32_t seq_len,
                                    uint64_t k_dst_addr, uint64_t v_dst_addr,
                                    uint8_t read_mode, uint8_t head_id) {
    OrbitDescKvcRead d{};
    d.h.type       = static_cast<uint8_t>(DescType::KVC_READ);
    d.h.flags      = 0;
    d.h.reserved0  = 0;
    d.h.length     = seq_len;
    d.h.next_desc  = 0;
    d.seq_id       = seq_id;
    d.layer_id     = layer_id;
    d.seq_start    = seq_start;
    d.seq_len      = seq_len;
    d.k_dst_addr   = k_dst_addr;
    d.v_dst_addr   = v_dst_addr;
    d.read_mode    = read_mode;
    d.head_id      = head_id;
    d.reserved     = 0;
    std::memset(d._pad, 0, sizeof(d._pad));
    append_desc(d);
}

// ============================================================
// add_kvc_write — type 0x0B
// ============================================================

void DescriptorQueue::add_kvc_write(uint32_t seq_id, uint32_t layer_id,
                                     uint32_t token_pos, uint32_t write_count,
                                     uint64_t k_src_addr, uint64_t v_src_addr,
                                     uint8_t write_mode, uint8_t alloc_blocks) {
    OrbitDescKvcWrite d{};
    d.h.type        = static_cast<uint8_t>(DescType::KVC_WRITE);
    d.h.flags       = 0;
    d.h.reserved0   = 0;
    d.h.length      = write_count;
    d.h.next_desc   = 0;
    d.seq_id        = seq_id;
    d.layer_id      = layer_id;
    d.token_pos     = token_pos;
    d.write_count   = write_count;
    d.k_src_addr    = k_src_addr;
    d.v_src_addr    = v_src_addr;
    d.write_mode    = write_mode;
    d.alloc_blocks  = alloc_blocks;
    d.reserved      = 0;
    std::memset(d._pad, 0, sizeof(d._pad));
    append_desc(d);
}

// ============================================================
// add_moe_route — type 0x0C
// ============================================================

void DescriptorQueue::add_moe_route(uint64_t logits_addr, uint64_t indices_addr,
                                     uint64_t scores_addr, uint32_t num_tokens,
                                     uint32_t num_experts, uint32_t top_k) {
    OrbitDescMoeRoute d{};
    d.h.type        = static_cast<uint8_t>(DescType::MOE_ROUTE);
    d.h.flags       = 0;
    d.h.reserved0   = 0;
    d.h.length      = num_tokens;
    d.h.next_desc   = 0;
    d.logits_addr   = logits_addr;
    d.indices_addr  = indices_addr;
    d.scores_addr   = scores_addr;
    d.num_tokens    = num_tokens;
    d.num_experts   = num_experts;
    d.top_k         = top_k;
    d.reserved      = 0;
    append_desc(d);
}

// ============================================================
// add_format_convert — type 0x05
// ============================================================

void DescriptorQueue::add_format_convert(uint64_t src, uint64_t dst, uint32_t element_count,
                                          DataFormat src_fmt, DataFormat dst_fmt,
                                          uint32_t options) {
    OrbitDescFormatConvert d{};
    d.h.type        = static_cast<uint8_t>(DescType::FORMAT_CONVERT);
    d.h.flags       = 0;
    d.h.reserved0   = 0;
    d.h.length      = element_count;
    d.h.next_desc   = 0;
    d.src_addr      = src;
    d.dst_addr      = dst;
    d.element_count = element_count;
    d.src_format    = static_cast<uint16_t>(src_fmt);
    d.dst_format    = static_cast<uint16_t>(dst_fmt);
    d.options       = options;
    d.reserved      = 0;
    std::memset(d._pad, 0, sizeof(d._pad));
    append_desc(d);
}

// ============================================================
// add_copy_2d_plus — type 0x04
// ============================================================

void DescriptorQueue::add_copy_2d_plus(uint64_t src, uint64_t dst,
                                        uint32_t width_bytes, uint32_t height,
                                        uint32_t src_stride, uint32_t dst_stride,
                                        uint32_t options) {
    OrbitDescCopy2DPlus d{};
    d.h.type       = static_cast<uint8_t>(DescType::COPY_2D_PLUS);
    d.h.flags      = 0;
    d.h.reserved0  = 0;
    d.h.length     = width_bytes * height;
    d.h.next_desc  = 0;
    d.src_addr     = src;
    d.dst_addr     = dst;
    d.width_bytes  = width_bytes;
    d.height       = height;
    d.src_stride   = src_stride;
    d.dst_stride   = dst_stride;
    d.options      = options;
    d.reserved     = 0;
    append_desc(d);
}

// ============================================================
// add_frame_fingerprint — type 0x06
// ============================================================

void DescriptorQueue::add_frame_fingerprint(uint64_t src, uint64_t result_addr,
                                             uint32_t byte_count, HashType hash_type) {
    OrbitDescFrameFingerprint d{};
    d.h.type        = static_cast<uint8_t>(DescType::FRAME_FINGERPRINT);
    d.h.flags       = 0;
    d.h.reserved0   = 0;
    d.h.length      = byte_count;
    d.h.next_desc   = 0;
    d.src_addr      = src;
    d.result_addr   = result_addr;
    d.byte_count    = byte_count;
    d.hash_type     = static_cast<uint32_t>(hash_type);
    d.reserved      = 0;
    std::memset(d._pad, 0, sizeof(d._pad));
    append_desc(d);
}

// ============================================================
// add_barrier — type 0x07
// ============================================================

void DescriptorQueue::add_barrier() {
    OrbitDescBarrier d{};
    d.h.type      = static_cast<uint8_t>(DescType::BARRIER);
    d.h.flags     = 0;
    d.h.reserved0 = 0;
    d.h.length    = 0;
    d.h.next_desc = 0;
    d.reserved1   = 0;
    d.reserved2   = 0;
    d.reserved3   = 0;
    append_desc(d);
}

// ============================================================
// add_event — type 0x08
// ============================================================

void DescriptorQueue::add_event(uint32_t event_id, uint32_t options) {
    OrbitDescEvent d{};
    d.h.type      = static_cast<uint8_t>(DescType::EVENT);
    d.h.flags     = 0;
    d.h.reserved0 = 0;
    d.h.length    = 0;
    d.h.next_desc = 0;
    d.event_id    = event_id;
    d.options     = options;
    d.reserved1   = 0;
    d.reserved2   = 0;
    std::memset(d._pad, 0, sizeof(d._pad));
    append_desc(d);
}

// ============================================================
// add_perf_snapshot — type 0x09
// ============================================================

void DescriptorQueue::add_perf_snapshot(uint64_t dst_addr) {
    OrbitDescPerfSnapshot d{};
    d.h.type      = static_cast<uint8_t>(DescType::PERF_SNAPSHOT);
    d.h.flags     = 0;
    d.h.reserved0 = 0;
    d.h.length    = 0;
    d.h.next_desc = 0;
    d.dst_addr    = dst_addr;
    d.reserved1   = 0;
    d.reserved2   = 0;
    std::memset(d._pad, 0, sizeof(d._pad));
    append_desc(d);
}

// ============================================================
// add_softmax — type 0x0F
// ============================================================

void DescriptorQueue::add_softmax(uint64_t src, uint64_t dst,
                                   uint32_t element_count, uint32_t num_rows,
                                   float scale) {
    OrbitDescSoftmax d{};
    d.h.type        = static_cast<uint8_t>(DescType::SOFTMAX);
    d.h.flags       = 0;
    d.h.reserved0   = 0;
    d.h.length      = element_count;
    d.h.next_desc   = 0;
    d.src_addr      = src;
    d.dst_addr      = dst;
    d.element_count = element_count;
    d.num_rows      = num_rows;
    d.scale_fp32    = float_to_bits(scale);
    d.reserved      = 0;
    std::memset(d._pad, 0, sizeof(d._pad));
    append_desc(d);
}

// ============================================================
// submit — bulk submit current batch
// ============================================================

SubmitCookie DescriptorQueue::submit() {
    if (batch_.empty()) {
        throw OrbitException("DescriptorQueue::submit: empty batch");
    }

    OrbitDescSubmit req{};
    req.queue_id     = queue_id_;
    req.count        = static_cast<uint32_t>(batch_.size());
    req.flags        = 0;
    req._pad         = 0;
    req.descs_ptr    = reinterpret_cast<uint64_t>(batch_.data());
    req.submit_cookie = 0;

    int ret = dev_.submit_desc(req);
    if (ret < 0) {
        throw OrbitException("DescriptorQueue::submit: ioctl failed", -ret);
    }

    SubmitCookie cookie = req.submit_cookie;
    batch_.clear();
    return cookie;
}

// ============================================================
// submit_and_wait
// ============================================================

void DescriptorQueue::submit_and_wait(uint32_t timeout_ms) {
    SubmitCookie cookie = submit();
    wait(cookie, timeout_ms);
}

// ============================================================
// wait
// ============================================================

void DescriptorQueue::wait(SubmitCookie cookie, uint32_t timeout_ms) {
    if (cookie == INVALID_SUBMIT_COOKIE) return;

    OrbitWaitDone req{};
    req.queue_id     = queue_id_;
    req.timeout_ms   = timeout_ms;
    req.submit_cookie = cookie;

    int ret = dev_.wait_done(req);
    if (ret < 0) {
        throw OrbitException("DescriptorQueue::wait: ioctl failed", -ret);
    }
}

// ============================================================
// reset_batch
// ============================================================

void DescriptorQueue::reset_batch() {
    batch_.clear();
}

// ============================================================
// batch_size
// ============================================================

size_t DescriptorQueue::batch_size() const noexcept {
    return batch_.size();
}

} // namespace orbit
