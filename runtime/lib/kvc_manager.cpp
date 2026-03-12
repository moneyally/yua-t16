/**
 * kvc_manager.cpp — KVCacheManager implementation
 *
 * PagedAttention-style KV-cache allocator for ORBIT-G1.
 *
 * Physical memory layout (from kvc.md §4):
 *   - GDDR6 pool divided into fixed-size physical pages.
 *   - Each page stores KV data for page_size_tokens tokens, all layers, all heads.
 *   - Page layout: layer-major → K/V-major → head-major → token-major.
 *
 * Address formula (kvc.md §4.3):
 *   block_base   = page_base_addr (from pool)
 *   layer_offset = layer_id  * (2 * num_heads * page_size * head_dim * dtype_bytes)
 *   kv_offset    = is_key ? 0 : (num_heads * page_size * head_dim * dtype_bytes)
 *   head_offset  = head_id * (page_size * head_dim * dtype_bytes)
 *   token_offset = (token_pos % page_size) * (head_dim * dtype_bytes)
 *   elem_addr    = block_base + layer_offset + kv_offset + head_offset + token_offset
 *
 * For GPT-OSS-20B (FP16): page_bytes = 32 * 32 * 16 * 128 * 2 * 2 = 8 MB
 */

#include "orbit.h"

#include <stdexcept>
#include <cassert>
#include <cstring>

namespace orbit {

// ============================================================
// dtype_bytes helper
// ============================================================

uint32_t KVCacheManager::dtype_bytes() const noexcept {
    switch (cfg_.dtype) {
        case DataType::FP16:  return 2;
        case DataType::BF16:  return 2;
        case DataType::INT8:  return 1;
        case DataType::INT4:  return 1;   // packed; treated as 1 byte per elem
        case DataType::INT16: return 2;
        case DataType::INT32: return 4;
        default:              return 2;
    }
}

// ============================================================
// page_bytes
// ============================================================

size_t KVCacheManager::page_bytes() const noexcept {
    // page_bytes = num_layers * 2(K+V) * num_heads * page_size_tokens * head_dim * dtype_bytes
    // = 32 * 2 * 32 * 16 * 128 * 2 = 8,388,608 bytes (8 MB) for GPT-OSS-20B FP16
    return static_cast<size_t>(cfg_.num_layers) *
           2u *
           cfg_.num_heads *
           cfg_.page_size_tokens *
           cfg_.head_dim *
           dtype_bytes();
}

// ============================================================
// token_offset_within_page — byte offset for one token's KV element
// ============================================================

uint64_t KVCacheManager::token_offset_within_page(uint32_t layer_id, uint32_t head_id,
                                                    uint32_t token_within_page,
                                                    bool is_key) const noexcept {
    uint32_t db = dtype_bytes();
    // Per kvc.md §4.3 formula:
    uint64_t per_head_tokens  = static_cast<uint64_t>(cfg_.page_size_tokens) * cfg_.head_dim * db;
    uint64_t per_layer_kv     = static_cast<uint64_t>(2) * cfg_.num_heads * per_head_tokens;
    uint64_t layer_offset     = static_cast<uint64_t>(layer_id) * per_layer_kv;
    uint64_t kv_select_offset = is_key ? 0ULL : (static_cast<uint64_t>(cfg_.num_heads) * per_head_tokens);
    uint64_t head_offset      = static_cast<uint64_t>(head_id) * per_head_tokens;
    uint64_t token_offset     = static_cast<uint64_t>(token_within_page) * cfg_.head_dim * db;

    return layer_offset + kv_select_offset + head_offset + token_offset;
}

// ============================================================
// Constructor — pre-allocate all pages from GDDR6
// ============================================================

KVCacheManager::KVCacheManager(Device& dev, MemoryPool& pool, const Config& cfg)
    : dev_(dev), pool_(pool), cfg_(cfg) {

    size_t pb = page_bytes();
    if (pb == 0) {
        throw OrbitException("KVCacheManager: page_bytes is zero (bad config)");
    }

    pages_.resize(cfg_.max_pages);

    // Pre-allocate all pages now to avoid allocation latency during inference.
    for (uint32_t i = 0; i < cfg_.max_pages; ++i) {
        // 256-byte aligned per kvc.md §11.
        MemHandle h = pool_.alloc(pb, 256);
        pages_[i].device_addr = pool_.device_addr(h);
        pages_[i].in_use      = false;
        free_page_ids_.push(i);
    }
}

// ============================================================
// Destructor
// ============================================================

KVCacheManager::~KVCacheManager() {
    // TODO: free all GDDR6 pages via pool_.free().
    // The MemoryPool destructor will clean up if pages were registered there,
    // but it's good practice to explicitly free here.
    // For now: no-op (MemoryPool owns the backing memory).
}

// ============================================================
// alloc_page — private, must be called with mutex held
// ============================================================

uint32_t KVCacheManager::alloc_page() {
    if (free_page_ids_.empty()) {
        throw OrbitException("KVCacheManager::alloc_page: out of KV pages (OOM)");
    }
    uint32_t id = free_page_ids_.front();
    free_page_ids_.pop();
    pages_[id].in_use = true;
    return id;
}

// ============================================================
// alloc_sequence
// ============================================================

KVCacheManager::SeqHandle KVCacheManager::alloc_sequence(uint32_t initial_capacity_tokens) {
    std::lock_guard<std::mutex> lk(mutex_);

    SeqHandle sh = next_seq_handle_++;
    SeqState& s  = seqs_[sh];
    s.num_tokens = 0;

    if (initial_capacity_tokens > 0) {
        uint32_t pages_needed = (initial_capacity_tokens + cfg_.page_size_tokens - 1)
                                / cfg_.page_size_tokens;
        s.page_ids.reserve(pages_needed);
        for (uint32_t i = 0; i < pages_needed; ++i) {
            uint32_t pid = alloc_page();
            s.page_ids.push_back(pid);
            s.page_table.page_addrs.push_back(pages_[pid].device_addr);
        }
        s.page_table.num_tokens = 0;
    }

    return sh;
}

// ============================================================
// extend_sequence
// ============================================================

bool KVCacheManager::extend_sequence(SeqHandle seq, uint32_t additional_tokens) {
    std::lock_guard<std::mutex> lk(mutex_);

    auto it = seqs_.find(seq);
    if (it == seqs_.end()) {
        return false;
    }

    SeqState& s = it->second;
    uint32_t pages_needed = (additional_tokens + cfg_.page_size_tokens - 1)
                            / cfg_.page_size_tokens;

    if (free_page_ids_.size() < pages_needed) {
        return false;   // Not enough pages
    }

    for (uint32_t i = 0; i < pages_needed; ++i) {
        uint32_t pid = alloc_page();
        s.page_ids.push_back(pid);
        s.page_table.page_addrs.push_back(pages_[pid].device_addr);
    }
    return true;
}

// ============================================================
// free_sequence
// ============================================================

void KVCacheManager::free_sequence(SeqHandle seq) {
    std::lock_guard<std::mutex> lk(mutex_);

    auto it = seqs_.find(seq);
    if (it == seqs_.end()) return;

    SeqState& s = it->second;
    for (uint32_t pid : s.page_ids) {
        pages_[pid].in_use = false;
        free_page_ids_.push(pid);
    }
    seqs_.erase(it);
}

// ============================================================
// kv_addr
// ============================================================

uint64_t KVCacheManager::kv_addr(SeqHandle seq, uint32_t layer_id, uint32_t head_id,
                                  uint32_t token_pos, bool is_key) const {
    std::lock_guard<std::mutex> lk(mutex_);

    auto it = seqs_.find(seq);
    if (it == seqs_.end()) {
        throw OrbitException("KVCacheManager::kv_addr: invalid sequence handle");
    }

    const SeqState& s = it->second;

    // Determine which physical page holds this token.
    uint32_t logical_page   = token_pos / cfg_.page_size_tokens;
    uint32_t token_in_page  = token_pos % cfg_.page_size_tokens;

    if (logical_page >= s.page_ids.size()) {
        throw OrbitException("KVCacheManager::kv_addr: token_pos " +
                             std::to_string(token_pos) +
                             " not covered by allocated pages (logical_page=" +
                             std::to_string(logical_page) + ")");
    }

    uint32_t phys_page_id = s.page_ids[logical_page];
    uint64_t page_base    = pages_[phys_page_id].device_addr;

    // Compute within-page byte offset using kvc.md §4.3 formula.
    uint64_t offset = token_offset_within_page(layer_id, head_id, token_in_page, is_key);

    return page_base + offset;
}

// ============================================================
// total_pages / free_pages
// ============================================================

uint32_t KVCacheManager::total_pages() const noexcept {
    return cfg_.max_pages;
}

uint32_t KVCacheManager::free_pages() const noexcept {
    std::lock_guard<std::mutex> lk(mutex_);
    return static_cast<uint32_t>(free_page_ids_.size());
}

// ============================================================
// page_table
// ============================================================

const KVCacheManager::PageTable& KVCacheManager::page_table(SeqHandle seq) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = seqs_.find(seq);
    if (it == seqs_.end()) {
        throw OrbitException("KVCacheManager::page_table: invalid sequence handle");
    }
    return it->second.page_table;
}

} // namespace orbit
