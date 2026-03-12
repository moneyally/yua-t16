/**
 * session.cpp — InferenceSession implementation
 *
 * Per-request state container for one autoregressive inference run.
 * Manages KV-cache sequence handle and scratch buffer addresses.
 * The Executor drives prefill and decode steps; the session just holds state.
 */

#include "orbit.h"

#include <stdexcept>
#include <cassert>
#include <cstring>

namespace orbit {

// ============================================================
// Static member
// ============================================================

std::atomic<uint32_t> InferenceSession::next_seq_id_{1};

// ============================================================
// Constructor
// ============================================================

InferenceSession::InferenceSession(KVCacheManager& kvc_mgr, const Config& cfg)
    : kvc_mgr_(kvc_mgr),
      kv_handle_(KVCacheManager::INVALID_SEQ),
      cfg_(cfg),
      seq_id_(next_seq_id_.fetch_add(1, std::memory_order_relaxed)),
      context_len_(0),
      done_(false),
      cancelled_(false),
      scratch_{} {

    // Allocate a KV-cache sequence handle.
    // Initial capacity: 0 (pages allocated lazily or by Executor before prefill).
    kv_handle_ = kvc_mgr_.alloc_sequence(0);
}

// ============================================================
// Destructor — frees KV pages
// ============================================================

InferenceSession::~InferenceSession() {
    if (kv_handle_ != KVCacheManager::INVALID_SEQ) {
        try {
            kvc_mgr_.free_sequence(kv_handle_);
        } catch (...) {
            // Destructor must not throw.
        }
        kv_handle_ = KVCacheManager::INVALID_SEQ;
    }
}

// ============================================================
// prefill
// ============================================================

void InferenceSession::prefill(const std::vector<int32_t>& input_token_ids) {
    if (cancelled_) {
        throw OrbitException("InferenceSession::prefill: session cancelled");
    }
    if (context_len_ > 0) {
        throw OrbitException("InferenceSession::prefill: already prefilled");
    }

    // Extend the KV-cache sequence to hold all input tokens.
    uint32_t prompt_len = static_cast<uint32_t>(input_token_ids.size());
    if (prompt_len > 0) {
        bool ok = kvc_mgr_.extend_sequence(kv_handle_, prompt_len);
        if (!ok) {
            throw OrbitException("InferenceSession::prefill: OOM — not enough KV pages");
        }
    }

    // The actual descriptor sequence for prefill is built and submitted by Executor.
    // We record the context length here.
    context_len_ = prompt_len;

    // TODO: store the input_token_ids for tokenizer / debug access if needed.
}

// ============================================================
// step — decode one token
// ============================================================

bool InferenceSession::step() {
    if (done_ || cancelled_) return false;

    // Check termination conditions.
    if (static_cast<uint32_t>(generated_tokens_.size()) >= cfg_.max_new_tokens) {
        done_ = true;
        return false;
    }

    // Extend KV-cache for the new token position.
    bool ok = kvc_mgr_.extend_sequence(kv_handle_, 1);
    if (!ok) {
        // OOM: stop generation gracefully.
        done_ = true;
        return false;
    }

    // The actual decode descriptor sequence is built and submitted by Executor.
    // Executor will call this method indirectly via run_decode_step().
    // After the descriptor sequence completes, Executor updates generated_tokens_.
    return true;
}

// ============================================================
// cancel
// ============================================================

void InferenceSession::cancel() {
    cancelled_ = true;
    done_      = true;
}

// ============================================================
// Accessors
// ============================================================

uint32_t InferenceSession::seq_id() const noexcept {
    return seq_id_;
}

uint32_t InferenceSession::context_length() const noexcept {
    return context_len_ + static_cast<uint32_t>(generated_tokens_.size());
}

const std::vector<int32_t>& InferenceSession::generated_tokens() const noexcept {
    return generated_tokens_;
}

int32_t InferenceSession::last_token() const noexcept {
    if (generated_tokens_.empty()) return -1;
    return generated_tokens_.back();
}

bool InferenceSession::is_done() const noexcept {
    return done_ || cancelled_;
}

KVCacheManager::SeqHandle InferenceSession::kv_handle() const noexcept {
    return kv_handle_;
}

const InferenceSession::ScratchBuffers& InferenceSession::scratch() const noexcept {
    return scratch_;
}

void InferenceSession::set_scratch(const ScratchBuffers& s) {
    scratch_ = s;
}

} // namespace orbit
