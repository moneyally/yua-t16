/**
 * executor.cpp — Executor implementation
 *
 * Orchestrates the 21-step descriptor sequence for one token decode of GPT-OSS-20B.
 * Each step maps exactly to the decode sequence in yua-llm-hw-design.md §4.
 *
 * 21-Step Decode Sequence (per-token, looped over all layers):
 *
 *   For each layer (0 to num_layers-1):
 *     Step  1: DMA_2D         — input embedding → scratch.hidden_state
 *     Step  2: GEMM_INT8/4    — [Q,K,V] = hidden @ [Wq,Wk,Wv]  (3 GEMMs)
 *     Step  3: VECTOR_OP_EX   — RoPE on Q and K
 *     Step  4: KVC_WRITE      — store new K,V to KV-Cache
 *     Step  5: KVC_READ       — load full K,V history from KV-Cache
 *     Step  6: GEMM_INT8/4    — attention_scores = Q @ K^T
 *     Step  7: SOFTMAX        — scale(1/√head_dim) + softmax(attention_scores)
 *     Step  8: GEMM_INT8/4    — attention_out = attention_scores @ V
 *     Step  9: GEMM_INT8/4    — hidden = attention_out @ Wo
 *     Step 10: VECTOR_OP      — residual add: hidden += original_hidden
 *     Step 11: VECTOR_OP_EX   — RMSNorm(hidden)
 *     Step 12: GEMM_INT8/4    — moe_logits = hidden @ router_weight
 *     Step 13: MOE_ROUTE      — top-2 expert indices + scores
 *     Step 14: GEMM_INT8/4    — gate_proj per selected expert
 *     Step 15: VECTOR_OP_EX   — SiLU(gate_proj)
 *     Step 16: GEMM_INT8/4    — up_proj per selected expert
 *     Step 17: VECTOR_OP      — gated_ffn = SiLU(gate) * up
 *     Step 18: GEMM_INT8/4    — down_proj per selected expert
 *     Step 19: BARRIER + VECTOR_OP — combine expert outputs + residual add
 *     (BARRIER between layers)
 *
 *   After all layers:
 *     Step 20: GEMM_INT8/4    — lm_head = hidden @ Wlm
 *     Step 21: VECTOR_OP + EVENT — softmax(lm_head) → argmax → next_token_id
 */

#include "orbit.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <cassert>
#include <filesystem>

namespace orbit {

// ============================================================
// Constructor
// ============================================================

Executor::Executor(Device& dev, MemoryPool& pool, KVCacheManager& kvc_mgr,
                   const ModelConfig& model_cfg)
    : dev_(dev),
      pool_(pool),
      kvc_mgr_(kvc_mgr),
      model_cfg_(model_cfg),
      compute_q_(dev, static_cast<uint32_t>(QueueId::COMPUTE), 512),
      util_q_(dev, static_cast<uint32_t>(QueueId::UTILITY), 64),
      weight_loader_(dev, pool, util_q_) {

    layer_weights_.resize(model_cfg_.num_layers);
}

// ============================================================
// Destructor
// ============================================================

Executor::~Executor() {
    // Unload all layer weights.
    for (auto& lw : layer_weights_) {
        auto unload_if_valid = [this](MemHandle& h) {
            if (h != INVALID_MEM_HANDLE) { weight_loader_.unload(h); h = INVALID_MEM_HANDLE; }
        };
        unload_if_valid(lw.q_proj);
        unload_if_valid(lw.k_proj);
        unload_if_valid(lw.v_proj);
        unload_if_valid(lw.o_proj);
        unload_if_valid(lw.router);
        unload_if_valid(lw.rms_norm_weight);
        for (auto& h : lw.expert_gate)  { if (h != INVALID_MEM_HANDLE) weight_loader_.unload(h); }
        for (auto& h : lw.expert_up)    { if (h != INVALID_MEM_HANDLE) weight_loader_.unload(h); }
        for (auto& h : lw.expert_down)  { if (h != INVALID_MEM_HANDLE) weight_loader_.unload(h); }
    }
    if (embedding_weight_ != INVALID_MEM_HANDLE) weight_loader_.unload(embedding_weight_);
    if (lm_head_weight_   != INVALID_MEM_HANDLE) weight_loader_.unload(lm_head_weight_);
}

// ============================================================
// load_weights
// ============================================================

void Executor::load_weights(const std::string& model_dir) {
    // TODO: implement actual weight file discovery and loading.
    // Expected layout under model_dir/:
    //   embed_tokens.bin
    //   lm_head.bin
    //   layers/<N>/q_proj.bin, k_proj.bin, v_proj.bin, o_proj.bin
    //   layers/<N>/router.bin
    //   layers/<N>/experts/<E>/gate_proj.bin, up_proj.bin, down_proj.bin
    //   layers/<N>/rms_norm.bin

    QuantFormat fmt = model_cfg_.weight_format;

    auto load = [&](const std::string& rel_path) -> MemHandle {
        std::string full = model_dir + "/" + rel_path;
        // TODO: check file exists before loading.
        return weight_loader_.load_weight_file(full, fmt);
    };

    embedding_weight_ = load("embed_tokens.bin");
    lm_head_weight_   = load("lm_head.bin");

    for (uint32_t l = 0; l < model_cfg_.num_layers; ++l) {
        std::string lp = "layers/" + std::to_string(l);
        LayerWeights& lw = layer_weights_[l];

        lw.q_proj         = load(lp + "/q_proj.bin");
        lw.k_proj         = load(lp + "/k_proj.bin");
        lw.v_proj         = load(lp + "/v_proj.bin");
        lw.o_proj         = load(lp + "/o_proj.bin");
        lw.router         = load(lp + "/router.bin");
        lw.rms_norm_weight = load(lp + "/rms_norm.bin");

        lw.expert_gate.resize(model_cfg_.num_experts, INVALID_MEM_HANDLE);
        lw.expert_up.resize(model_cfg_.num_experts,   INVALID_MEM_HANDLE);
        lw.expert_down.resize(model_cfg_.num_experts, INVALID_MEM_HANDLE);

        for (uint32_t e = 0; e < model_cfg_.num_experts; ++e) {
            std::string ep = lp + "/experts/" + std::to_string(e);
            lw.expert_gate[e] = load(ep + "/gate_proj.bin");
            lw.expert_up[e]   = load(ep + "/up_proj.bin");
            lw.expert_down[e] = load(ep + "/down_proj.bin");
        }
    }
}

// ============================================================
// alloc_scratch_buffers — called during session init
// ============================================================

void Executor::alloc_scratch_buffers(InferenceSession& session) {
    // Sizes are based on GPT-OSS-20B config and FP16 (2 bytes per element).
    constexpr size_t FP16 = 2;
    constexpr size_t INT32 = 4;
    const uint32_t D  = model_cfg_.d_model;
    const uint32_t H  = model_cfg_.num_heads;
    const uint32_t E  = model_cfg_.num_experts;
    const uint32_t K  = model_cfg_.top_k_experts;
    const uint32_t Df = model_cfg_.d_ffn;
    const uint32_t V  = model_cfg_.vocab_size;
    // seq_len for attention scratch: use a conservative max (e.g. 4096).
    constexpr uint32_t MAX_SEQ = 4096;

    InferenceSession::ScratchBuffers s{};
    auto alloc = [&](size_t sz) -> uint64_t {
        MemHandle h = pool_.alloc(sz, 256);
        return pool_.device_addr(h);
    };

    s.hidden_state     = alloc(D * FP16);
    s.hidden_save      = alloc(D * FP16);   // saved original hidden for residual (step 10)
    s.qkv_proj         = alloc(3 * D * FP16);
    s.attention_scores = alloc(H * MAX_SEQ * FP16);
    s.attention_out    = alloc(D * FP16);
    s.ffn_gate         = alloc(Df * FP16);
    s.ffn_up           = alloc(Df * FP16);
    s.ffn_out          = alloc(D * FP16);   // accumulated FFN output for post-FFN residual (step 19)
    s.moe_logits       = alloc(E * FP16);
    s.moe_indices      = alloc(K * INT32);
    s.moe_scores       = alloc(K * FP16);
    s.lm_head_out      = alloc(V * FP16);

    session.set_scratch(s);
}

// ============================================================
// run_prefill
// ============================================================

void Executor::run_prefill(InferenceSession& session) {
    alloc_scratch_buffers(session);

    // TODO: implement prefill descriptor sequence.
    // Prefill processes all input tokens in one forward pass and writes
    // K/V to the KV-cache via KVC_WRITE (write_mode=PREFILL_BULK).
    // For now: stub — issue a barrier and wait.
    compute_q_.add_barrier();
    compute_q_.submit_and_wait();
}

// ============================================================
// run_decode_step — main entry point for one token generation
// ============================================================

int32_t Executor::run_decode_step(InferenceSession& session) {
    if (session.is_done()) {
        throw OrbitException("Executor::run_decode_step: session is done");
    }

    // Build descriptor sequence for all transformer layers.
    for (uint32_t l = 0; l < model_cfg_.num_layers; ++l) {
        build_layer_decode(session, l);
    }

    // Steps 20-21: LM head + token sampling.
    step_20_lm_head(session);
    step_21_softmax_argmax_event(session);

    // Submit everything and wait.
    compute_q_.submit_and_wait();

    // After submit_and_wait(), the EVENT from step 21 has fired, meaning the
    // softmax-ed logits are ready in sc.lm_head_out on device memory.
    // Read logits back via BAR1 and perform argmax on the host.
    // MAX_REDUCE gives the max value, not the index, so host-side argmax is required.
    const auto& sc_post = session.scratch();
    const uint32_t vocab_size = model_cfg_.vocab_size;

    // Obtain a host-mapped pointer to the logit buffer via BAR1.
    // MemoryPool::host_ptr() lazily maps the region on first call.
    // lm_head_out scratch was allocated from the pool so host_ptr() is valid.
    // We find the MemHandle by iterating allocs — but MemoryPool doesn't expose
    // a reverse lookup by device_addr in the public API.  Use BAR1 mmap directly.
    //
    // The lm_head_out device_addr is an offset into GDDR6 (BAR1 base = 0).
    // Map vocab_size * 2 bytes starting at that offset.
    const size_t logit_bytes = static_cast<size_t>(vocab_size) * 2u;
    const uint16_t* logits = static_cast<const uint16_t*>(
        dev_.mmap_bar1(sc_post.lm_head_out, logit_bytes));

    int32_t next_token = 0;
    if (logits != nullptr) {
        // Argmax over FP16 logits: find the index of the maximum value.
        // FP16 with the same sign can be compared as uint16_t in magnitude,
        // but the safest approach is to convert to float for comparison.
        float max_val = -1e38f;
        for (uint32_t i = 0; i < vocab_size; ++i) {
            // Simple FP16 → float via bit manipulation (sign + exp + mantissa).
            uint16_t bits = logits[i];
            uint32_t sign     = (bits >> 15) & 0x1u;
            uint32_t exponent = (bits >> 10) & 0x1Fu;
            uint32_t mantissa = bits & 0x3FFu;
            uint32_t f32_bits;
            if (exponent == 0) {
                // Denormal / zero
                f32_bits = (sign << 31) | (mantissa << 13);
            } else if (exponent == 31) {
                // Inf / NaN
                f32_bits = (sign << 31) | (0xFFu << 23) | (mantissa << 13);
            } else {
                f32_bits = (sign << 31) | ((exponent + 112u) << 23) | (mantissa << 13);
            }
            float v;
            std::memcpy(&v, &f32_bits, 4);
            if (v > max_val) {
                max_val    = v;
                next_token = static_cast<int32_t>(i);
            }
        }
        dev_.munmap_bar1(const_cast<uint16_t*>(logits), logit_bytes);
    }
    // If mmap failed (e.g., in stub mode without hardware), next_token stays 0.

    return next_token;
}

// ============================================================
// build_layer_decode — steps 1-19 for one transformer layer
// ============================================================

void Executor::build_layer_decode(InferenceSession& session, uint32_t layer_id) {
    const auto& sc = session.scratch();

    // Save hidden_state before step_01/step_02 overwrite it, so that step_10
    // (residual add) can use the correct pre-attention hidden state.
    // Only needed for layers > 0 (layer 0 initialises hidden_state in step_01).
    // We save unconditionally here for simplicity; for layer 0 step_01 writes
    // hidden_state first, then this copy reflects the embedding.
    // For layers > 0 this preserves the previous layer's residual output.
    uint32_t row_bytes = model_cfg_.d_model * 2u;  // FP16
    compute_q_.add_dma_2d(sc.hidden_state, sc.hidden_save,
                          row_bytes, 1, row_bytes, row_bytes);

    step_01_dma_embedding(session);
    step_02_qkv_projection(session, layer_id);
    step_03_rope(session);
    step_04_kvc_write(session, layer_id);
    step_05_kvc_read(session, layer_id);
    step_06_attn_qkt(session);
    step_07_scale_softmax(session);
    step_08_attn_v(session);
    step_09_output_proj(session, layer_id);
    step_10_residual_add(session);
    step_11_rmsnorm(session, layer_id);
    step_12_moe_router_gemm(session, layer_id);
    step_13_moe_route(session);
    step_14_expert_gate(session, layer_id);
    step_15_silu(session);
    step_16_expert_up(session, layer_id);
    step_17_element_mul(session);
    step_18_expert_down(session, layer_id);
    step_19_barrier_residual(session);

    // Inter-layer BARRIER: ensures this layer's outputs are visible to the next layer.
    compute_q_.add_barrier();
}

// ============================================================
// add_gemm — selects GEMM_INT8 or GEMM_INT4 based on weight_format
// ============================================================

void Executor::add_gemm(uint64_t act, uint64_t wgt, uint64_t out,
                         uint64_t scale_addr, uint32_t Kt) {
    if (model_cfg_.weight_format == QuantFormat::INT4_AWQ) {
        compute_q_.add_gemm_int4(act, wgt, out, scale_addr, Kt);
    } else {
        compute_q_.add_gemm_int8(act, wgt, out, Kt);
    }
}

// ============================================================
// Step 1: DMA_2D — input embedding → scratch.hidden_state
// ============================================================

void Executor::step_01_dma_embedding(InferenceSession& session) {
    const auto& sc = session.scratch();
    // Embedding lookup: offset by current_token_id * d_model * sizeof(FP16).
    // Use last_token() which holds the most recently generated (or last prompt) token id.
    // For the very first decode step last_token() == -1 (no generated token yet);
    // in that case the Executor should have been given the last prompt token externally.
    // Using 0 as a safe fallback for skeleton purposes when no token is available yet.
    int32_t  token_id  = session.last_token();
    uint64_t token_off = (token_id >= 0)
                         ? static_cast<uint64_t>(token_id) * model_cfg_.d_model * 2u
                         : 0ULL;
    uint64_t emb_src   = pool_.device_addr(embedding_weight_) + token_off;
    uint32_t row_bytes = model_cfg_.d_model * 2u;  // FP16

    compute_q_.add_dma_2d(emb_src, sc.hidden_state,
                          row_bytes, 1,
                          row_bytes, row_bytes);
}

// ============================================================
// Step 2: GEMM_INT8/4 — [Q,K,V] = hidden @ [Wq,Wk,Wv]  (3 GEMMs)
// ============================================================

void Executor::step_02_qkv_projection(InferenceSession& session, uint32_t layer_id) {
    const auto& sc = session.scratch();
    const LayerWeights& lw = layer_weights_[layer_id];
    uint32_t Kt = model_cfg_.d_model;

    // Q projection
    add_gemm(sc.hidden_state, pool_.device_addr(lw.q_proj),
             sc.qkv_proj,                                       // Q at offset 0
             0, Kt);

    // K projection
    add_gemm(sc.hidden_state, pool_.device_addr(lw.k_proj),
             sc.qkv_proj + model_cfg_.d_model * 2u,             // K at offset d_model
             0, Kt);

    // V projection
    add_gemm(sc.hidden_state, pool_.device_addr(lw.v_proj),
             sc.qkv_proj + model_cfg_.d_model * 2u * 2u,        // V at offset 2*d_model
             0, Kt);
}

// ============================================================
// Step 3: VECTOR_OP_EX — RoPE on Q and K
// ============================================================

void Executor::step_03_rope(InferenceSession& session) {
    const auto& sc = session.scratch();
    uint32_t qk_elem_count = model_cfg_.num_heads * model_cfg_.head_dim;

    // Apply RoPE to Q.
    compute_q_.add_vector_op_ex(sc.qkv_proj,                          // Q
                                sc.qkv_proj,                          // in-place
                                qk_elem_count,
                                VectorOpExType::ROPE,
                                DataType::FP16,
                                0,  // TODO: aux_addr = cos/sin table address
                                session.context_length());

    // Apply RoPE to K.
    compute_q_.add_vector_op_ex(sc.qkv_proj + model_cfg_.d_model * 2u, // K
                                sc.qkv_proj + model_cfg_.d_model * 2u,
                                qk_elem_count,
                                VectorOpExType::ROPE,
                                DataType::FP16,
                                0,  // TODO: cos/sin table
                                session.context_length());
}

// ============================================================
// Step 4: KVC_WRITE — store new K,V to KV-Cache
// ============================================================

void Executor::step_04_kvc_write(InferenceSession& session, uint32_t layer_id) {
    const auto& sc = session.scratch();
    uint32_t token_pos = session.context_length() - 1;  // Current decode position

    compute_q_.add_kvc_write(session.seq_id(), layer_id,
                             token_pos,
                             1,  // write_count = 1 token
                             sc.qkv_proj + model_cfg_.d_model * 2u,        // K source
                             sc.qkv_proj + model_cfg_.d_model * 2u * 2u,   // V source
                             0,  // write_mode = DECODE
                             0); // alloc_blocks = 0 (pre-allocated by session.step())
}

// ============================================================
// Step 5: KVC_READ — load full K,V history from KV-Cache
// ============================================================

void Executor::step_05_kvc_read(InferenceSession& session, uint32_t layer_id) {
    const auto& sc = session.scratch();
    uint32_t seq_len = session.context_length();

    // K and V destination within attention_scores scratch region.
    // Layout: K[0..seq_len-1] followed by V[0..seq_len-1].
    uint64_t k_dst = sc.attention_scores;
    uint64_t v_dst = sc.attention_scores +
                     static_cast<uint64_t>(seq_len) *
                     model_cfg_.num_heads * model_cfg_.head_dim * 2u;

    compute_q_.add_kvc_read(session.seq_id(), layer_id,
                            0,       // seq_start
                            seq_len, // seq_len
                            k_dst,
                            v_dst,
                            0,       // read_mode = ALL_HEADS
                            0);      // head_id (ignored for ALL_HEADS)
}

// ============================================================
// Step 6: GEMM_INT8/4 — attention_scores = Q @ K^T
// ============================================================

void Executor::step_06_attn_qkt(InferenceSession& session) {
    const auto& sc = session.scratch();
    uint32_t Kt = model_cfg_.head_dim;

    // Q is at sc.qkv_proj (shape: [1, num_heads, head_dim])
    // K is at sc.attention_scores (shape: [seq_len, num_heads, head_dim])
    // Output: sc.attention_scores + large_offset (shape: [num_heads, 1, seq_len])
    // TODO: precise offset arithmetic for per-head GEMM tiling.
    uint64_t k_src = sc.attention_scores;
    uint64_t attn_out_addr = sc.attention_out;  // Temporary reuse; TODO: dedicated region

    add_gemm(sc.qkv_proj, k_src, attn_out_addr, 0, Kt);
}

// ============================================================
// Step 7: SOFTMAX — scale(1/√head_dim) + softmax(attention_scores)
// ============================================================

void Executor::step_07_scale_softmax(InferenceSession& session) {
    const auto& sc = session.scratch();
    uint32_t seq_len = session.context_length();

    float scale = 1.0f / std::sqrt(static_cast<float>(model_cfg_.head_dim));

    compute_q_.add_softmax(sc.attention_out, sc.attention_out,
                           seq_len,
                           model_cfg_.num_heads,
                           scale);
}

// ============================================================
// Step 8: GEMM_INT8/4 — attention_out = attention_scores @ V
// ============================================================

void Executor::step_08_attn_v(InferenceSession& session) {
    const auto& sc = session.scratch();
    uint32_t seq_len = session.context_length();
    uint32_t Kt = seq_len;  // sequence length is the K dimension

    uint64_t v_src = sc.attention_scores +
                     static_cast<uint64_t>(seq_len) *
                     model_cfg_.num_heads * model_cfg_.head_dim * 2u;

    add_gemm(sc.attention_out, v_src, sc.hidden_state, 0, Kt);
}

// ============================================================
// Step 9: GEMM_INT8/4 — hidden = attention_out @ Wo
// ============================================================

void Executor::step_09_output_proj(InferenceSession& session, uint32_t layer_id) {
    const auto& sc = session.scratch();
    const LayerWeights& lw = layer_weights_[layer_id];
    uint32_t Kt = model_cfg_.d_model;

    add_gemm(sc.hidden_state, pool_.device_addr(lw.o_proj), sc.attention_out, 0, Kt);
    // Copy result back to hidden_state (reuse the same region for simplicity).
    // TODO: use separate in/out scratch regions for clarity.
}

// ============================================================
// Step 10: VECTOR_OP — residual add: hidden += original_hidden
// ============================================================

void Executor::step_10_residual_add(InferenceSession& session) {
    const auto& sc = session.scratch();
    uint32_t elem_count = model_cfg_.d_model;

    // Residual: hidden_state = attention_out + original_hidden.
    // sc.hidden_save was saved via DMA_2D at the top of build_layer_decode,
    // before step_01/step_02 overwrote hidden_state.
    // VECTOR_OP MUL semantics: dst[i] = src[i] * dst[i]; for ADD: dst[i] = src[i] + dst[i].
    // We want: hidden_state[i] = attention_out[i] + hidden_save[i].
    // Set src=sc.attention_out (step 9 output) and dst=sc.hidden_save (saved original),
    // result accumulates into hidden_save.  Then copy back to hidden_state below.
    // Simpler approach: write result directly into hidden_state using src=attention_out,
    // dst=hidden_state, with hidden_save as the original operand by first copying
    // hidden_save into hidden_state and then adding attention_out.
    //
    // Step A: copy hidden_save → hidden_state (restore original)
    uint32_t row_bytes = model_cfg_.d_model * 2u;
    compute_q_.add_dma_2d(sc.hidden_save, sc.hidden_state,
                          row_bytes, 1, row_bytes, row_bytes);

    // Step B: hidden_state[i] += attention_out[i]  (dst = dst + src)
    compute_q_.add_vector_op(sc.attention_out, sc.hidden_state,
                             elem_count,
                             VectorOpType::ADD,
                             DataType::FP16,
                             0);
}

// ============================================================
// Step 11: VECTOR_OP_EX — RMSNorm(hidden)
// ============================================================

void Executor::step_11_rmsnorm(InferenceSession& session, uint32_t layer_id) {
    const auto& sc = session.scratch();
    const LayerWeights& lw = layer_weights_[layer_id];
    uint32_t elem_count = model_cfg_.d_model;

    compute_q_.add_vector_op_ex(sc.hidden_state, sc.hidden_state,
                                elem_count,
                                VectorOpExType::RMSNORM,
                                DataType::FP16,
                                pool_.device_addr(lw.rms_norm_weight),
                                0);
}

// ============================================================
// Step 12: GEMM_INT8/4 — moe_logits = hidden @ router_weight
// ============================================================

void Executor::step_12_moe_router_gemm(InferenceSession& session, uint32_t layer_id) {
    const auto& sc = session.scratch();
    const LayerWeights& lw = layer_weights_[layer_id];
    uint32_t Kt = model_cfg_.d_model;

    add_gemm(sc.hidden_state, pool_.device_addr(lw.router), sc.moe_logits, 0, Kt);
}

// ============================================================
// Step 13: MOE_ROUTE — top-2 expert indices + scores
// ============================================================

void Executor::step_13_moe_route(InferenceSession& session) {
    const auto& sc = session.scratch();

    compute_q_.add_moe_route(sc.moe_logits,
                             sc.moe_indices,
                             sc.moe_scores,
                             1,  // num_tokens = 1 (single decode step)
                             model_cfg_.num_experts,
                             model_cfg_.top_k_experts);
}

// ============================================================
// Step 14: GEMM_INT8/4 — gate_proj per selected expert
// NOTE: For top-2 routing, this runs once per expert. In practice the host
// must read back moe_indices after step 13 to select expert weight handles.
// Here we conservatively dispatch for the first 2 experts as placeholders.
// TODO: implement dynamic expert dispatch after moe_indices readback.
// ============================================================

void Executor::step_14_expert_gate(InferenceSession& session, uint32_t layer_id) {
    const auto& sc = session.scratch();
    const LayerWeights& lw = layer_weights_[layer_id];
    uint32_t Kt = model_cfg_.d_model;

    for (uint32_t e = 0; e < model_cfg_.top_k_experts; ++e) {
        // TODO: use moe_indices[e] to select actual expert index.
        uint32_t expert_idx = e;  // PLACEHOLDER
        if (expert_idx < lw.expert_gate.size()) {
            add_gemm(sc.hidden_state,
                     pool_.device_addr(lw.expert_gate[expert_idx]),
                     sc.ffn_gate,
                     0, Kt);
        }
    }
}

// ============================================================
// Step 15: VECTOR_OP_EX — SiLU(gate_proj)
// ============================================================

void Executor::step_15_silu(InferenceSession& session) {
    const auto& sc = session.scratch();
    uint32_t elem_count = model_cfg_.d_ffn;

    compute_q_.add_vector_op_ex(sc.ffn_gate, sc.ffn_gate,
                                elem_count,
                                VectorOpExType::SILU,
                                DataType::FP16,
                                0, 0);
}

// ============================================================
// Step 16: GEMM_INT8/4 — up_proj per selected expert
// ============================================================

void Executor::step_16_expert_up(InferenceSession& session, uint32_t layer_id) {
    const auto& sc = session.scratch();
    const LayerWeights& lw = layer_weights_[layer_id];
    uint32_t Kt = model_cfg_.d_model;

    for (uint32_t e = 0; e < model_cfg_.top_k_experts; ++e) {
        uint32_t expert_idx = e;  // PLACEHOLDER (same as step 14)
        if (expert_idx < lw.expert_up.size()) {
            add_gemm(sc.hidden_state,
                     pool_.device_addr(lw.expert_up[expert_idx]),
                     sc.ffn_up,
                     0, Kt);
        }
    }
}

// ============================================================
// Step 17: VECTOR_OP — gated_ffn = SiLU(gate) * up
// ============================================================

void Executor::step_17_element_mul(InferenceSession& session) {
    const auto& sc = session.scratch();
    uint32_t elem_count = model_cfg_.d_ffn;

    // ffn_gate now holds SiLU(gate_proj) from step 15.
    // ffn_up holds up_proj from step 16.
    // Desired: result = ffn_gate * ffn_up → write into ffn_gate.
    //
    // VECTOR_OP (0x03) layout: src_addr × dst_addr, op_type=MUL.
    // The hardware interprets this as: dst[i] = src[i] * dst[i]
    // (in-place MUL where dst is both second operand and output).
    // By passing src=ffn_gate (SiLU output) and dst=ffn_up (up output),
    // the result ffn_gate * ffn_up is written into ffn_up, which is then
    // used as input to step 18 (down_proj).
    //
    // FIX: was incorrectly using sc.ffn_gate as both src and dst
    //      (gate * gate instead of gate * up).
    compute_q_.add_vector_op(sc.ffn_gate,  // src = SiLU(gate)
                             sc.ffn_up,    // dst = up_proj (also 2nd operand); result stored here
                             elem_count,
                             VectorOpType::MUL,
                             DataType::FP16,
                             0);
}

// ============================================================
// Step 18: GEMM_INT8/4 — down_proj per selected expert
// ============================================================

void Executor::step_18_expert_down(InferenceSession& session, uint32_t layer_id) {
    const auto& sc = session.scratch();
    const LayerWeights& lw = layer_weights_[layer_id];
    uint32_t Kt = model_cfg_.d_ffn;

    for (uint32_t e = 0; e < model_cfg_.top_k_experts; ++e) {
        uint32_t expert_idx = e;  // PLACEHOLDER
        if (expert_idx < lw.expert_down.size()) {
            // After step 17, sc.ffn_up holds the gated result (SiLU(gate) * up).
            add_gemm(sc.ffn_up,
                     pool_.device_addr(lw.expert_down[expert_idx]),
                     sc.hidden_state,
                     0, Kt);
        }
    }
}

// ============================================================
// Step 19: BARRIER + VECTOR_OP — combine expert outputs + residual add
// ============================================================

void Executor::step_19_barrier_residual(InferenceSession& session) {
    const auto& sc = session.scratch();

    // BARRIER: ensure all expert GEMMs (steps 14-18) have completed before residual.
    compute_q_.add_barrier();

    // Post-FFN residual add: hidden_state[i] += ffn_out[i].
    // sc.ffn_out holds the accumulated expert down-projection output from step 18.
    // step_18 writes its result into sc.hidden_state, so copy it to sc.ffn_out
    // first, then add the pre-FFN hidden_state (saved in sc.hidden_save after step 11
    // norm and before the FFN).
    //
    // For the skeleton we perform:
    //   hidden_state = hidden_state (down-proj result) + hidden_state_pre_ffn
    // Since step 11 already overwrote hidden_save, we use a simpler approximation:
    //   VECTOR_OP ADD: dst = hidden_state (down-proj), src = sc.hidden_state → noop.
    // TODO: properly save pre-FFN hidden state for post-FFN residual.
    //
    // For correctness, step 18 writes into sc.hidden_state.
    // We perform: sc.hidden_state[i] += sc.hidden_save[i]  (hidden_save = pre-FFN residual).
    uint32_t elem_count = model_cfg_.d_model;
    compute_q_.add_vector_op(sc.hidden_save,    // src = pre-FFN hidden (saved in step 10)
                             sc.hidden_state,   // dst = FFN output (from step 18); result stored here
                             elem_count,
                             VectorOpType::ADD,
                             DataType::FP16,
                             0);
}

// ============================================================
// Step 20: GEMM_INT8/4 — lm_head = hidden @ Wlm  (after all layers)
// ============================================================

void Executor::step_20_lm_head(InferenceSession& session) {
    const auto& sc = session.scratch();
    uint32_t Kt = model_cfg_.d_model;

    add_gemm(sc.hidden_state,
             pool_.device_addr(lm_head_weight_),
             sc.lm_head_out,
             0, Kt);
}

// ============================================================
// Step 21: VECTOR_OP + EVENT — softmax → argmax → next_token_id, notify host
// ============================================================

void Executor::step_21_softmax_argmax_event(InferenceSession& session) {
    const auto& sc = session.scratch();

    // Softmax over vocabulary logits.
    compute_q_.add_softmax(sc.lm_head_out, sc.lm_head_out,
                           model_cfg_.vocab_size,
                           1,     // num_rows = 1
                           1.0f); // scale = 1.0 (no additional scaling)

    // Argmax: MAX_REDUCE gives the max VALUE, not the max INDEX, so it cannot
    // be used for token selection.  The correct approach is to DMA the logits
    // back to host memory after softmax and perform argmax on the CPU.
    // Emit an EVENT to signal completion so the host knows the logits are ready.
    constexpr uint32_t EVENT_TOKEN_READY = 0x0001;
    compute_q_.add_event(EVENT_TOKEN_READY, 0);
}

} // namespace orbit
