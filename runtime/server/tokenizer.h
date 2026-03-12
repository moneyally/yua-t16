#pragma once
// orbit_server — Tokenizer interface (BPE stub)
// SSOT: spec/driver-runtime-design.md §4.6
//
// This is a skeleton/stub.  A real implementation would load a
// HuggingFace tokenizer.json and run BPE encode/decode.

#include <cstdint>
#include <string>
#include <vector>

namespace orbit {

// Minimal BPE tokenizer interface.
// encode() converts UTF-8 text to token-id sequences.
// decode() converts a single token id back to its UTF-8 piece.
//
// The stub implementation uses a trivial byte-level mapping so that
// the server compiles and runs without a real vocabulary file.
class Tokenizer {
public:
    // Load vocabulary from a HuggingFace tokenizer.json.
    // Returns true on success; on failure the stub fallback is used.
    bool load(const std::string& tokenizer_json_path);

    // Apply a chat template and encode to token ids.
    // messages: list of {"role": ..., "content": ...} objects as raw JSON strings
    // The result is a flat token-id sequence ready for prefill.
    std::vector<int32_t> encode_chat(const std::vector<std::pair<std::string,std::string>>& messages) const;

    // Encode a plain text string to token ids.
    std::vector<int32_t> encode(const std::string& text) const;

    // Decode a single token id to its UTF-8 text piece.
    // May return empty string for special tokens.
    std::string decode_token(int32_t token_id) const;

    // Decode a sequence of token ids to a UTF-8 string.
    std::string decode(const std::vector<int32_t>& token_ids) const;

    // Vocabulary size (stub: 32000)
    int32_t vocab_size() const { return vocab_size_; }

    // EOS token id (stub: 2)
    int32_t eos_token_id() const { return eos_token_id_; }

    bool is_loaded() const { return loaded_; }

private:
    bool    loaded_     = false;
    int32_t vocab_size_ = 32000;
    int32_t eos_token_id_ = 2;

    // Vocabulary table: token_id → UTF-8 piece
    std::vector<std::string> id_to_piece_;

    // Stub chat template (Llama-3 style)
    std::string apply_chat_template(const std::vector<std::pair<std::string,std::string>>& messages) const;
};

} // namespace orbit
