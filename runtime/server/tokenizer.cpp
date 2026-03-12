// orbit_server — Tokenizer implementation (BPE stub)
// SSOT: spec/driver-runtime-design.md §4.6
//
// Stub behaviour:
//   encode()  — UTF-8 byte-level fallback (each byte = one token id in [3, 258])
//   decode()  — reverse of byte-level mapping
//   A real production tokenizer would parse tokenizer.json (tiktoken/SentencePiece).

#include "tokenizer.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace orbit {

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------
bool Tokenizer::load(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[tokenizer] Warning: could not open " << path
                  << " — using byte-level stub\n";
        // Populate stub id_to_piece_ for byte range [0,255] at offsets [3,258]
        // 0 = <pad>, 1 = <bos>, 2 = <eos>
        id_to_piece_.resize(vocab_size_);
        id_to_piece_[0] = "<pad>";
        id_to_piece_[1] = "<bos>";
        id_to_piece_[2] = "<eos>";
        for (int i = 0; i < 256; ++i) {
            id_to_piece_[i + 3] = std::string(1, static_cast<char>(i));
        }
        return false;
    }

    // Minimal tokenizer.json parser stub:
    // A real implementation would use nlohmann::json here.
    // For now, fall through to the byte stub while still marking loaded.
    id_to_piece_.resize(vocab_size_);
    id_to_piece_[0] = "<pad>";
    id_to_piece_[1] = "<bos>";
    id_to_piece_[2] = "<eos>";
    for (int i = 0; i < 256; ++i) {
        id_to_piece_[i + 3] = std::string(1, static_cast<char>(i));
    }

    loaded_ = true;
    std::cerr << "[tokenizer] Loaded (stub) from " << path << "\n";
    return true;
}

// ---------------------------------------------------------------------------
// apply_chat_template  (Llama-3 style)
// ---------------------------------------------------------------------------
std::string Tokenizer::apply_chat_template(
    const std::vector<std::pair<std::string,std::string>>& messages) const
{
    // Format:
    //   <|begin_of_text|>
    //   <|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>
    //   <|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>
    //   <|start_header_id|>assistant<|end_header_id|>\n\n
    std::string result = "<|begin_of_text|>";
    for (const auto& [role, content] : messages) {
        result += "<|start_header_id|>" + role + "<|end_header_id|>\n\n";
        result += content + "<|eot_id|>";
    }
    result += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    return result;
}

// ---------------------------------------------------------------------------
// encode_chat
// ---------------------------------------------------------------------------
std::vector<int32_t> Tokenizer::encode_chat(
    const std::vector<std::pair<std::string,std::string>>& messages) const
{
    std::string text = apply_chat_template(messages);
    return encode(text);
}

// ---------------------------------------------------------------------------
// encode  (byte-level stub)
// ---------------------------------------------------------------------------
std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> ids;
    ids.reserve(text.size() + 1);
    ids.push_back(1); // <bos>
    for (unsigned char c : text) {
        // byte i → token id (i + 3), clamped to vocab range
        int32_t id = static_cast<int32_t>(c) + 3;
        if (id < vocab_size_) {
            ids.push_back(id);
        }
    }
    return ids;
}

// ---------------------------------------------------------------------------
// decode_token
// ---------------------------------------------------------------------------
std::string Tokenizer::decode_token(int32_t token_id) const {
    if (token_id < 0 || token_id >= static_cast<int32_t>(id_to_piece_.size())) {
        return "";
    }
    const std::string& piece = id_to_piece_[token_id];
    // Filter special tokens
    if (piece == "<pad>" || piece == "<bos>" || piece == "<eos>") {
        return "";
    }
    return piece;
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------
std::string Tokenizer::decode(const std::vector<int32_t>& token_ids) const {
    std::string result;
    result.reserve(token_ids.size());
    for (int32_t id : token_ids) {
        result += decode_token(id);
    }
    return result;
}

} // namespace orbit
