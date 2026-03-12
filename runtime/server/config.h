#pragma once
// orbit_server — Server configuration struct
// SSOT: spec/driver-runtime-design.md §4.7

#include <cstdint>
#include <string>

namespace orbit {

struct ServerConfig {
    // Path to model weights directory (contains tokenizer.json, config.json, *.bin or *.safetensors)
    std::string model_dir;

    // /dev/orbit_g1_N character device opened by liborbit
    std::string device_node = "/dev/orbit_g1_0";

    // HTTP listen port
    uint16_t port = 8080;

    // Bind address (0.0.0.0 = all interfaces)
    std::string host = "0.0.0.0";

    // Maximum input + output tokens per request
    uint32_t max_batch_tokens = 4096;

    // Maximum KV-cache pages — caps concurrent in-flight sessions
    // GPT-OSS-20B: 8 MB/page, so 128 pages = 1 GB KV budget
    uint32_t max_kv_pages = 128;

    // Request queue capacity before returning HTTP 503
    uint32_t queue_capacity = 64;

    // Use GEMM_INT4 descriptors if the hardware supports them
    bool enable_int4 = true;

    // Log verbosity: "debug", "info", "warn", "error"
    std::string log_level = "info";

    // Worker thread count.
    // ORBIT-G1 has a single compute queue, so default is 1.
    // Future multi-queue builds may raise this.
    uint32_t worker_threads = 1;

    // Timeout (ms) waiting for a decode step to complete
    uint32_t decode_timeout_ms = 30000;

    // EOS token id (model-specific, loaded from config.json at runtime)
    int32_t eos_token_id = 2;   // LLaMA default
};

} // namespace orbit
