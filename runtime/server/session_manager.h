#pragma once
// orbit_server — SessionManager: concurrent request queue + session lifecycle
// SSOT: spec/driver-runtime-design.md §4.5

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <vector>

namespace orbit {

// ---------------------------------------------------------------------------
// InferenceRequest — one pending chat-completion job
// ---------------------------------------------------------------------------
struct InferenceRequest {
    uint64_t    request_id;         // monotonic, assigned at enqueue time
    std::string request_json;       // raw serialized ChatCompletionRequest JSON
    bool        stream;             // SSE streaming requested
    std::atomic<bool> cancelled{false}; // set by HTTP layer on client disconnect

    // Copy/move: atomic<bool> is not copyable, so we delete copy.
    InferenceRequest() = default;
    InferenceRequest(const InferenceRequest&) = delete;
    InferenceRequest& operator=(const InferenceRequest&) = delete;
    InferenceRequest(InferenceRequest&& o) noexcept
        : request_id(o.request_id),
          request_json(std::move(o.request_json)),
          stream(o.stream)
    {
        cancelled.store(o.cancelled.load(std::memory_order_relaxed),
                        std::memory_order_relaxed);
    }
};

// Shared ownership handle passed between HTTP layer and worker thread.
using RequestHandle = std::shared_ptr<InferenceRequest>;

// ---------------------------------------------------------------------------
// SessionManager
// ---------------------------------------------------------------------------
// Thread-safe FIFO queue between HTTP handler threads and the single
// worker thread that drives liborbit (Executor / InferenceSession).
//
// Design constraints from the spec:
//   - ORBIT-G1 has one compute queue → 1 worker thread, serial execution.
//   - Queue capacity defaults to 64; push() returns false when full.
//   - Worker calls pop() which blocks until a request arrives or shutdown.
// ---------------------------------------------------------------------------
class SessionManager {
public:
    explicit SessionManager(uint32_t capacity = 64);
    ~SessionManager();

    // Enqueue a request.  Returns request_id on success, 0 if queue is full
    // and block=false, or if the manager has been shut down.
    // If block=true, waits until a slot becomes available.
    uint64_t enqueue(std::string request_json,
                     bool stream,
                     bool block = false);

    // Cancel a previously enqueued request.  Safe to call from any thread.
    // Returns true if the request was found and marked cancelled.
    bool cancel(uint64_t request_id);

    // Worker thread: block until a request is available.
    // Returns nullptr when shutdown has been requested.
    RequestHandle pop();

    // Signal the worker to stop after draining.
    void shutdown();

    bool is_shutdown() const { return shutdown_.load(); }

    // Approximate queue depth (for metrics / 503 decisions).
    size_t depth() const;

private:
    const uint32_t capacity_;
    std::atomic<uint64_t> next_id_{1};
    std::atomic<bool>     shutdown_{false};

    mutable std::mutex              mu_;
    std::condition_variable         cv_push_;  // notified when slot frees
    std::condition_variable         cv_pop_;   // notified when request added
    std::queue<RequestHandle>       queue_;
};

} // namespace orbit
