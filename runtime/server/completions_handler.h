#pragma once
// orbit_server — POST /v1/chat/completions handler
// SSOT: spec/driver-runtime-design.md §4.3

// Forward declarations to avoid pulling httplib.h into every TU that
// includes this header.
namespace httplib { struct Request; struct Response; }

#include "config.h"
#include "session_manager.h"
#include "tokenizer.h"

#include <memory>
#include <string>

namespace orbit {

// ---------------------------------------------------------------------------
// CompletionsHandler
// ---------------------------------------------------------------------------
// Owns the HTTP-facing logic for:
//   POST /v1/chat/completions  — full response and SSE streaming
//   GET  /v1/models            — list available model
//   GET  /health               — liveness probe
//
// The handler itself does NOT drive the inference loop directly.
// It serialises the request into the SessionManager queue and then
// either waits for a result (non-streaming) or streams SSE chunks
// back over the chunked HTTP response (streaming).
//
// For the skeleton the "inference loop" is stubbed: the worker thread
// (started in main.cpp) calls run_worker() which pops from the queue
// and invokes execute_request().
// ---------------------------------------------------------------------------
class CompletionsHandler {
public:
    explicit CompletionsHandler(const ServerConfig& cfg,
                                 SessionManager& session_mgr,
                                 Tokenizer& tokenizer);

    // Register routes on an httplib::Server.
    // Called once from main() after the Server object is created.
    // We template on ServerT to avoid a hard httplib.h dependency here;
    // the .cpp includes httplib.h directly.
    template<typename ServerT>
    void register_routes(ServerT& srv);

    // Worker-thread entry point.  Loops until SessionManager shuts down.
    // In production this would drive the Executor / InferenceSession from liborbit.
    // In this skeleton it emits a short stub stream.
    void run_worker();

private:
    const ServerConfig& cfg_;
    SessionManager&     session_mgr_;
    Tokenizer&          tokenizer_;

    // Build an OpenAI-format error JSON body.
    static std::string make_error(const std::string& message,
                                   const std::string& type    = "server_error",
                                   const std::string& code    = "internal_error");

    // Process one request end-to-end (called from worker thread).
    void execute_request(RequestHandle req);

    // Format one SSE chunk matching OpenAI wire format exactly.
    // data: {"id":"chatcmpl-<id>","object":"chat.completion.chunk",
    //        "model":"<model>","choices":[{"index":0,"delta":{"content":"<tok>"},
    //        "finish_reason":null}]}
    static std::string make_sse_chunk(const std::string& completion_id,
                                       const std::string& model,
                                       const std::string& content_piece,
                                       bool last = false);

    // Format the final [DONE] SSE line.
    static std::string make_sse_done();

    // Build a full (non-streaming) chat completion JSON response.
    static std::string make_full_response(const std::string& completion_id,
                                           const std::string& model,
                                           const std::string& full_text,
                                           int32_t prompt_tokens,
                                           int32_t completion_tokens);

    // Generate a unique completion id like "chatcmpl-abcdef123456"
    static std::string new_completion_id();
};

} // namespace orbit
