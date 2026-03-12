// orbit_server — POST /v1/chat/completions handler implementation
// SSOT: spec/driver-runtime-design.md §4.3, §4.4
//
// JSON library: nlohmann/json (header-only, include directly)
// HTTP library: cpp-httplib (header-only, include directly)
//
// SSE wire format (must match OpenAI spec exactly):
//   data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","model":"...","choices":[{"index":0,"delta":{"content":"token"},"finish_reason":null}]}
//   <blank line>
//   data: [DONE]
//   <blank line>

#define CPPHTTPLIB_OPENSSL_SUPPORT 0
#include "httplib.h"
#include "json.hpp"

#include "completions_handler.h"
#include "session_manager.h"
#include "tokenizer.h"

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>

using json = nlohmann::json;

namespace orbit {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
CompletionsHandler::CompletionsHandler(const ServerConfig& cfg,
                                        SessionManager& session_mgr,
                                        Tokenizer& tokenizer)
    : cfg_(cfg), session_mgr_(session_mgr), tokenizer_(tokenizer) {}

// ---------------------------------------------------------------------------
// new_completion_id
// ---------------------------------------------------------------------------
std::string CompletionsHandler::new_completion_id() {
    // Thread-safe 12-char hex suffix
    static std::atomic<uint64_t> counter{0};
    uint64_t id = counter.fetch_add(1, std::memory_order_relaxed);
    // Mix with a random seed to avoid predictable ids across restarts
    static const uint64_t seed = []() -> uint64_t {
        std::random_device rd;
        return (static_cast<uint64_t>(rd()) << 32) | rd();
    }();
    std::ostringstream ss;
    ss << "chatcmpl-"
       << std::hex << std::setfill('0') << std::setw(12) << (seed ^ id);
    return ss.str();
}

// ---------------------------------------------------------------------------
// make_error — OpenAI error format
// {"error":{"message":"...","type":"...","code":"..."}}
// ---------------------------------------------------------------------------
std::string CompletionsHandler::make_error(const std::string& message,
                                             const std::string& type,
                                             const std::string& code)
{
    json body = {
        {"error", {
            {"message", message},
            {"type",    type},
            {"code",    code}
        }}
    };
    return body.dump();
}

// ---------------------------------------------------------------------------
// make_sse_chunk — one SSE data line
// SSE format: "data: <json>\n\n"  (two newlines = end of event)
// ---------------------------------------------------------------------------
std::string CompletionsHandler::make_sse_chunk(const std::string& completion_id,
                                                const std::string& model,
                                                const std::string& content_piece,
                                                bool last)
{
    json delta;
    if (!last) {
        delta = {{"content", content_piece}};
    } else {
        delta = json::object();  // empty delta on last chunk
    }

    json chunk = {
        {"id",      completion_id},
        {"object",  "chat.completion.chunk"},
        {"model",   model},
        {"choices", json::array({
            {
                {"index",         0},
                {"delta",         delta},
                {"finish_reason", last ? json("stop") : json(nullptr)}
            }
        })}
    };

    return "data: " + chunk.dump() + "\n\n";
}

// ---------------------------------------------------------------------------
// make_sse_done — final [DONE] sentinel
// ---------------------------------------------------------------------------
std::string CompletionsHandler::make_sse_done() {
    return "data: [DONE]\n\n";
}

// ---------------------------------------------------------------------------
// make_full_response — non-streaming JSON body
// ---------------------------------------------------------------------------
std::string CompletionsHandler::make_full_response(const std::string& completion_id,
                                                     const std::string& model,
                                                     const std::string& full_text,
                                                     int32_t prompt_tokens,
                                                     int32_t completion_tokens)
{
    json resp = {
        {"id",      completion_id},
        {"object",  "chat.completion"},
        {"model",   model},
        {"choices", json::array({
            {
                {"index",         0},
                {"message",       {
                    {"role",    "assistant"},
                    {"content", full_text}
                }},
                {"finish_reason", "stop"}
            }
        })},
        {"usage", {
            {"prompt_tokens",     prompt_tokens},
            {"completion_tokens", completion_tokens},
            {"total_tokens",      prompt_tokens + completion_tokens}
        }}
    };
    return resp.dump();
}

// ---------------------------------------------------------------------------
// register_routes
// ---------------------------------------------------------------------------
template<typename ServerT>
void CompletionsHandler::register_routes(ServerT& srv) {

    // ── GET /health ─────────────────────────────────────────────────────────
    srv.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        json body = {{"status", "ok"}};
        res.set_content(body.dump(), "application/json");
    });

    // ── GET /v1/models ──────────────────────────────────────────────────────
    srv.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
        json body = {
            {"object", "list"},
            {"data", json::array({
                {
                    {"id",       "orbit-g1"},
                    {"object",   "model"},
                    {"owned_by", "orbit"}
                }
            })}
        };
        res.set_content(body.dump(), "application/json");
    });

    // ── POST /v1/chat/completions ────────────────────────────────────────────
    srv.Post("/v1/chat/completions",
        [this](const httplib::Request& req, httplib::Response& res)
    {
        // --- 1. Parse JSON body ---
        json body;
        try {
            body = json::parse(req.body);
        } catch (const json::exception& e) {
            res.status = 400;
            res.set_content(
                make_error(std::string("JSON parse error: ") + e.what(),
                            "invalid_request_error", "parse_error"),
                "application/json");
            return;
        }

        // --- 2. Validate required fields ---
        if (!body.contains("messages") || !body["messages"].is_array()) {
            res.status = 400;
            res.set_content(
                make_error("'messages' field is required and must be an array",
                            "invalid_request_error", "missing_required_field"),
                "application/json");
            return;
        }

        // --- 3. Extract parameters (OpenAI field names) ---
        std::string model       = body.value("model",       "orbit-g1");
        float       temperature = body.value("temperature", 1.0f);
        float       top_p       = body.value("top_p",       1.0f);
        int32_t     max_tokens  = body.value("max_tokens",  512);
        bool        stream      = body.value("stream",      false);

        // Clamp max_tokens
        if (max_tokens <= 0 || static_cast<uint32_t>(max_tokens) > cfg_.max_batch_tokens) {
            max_tokens = static_cast<int32_t>(cfg_.max_batch_tokens);
        }

        // --- 4. Build messages vector for tokenizer ---
        std::vector<std::pair<std::string,std::string>> messages;
        for (const auto& msg : body["messages"]) {
            std::string role    = msg.value("role",    "user");
            std::string content = msg.value("content", "");
            messages.emplace_back(role, content);
        }

        // --- 5. Enqueue request ---
        // We re-serialise with normalised fields so the worker has clean data.
        json norm_req = {
            {"model",       model},
            {"messages",    body["messages"]},
            {"temperature", temperature},
            {"top_p",       top_p},
            {"max_tokens",  max_tokens},
            {"stream",      stream}
        };

        if (session_mgr_.depth() >= cfg_.queue_capacity) {
            res.status = 503;
            res.set_content(
                make_error("Request queue is full. Try again later.",
                            "server_error", "queue_full"),
                "application/json");
            return;
        }

        uint64_t req_id = session_mgr_.enqueue(norm_req.dump(), stream, /*block=*/false);
        if (req_id == 0) {
            res.status = 503;
            res.set_content(
                make_error("Server is shutting down.",
                            "server_error", "shutting_down"),
                "application/json");
            return;
        }

        // --- 6. Streaming response ---
        if (stream) {
            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.set_header("Connection",    "keep-alive");
            res.set_header("X-Accel-Buffering", "no");

            std::string completion_id = new_completion_id();

            // In a full implementation the worker thread owns the InferenceSession
            // and pushes tokens back via a shared channel.  For this skeleton we
            // run a short stub decode loop inline to demonstrate the correct SSE
            // wire format.
            //
            // Production path: replace stub_tokens with actual tokens emitted
            // by orbit::Executor::run_decode_step() via a shared queue.

            std::vector<std::string> stub_tokens = {
                "Hello", ",", " I", " am", " ORBIT", "-", "G1", "."
            };

            // Use chunked content provider for SSE
            res.set_chunked_content_provider(
                "text/event-stream",
                [stub_tokens, completion_id, model, this](size_t /*offset*/,
                                                           httplib::DataSink& sink) -> bool
                {
                    for (size_t i = 0; i < stub_tokens.size(); ++i) {
                        std::string chunk = make_sse_chunk(
                            completion_id, model, stub_tokens[i], false);
                        if (!sink.write(chunk.c_str(), chunk.size())) return false;
                        // Small delay to simulate decode latency
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                    // Final chunk with finish_reason=stop
                    std::string last_chunk = make_sse_chunk(
                        completion_id, model, "", /*last=*/true);
                    if (!sink.write(last_chunk.c_str(), last_chunk.size())) return false;

                    // DONE sentinel
                    std::string done = make_sse_done();
                    if (!sink.write(done.c_str(), done.size())) return false;

                    sink.done();
                    return true;
                });

        } else {
            // --- 7. Non-streaming: return full JSON ---
            // Stub: decode all tokens at once then return
            std::string stub_text = "Hello, I am ORBIT-G1.";

            std::vector<int32_t> prompt_ids = tokenizer_.encode_chat(messages);
            int32_t prompt_tokens     = static_cast<int32_t>(prompt_ids.size());
            // Approximate token count: ~4 bytes per token (BPE heuristic).
            int32_t completion_tokens = static_cast<int32_t>(stub_text.size() / 4 + 1);

            std::string completion_id = new_completion_id();
            std::string response_body = make_full_response(
                completion_id, model, stub_text, prompt_tokens, completion_tokens);

            res.status = 200;
            res.set_content(response_body, "application/json");
        }
    });
}

// Explicit instantiation for httplib::Server
template void CompletionsHandler::register_routes<httplib::Server>(httplib::Server&);

// ---------------------------------------------------------------------------
// run_worker — worker thread that drives liborbit
// ---------------------------------------------------------------------------
void CompletionsHandler::run_worker() {
    std::cout << "[worker] Started\n";

    while (true) {
        RequestHandle req = session_mgr_.pop();
        if (!req) {
            std::cout << "[worker] Shutdown received\n";
            break;
        }

        if (req->cancelled.load(std::memory_order_relaxed)) {
            std::cout << "[worker] Request " << req->request_id << " cancelled, skipping\n";
            continue;
        }

        execute_request(req);
    }
}

// ---------------------------------------------------------------------------
// execute_request — run one inference request (stub)
// ---------------------------------------------------------------------------
void CompletionsHandler::execute_request(RequestHandle req) {
    // In production:
    //   1. Parse req->request_json
    //   2. tokenizer_.encode_chat(messages) → token_ids
    //   3. Create orbit::InferenceSession via KVCacheManager
    //   4. executor_.run_prefill(session)
    //   5. Loop: token = executor_.run_decode_step(session)
    //            push token text to SSE channel
    //   6. session.cancel() on req->cancelled
    //   7. Free KV pages
    //
    // This skeleton logs the request and returns immediately.
    std::cout << "[worker] Executing request " << req->request_id << "\n";
    // Parse to show we can at least handle the JSON
    try {
        json body = json::parse(req->request_json);
        std::string model = body.value("model", "orbit-g1");
        int32_t max_tokens = body.value("max_tokens", 512);
        std::cout << "[worker] model=" << model
                  << " max_tokens=" << max_tokens << "\n";
    } catch (...) {
        std::cerr << "[worker] Failed to parse request JSON\n";
    }
}

} // namespace orbit
