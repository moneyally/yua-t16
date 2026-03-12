// orbit_server — HTTP/1.1 server implementation
// SSOT: spec/driver-runtime-design.md §4.1, §4.2
//
// HTTP library: cpp-httplib (header-only)
//   Single-include; place httplib.h alongside this file or in the include path.
//   Repository: https://github.com/yhirose/cpp-httplib
//
// JSON library: nlohmann/json (header-only)
//   Single-include json.hpp.

#define CPPHTTPLIB_OPENSSL_SUPPORT 0
#include "httplib.h"
#include "json.hpp"

#include "http_server.h"

#include <iostream>
#include <stdexcept>

using json = nlohmann::json;

namespace orbit {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
HttpServer::HttpServer(const ServerConfig& cfg)
    : cfg_(cfg),
      session_mgr_(cfg.queue_capacity),
      handler_(cfg, session_mgr_, tokenizer_)
{
    // Create the httplib server instance.
    srv_ = std::make_unique<httplib::Server>();

    // httplib thread pool for concurrent HTTP handler threads.
    // Each handler enqueues to SessionManager; actual inference is serial.
    srv_->new_task_queue = [cfg]() {
        return new httplib::ThreadPool(/* thread_count = */ 4);
    };

    // Load tokenizer (best-effort; stub fallback if not found)
    std::string tok_path = cfg_.model_dir + "/tokenizer.json";
    tokenizer_.load(tok_path);

    // Register all HTTP routes
    handler_.register_routes(*srv_);

    // Error handler — return OpenAI-format error on httplib exceptions
    srv_->set_error_handler([](const httplib::Request& /*req*/,
                                httplib::Response& res) {
        json err = {
            {"error", {
                {"message", "Internal server error"},
                {"type",    "server_error"},
                {"code",    "internal_error"}
            }}
        };
        res.set_content(err.dump(), "application/json");
    });

    // Exception handler
    srv_->set_exception_handler([](const httplib::Request& /*req*/,
                                    httplib::Response& res,
                                    std::exception_ptr ep) {
        std::string what = "Unknown exception";
        try { std::rethrow_exception(ep); }
        catch (const std::exception& e) { what = e.what(); }
        catch (...) {}

        std::cerr << "[http] Unhandled exception: " << what << "\n";

        json err = {
            {"error", {
                {"message", what},
                {"type",    "server_error"},
                {"code",    "internal_error"}
            }}
        };
        res.status = 500;
        res.set_content(err.dump(), "application/json");
    });
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------
HttpServer::~HttpServer() {
    stop();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

// ---------------------------------------------------------------------------
// run — start listening (blocking)
// ---------------------------------------------------------------------------
int HttpServer::run() {
    // Start the inference worker thread
    worker_thread_ = std::thread([this] {
        handler_.run_worker();
    });

    running_.store(true);

    std::cout << "[http] orbit_server listening on "
              << cfg_.host << ":" << cfg_.port << "\n";
    std::cout << "[http] Endpoints:\n"
              << "  POST /v1/chat/completions\n"
              << "  GET  /v1/models\n"
              << "  GET  /health\n";

    // Blocks until srv_->stop() is called
    bool ok = srv_->listen(cfg_.host.c_str(), cfg_.port);

    running_.store(false);

    // Signal the worker to drain and exit
    session_mgr_.shutdown();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// stop
// ---------------------------------------------------------------------------
void HttpServer::stop() {
    if (srv_) {
        srv_->stop();
    }
    session_mgr_.shutdown();
}

} // namespace orbit
