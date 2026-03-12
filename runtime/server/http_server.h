#pragma once
// orbit_server — HTTP/1.1 server wrapper (cpp-httplib)
// SSOT: spec/driver-runtime-design.md §4.1, §4.2

#include "config.h"
#include "completions_handler.h"
#include "session_manager.h"
#include "tokenizer.h"

#include <atomic>
#include <memory>
#include <thread>

// Forward-declare httplib::Server so consumers of this header
// do not need to include httplib.h directly.
namespace httplib { class Server; }

namespace orbit {

// ---------------------------------------------------------------------------
// HttpServer
// ---------------------------------------------------------------------------
// Thin wrapper around cpp-httplib that:
//   1. Creates the httplib::Server instance
//   2. Registers all routes via CompletionsHandler
//   3. Starts the HTTP listener + worker thread
//   4. Handles graceful shutdown on SIGINT/SIGTERM (coordinated with main())
// ---------------------------------------------------------------------------
class HttpServer {
public:
    explicit HttpServer(const ServerConfig& cfg);
    ~HttpServer();

    // Start listening.  Blocks until stop() is called.
    // Returns 0 on clean shutdown, non-zero on bind/listen error.
    int run();

    // Signal the server to stop.  Thread-safe; may be called from signal handler.
    void stop();

    bool is_running() const { return running_.load(); }

private:
    const ServerConfig& cfg_;

    Tokenizer           tokenizer_;
    SessionManager      session_mgr_;
    CompletionsHandler  handler_;

    // httplib server (heap-allocated to keep httplib.h out of this header)
    std::unique_ptr<httplib::Server> srv_;

    std::atomic<bool> running_{false};
    std::thread       worker_thread_;
};

} // namespace orbit
