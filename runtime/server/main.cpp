// orbit_server — Entry point
// SSOT: spec/driver-runtime-design.md §4.1, §4.7
//
// Usage:
//   orbit_server [OPTIONS]
//
//   -p <port>          HTTP listen port (default: 8080)
//   -H <host>          Bind address (default: 0.0.0.0)
//   -m <model_dir>     Path to model weights directory (required for real inference)
//   -d <device_node>   ORBIT-G1 device node (default: /dev/orbit_g1_0)
//   -c <max_context>   Max tokens per request (default: 4096)
//   --int4             Enable GEMM_INT4 descriptors
//   --no-int4          Disable GEMM_INT4 (use INT8 only)
//   -l <level>         Log level: debug|info|warn|error (default: info)
//   -h, --help         Print this help

#include "config.h"
#include "http_server.h"

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

// ---------------------------------------------------------------------------
// Global server pointer for signal handling
// ---------------------------------------------------------------------------
static orbit::HttpServer* g_server = nullptr;
static std::atomic<bool>  g_stop{false};

static void signal_handler(int signum) {
    std::cout << "\n[main] Received signal " << signum << ", shutting down...\n";
    g_stop.store(true);
    if (g_server) {
        g_server->stop();
    }
}

// ---------------------------------------------------------------------------
// print_usage
// ---------------------------------------------------------------------------
static void print_usage(const char* argv0) {
    std::cout <<
        "orbit_server — OpenAI-compatible inference server for ORBIT-G1\n\n"
        "Usage: " << argv0 << " [OPTIONS]\n\n"
        "Options:\n"
        "  -p <port>          Listen port (default: 8080)\n"
        "  -H <host>          Bind address (default: 0.0.0.0)\n"
        "  -m <model_dir>     Model weights directory\n"
        "  -d <device_node>   ORBIT-G1 device node (default: /dev/orbit_g1_0)\n"
        "  -c <max_context>   Max tokens per request (default: 4096)\n"
        "  --int4             Enable GEMM_INT4 (default: on)\n"
        "  --no-int4          Disable GEMM_INT4\n"
        "  -l <level>         Log level: debug|info|warn|error\n"
        "  -h, --help         Print this help\n";
}

// ---------------------------------------------------------------------------
// parse_args
// ---------------------------------------------------------------------------
static orbit::ServerConfig parse_args(int argc, char* argv[]) {
    orbit::ServerConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        auto require_next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "[main] Error: " << flag << " requires an argument\n";
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "-p") {
            int port;
            try {
                port = std::stoi(require_next("-p"));
            } catch (const std::exception& e) {
                std::cerr << "[main] Invalid port value: " << e.what() << "\n";
                std::exit(1);
            }
            if (port < 1 || port > 65535) {
                std::cerr << "[main] Invalid port: " << port << "\n";
                std::exit(1);
            }
            cfg.port = static_cast<uint16_t>(port);

        } else if (arg == "-H") {
            cfg.host = require_next("-H");

        } else if (arg == "-m") {
            cfg.model_dir = require_next("-m");

        } else if (arg == "-d") {
            cfg.device_node = require_next("-d");

        } else if (arg == "-c") {
            try {
                cfg.max_batch_tokens = static_cast<uint32_t>(
                    std::stoul(require_next("-c")));
            } catch (const std::exception& e) {
                std::cerr << "[main] Invalid -c value: " << e.what() << "\n";
                std::exit(1);
            }

        } else if (arg == "--int4") {
            cfg.enable_int4 = true;

        } else if (arg == "--no-int4") {
            cfg.enable_int4 = false;

        } else if (arg == "-l") {
            cfg.log_level = require_next("-l");

        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);

        } else {
            std::cerr << "[main] Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    return cfg;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    orbit::ServerConfig cfg = parse_args(argc, argv);

    // Print startup configuration
    std::cout << "=== orbit_server ===\n"
              << "  port:        " << cfg.port                << "\n"
              << "  host:        " << cfg.host                << "\n"
              << "  model_dir:   " << (cfg.model_dir.empty() ? "(none — stub mode)" : cfg.model_dir) << "\n"
              << "  device_node: " << cfg.device_node         << "\n"
              << "  max_tokens:  " << cfg.max_batch_tokens    << "\n"
              << "  int4:        " << (cfg.enable_int4 ? "yes" : "no") << "\n"
              << "  log_level:   " << cfg.log_level           << "\n"
              << "====================\n";

    // Install signal handlers
    std::signal(SIGINT,  signal_handler);
    std::signal(SIGTERM, signal_handler);
    // Ignore SIGPIPE (common on SSE client disconnect)
    std::signal(SIGPIPE, SIG_IGN);

    // Construct and run the server
    orbit::HttpServer server(cfg);
    g_server = &server;

    int exit_code = server.run();

    g_server = nullptr;
    std::cout << "[main] orbit_server exited with code " << exit_code << "\n";
    return exit_code;
}
