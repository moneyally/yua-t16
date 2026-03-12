cmake_minimum_required(VERSION 3.16)
project(orbit_server CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Compiler options
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        -O2
    )
endif()

# ---------------------------------------------------------------------------
# Third-party header-only libraries
# ---------------------------------------------------------------------------
# cpp-httplib: https://github.com/yhirose/cpp-httplib  (httplib.h)
# nlohmann/json: https://github.com/nlohmann/json      (json.hpp)
#
# Both are expected to be placed alongside this CMakeLists.txt under:
#   runtime/server/httplib.h
#   runtime/server/json.hpp
#
# Alternatively, set ORBIT_HTTPLIB_DIR and ORBIT_JSON_DIR to point at
# directories containing these single headers.

if(NOT DEFINED ORBIT_HTTPLIB_DIR)
    set(ORBIT_HTTPLIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

if(NOT DEFINED ORBIT_JSON_DIR)
    set(ORBIT_JSON_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# ---------------------------------------------------------------------------
# liborbit (runtime library)
# ---------------------------------------------------------------------------
# The parent CMakeLists.txt in runtime/ is expected to define the
# orbit_runtime target (shared or static).  If we are being built
# standalone, provide a stub interface target.
if(NOT TARGET orbit_runtime)
    message(STATUS "[orbit_server] orbit_runtime target not found — using stub")
    add_library(orbit_runtime INTERFACE)
    # Stub: point at the runtime include directory so stub headers compile
    target_include_directories(orbit_runtime INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include/orbit"
    )
endif()

# ---------------------------------------------------------------------------
# orbit_server executable
# ---------------------------------------------------------------------------
add_executable(orbit_server
    main.cpp
    http_server.cpp
    completions_handler.cpp
    session_manager.cpp
    tokenizer.cpp
)

target_include_directories(orbit_server PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"   # config.h, openai_types.h, ...
    "${ORBIT_HTTPLIB_DIR}"          # httplib.h
    "${ORBIT_JSON_DIR}"             # json.hpp
)

# Link liborbit (provides orbit::Device, orbit::Executor, etc.)
target_link_libraries(orbit_server PRIVATE
    orbit_runtime
)

# pthreads (required by cpp-httplib thread pool + our own mutex/cv usage)
find_package(Threads REQUIRED)
target_link_libraries(orbit_server PRIVATE Threads::Threads)

# ---------------------------------------------------------------------------
# Optional: OpenSSL for HTTPS support (disabled by default)
# ---------------------------------------------------------------------------
option(ORBIT_SERVER_TLS "Enable HTTPS via OpenSSL" OFF)
if(ORBIT_SERVER_TLS)
    find_package(OpenSSL REQUIRED)
    target_compile_definitions(orbit_server PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT=1)
    target_link_libraries(orbit_server PRIVATE OpenSSL::SSL OpenSSL::Crypto)
else()
    target_compile_definitions(orbit_server PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT=0)
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS orbit_server DESTINATION bin)

# ---------------------------------------------------------------------------
# Status summary
# ---------------------------------------------------------------------------
message(STATUS "orbit_server build configuration:")
message(STATUS "  C++ standard : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "  httplib dir  : ${ORBIT_HTTPLIB_DIR}")
message(STATUS "  json dir     : ${ORBIT_JSON_DIR}")
message(STATUS "  TLS (HTTPS)  : ${ORBIT_SERVER_TLS}")
