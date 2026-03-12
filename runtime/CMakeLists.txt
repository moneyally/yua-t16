cmake_minimum_required(VERSION 3.16)
project(liborbit VERSION 1.0.0 LANGUAGES CXX)

# ============================================================
# Compiler settings
# ============================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Position-independent code for shared library.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Warnings
add_compile_options(
    -Wall
    -Wextra
    -Wshadow
    -Wnon-virtual-dtor
    -Wcast-align
    -Woverloaded-virtual
    -Wpedantic
    -Wno-unused-parameter   # suppress for TODO stubs
)

# ============================================================
# Include paths
# ============================================================

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include      # orbit.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include   # upstream uapi (orbit_g1_uapi.h)
)

# ============================================================
# Source files
# ============================================================

set(LIBORBIT_SOURCES
    lib/device.cpp
    lib/memory_pool.cpp
    lib/descriptor_queue.cpp
    lib/weight_loader.cpp
    lib/kvc_manager.cpp
    lib/session.cpp
    lib/executor.cpp
)

# ============================================================
# Shared library: liborbit.so
# ============================================================

add_library(orbit_shared SHARED ${LIBORBIT_SOURCES})
set_target_properties(orbit_shared PROPERTIES
    OUTPUT_NAME   orbit
    VERSION       ${PROJECT_VERSION}
    SOVERSION     1
)
target_include_directories(orbit_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(orbit_shared
    PRIVATE
        pthread
)

# ============================================================
# Static library: liborbit.a
# ============================================================

add_library(orbit_static STATIC ${LIBORBIT_SOURCES})
set_target_properties(orbit_static PROPERTIES
    OUTPUT_NAME orbit
)
target_include_directories(orbit_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(orbit_static
    PRIVATE
        pthread
)

# ============================================================
# Tests
# ============================================================

option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    enable_testing()

    add_executable(test_basic tests/test_basic.cpp)
    target_include_directories(test_basic PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(test_basic
        PRIVATE orbit_static pthread
    )
    target_compile_options(test_basic PRIVATE -Wno-unused-parameter)

    add_test(NAME test_basic COMMAND test_basic)
endif()

# ============================================================
# Install
# ============================================================

include(GNUInstallDirs)

install(TARGETS orbit_shared orbit_static
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES include/orbit.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# ============================================================
# Package config (optional: for find_package(orbit))
# ============================================================

include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/orbitConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/orbitConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/orbit
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/orbitConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/orbitConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/orbitConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/orbit
)
