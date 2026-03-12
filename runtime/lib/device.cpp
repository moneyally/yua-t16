/**
 * device.cpp — Device class implementation
 *
 * Opens /dev/orbit_g1_N, issues ioctl wrappers, handles BAR1 mmap.
 * Hardware: ORBIT-G1 v2, driver: orbit_g1.ko
 */

#include "orbit.h"

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <linux/ioctl.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

// ioctl numbers (mirrors orbit_g1_uapi.h)
// Using raw _IOWR macros here to avoid a kernel header dependency.
// Magic: 'O' = 0x4F

#define ORBIT_IOC_MAGIC  'O'

// These match orbit_g1_uapi.h exactly.
// struct sizes must match the kernel uapi structs.
static_assert(sizeof(orbit::OrbitDescSubmit) == 32, "OrbitDescSubmit size mismatch");
static_assert(sizeof(orbit::OrbitWaitDone)   == 16, "OrbitWaitDone size mismatch");
static_assert(sizeof(orbit::OrbitMemAlloc)   == 32, "OrbitMemAlloc size mismatch");
static_assert(sizeof(orbit::OrbitMemFree)    == 8,  "OrbitMemFree size mismatch");
static_assert(sizeof(orbit::OrbitDeviceInfo) == 80, "OrbitDeviceInfo size mismatch");
static_assert(sizeof(orbit::OrbitQueueReset) == 8,  "OrbitQueueReset size mismatch");

#define _ORBIT_IOC_SUBMIT_DESC  _IOWR(ORBIT_IOC_MAGIC, 0x01, orbit::OrbitDescSubmit)
#define _ORBIT_IOC_WAIT_DONE    _IOWR(ORBIT_IOC_MAGIC, 0x02, orbit::OrbitWaitDone)
#define _ORBIT_IOC_ALLOC_MEM    _IOWR(ORBIT_IOC_MAGIC, 0x03, orbit::OrbitMemAlloc)
#define _ORBIT_IOC_FREE_MEM     _IOW (ORBIT_IOC_MAGIC, 0x04, orbit::OrbitMemFree)
#define _ORBIT_IOC_GET_INFO     _IOR (ORBIT_IOC_MAGIC, 0x05, orbit::OrbitDeviceInfo)
#define _ORBIT_IOC_RESET_QUEUE  _IOW (ORBIT_IOC_MAGIC, 0x06, orbit::OrbitQueueReset)

namespace orbit {

// ============================================================
// Device::Device (private constructor)
// ============================================================

Device::Device(int fd, const OrbitDeviceInfo& info)
    : fd_(fd), info_(info) {}

// ============================================================
// Device::open (factory)
// ============================================================

std::unique_ptr<Device> Device::open(int card_index) {
    std::string path = "/dev/orbit_g1_" + std::to_string(card_index);

    int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        throw OrbitException("Device::open: failed to open " + path +
                             ": " + std::strerror(errno), errno);
    }

    // Query device info
    OrbitDeviceInfo info{};
    if (::ioctl(fd, _ORBIT_IOC_GET_INFO, &info) < 0) {
        int saved = errno;
        ::close(fd);
        throw OrbitException("Device::open: ORBIT_IOC_GET_INFO failed: " +
                             std::string(std::strerror(saved)), saved);
    }

    return std::unique_ptr<Device>(new Device(fd, info));
}

// ============================================================
// Device::~Device
// ============================================================

Device::~Device() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

// ============================================================
// Move semantics
// ============================================================

Device::Device(Device&& other) noexcept
    : fd_(other.fd_), info_(other.info_) {
    other.fd_ = -1;
}

Device& Device::operator=(Device&& other) noexcept {
    if (this != &other) {
        if (fd_ >= 0) ::close(fd_);
        fd_   = other.fd_;
        info_ = other.info_;
        other.fd_ = -1;
    }
    return *this;
}

// ============================================================
// Device::info
// ============================================================

const OrbitDeviceInfo& Device::info() const noexcept {
    return info_;
}

// ============================================================
// Device::submit_desc
// ============================================================

int Device::submit_desc(OrbitDescSubmit& req) {
    // descs_ptr must point to userspace descriptor buffer before calling.
    if (::ioctl(fd_, _ORBIT_IOC_SUBMIT_DESC, &req) < 0) {
        return -errno;
    }
    return 0;
}

// ============================================================
// Device::wait_done
// ============================================================

int Device::wait_done(OrbitWaitDone& req) {
    if (::ioctl(fd_, _ORBIT_IOC_WAIT_DONE, &req) < 0) {
        return -errno;
    }
    return 0;
}

// ============================================================
// Device::alloc_gddr
// ============================================================

int Device::alloc_gddr(size_t size_bytes, size_t align, OrbitMemAlloc* out) {
    // TODO: validate align is power-of-2 and >= 4096
    out->size_bytes  = static_cast<uint64_t>(size_bytes);
    out->align_bytes = static_cast<uint64_t>(align);
    out->device_addr = 0;
    out->handle      = 0;

    if (::ioctl(fd_, _ORBIT_IOC_ALLOC_MEM, out) < 0) {
        return -errno;
    }
    return 0;
}

// ============================================================
// Device::free_gddr
// ============================================================

int Device::free_gddr(uint64_t handle) {
    OrbitMemFree req{handle};
    if (::ioctl(fd_, _ORBIT_IOC_FREE_MEM, &req) < 0) {
        return -errno;
    }
    return 0;
}

// ============================================================
// Device::reset_queue
// ============================================================

int Device::reset_queue(uint32_t queue_id) {
    OrbitQueueReset req{queue_id, 0};
    if (::ioctl(fd_, _ORBIT_IOC_RESET_QUEUE, &req) < 0) {
        return -errno;
    }
    return 0;
}

// ============================================================
// Device::mmap_bar1
// ============================================================

void* Device::mmap_bar1(uint64_t bar1_offset, size_t length) {
    // BAR1 is mapped using the device fd with PROT_READ|PROT_WRITE|MAP_SHARED.
    // The offset is relative to BAR1 aperture start (as documented in
    // orbit_g1_mmap.c: vm_pgoff encodes the BAR1 byte offset >> PAGE_SHIFT).

    if (bar1_offset + length > info_.bar1_size_bytes) {
        // TODO: log or set errno
        return nullptr;
    }

    void* ptr = ::mmap(nullptr, length,
                       PROT_READ | PROT_WRITE,
                       MAP_SHARED,
                       fd_,
                       static_cast<off_t>(bar1_offset));

    if (ptr == MAP_FAILED) {
        return nullptr;
    }
    return ptr;
}

// ============================================================
// Device::munmap_bar1
// ============================================================

void Device::munmap_bar1(void* ptr, size_t length) {
    if (ptr && ptr != MAP_FAILED) {
        ::munmap(ptr, length);
    }
}

} // namespace orbit
