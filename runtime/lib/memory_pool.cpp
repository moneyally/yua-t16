/**
 * memory_pool.cpp — MemoryPool implementation
 *
 * Client-side GDDR6 memory manager with a slab front-end.
 * - Small allocations (< 1 MB): sub-allocated from 64 MB slabs (avoids ioctl per alloc).
 * - Large allocations (>= 1 MB): directly via ORBIT_IOC_ALLOC_MEM.
 * - host_ptr(): BAR1 mmap, lazy, cached.
 * - RAII: destructor frees all outstanding allocations.
 */

#include "orbit.h"

#include <stdexcept>
#include <cassert>
#include <cstring>

namespace orbit {

// ============================================================
// Constructor / Destructor
// ============================================================

MemoryPool::MemoryPool(Device& dev)
    : dev_(dev) {}

MemoryPool::~MemoryPool() {
    // Free all outstanding allocations in reverse insertion order.
    // We iterate by copy to avoid iterator invalidation.
    std::vector<MemHandle> handles;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        handles.reserve(allocs_.size());
        for (auto& [h, _] : allocs_) handles.push_back(h);
    }
    for (auto h : handles) {
        try { free(h); } catch (...) { /* best effort */ }
    }
}

// ============================================================
// alloc
// ============================================================

MemHandle MemoryPool::alloc(size_t size_bytes, size_t align) {
    if (size_bytes == 0) return INVALID_MEM_HANDLE;

    if (size_bytes >= LARGE_THRESHOLD) {
        return alloc_large(size_bytes, align);
    }
    return alloc_from_slab(size_bytes, align);
}

// ============================================================
// alloc_large — direct ioctl path for >= 1 MB
// ============================================================

MemHandle MemoryPool::alloc_large(size_t size_bytes, size_t align) {
    OrbitMemAlloc ma{};
    int ret = dev_.alloc_gddr(size_bytes, align, &ma);
    if (ret < 0) {
        throw OrbitException("MemoryPool::alloc_large: GDDR6 alloc failed", -ret);
    }

    std::lock_guard<std::mutex> lk(mutex_);
    MemHandle h = next_handle_++;
    AllocEntry e;
    e.device_addr   = ma.device_addr;
    e.kernel_handle = ma.handle;
    e.size          = size_bytes;
    e.host_ptr      = nullptr;
    e.bar1_mapped   = false;
    allocs_[h] = e;
    return h;
}

// ============================================================
// alloc_from_slab — sub-allocator for small allocations
// ============================================================

MemHandle MemoryPool::alloc_from_slab(size_t size_bytes, size_t align) {
    std::lock_guard<std::mutex> lk(mutex_);

    // Align size_bytes to alignment boundary.
    auto aligned_size = [](size_t n, size_t a) -> size_t {
        return (n + a - 1) & ~(a - 1);
    };
    size_t sz = aligned_size(size_bytes, align);

    // Find a slab with enough headroom.
    for (auto& slab : slabs_) {
        size_t aligned_used = aligned_size(slab.used, align);
        if (aligned_used + sz <= slab.slab_size) {
            // Check alignment of proposed sub-address.
            uint64_t sub_addr = slab.base_device_addr + aligned_used;
            if (sub_addr % align == 0) {
                slab.used = aligned_used + sz;

                MemHandle h = next_handle_++;
                AllocEntry se;
                se.device_addr   = sub_addr;
                se.kernel_handle = 0;   // sub-alloc: no direct kernel handle
                se.size          = size_bytes;
                se.host_ptr      = nullptr;
                se.bar1_mapped   = false;
                allocs_[h] = se;
                return h;
            }
        }
    }

    // No suitable slab; allocate a new 64 MB slab.
    OrbitMemAlloc ma{};
    size_t slab_sz = std::max(SLAB_SIZE, sz);
    {
        // Temporarily unlock to avoid potential ioctl deadlock.
        // (In practice the mutex is reentrant within this thread so this is safe.)
        // TODO: use a proper lock hierarchy if needed.
    }
    int ret = dev_.alloc_gddr(slab_sz, 4096, &ma);
    if (ret < 0) {
        throw OrbitException("MemoryPool::alloc_from_slab: slab alloc failed", -ret);
    }

    // Register slab as a large allocation so it gets freed on destruction.
    MemHandle slab_handle = next_handle_++;
    {
        AllocEntry ae;
        ae.device_addr   = ma.device_addr;
        ae.kernel_handle = ma.handle;
        ae.size          = slab_sz;
        ae.host_ptr      = nullptr;
        ae.bar1_mapped   = false;
        allocs_[slab_handle] = ae;
    }

    {
        Slab sl;
        sl.parent_handle    = slab_handle;
        sl.base_device_addr = ma.device_addr;
        sl.slab_size        = slab_sz;
        sl.used             = sz;
        slabs_.push_back(sl);
    }

    MemHandle h = next_handle_++;
    {
        AllocEntry ae2;
        ae2.device_addr   = ma.device_addr;  // Sub-alloc starts at base
        ae2.kernel_handle = 0;
        ae2.size          = size_bytes;
        ae2.host_ptr      = nullptr;
        ae2.bar1_mapped   = false;
        allocs_[h] = ae2;
    }
    return h;
}

// ============================================================
// free
// ============================================================

void MemoryPool::free(MemHandle handle) {
    if (handle == INVALID_MEM_HANDLE) return;

    std::lock_guard<std::mutex> lk(mutex_);
    auto it = allocs_.find(handle);
    if (it == allocs_.end()) {
        // Double-free or invalid handle — ignore (or TODO: warn).
        return;
    }

    AllocEntry& entry = it->second;

    // Unmap BAR1 if mmap'd.
    if (entry.bar1_mapped && entry.host_ptr) {
        dev_.munmap_bar1(entry.host_ptr, entry.size);
        entry.host_ptr   = nullptr;
        entry.bar1_mapped = false;
    }

    // Free kernel allocation if we own the kernel handle.
    if (entry.kernel_handle != 0) {
        dev_.free_gddr(entry.kernel_handle);
    }

    allocs_.erase(it);
}

// ============================================================
// device_addr
// ============================================================

uint64_t MemoryPool::device_addr(MemHandle handle) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = allocs_.find(handle);
    if (it == allocs_.end()) {
        throw OrbitException("MemoryPool::device_addr: invalid handle");
    }
    return it->second.device_addr;
}

// ============================================================
// host_ptr — lazy BAR1 mmap
// ============================================================

void* MemoryPool::host_ptr(MemHandle handle) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = allocs_.find(handle);
    if (it == allocs_.end()) {
        throw OrbitException("MemoryPool::host_ptr: invalid handle");
    }

    AllocEntry& entry = it->second;

    if (entry.bar1_mapped) {
        return entry.host_ptr;
    }

    // Check if device_addr falls within BAR1 aperture.
    const OrbitDeviceInfo& di = dev_.info();
    uint64_t bar1_end = di.bar1_size_bytes;  // BAR1 window covers [0, bar1_size_bytes)

    if (entry.device_addr >= bar1_end) {
        // Outside BAR1 window — cannot mmap.
        return nullptr;
    }
    if (entry.device_addr + entry.size > bar1_end) {
        // Partially outside; mmap what fits or return nullptr for safety.
        return nullptr;
    }

    void* ptr = dev_.mmap_bar1(entry.device_addr, entry.size);
    if (!ptr) {
        return nullptr;
    }

    entry.host_ptr   = ptr;
    entry.bar1_mapped = true;
    return ptr;
}

// ============================================================
// total_bytes / free_bytes — approximate
// ============================================================

uint64_t MemoryPool::total_bytes() const noexcept {
    return dev_.info().gddr_size_bytes;
}

uint64_t MemoryPool::free_bytes() const noexcept {
    // TODO: query driver for exact free bytes via a dedicated ioctl or debugfs.
    // For now return a conservative estimate.
    std::lock_guard<std::mutex> lk(mutex_);
    uint64_t used = 0;
    for (auto& [h, e] : allocs_) {
        if (e.kernel_handle != 0) {
            used += static_cast<uint64_t>(e.size);
        }
    }
    uint64_t total = dev_.info().gddr_size_bytes;
    return (used < total) ? (total - used) : 0;
}

} // namespace orbit
