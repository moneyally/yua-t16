/**
 * weight_loader.cpp — WeightLoader implementation
 *
 * Loads INT8 / INT4-AWQ quantized model weights from host filesystem into GDDR6.
 *
 * Upload strategy (from driver-runtime-design.md §3.5):
 *   1. alloc GDDR6 region via MemoryPool
 *   2. If within BAR1 window: memcpy directly (zero-copy, fast path)
 *   3. If outside BAR1: issue DMA_2D descriptor via util_queue (slow path)
 *   4. BARRIER before returning handle
 *
 * INT4-AWQ: packed INT4 bytes uploaded as-is; GEMM_INT4 handles dequant on-chip.
 *           Group scales/zeros uploaded to a companion GDDR6 region.
 */

#include "orbit.h"

#include <fstream>
#include <stdexcept>
#include <cstring>
#include <thread>
#include <cassert>

namespace orbit {

// ============================================================
// Constructor
// ============================================================

WeightLoader::WeightLoader(Device& dev, MemoryPool& pool, DescriptorQueue& util_queue)
    : dev_(dev), pool_(pool), util_queue_(util_queue) {}

// ============================================================
// load_weight_file
// ============================================================

MemHandle WeightLoader::load_weight_file(const std::string& path, QuantFormat format) {
    // Open and read the weight file into a host buffer.
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        throw OrbitException("WeightLoader::load_weight_file: cannot open " + path, ENOENT);
    }

    std::streamsize size = ifs.tellg();
    if (size <= 0) {
        throw OrbitException("WeightLoader::load_weight_file: empty or invalid file " + path);
    }
    ifs.seekg(0, std::ios::beg);

    std::vector<uint8_t> buf(static_cast<size_t>(size));
    if (!ifs.read(reinterpret_cast<char*>(buf.data()), size)) {
        throw OrbitException("WeightLoader::load_weight_file: read error " + path, EIO);
    }
    ifs.close();

    return load_weight_buffer(buf.data(), buf.size(), format);
}

// ============================================================
// load_weight_buffer
// ============================================================

MemHandle WeightLoader::load_weight_buffer(const void* data, size_t size_bytes, QuantFormat format) {
    // Choose upload method based on BAR1 availability.
    MemHandle h;

    // Allocate GDDR6 region. 256-byte alignment for GDDR6 burst efficiency.
    // (kvc.md §11: all GDDR6 accesses must be 256-byte aligned)
    constexpr size_t GDDR6_ALIGN = 256;
    MemHandle region_h = pool_.alloc(size_bytes, GDDR6_ALIGN);

    void* bar1_ptr = pool_.host_ptr(region_h);
    if (bar1_ptr) {
        h = upload_via_mmap_region(data, size_bytes, region_h, bar1_ptr);
    } else {
        h = upload_via_dma_region(data, size_bytes, region_h);
    }

    // Issue BARRIER descriptor to ensure all writes are complete before the
    // caller issues GEMM descriptors referencing this weight.
    util_queue_.add_barrier();
    util_queue_.submit_and_wait();

    // Register metadata.
    std::lock_guard<std::mutex> lk(mutex_);
    WeightInfo wi;
    wi.handle       = h;
    wi.device_addr  = pool_.device_addr(h);
    wi.size_bytes   = size_bytes;
    wi.format       = format;
    wi.shape        = {};   // TODO: parse shape from weight metadata
    wi.scale_handle = INVALID_MEM_HANDLE;
    loaded_[h] = wi;

    return h;
}

// ============================================================
// upload_via_mmap_region — fast path: direct memcpy over BAR1
// ============================================================

MemHandle WeightLoader::upload_via_mmap_region(const void* data, size_t size_bytes,
                                                MemHandle region_h, void* bar1_ptr) {
    // Direct write through PCIe write-combining BAR1 window.
    std::memcpy(bar1_ptr, data, size_bytes);
    // Memory fence to ensure all writes reach the device.
    __sync_synchronize();
    return region_h;
}

// ============================================================
// upload_via_dma_region — fallback: DMA_2D descriptor path
// ============================================================

MemHandle WeightLoader::upload_via_dma_region(const void* data, size_t size_bytes,
                                               MemHandle region_h) {
    // TODO: Allocate a pinned (DMA-coherent) host buffer via a driver ioctl or
    // posix_memalign + mlock, copy weight data there, then issue a DMA_2D
    // descriptor to transfer it to GDDR6.
    //
    // For now, this is a stub that illustrates the intent.  Real implementation
    // requires a PINNED_ALLOC ioctl (not yet in the uapi header — to be added).

    uint64_t dst_addr = pool_.device_addr(region_h);

    // Placeholder: treat host data pointer as a GDDR6-bus address.
    // THIS IS NOT CORRECT in real hardware — replace with actual pinned DMA path.
    // TODO: implement ORBIT_IOC_PIN_HOST_MEM + DMA_2D submit.
    uint64_t src_bus_addr = reinterpret_cast<uint64_t>(data);  // PLACEHOLDER

    util_queue_.add_dma_2d(src_bus_addr, dst_addr,
                           static_cast<uint32_t>(size_bytes), 1,
                           static_cast<uint32_t>(size_bytes),
                           static_cast<uint32_t>(size_bytes));
    return region_h;
}

// ============================================================
// Compatibility wrappers (matching header declarations)
// ============================================================

MemHandle WeightLoader::upload_via_mmap(const void* data, size_t size_bytes) {
    // Allocate and upload; fallback if no BAR1 available.
    constexpr size_t GDDR6_ALIGN = 256;
    MemHandle h = pool_.alloc(size_bytes, GDDR6_ALIGN);
    void* bar1 = pool_.host_ptr(h);
    if (bar1) {
        return upload_via_mmap_region(data, size_bytes, h, bar1);
    }
    return upload_via_dma_region(data, size_bytes, h);
}

MemHandle WeightLoader::upload_via_dma(const void* data, size_t size_bytes) {
    constexpr size_t GDDR6_ALIGN = 256;
    MemHandle h = pool_.alloc(size_bytes, GDDR6_ALIGN);
    return upload_via_dma_region(data, size_bytes, h);
}

// ============================================================
// load_weight_async
// ============================================================

void WeightLoader::load_weight_async(const std::string& path, QuantFormat format,
                                      LoadCallback cb) {
    // Launch a detached thread for async loading.
    // TODO: replace with a proper thread pool or io_uring for production use.
    std::thread([this, path, format, cb]() {
        MemHandle h = INVALID_MEM_HANDLE;
        int err = 0;
        try {
            h = load_weight_file(path, format);
        } catch (const OrbitException& ex) {
            err = ex.error_code();
            if (err == 0) err = -1;
        } catch (...) {
            err = -1;
        }
        cb(h, err);
    }).detach();
}

// ============================================================
// weight_info
// ============================================================

const WeightLoader::WeightInfo& WeightLoader::weight_info(MemHandle handle) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = loaded_.find(handle);
    if (it == loaded_.end()) {
        throw OrbitException("WeightLoader::weight_info: invalid handle");
    }
    return it->second;
}

// ============================================================
// unload
// ============================================================

void WeightLoader::unload(MemHandle handle) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = loaded_.find(handle);
    if (it == loaded_.end()) return;

    WeightInfo& wi = it->second;

    // Free companion scale region if present (INT4-AWQ).
    if (wi.scale_handle != INVALID_MEM_HANDLE) {
        pool_.free(wi.scale_handle);
        wi.scale_handle = INVALID_MEM_HANDLE;
    }

    pool_.free(handle);
    loaded_.erase(it);
}

} // namespace orbit
