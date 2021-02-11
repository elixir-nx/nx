#ifndef EXLA_ALLOCATOR_H_
#define EXLA_ALLOCATOR_H_

#include <string>
#include <memory>

#include "tensorflow/compiler/xla/exla/exla_nif_util.h"
#include "tensorflow/compiler/xla/exla/exla_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/common_runtime/device/device_host_allocator.h"
#include "tensorflow/core/common_runtime/device/device_mem_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/tf_allocator_adapter.h"

namespace exla {

namespace se = tensorflow::se;

namespace allocator {

// Allocator which allocates directly on ERTS. This is
// used as the default host-memory allocator if no other
// allocator is specified. Host-memory allocators are used
// for staging host-to-device transfers. We treat the VM
// as the "Host" in this case.
class ExlaErtsAllocator : public tensorflow::Allocator {
 public:
    ExlaErtsAllocator() = default;

    std::string Name() override { return "erts"; }

    // Allocates an uninitialized block of memory that is "num_bytes"
    // bytes in size. Usually, alignment is used to enforce an alignment
    // that is a multiple of "alignment". `enif_alloc` guarantees that
    // the memory is sufficiently aligned for any built-in type. We use
    // this for allocating memory for tensors, so we can let the VM
    // take care of alignment.
    void* AllocateRaw(size_t alignment, size_t num_bytes) override {
      return enif_alloc(num_bytes);
    }

    // Deallocates a block of memory pointed to by ptr. Method requires
    // that the memory was allocated by a call to `AllocateRaw`, so we
    // can guarantee that `ptr` was created by a call to `enif_alloc`
    // and can safely be freed by `enif_free`.
    void DeallocateRaw(void* ptr) override {
      return enif_free(ptr);
    }
};

// Creates a multi-device "best-fit with coalescing" allocator in the
// same manner as PjRt. This is the allocator used on GPUs. See the
// TensorFlow repository for a dicsussion on BFC Allocators.
xla::StatusOr<std::unique_ptr<se::MultiDeviceAdapter>>
CreateBFCAllocator(absl::Span<std::unique_ptr<ExlaDevice> const> devices,
                   double memory_fraction,
                   bool preallocate);

// Returns a valid device memory allocator for the given GPU devices.
// memory_fraction controls the fraction of device memory available
// to this allocator. Preallocate determines whether or not to allocate
// memory in advance.
xla::StatusOr<std::unique_ptr<se::DeviceMemoryAllocator>>
GetGpuDeviceAllocator(absl::Span<std::unique_ptr<ExlaDevice> const> devices,
                      double memory_fraction,
                      bool preallocate);

// Creates a "best-fit with coalescing" host-memory allocator which
// makes some host RAM known to the GPU. This is used for staging
// host to device transfers.
std::unique_ptr<tensorflow::BFCAllocator>
GetGpuHostAllocator(se::StreamExecutor* executor);

}  // namespace allocator
}  // namespace exla

#endif
