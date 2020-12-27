#ifndef EXLA_ALLOCATOR_H_
#define EXLA_ALLOCATOR_H_

#include <string>
#include <memory>

#include "tensorflow/compiler/xla/exla/exla_nif_util.h"
#include "tensorflow/compiler/xla/exla/exla_device.h"

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/common_runtime/gpu/gpu_host_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_mem_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"

#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/tf_allocator_adapter.h"

namespace exla {

namespace se = tensorflow::se;

namespace allocator {

/*
 * Allocator which allocates/deallocates directly on ERTS.
 */
class ExlaErtsAllocator : public tensorflow::Allocator {
 public:
    ExlaErtsAllocator() = default;

    std::string Name() override { return "erts"; }

    void* AllocateRaw(size_t alignment, size_t num_bytes) override {
      return enif_alloc(num_bytes);
    }

    void DeallocateRaw(void* ptr) override {
      return enif_free(ptr);
    }
};

xla::StatusOr<std::unique_ptr<se::MultiDeviceAdapter>> CreateBFCAllocator(absl::Span<std::unique_ptr<ExlaDevice> const> devices,
                                                                          double memory_fraction,
                                                                          bool preallocate);

xla::StatusOr<std::unique_ptr<se::DeviceMemoryAllocator>> GetGpuDeviceAllocator(absl::Span<std::unique_ptr<ExlaDevice> const> devices,
                                                                                double memory_fraction,
                                                                                bool preallocate);

std::unique_ptr<tensorflow::BFCAllocator> GetGpuHostAllocator(se::StreamExecutor* executor);

}  // namespace allocator
}  // namespace exla

#endif
