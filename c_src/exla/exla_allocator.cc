#include "tensorflow/compiler/xla/exla/exla_allocator.h"

#include "tensorflow/core/util/env_var.h"

namespace exla {
namespace allocator {

  // See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/pjrt/nvidia_gpu_device.cc#L85
  xla::StatusOr<std::unique_ptr<se::MultiDeviceAdapter>> CreateBFCAllocator(absl::Span<std::unique_ptr<ExlaDevice> const> devices,
                                                                            double memory_fraction, bool preallocate) {
    const se::Platform* platform = devices.front()->executor()->platform();
    std::vector<se::MultiDeviceAdapter::AllocatorWithStream> allocators;
    bool enable_unified_memory;
    xla::Status status = tensorflow::ReadBoolFromEnvVar("TF_FORCE_UNIFIED_MEMORY",
                                                   false, &enable_unified_memory);
    if (!status.ok()) {
      LOG(ERROR) << "Unable to read TF_FORCE_UNIFIED_MEMORY: "
                 << status.error_message();
    }

    for (auto& device : devices) {
      se::StreamExecutor* executor = device->executor();
      int device_ordinal = executor->device_ordinal();
      auto sub_allocator = absl::make_unique<tensorflow::GPUMemAllocator>(executor, tensorflow::PlatformGpuId(device_ordinal),
                                                                          /*use_unified_memory=*/enable_unified_memory,
                                                                          /*alloc_visitors=*/std::vector<tensorflow::SubAllocator::Visitor>(),
                                                                          /*free_visitors=*/std::vector<tensorflow::SubAllocator::Visitor>());

      tensorflow::int64 free_memory;
      tensorflow::int64 total_memory;
      if (!executor->DeviceMemoryUsage(&free_memory, &total_memory)) {
        return tensorflow::errors::Unavailable("Failed to query available memory from device %i",
                         device_ordinal);
      }
      // To allow full GPU memory to be visible to the BFC allocator if using
      // unified memory.
      size_t allocator_memory =
        enable_unified_memory ? total_memory : free_memory * memory_fraction;
      if (preallocate) {
        LOG(INFO) << "XLA backend allocating " << allocator_memory
                  << " bytes on device " << device_ordinal
                  << " for BFCAllocator.";
      } else {
        LOG(INFO) << "XLA backend will use up to " << allocator_memory
                  << " bytes on device " << device_ordinal
                  << " for BFCAllocator.";
      }
      auto gpu_bfc_allocator = absl::make_unique<tensorflow::BFCAllocator>(sub_allocator.release(),
                                                                         allocator_memory,
                                                                        /*allow_growth=*/!preallocate,
                                                                        absl::StrCat("GPU_", device_ordinal, "_bfc"),
                                                                        false);

      allocators.emplace_back(std::move(gpu_bfc_allocator), device->compute_stream());
    }
    return absl::make_unique<se::MultiDeviceAdapter>(platform, std::move(allocators));
  }

  // See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/pjrt/nvidia_gpu_device.cc#L140
  xla::StatusOr<std::unique_ptr<se::DeviceMemoryAllocator>> GetGpuDeviceAllocator(absl::Span<std::unique_ptr<ExlaDevice> const> devices,
                                                                                  double memory_fraction,
                                                                                  bool preallocate) {
    EXLA_ASSIGN_OR_RETURN(std::unique_ptr<se::DeviceMemoryAllocator> allocator,
      CreateBFCAllocator(devices, memory_fraction, preallocate));
    return std::move(allocator);
  }

  // See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/pjrt/nvidia_gpu_device.cc#L155
  // TODO(seanmor5): Rather than pin to Host memory, pin to ERTS memory.
  std::unique_ptr<tensorflow::BFCAllocator> GetGpuHostAllocator(se::StreamExecutor* executor) {
    tensorflow::SubAllocator* sub_allocator = new tensorflow::GpuHostAllocator(executor, 0, {}, {});
    const tensorflow::int64 kHostMemoryLimitBytes = 64 * (1LL << 30);
    return absl::make_unique<tensorflow::BFCAllocator>(sub_allocator,
                                                       kHostMemoryLimitBytes,
                                                       true,
                                                       "xla_gpu_host_bfc");
  }

}  // namespace allocator
}  // namespace exla
