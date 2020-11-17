#include "tensorflow/compiler/xla/exla/exla_client.h"
#include "tensorflow/core/common_runtime/gpu/gpu_host_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_mem_allocator.h"
#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "absl/memory/memory.h"

namespace exla {

  // See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/pjrt/pjrt_client.cc#L171-L183
  // TODO: Move to `exla_allocator.h` and consider implementation details
  class CpuAllocator : public tensorflow::Allocator {
  public:
    CpuAllocator() = default;

    std::string Name() override { return "cpu"; }

    void* AllocateRaw(size_t alignment, size_t num_bytes) override {
      return tensorflow::port::AlignedMalloc(num_bytes, alignment);
    }
    void DeallocateRaw(void* ptr) override {
      return tensorflow::port::AlignedFree(ptr);
    }
  };

  ExlaClient::ExlaClient(xla::LocalClient* client,
                         int host_id,
                         std::unique_ptr<se::DeviceMemoryAllocator> allocator,
                         std::unique_ptr<tensorflow::Allocator> host_memory_allocator,
                         std::unique_ptr<xla::GpuExecutableRunOptions> gpu_run_options) : client_(client),
                                                                                          host_id_(host_id),
                                                                                          owned_allocator_(std::move(allocator)),
                                                                                          host_memory_allocator_(std::move(host_memory_allocator)),
                                                                                          gpu_run_options_(std::move(gpu_run_options)) {
    if (owned_allocator_ != nullptr){
      allocator_ = owned_allocator_.get();
    } else {
      allocator_ = client_->backend().memory_allocator();
    }

    if(!host_memory_allocator) {
      host_memory_allocator_ = std::make_unique<CpuAllocator>();
    }

  }

  xla::StatusOr<ExlaClient*> GetCpuClient() {
    // TODO: Handle StatusOr
    stream_executor::Platform *platform = xla::PlatformUtil::GetPlatform("Host").ConsumeValueOrDie();
    if(platform->VisibleDeviceCount() <= 0){
      return xla::FailedPrecondition("CPU platform has no visible devices.");
    }

    xla::LocalClientOptions options;
    options.set_platform(platform);

    // TODO: Handle StatusOr
    // TODO: Individual device configuration similar to: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/pjrt/cpu_device.cc
    xla::LocalClient* client = xla::ClientLibrary::GetOrCreateLocalClient(options).ConsumeValueOrDie();

    return new ExlaClient(client, /*host_id*/0, /*allocator*/nullptr, /*host_memory_allocator*/nullptr, /*gpu_run_options*/nullptr);
  }

  // TODO: Move this to `exla_allocator`
  // See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/pjrt/nvidia_gpu_device.cc#L155
  // TODO: Consider a different approach, it might not be necessary, but it's worth thinking about later on.
  std::unique_ptr<tensorflow::BFCAllocator> GetGpuHostAllocator(se::StreamExecutor* executor) {
    tensorflow::SubAllocator* sub_allocator = new tensorflow::GpuHostAllocator(executor, 0, {}, {});
    const tensorflow::int64 kGpuHostMemoryLimitBytes = 64 * (1LL << 30);
    return absl::make_unique<tensorflow::BFCAllocator>(sub_allocator, kGpuHostMemoryLimitBytes, true, "xla_gpu_host_bfc");
  }

  xla::StatusOr<ExlaClient*> GetGpuClient() {
    // TODO: Handle StatusOr
    stream_executor::Platform *platform = xla::PlatformUtil::GetPlatform("CUDA").ConsumeValueOrDie();
    if(platform->VisibleDeviceCount() <= 0){
      return xla::FailedPrecondition("CUDA Platform has no visible devices.");
    }

    xla::LocalClientOptions options;
    options.set_platform(platform);

    // TODO: Handle StatusOr
    // TODO: Individual device configuration similar to: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/pjrt/nvidia_gpu_device.cc
    xla::LocalClient* client = xla::ClientLibrary::GetOrCreateLocalClient(options).ConsumeValueOrDie();

    se::StreamExecutorConfig config;
    // TODO: When we go to handling multiple GPUs, this needs to be adjusted
    config.ordinal = 0;
    config.device_options.non_portable_tags["host_thread_stack_size_in_bytes"] = absl::StrCat(8192 * 1024);
    // TODO: Handle StatusOr
    auto executor = (platform->GetExecutor(config)).ConsumeValueOrDie();

    auto host_memory_allocator = GetGpuHostAllocator(executor);

    return new ExlaClient(client, /*host_id*/0, /*allocator*/nullptr, /*host_memory_allcoator*/std::move(host_memory_allocator), /*gpu_run_options*/nullptr);
  }
}
