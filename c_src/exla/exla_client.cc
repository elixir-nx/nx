#include "tensorflow/compiler/xla/exla/exla_client.h"
#include "tensorflow/core/common_runtime/gpu/gpu_host_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_mem_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/stream_executor/tf_allocator_adapter.h"
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
                         std::vector<std::unique_ptr<ExlaDevice>> devices,
                         std::unique_ptr<se::DeviceMemoryAllocator> allocator,
                         std::unique_ptr<tensorflow::Allocator> host_memory_allocator,
                         std::unique_ptr<xla::GpuExecutableRunOptions> gpu_run_options) : client_(client),
                                                                                          host_id_(host_id),
                                                                                          devices_(std::move(devices)),
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

  xla::StatusOr<ExlaClient*> GetCpuClient(int num_replicas, int intra_op_parallelism_threads) {
    // TODO: Handle StatusOr
    se::Platform *platform = xla::PlatformUtil::GetPlatform("Host").ConsumeValueOrDie();
    if(platform->VisibleDeviceCount() <= 0){
      return xla::FailedPrecondition("CPU platform has no visible devices.");
    }

    xla::LocalClientOptions options;
    options.set_platform(platform);
    options.set_number_of_replicas(num_replicas);
    options.set_intra_op_parallelism_threads(intra_op_parallelism_threads);

    // TODO: Handle StatusOr
    // TODO: Individual device configuration similar to: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/pjrt/cpu_device.cc
    xla::LocalClient* client = xla::ClientLibrary::GetOrCreateLocalClient(options).ConsumeValueOrDie();

    std::vector<std::unique_ptr<ExlaDevice>> devices;
    for(int i = 0; i < client->device_count(); ++i) {
      se::StreamExecutorConfig config;
      config.ordinal = i;
      config.device_options.non_portable_tags["host_thread_stack_size_in_bytes"] = absl::StrCat(8192*1024);
      // TODO: Handle StatusOr
      se::StreamExecutor* executor = platform->GetExecutor(config).ConsumeValueOrDie();
      auto device = absl::make_unique<ExlaDevice>(i, executor, client);
      devices.push_back(std::move(device));
    }
    return new ExlaClient(client, /*host_id*/0,
                          /*devices*/std::move(devices),
                          /*allocator*/nullptr,
                          /*host_memory_allocator*/nullptr,
                          /*gpu_run_options*/nullptr);
  }

  // TODO: Move this to `exla_allocator`
  // See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/pjrt/nvidia_gpu_device.cc#L85
  // TODO: Consider a different approach, it might not be necessary, but it's worth thinking about later on.
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
                                                                        absl::StrCat("GPU_", device_ordinal, "_bfc"));

      allocators.emplace_back(std::move(gpu_bfc_allocator), device->compute_stream());
    }
    return absl::make_unique<se::MultiDeviceAdapter>(platform, std::move(allocators));
  }

  // TODO: Move this to `exla_allocator`
  // See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/pjrt/nvidia_gpu_device.cc#L140
  // TODO: Consider a different approach, it might not be necessary, but it's worth thinking about later on.
  std::unique_ptr<se::DeviceMemoryAllocator> GetGpuDeviceAllocator(absl::Span<std::unique_ptr<ExlaDevice> const> devices,
                                                                   double memory_fraction, bool preallocate) {
    // TODO: Handle StatusOr
    auto allocator = CreateBFCAllocator(devices, memory_fraction, preallocate).ConsumeValueOrDie();
    return std::move(allocator);
  }

  // TODO: Move this to `exla_allocator`
  // See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/pjrt/nvidia_gpu_device.cc#L155
  // TODO: Consider a different approach, it might not be necessary, but it's worth thinking about later on.
  std::unique_ptr<tensorflow::BFCAllocator> GetGpuHostAllocator(se::StreamExecutor* executor) {
    tensorflow::SubAllocator* sub_allocator = new tensorflow::GpuHostAllocator(executor, 0, {}, {});
    const tensorflow::int64 kGpuHostMemoryLimitBytes = 64 * (1LL << 30);
    return absl::make_unique<tensorflow::BFCAllocator>(sub_allocator, kGpuHostMemoryLimitBytes, true, "xla_gpu_host_bfc");
  }

  xla::StatusOr<ExlaClient*> GetGpuClient(int num_replicas, int intra_op_parallelism_threads) {
    // TODO: Handle StatusOr
    stream_executor::Platform *platform = xla::PlatformUtil::GetPlatform("CUDA").ConsumeValueOrDie();
    if(platform->VisibleDeviceCount() <= 0){
      return xla::FailedPrecondition("CUDA Platform has no visible devices.");
    }

    xla::LocalClientOptions options;
    options.set_platform(platform);
    options.set_number_of_replicas(num_replicas);
    options.set_intra_op_parallelism_threads(intra_op_parallelism_threads);

    // TODO: Handle StatusOr
    // TODO: Individual device configuration similar to: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/pjrt/nvidia_gpu_device.cc
    xla::LocalClient* client = xla::ClientLibrary::GetOrCreateLocalClient(options).ConsumeValueOrDie();

    std::vector<std::unique_ptr<ExlaDevice>> devices;
    for(int i = 0; i < client->device_count(); ++i) {
      se::StreamExecutor* executor = client->backend().stream_executor(i).ValueOrDie();
      int device_ordinal = executor->device_ordinal();
      devices.push_back(absl::make_unique<ExlaDevice>(device_ordinal, executor, client));
    }

    // TODO: Allocator options should be a configuration option.
    auto allocator = GetGpuDeviceAllocator(devices, 0.9, true);
    auto host_memory_allocator = GetGpuHostAllocator(devices.front()->executor());

    auto gpu_run_options = absl::make_unique<xla::GpuExecutableRunOptions>();

    return new ExlaClient(client, /*host_id*/0,
                          /*devices*/std::move(devices),
                          /*allocator*/std::move(allocator),
                          /*host_memory_allcoator*/std::move(host_memory_allocator),
                          /*gpu_run_options*/std::move(gpu_run_options));
  }
}
