#include "tensorflow/compiler/xla/exla/exla_client.h"
#include "tensorflow/compiler/xla/cpu_function_runtime.h"
#include "tensorflow/compiler/xla/exla/exla_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_host_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_mem_allocator.h"
#include "tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/stream_executor/tf_allocator_adapter.h"
#include "absl/memory/memory.h"

namespace exla {

  using int64 = tensorflow::int64;

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
      host_memory_allocator_ = std::make_unique<ExlaErtsAllocator>();
    }

  }

  xla::StatusOr<xla::ScopedShapedBuffer> AllocateDestinationBuffer(const xla::Shape& on_host_shape,
                                                                                    ExlaDevice* device,
                                                                                    ExlaClient* client) {
    xla::TransferManager* transfer_manager = client->client()->backend().transfer_manager();
    xla::ScopedShapedBuffer buffer = transfer_manager->AllocateScopedShapedBuffer(on_host_shape, client->allocator(),
                                                                                  device->id()).ConsumeValueOrDie();
    return buffer;
  }

  xla::StatusOr<ErlNifBinary> ExlaClient::ErlBinFromBuffer(const xla::ShapedBuffer& buffer,
                                                           ExlaDevice* device) {
    bool is_cpu_platform = device->executor()->platform()->id() == stream_executor::host::kHostPlatformId;

    // Special case where we can just point to memory address
    if(is_cpu_platform) {
      // Allocate enough space for the binary
      long long int size = xla::ShapeUtil::ByteSizeOf(buffer.on_host_shape());
      ErlNifBinary binary;
      enif_alloc_binary(size, &binary);

      // Get the result buffer
      const stream_executor::DeviceMemoryBase mem_buffer = buffer.root_buffer();

      // No need to copy, just point to the underlying bytes in memory
      void* src_mem = const_cast<void *>(mem_buffer.opaque());
      binary.data = (unsigned char*) src_mem;

      return binary;
    }

    // Otherwise we have to do the transfer
    xla::TransferManager* transfer_manager = client()->backend().transfer_manager();
    xla::StatusOr<xla::Literal> transfer_status = transfer_manager->TransferLiteralFromDevice(device->device_to_host_stream(), buffer, nullptr);

    // Something went wrong
    if(!transfer_status.ok()) {
      return transfer_status.status();
    }

    xla::Literal literal = transfer_status.ConsumeValueOrDie();
    // Allocate enough space for the binary
    long long int size = literal.size_bytes();
    ErlNifBinary binary;
    enif_alloc_binary(size, &binary);

    // No need to copy, just point to the underlying bytes in memory
    const void *src_mem = literal.untyped_data();
    binary.data = (unsigned char*) src_mem;

    return binary;
  }

  xla::StatusOr<std::unique_ptr<xla::ScopedShapedBuffer>> ExlaClient::BufferFromErlBin(const ErlNifBinary bin,
                                                                                       const xla::Shape& shape,
                                                                                       ExlaDevice* device) {
    // Get the expected size of the given shape
    int64 size = xla::ShapeUtil::ByteSizeOf(shape);
    // Validate the expected size and actual data size are the same
    // If they are not, we need to return an error because otherwise we'll be trying to read
    // from invalid memory
    if(size != bin.size) {
      return tensorflow::errors::InvalidArgument("Expected %d bytes from binary but got %d.", size, bin.size);
    }
    // Transfer Manager will manager the "transfer" to the device
    xla::TransferManager* transfer_manager = client()->backend().transfer_manager();
    // Ask for shape which has a compact layout on the device, in other words the anticipated shape of the data
    // on the device
    // TODO: Handle StatusOr
    xla::Shape compact_shape = transfer_manager->ChooseCompactLayoutForShape(shape).ConsumeValueOrDie();
    // CPU Platform allows for zero-copy transfers, we can just read directly from the binary
    bool is_cpu_platform = device->executor()->platform()->id() == se::host::kHostPlatformId;

    if(is_cpu_platform) {
      // Validates that the data is sufficiently aligned for a zero-copy transfer
      // XLA enforces a 16-byte alignment
      bool can_use_zero_copy = (absl::bit_cast<std::uintptr_t>(bin.data) & (xla::cpu_function_runtime::kMinAlign - 1)) == 0;
      // Ensure the shapes match, I think this avoids illegal memory errors
      if(shape.layout() == compact_shape.layout()) {
        se::DeviceMemoryBase buffer;

        if(can_use_zero_copy) {
          // Point directly to binary data!
          buffer = se::DeviceMemoryBase(const_cast<unsigned char*>(bin.data), size);
        } else {
          // Otherwise we stage on the VM and copy between
          void* staging_buffer = host_memory_allocator()->AllocateRaw(xla::cpu_function_runtime::kMinAlign, size);
          buffer = se::DeviceMemoryBase(staging_buffer, size);
          std::memcpy(staging_buffer, bin.data, size);
        }
        auto device_buffer = absl::make_unique<xla::ScopedShapedBuffer>(compact_shape, allocator(), device->id());
        auto memory = se::OwningDeviceMemory(buffer, device->id(), allocator());
        device_buffer->set_buffer(std::move(memory), {});
        return std::move(device_buffer);
      }
    }
    // Allocate space on the GPU
    xla::ScopedShapedBuffer device_buffer = AllocateDestinationBuffer(compact_shape, device, this).ConsumeValueOrDie();
    // Read directly from binary data into a `BorrowingLiteral`, this is zero-copy again
    xla::BorrowingLiteral literal(const_cast<char*>((char*) bin.data), shape);
    // Transfer literal to the device in the allocated buffer
    transfer_manager->TransferLiteralToDevice(device->host_to_device_stream(), literal, device_buffer);
    return absl::make_unique<xla::ScopedShapedBuffer>(std::move(device_buffer));
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
