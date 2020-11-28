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

  ERL_NIF_TERM ErlTupleFromLiteral(ErlNifEnv* env, xla::Literal& literal) {
    std::vector<xla::Literal> literals = literal.DecomposeTuple();
    int elems = literals.size();
    ERL_NIF_TERM data[elems];

    for(int i=0;i<elems;i++) {
      xla::Literal lit(std::move(literals.at(i)));
      if(lit.shape().IsTuple()) {
        ERL_NIF_TERM term = ErlTupleFromLiteral(env, lit);
        data[i] = term;
      } else {
        long long int size = lit.size_bytes();
        ErlNifBinary binary;
        enif_alloc_binary(size, &binary);

        // No need to copy, just move to the underlying bytes in memory
        void *src_mem = const_cast<void*>(lit.untyped_data());
        std::memmove(binary.data, src_mem, size);

        ERL_NIF_TERM term = enif_make_binary(env, &binary);
        data[i] = term;
      }
    }
    return enif_make_tuple_from_array(env, data, elems);
  }

  xla::StatusOr<ERL_NIF_TERM> ExlaClient::ErlTupleFromBuffer(ErlNifEnv* env, exla::ExlaBuffer* buffer) {
    if(buffer->empty()) {
      return tensorflow::errors::FailedPrecondition("Attempt to read from empty buffer.");
    }

    if(!buffer->is_tuple()) {
      return tensorflow::errors::FailedPrecondition("Attempt to extract tuple from non-tuple buffer.");
    }

    xla::TransferManager* transfer_manager = client()->backend().transfer_manager();
    xla::StatusOr<xla::Literal> transfer_status = transfer_manager->TransferLiteralFromDevice(buffer->device()->device_to_host_stream(),
                                                                                              *(buffer->buffer()),
                                                                                              nullptr);

    // Something went wrong
    if(!transfer_status.ok()) {
      return transfer_status.status();
    }

    xla::Literal literal = transfer_status.ConsumeValueOrDie();
    ERL_NIF_TERM list = ErlTupleFromLiteral(env, literal);

    return list;
  }

  xla::StatusOr<ErlNifBinary> ExlaClient::ErlBinFromBuffer(exla::ExlaBuffer* buffer) {
    if(buffer->empty()) {
      return tensorflow::errors::Aborted("Attempt to read from empty buffer.");
    }

    bool is_cpu_platform = buffer->device()->executor()->platform()->id() == stream_executor::host::kHostPlatformId;

    if(is_cpu_platform) {
      // Allocate enough space for the binary
      long long int size = xla::ShapeUtil::ByteSizeOf(buffer->on_host_shape());
      ErlNifBinary binary;
      enif_alloc_binary(size, &binary);

      // Get the result buffer
      const stream_executor::DeviceMemoryBase mem_buffer = buffer->buffer()->root_buffer();

      // No need to copy, just move the underlying bytes in memory
      void* src_mem = const_cast<void *>(mem_buffer.opaque());
      std::memmove(binary.data, src_mem, size);

      return binary;
    }

    // Otherwise we have to do the transfer
    xla::TransferManager* transfer_manager = client()->backend().transfer_manager();
    xla::StatusOr<xla::Literal> transfer_status = transfer_manager->TransferLiteralFromDevice(buffer->device()->device_to_host_stream(),
                                                                                              *(buffer->buffer()),
                                                                                              nullptr);

    // Something went wrong
    if(!transfer_status.ok()) {
      return transfer_status.status();
    }

    xla::Literal literal = transfer_status.ConsumeValueOrDie();
    // Allocate enough space for the binary
    long long int size = literal.size_bytes();
    ErlNifBinary binary;
    enif_alloc_binary(size, &binary);

    // No need to copy, just move to the underlying bytes in memory
    void *src_mem = const_cast<void*>(literal.untyped_data());
    std::memmove(binary.data, src_mem, size);

    return binary;
  }

  bool CanUseZeroCopy(ErlNifBinary bin,
                      const xla::Shape& shape,
                      const xla::Shape& compact_shape,
                      ExlaDevice* device) {
    // Only possible on CPUs
    bool is_cpu_platform = device->executor()->platform()->id() == se::host::kHostPlatformId;
    // With well-aligned data
    bool is_well_aligned = (absl::bit_cast<std::uintptr_t>(bin.data) & (xla::cpu_function_runtime::kMinAlign - 1)) == 0;
    // With matching layouts
    bool has_same_layout = shape.layout() == compact_shape.layout();
    return is_cpu_platform && is_well_aligned && has_same_layout;
  }

  xla::ScopedShapedBuffer* ZeroCopyTransferBinToBuffer(const ErlNifBinary bin,
                                                       const xla::Shape& shape,
                                                       const xla::Shape& compact_shape,
                                                       ExlaDevice* device,
                                                       ExlaClient* client) {
    // Initialize a buffer to point to the same data as the binary
    se::DeviceMemoryBase buffer;
    buffer = se::DeviceMemoryBase(const_cast<unsigned char*>(bin.data), bin.size);
    // Make a new ScopedShapedBuffer
    auto device_buffer = new xla::ScopedShapedBuffer(compact_shape, client->allocator(), device->id());
    // Tell it to point to the buffer we made above
    auto memory = se::OwningDeviceMemory(buffer, device->id(), client->allocator());
    device_buffer->set_buffer(std::move(memory), {});
    return device_buffer;
  }

  xla::ScopedShapedBuffer* TransferBinToBuffer(const ErlNifBinary bin,
                                               const xla::Shape& shape,
                                               const xla::Shape& compact_shape,
                                               ExlaDevice* device,
                                               ExlaClient* client) {
    // Allocate space on the device
    xla::ScopedShapedBuffer device_buffer = AllocateDestinationBuffer(compact_shape, device, client).ConsumeValueOrDie();
    // Read directly from binary data into a `BorrowingLiteral`, this is zero-copy again
    xla::BorrowingLiteral literal(const_cast<char*>((char*) bin.data), shape);
    // Transfer literal to the device in the allocated buffer
    client->client()->backend().transfer_manager()->TransferLiteralToDevice(device->host_to_device_stream(), literal, device_buffer);
    return new xla::ScopedShapedBuffer(std::move(device_buffer));
  }

  xla::StatusOr<ExlaBuffer*> ExlaClient::BufferFromErlBin(const ErlNifBinary bin,
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

    // Can we use a zero copy transfer?
    bool can_use_zero_copy = CanUseZeroCopy(bin, shape, compact_shape, device);
    if(can_use_zero_copy) {
      xla::ScopedShapedBuffer* device_buffer = ZeroCopyTransferBinToBuffer(bin, shape, compact_shape, device, this);
      return new ExlaBuffer(/*buffer=*/device_buffer, /*device=*/device, /*zero_copy=*/true);
    } else {
      xla::ScopedShapedBuffer* device_buffer = TransferBinToBuffer(bin, shape, compact_shape, device, this);
      return new ExlaBuffer(/*buffer=*/device_buffer, /*device=*/device, /*zero_copy=*/false);
    }
  }

  xla::StatusOr<xla::ScopedShapedBuffer> ExlaClient::Run(xla::LocalExecutable* executable,
                                                         std::vector<std::pair<ExlaBuffer*, ExlaBuffer**>>& buffers,
                                                         xla::ExecutableRunOptions& options) {
    std::vector<xla::ShapedBuffer*> inputs;
    for(auto buf : buffers) {
      xla::ShapedBuffer* inp = (xla::ShapedBuffer*) (buf.first)->buffer();
      inputs.push_back(inp);
    }

    xla::StatusOr<xla::ScopedShapedBuffer> result = executable->Run(inputs, options);

    for(auto buf : buffers) {
      if(*buf.second != NULL) {
        delete *buf.second;
        *buf.second = NULL;
      }
    }

    return result;
  }

  xla::StatusOr<ExlaClient*> getHostClient(int num_replicas, int intra_op_parallelism_threads) {
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
                                                                        absl::StrCat("GPU_", device_ordinal, "_bfc"),
                                                                        false);

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

  xla::StatusOr<ExlaClient*> getCUDAClient(int num_replicas, int intra_op_parallelism_threads) {
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
