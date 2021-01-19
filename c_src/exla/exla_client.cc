#include "tensorflow/compiler/xla/exla/exla_client.h"
#include "tensorflow/compiler/xla/exla/exla_allocator.h"

#include "tensorflow/compiler/xla/cpu_function_runtime.h"
#include "tensorflow/compiler/xla/literal_util.h"

namespace exla {

// ExlaBuffer functions
xla::Status ExlaBuffer::Deallocate() {
  if (!empty()) {
    switch (type_) {
      case BufferType::kZeroCopy:
        ReleaseMemoryOwnership();
        state_ = BufferState::kDeallocated;
        return xla::Status::OK();
      case BufferType::kReference:
      case BufferType::kTemporary:
      {
        int device_ordinal = device_->device_ordinal();
        for (const se::DeviceMemoryBase& buffer : device_memory_) {
          xla::Status status = client_->allocator()->Deallocate(device_ordinal, buffer);
          if (!status.ok()) {
            LOG(WARNING) << "Buffer deallocation failed: " << status;
            return status;
          }
        }
        state_ = BufferState::kDeallocated;
        return xla::Status::OK();
      }
    }
    LOG(ERROR) << "Internal error.";
  }

  return xla::FailedPrecondition("Attempt to deallocate already deallocated buffer.");
}

void ExlaBuffer::AddToInputAsImmutable(xla::ShapeTree<xla::MaybeOwningDeviceMemory>::iterator* iterator,
                                       xla::ShapeTree<xla::MaybeOwningDeviceMemory>::iterator& end) {
  for (const se::DeviceMemoryBase& buf : device_memory_) {
    CHECK(*iterator != end);
    (*iterator)->second = xla::MaybeOwningDeviceMemory(buf);
    ++(*iterator);
  }
}

void ExlaBuffer::AddToInputAsDonated(xla::ShapeTree<xla::MaybeOwningDeviceMemory>::iterator* iterator,
                                     xla::ShapeTree<xla::MaybeOwningDeviceMemory>::iterator& end,
                                     xla::ExecutionInput* input) {
  se::DeviceMemoryAllocator* allocator = client_->allocator();
  int device_ordinal = device_->device_ordinal();

  for (const se::DeviceMemoryBase& buf : device_memory_) {
    CHECK(*iterator != end);

    (*iterator)->second = xla::MaybeOwningDeviceMemory(
      se::OwningDeviceMemory(buf, device_ordinal, allocator));
    input->SetUnownedIndex((*iterator)->first);
    ++(*iterator);
  }
}

void ExlaBuffer::AddToInput(xla::ShapeTree<xla::MaybeOwningDeviceMemory>::iterator* iterator,
                            xla::ShapeTree<xla::MaybeOwningDeviceMemory>::iterator& end,
                            xla::ExecutionInput* input) {
  switch (type_) {
    case BufferType::kZeroCopy:
    case BufferType::kReference:
      AddToInputAsImmutable(iterator, end);
      return;
    case BufferType::kTemporary:
      AddToInputAsDonated(iterator, end, input);
      return;
  }
}

xla::ShapedBuffer ExlaBuffer::AsShapedBuffer() {
  int32 device_ordinal = device_->device_ordinal();
  xla::ShapedBuffer shaped_buffer(on_host_shape_, on_device_shape_, device_ordinal);
  xla::ShapeTree<se::DeviceMemoryBase>::iterator iterator =
    shaped_buffer.buffers().begin();
  for (const se::DeviceMemoryBase& buf : device_memory_) {
    CHECK(iterator != shaped_buffer.buffers().end());
    iterator->second = buf;
    ++iterator;
  }
  CHECK(iterator == shaped_buffer.buffers().end());

  return shaped_buffer;
}

ExlaBuffer* ExlaBuffer::FromScopedShapedBuffer(xla::ScopedShapedBuffer* shaped_buffer,
                                               ExlaDevice* device,
                                               ExlaClient* client,
                                               BufferType type) {
  xla::ShapeTree<se::DeviceMemoryBase>::iterator iterator =
    shaped_buffer->buffers().begin();

  std::vector<se::DeviceMemoryBase> buffers;
  buffers.reserve(1);

  xla::ShapeUtil::ForEachSubshape(
      shaped_buffer->on_device_shape(), [&](const xla::Shape&, const xla::ShapeIndex&) {
        CHECK(iterator != shaped_buffer->buffers().end());
        buffers.push_back(iterator->second);
        iterator->second = se::DeviceMemoryBase();
        ++iterator;
      });

  CHECK(iterator == shaped_buffer->buffers().end());

  xla::Shape on_host_shape = shaped_buffer->on_host_shape();
  xla::Shape on_device_shape = shaped_buffer->on_device_shape();

  return new ExlaBuffer(/*device_memory=*/absl::Span<se::DeviceMemoryBase>(buffers),
                        /*on_host_shape=*/on_host_shape,
                        /*on_device_shape=*/on_device_shape,
                        /*device=*/device,
                        /*client=*/client,
                        /*type=*/type);
}

xla::StatusOr<ErlNifBinary> ExlaBuffer::ToBinary() {
  if (empty()) {
    return xla::FailedPrecondition("Attempt to read from deallocated buffer.");
  }

  bool is_cpu_platform =
    (device_->executor()->platform()->id() ==
      stream_executor::host::kHostPlatformId);

  if (is_cpu_platform) {
    int64 size = xla::ShapeUtil::ByteSizeOf(on_host_shape());
    ErlNifBinary binary;
    enif_alloc_binary(size, &binary);

    se::DeviceMemoryBase mem_buffer = device_memory_.at(0);

    void* src_mem = const_cast<void *>(mem_buffer.opaque());
    std::memcpy(binary.data, src_mem, size);

    return binary;
  }

  xla::ShapedBuffer shaped_buffer = AsShapedBuffer();

  xla::TransferManager* transfer_manager =
    client_->client()->backend().transfer_manager();

  EXLA_ASSIGN_OR_RETURN(xla::Literal literal,
    transfer_manager->TransferLiteralFromDevice(
      device_->device_to_host_stream(),
      shaped_buffer,
      nullptr));

  int64 size = literal.size_bytes();
  ErlNifBinary binary;
  enif_alloc_binary(size, &binary);

  void *src_mem = const_cast<void*>(literal.untyped_data());
  std::memcpy(binary.data, src_mem, size);

  return binary;
}

xla::Status ExlaBuffer::BlockHostUntilReady() {
  // TODO(seanmor5): Error check?
  creation_stream_->ThenWaitFor(definition_event_.get());
  return xla::Status::OK();
}

xla::StatusOr<xla::ScopedShapedBuffer> AllocateDestinationBuffer(const xla::Shape& on_host_shape,
                                                                 ExlaDevice* device,
                                                                 ExlaClient* client) {
  xla::TransferManager* transfer_manager =
    client->client()->backend().transfer_manager();

  EXLA_ASSIGN_OR_RETURN(
    xla::ScopedShapedBuffer buffer,
    transfer_manager->AllocateScopedShapedBuffer(on_host_shape,
                                                 client->allocator(),
                                                 device->id()));

  return buffer;
}

xla::StatusOr<std::vector<ExlaBuffer*>> UnpackRunArguments(ErlNifEnv* env,
                                                           ERL_NIF_TERM list,
                                                           ExlaDevice* device,
                                                           ExlaClient* client) {
  uint32 length;
  if (!enif_get_list_length(env, list, &length)) {
    return xla::InvalidArgument("Argument is not a list.");
  }

  std::vector<ExlaBuffer*> arguments;
  arguments.reserve(length);

  ERL_NIF_TERM head, tail;
  while (enif_get_list_cell(env, list, &head, &tail)) {
    const ERL_NIF_TERM* tuple;
    int arity;
    ExlaBuffer** buffer;
    if (enif_get_tuple(env, head, &arity, &tuple)) {
      ErlNifBinary data;
      xla::Shape* shape;
      if (!nif::get_binary(env, tuple[0], &data)) {
        return xla::InvalidArgument("Arguments must either be buffer or tuple of shape, binary.");
      }
      if (!nif::get<xla::Shape>(env, tuple[1], shape)) {
        return xla::InvalidArgument("Arguments must either be buffer or tuple of shape, binary.");
      }
      EXLA_ASSIGN_OR_RETURN(ExlaBuffer* buf,
        client->BufferFromBinary(data, *shape, device, true));
      arguments.push_back(buf);
    } else if (nif::get<ExlaBuffer*>(env, head, buffer)) {
      arguments.push_back(*buffer);
    } else {
      return xla::InvalidArgument("Arguments must either be buffer or tuple of shape, binary.");
    }
    list = tail;
  }
  return arguments;
}

// ExlaExecutable Functions
ExlaExecutable::ExlaExecutable(std::vector<std::unique_ptr<xla::LocalExecutable>> executables,
                               std::shared_ptr<xla::DeviceAssignment> device_assignment,
                               std::vector<std::pair<int, int>> local_logical_device_ids,
                               std::vector<ExlaDevice*> local_devices,
                               ExlaClient* client)
                               : client_(client),
                                 device_assignment_(std::move(device_assignment)),
                                 local_logical_device_ids_(std::move(local_logical_device_ids)),
                                 local_devices_(std::move(local_devices)) {
  executables_.reserve(executables.size());
  for (auto& executable : executables) {
    executables_.emplace_back(std::move(executable));
  }

  int num_partitions;
  if (device_assignment_ == nullptr) {
    // Executable portable single-core
    num_partitions = 1;
    CHECK(local_devices_.empty());
  } else {
    // Executable with a device_assignment
    CHECK_GE(local_devices_.size(), 1) << device_assignment_->ToString();
    CHECK_LE(local_devices_.size(),
             client_->device_count()) << "Inconsistent local device count.";
    num_partitions = device_assignment_->computation_count();
  }

  // SPMD sharding produces a single executable for multiple partitions.
  if (executables_.size() > 1) {
    CHECK_EQ(num_partitions, executables_.size())
        << "Number of executables " << executables_.size()
        << " did not match number of partitions " << num_partitions;
  }
}

xla::StatusOr<std::vector<xla::ExecutionInput>>
ExlaExecutable::PopulateInputBuffers(absl::Span<ExlaBuffer* const> argument_handles) {
  std::vector<xla::ExecutionInput> execution_inputs;
  execution_inputs.reserve(argument_handles.size());

  for (int i = 0; i < argument_handles.size(); ++i) {
    ExlaBuffer* handle = argument_handles[i];
    execution_inputs.emplace_back(handle->on_host_shape(),
                                  handle->on_device_shape());

    xla::ExecutionInput& execution_input = execution_inputs.back();
    xla::ShapeTree<xla::MaybeOwningDeviceMemory>::iterator input_iterator =
      execution_input.MutableBuffers()->begin();
    xla::ShapeTree<xla::MaybeOwningDeviceMemory>::iterator iterator_end =
      execution_input.MutableBuffers()->end();

    handle->AddToInput(&input_iterator, iterator_end, &execution_input);
    CHECK(input_iterator == iterator_end);
  }

  return execution_inputs;
}

xla::StatusOr<ERL_NIF_TERM> ExlaExecutable::Run(ErlNifEnv* env,
                                                ERL_NIF_TERM argument_terms,
                                                xla::Shape& output_shape,
                                                int replica,
                                                int partition,
                                                int run_id,
                                                int rng_seed,
                                                int launch_id,
                                                ExlaDevice* device,
                                                bool keep_on_device) {
  std::shared_ptr<xla::DeviceAssignment> device_assignment;
  if (device == nullptr) {
    CHECK(device_assignment_ != nullptr);
    const int device_id = (*device_assignment_)(replica - 1, partition - 1);
    device = client_->device(device_id);
    device_assignment = device_assignment_;
  } else {
    device_assignment = std::make_shared<xla::DeviceAssignment>(1, 1);
    (*device_assignment)(0, 0) = device->id();
  }

  int device_ordinal = device->device_ordinal();
  int executable_idx = executables_.size() > 1 ? partition : 0;

  xla::RunId run_id_obj(run_id);
  xla::ExecutableRunOptions run_options;
  run_options.set_stream(device->compute_stream());
  run_options.set_host_to_device_stream(device->host_to_device_stream());
  run_options.set_allocator(client_->allocator());
  run_options.set_intra_op_thread_pool(client_->client()->backend().eigen_intra_op_thread_pool_device());
  run_options.set_device_assignment(device_assignment.get());
  run_options.set_run_id(run_id_obj);
  run_options.set_rng_seed(rng_seed);
  run_options.set_gpu_executable_run_options(client_->gpu_run_options());
  run_options.set_launch_id(launch_id);

  std::shared_ptr<xla::LocalExecutable> executable = executables_.at(executable_idx);

  EXLA_ASSIGN_OR_RETURN(std::vector<ExlaBuffer*> arguments,
                        UnpackRunArguments(env, argument_terms, device, client_));

  EXLA_ASSIGN_OR_RETURN(std::vector<xla::ExecutionInput> inputs,
                        PopulateInputBuffers(arguments));

  xla::StatusOr<xla::ExecutionOutput> exec_status =
    executable->RunAsync(std::move(inputs), run_options);

  device->compute_stream()->BlockHostUntilDone();

  xla::ExecutionOutput results = exec_status.ConsumeValueOrDie();
  xla::ScopedShapedBuffer result_buffer = results.ConsumeResult();

  ExlaBuffer* buffer_ref = ExlaBuffer::FromScopedShapedBuffer(&result_buffer, device, client_, ExlaBuffer::BufferType::kReference);

  if (keep_on_device && buffer_ref->is_tuple()) {
    return nif::ok(env);
  } else if (keep_on_device) {
    return nif::make<ExlaBuffer*>(env, buffer_ref);
  } else if (buffer_ref->is_tuple()) {
    return nif::ok(env);
  } else {
    EXLA_ASSIGN_OR_RETURN_NIF(ErlNifBinary binary,
      buffer_ref->ToBinary(), env);
    delete buffer_ref;
    return nif::make(env, binary);
  }
}

// ExlaClient Functions
ExlaClient::ExlaClient(xla::LocalClient* client,
                       int host_id,
                       std::vector<std::unique_ptr<ExlaDevice>> devices,
                       std::unique_ptr<se::DeviceMemoryAllocator> allocator,
                       std::unique_ptr<tensorflow::Allocator> host_memory_allocator,
                       std::unique_ptr<xla::gpu::GpuExecutableRunOptions> gpu_run_options)
                        : client_(client),
                          host_id_(host_id),
                          devices_(std::move(devices)),
                          owned_allocator_(std::move(allocator)),
                          host_memory_allocator_(std::move(host_memory_allocator)),
                          gpu_run_options_(std::move(gpu_run_options)) {
  if (owned_allocator_ != nullptr) {
    allocator_ = owned_allocator_.get();
  } else {
    allocator_ = client_->backend().memory_allocator();
  }

  if (!host_memory_allocator) {
    host_memory_allocator_ = std::make_unique<allocator::ExlaErtsAllocator>();
  }
}

xla::StatusOr<xla::DeviceAssignment>
ExlaClient::GetDefaultDeviceAssignment(int num_replicas,
                                       int num_partitions) {
  return client_->backend().computation_placer()->AssignDevices(num_replicas, num_partitions);
}

bool CanUseZeroCopy(const ErlNifBinary& bin,
                    const xla::Shape& shape,
                    const xla::Shape& compact_shape,
                    ExlaDevice* device) {
  bool is_cpu_platform =
    device->executor()->platform()->id() == se::host::kHostPlatformId;
  bool is_well_aligned =
    (absl::bit_cast<std::uintptr_t>(bin.data) &
      (xla::cpu_function_runtime::kMinAlign - 1)) == 0;
  bool has_same_layout = shape.layout() == compact_shape.layout();
  return is_cpu_platform && is_well_aligned && has_same_layout;
}

xla::StatusOr<ExlaBuffer*>
ExlaClient::BufferFromBinary(const ErlNifBinary& binary,
                             xla::Shape& on_host_shape,
                             ExlaDevice* device,
                             bool transfer_for_run) {
  int64 size = xla::ShapeUtil::ByteSizeOf(on_host_shape);
  if (size != binary.size) {
    return xla::InvalidArgument("Expected %d bytes from binary but got %d.", size, binary.size);
  }
  xla::TransferManager* transfer_manager =
    client_->backend().transfer_manager();

  EXLA_ASSIGN_OR_RETURN(xla::Shape on_device_shape,
    transfer_manager->ChooseCompactLayoutForShape(on_host_shape));

  bool can_use_zero_copy = CanUseZeroCopy(binary, on_host_shape, on_device_shape, device);

  if (can_use_zero_copy && transfer_for_run) {
    se::DeviceMemoryBase buffer;
    buffer = se::DeviceMemoryBase(const_cast<unsigned char*>(binary.data), binary.size);

    return new ExlaBuffer(/*device_memory=*/absl::Span<se::DeviceMemoryBase const>({buffer}),
                          /*on_host_shape=*/on_host_shape,
                          /*on_device_shape=*/on_device_shape,
                          /*device=*/device,
                          /*client=*/this,
                          /*type=*/ExlaBuffer::BufferType::kZeroCopy);
  } else  {
    ExlaBuffer::BufferType type = transfer_for_run ? ExlaBuffer::BufferType::kTemporary : ExlaBuffer::BufferType::kReference;

    EXLA_ASSIGN_OR_RETURN(xla::ScopedShapedBuffer device_buffer,
      AllocateDestinationBuffer(on_device_shape, device, this));

    xla::BorrowingLiteral literal(const_cast<char*>(reinterpret_cast<char*>(binary.data)), on_device_shape);

    client_->backend().transfer_manager()->TransferLiteralToDevice(device->host_to_device_stream(), literal, device_buffer);

    ExlaBuffer* buffer = ExlaBuffer::FromScopedShapedBuffer(&device_buffer, device, this, type);

    return buffer;
  }
}

xla::StatusOr<ExlaExecutable*> ExlaClient::Compile(const xla::XlaComputation& computation,
                                                   std::vector<xla::Shape*> argument_layouts,
                                                   xla::ExecutableBuildOptions& options,
                                                   bool compile_portable_executable) {
  if (!options.device_allocator()) {
    options.set_device_allocator(allocator());
  }

  int32 num_replicas, num_partitions;
  std::shared_ptr<xla::DeviceAssignment> device_assignment;
  if (compile_portable_executable) {
    if (options.has_device_assignment()) {
      return xla::InvalidArgument("Requested portable executable but specified device assignment.");
    }
    num_replicas = 1;
    num_partitions = 1;
  } else {
    if (!options.has_device_assignment()) {
      // Compiling with default device assignment
      EXLA_ASSIGN_OR_RETURN(
        xla::DeviceAssignment device_assignment,
        GetDefaultDeviceAssignment(options.num_replicas(),
                                   options.num_partitions()));
      options.set_device_assignment(device_assignment);
    }
    num_replicas = options.device_assignment().replica_count();
    num_partitions = options.device_assignment().computation_count();
    device_assignment =
      std::make_shared<xla::DeviceAssignment>(options.device_assignment());
  }

  std::vector<std::pair<int, int>> local_logical_device_ids;
  std::vector<ExlaDevice*> local_devices;
  if (device_assignment != nullptr) {
    for (int replica = 0; replica < num_replicas; ++replica) {
      for (int partition = 0; partition < num_partitions; ++partition) {
        int device_id = (*device_assignment)(replica, partition);
        ExlaDevice* device = this->device(device_id);
        local_logical_device_ids.emplace_back(replica, partition);
        local_devices.push_back(device);
      }
    }
    if (local_devices.empty()) {
      return xla::InvalidArgument(
          "Device assignment (%s) does not have any local devices.",
          device_assignment->ToString());
    }
  }

  EXLA_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<xla::LocalExecutable>> local_executables,
                        client()->Compile(computation, argument_layouts, options));

  ExlaExecutable* executable = new ExlaExecutable(std::move(local_executables),
                                                  std::move(device_assignment),
                                                  std::move(local_logical_device_ids),
                                                  std::move(local_devices),
                                                  this);
  return executable;
}

xla::StatusOr<ExlaClient*> GetHostClient(int num_replicas,
                                         int intra_op_parallelism_threads) {
  EXLA_ASSIGN_OR_RETURN(se::Platform *platform,
    xla::PlatformUtil::GetPlatform("Host"));

  if (platform->VisibleDeviceCount() <= 0) {
    return xla::FailedPrecondition("CPU platform has no visible devices.");
  }

  xla::LocalClientOptions options;
  options.set_platform(platform);
  options.set_number_of_replicas(num_replicas);
  options.set_intra_op_parallelism_threads(intra_op_parallelism_threads);

  EXLA_ASSIGN_OR_RETURN(xla::LocalClient* client,
    xla::ClientLibrary::GetOrCreateLocalClient(options));

  std::vector<std::unique_ptr<ExlaDevice>> devices;
  int num_devices = client->device_count();
  devices.reserve(num_devices);
  for (int i = 0; i < num_devices; ++i) {
    se::StreamExecutorConfig config;
    config.ordinal = i;
    config.device_options.non_portable_tags["host_thread_stack_size_in_bytes"] = absl::StrCat(8192*1024);

    EXLA_ASSIGN_OR_RETURN(se::StreamExecutor* executor,
      platform->GetExecutor(config));

    auto device = std::make_unique<ExlaDevice>(i, executor, client);
    devices.emplace_back(std::move(device));
  }
  return new ExlaClient(client,
                        /*host_id*/0,
                        /*devices*/std::move(devices),
                        /*allocator*/nullptr,
                        /*host_memory_allocator*/nullptr,
                        /*gpu_run_options*/nullptr);
}

xla::StatusOr<ExlaClient*> GetGpuClient(int num_replicas,
                                        int intra_op_parallelism_threads,
                                        const char* platform_name) {
  EXLA_ASSIGN_OR_RETURN(stream_executor::Platform *platform,
    xla::PlatformUtil::GetPlatform(std::string(platform_name)));

  if (platform->VisibleDeviceCount() <= 0) {
    return xla::FailedPrecondition("%s Platform has no visible devices.",
                                   platform_name);
  }

  xla::LocalClientOptions options;
  options.set_platform(platform);
  options.set_number_of_replicas(num_replicas);
  options.set_intra_op_parallelism_threads(intra_op_parallelism_threads);

  EXLA_ASSIGN_OR_RETURN(xla::LocalClient* client,
    xla::ClientLibrary::GetOrCreateLocalClient(options));

  std::vector<std::unique_ptr<ExlaDevice>> devices;
  int num_devices = client->device_count();
  devices.reserve(num_devices);
  for (int i = 0; i < num_devices; ++i) {
    EXLA_ASSIGN_OR_RETURN(se::StreamExecutor* executor,
      client->backend().stream_executor(i));

    int device_ordinal = executor->device_ordinal();
    devices.emplace_back(std::make_unique<ExlaDevice>(device_ordinal,
                                                    executor,
                                                    client));
  }

  // TODO(seanmor5): Allocator options should be a configuration option.
  EXLA_ASSIGN_OR_RETURN(std::unique_ptr<se::DeviceMemoryAllocator> allocator,
    allocator::GetGpuDeviceAllocator(devices, 0.9, true));

  std::unique_ptr<tensorflow::BFCAllocator> host_memory_allocator =
    allocator::GetGpuHostAllocator(devices.front()->executor());

  auto gpu_run_options = std::make_unique<xla::gpu::GpuExecutableRunOptions>();

  return new ExlaClient(client,
                        /*host_id*/0,
                        /*devices*/std::move(devices),
                        /*allocator*/std::move(allocator),
                        /*host_memory_allcoator*/std::move(host_memory_allocator),
                        /*gpu_run_options*/std::move(gpu_run_options));
}

}  // namespace exla
