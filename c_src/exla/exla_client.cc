#include "tensorflow/compiler/xla/exla/exla_client.h"
#include "tensorflow/compiler/xla/exla/exla_allocator.h"

#include "tensorflow/compiler/xla/cpu_function_runtime.h"
#include "tensorflow/compiler/xla/literal_util.h"

#include "absl/memory/memory.h"

namespace exla {

using int64 = tensorflow::int64;

/*
 * ExlaBuffer Functions
 */
xla::Status ExlaBuffer::Deallocate() {
  if (!empty()) {
    if (zero_copy_) {
      buffer_->release();
      buffer_ = nullptr;
    } else {
      delete buffer_;
      buffer_ = nullptr;
    }
    return tensorflow::Status::OK();
  }
  return xla::FailedPrecondition("Attempt to deallocate already deallocated buffer.");
}

xla::StatusOr<std::vector<ExlaBuffer*>> ExlaBuffer::DecomposeTuple() {
  if (!is_tuple()) {
    return xla::FailedPrecondition("Buffer is not a Tuple.");
  }

  std::vector<ExlaBuffer*> buffers;
  int64 tuple_elements = xla::ShapeUtil::TupleElementCount(on_device_shape());
  buffers.reserve(tuple_elements);
  for (int i=0; i < tuple_elements; i++) {
    xla::ScopedShapedBuffer* sub_buffer =
      new xla::ScopedShapedBuffer(std::move(buffer_->TakeSubTree({i})));
    buffers.emplace_back(new ExlaBuffer(sub_buffer, device_, false));
  }

  return buffers;
}
/*
 * ExlaExecutable Functions
 */
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

xla::StatusOr<xla::ExecutionOutput> ExlaExecutable::Run(ErlNifEnv* env,
                                                        ERL_NIF_TERM arguments,
                                                        std::vector<ExlaBuffer**>& buffers,
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

  // Track buffers that need to be released after `Run`
  std::vector<xla::ExecutionInput> inputs;

  ERL_NIF_TERM head, tail, list;
  list = arguments;

  while (enif_get_list_cell(env, list, &head, &tail)) {
    const ERL_NIF_TERM* tuple;
    int arity;
    exla::ExlaBuffer** buffer;

    if (enif_get_tuple(env, head, &arity, &tuple)) {
      ErlNifBinary data;
      xla::Shape* shape;

      if (!get_binary(env, tuple[0], &data)) {
        return xla::InvalidArgument("Unable to read binary data from input.");
      }
      if (!get<xla::Shape>(env, tuple[1], shape)) {
        return xla::InvalidArgument("Unable to read shape from input.");
      }

      EXLA_ASSIGN_OR_RETURN(ExlaBuffer* buf,
        client_->BufferFromErlBin(data, *shape, device, true));

      xla::ExecutionInput inp = xla::ExecutionInput(buf->on_device_shape());

      const xla::ShapeTree<se::DeviceMemoryBase> bufs =
        buf->buffer()->buffers();

      bufs.ForEachElement(
        [&](const xla::ShapeIndex& index, const se::DeviceMemoryBase& mem){
          inp.SetBuffer(index, xla::MaybeOwningDeviceMemory(mem));
        });

      inputs.push_back(std::move(inp));
      buffers.push_back(&buf);

    } else if (get<ExlaBuffer*>(env, head, buffer)) {
      if (*buffer == NULL) {
        return xla::FailedPrecondition("Attempt to re-use a previously deallocated device buffer.");
      }
      xla::ExecutionInput inp = xla::ExecutionInput((*buffer)->on_device_shape());

      const xla::ShapeTree<se::DeviceMemoryBase> bufs = (*buffer)->buffer()->buffers();

      bufs.ForEachElement(
        [&](const xla::ShapeIndex& index, const se::DeviceMemoryBase& mem){
          inp.SetBuffer(index, xla::MaybeOwningDeviceMemory(mem));
        });

      inputs.push_back(std::move(inp));
    } else {
      return xla::InvalidArgument("Invalid input passed to run.");
    }
    list = tail;
  }

  xla::StatusOr<xla::ExecutionOutput> exec_result_status =
    executable->RunAsync(std::move(inputs), run_options);

  return exec_result_status;
}

/*
 * ExlaClient Functions
 */
ExlaClient::ExlaClient(xla::LocalClient* client,
                       int host_id,
                       std::vector<std::unique_ptr<ExlaDevice>> devices,
                       std::unique_ptr<se::DeviceMemoryAllocator> allocator,
                       std::unique_ptr<tensorflow::Allocator> host_memory_allocator,
                       std::unique_ptr<xla::GpuExecutableRunOptions> gpu_run_options)
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

xla::StatusOr<xla::DeviceAssignment> ExlaClient::GetDefaultDeviceAssignment(int num_replicas, int num_partitions) {
  return client_->backend().computation_placer()->AssignDevices(num_replicas, num_partitions);
}

xla::StatusOr<ERL_NIF_TERM> ExlaClient::DecomposeBuffer(ErlNifEnv* env,
                                                        ExlaBuffer* buffer) {
  if (!buffer->is_tuple()) {
    return make<ExlaBuffer*>(env, buffer);
  } else {
    EXLA_ASSIGN_OR_RETURN(std::vector<ExlaBuffer*> sub_buffers,
      buffer->DecomposeTuple());

    int num_elems = sub_buffers.size();
    std::vector<ERL_NIF_TERM> terms;
    terms.reserve(num_elems);
    for (int i=0; i < num_elems; i++) {
      EXLA_ASSIGN_OR_RETURN(ERL_NIF_TERM term_at, DecomposeBuffer(env, sub_buffers.at(i)));
      terms.emplace_back(term_at);
    }
    return enif_make_list_from_array(env, &terms[0], num_elems);
  }
}

ERL_NIF_TERM ErlListFromLiteral(ErlNifEnv* env, xla::Literal& literal) {
  std::vector<xla::Literal> literals = literal.DecomposeTuple();
  int elems = literals.size();
  std::vector<ERL_NIF_TERM> data;
  data.reserve(elems);

  for (int i=0; i < elems; i++) {
    xla::Literal lit(std::move(literals.at(i)));
    if (lit.shape().IsTuple()) {
      ERL_NIF_TERM term = ErlListFromLiteral(env, lit);
      data.emplace_back(term);
    } else {
      int64 size = lit.size_bytes();
      ErlNifBinary binary;
      enif_alloc_binary(size, &binary);

      // No need to copy, just move to the underlying bytes in memory
      void *src_mem = const_cast<void*>(lit.untyped_data());
      std::memmove(binary.data, src_mem, size);

      ERL_NIF_TERM term = enif_make_binary(env, &binary);
      data.emplace_back(term);
    }
  }
  return enif_make_list_from_array(env, &data[0], elems);
}

xla::StatusOr<ERL_NIF_TERM> ExlaClient::ErlListFromBuffer(ErlNifEnv* env,
                                                          ExlaBuffer* buffer) {
  if (buffer->empty()) {
    return xla::FailedPrecondition("Attempt to read from deallocated buffer.");
  }

  if (!buffer->is_tuple()) {
    return xla::FailedPrecondition("Attempt to extract tuple from non-tuple buffer.");
  }

  xla::TransferManager* transfer_manager =
    client()->backend().transfer_manager();

  EXLA_ASSIGN_OR_RETURN(xla::Literal literal,
    transfer_manager->TransferLiteralFromDevice(
      buffer->device()->device_to_host_stream(),
      *(buffer->buffer()),
      nullptr));

  ERL_NIF_TERM list = ErlListFromLiteral(env, literal);

  return list;
}

xla::StatusOr<ErlNifBinary> ExlaClient::ErlBinFromBuffer(ExlaBuffer* buffer) {
  if (buffer->empty()) {
    return xla::FailedPrecondition("Attempt to read from deallocated buffer.");
  }

  bool is_cpu_platform =
    (buffer->device()->executor()->platform()->id() ==
      stream_executor::host::kHostPlatformId);

  if (is_cpu_platform) {
    // Allocate enough space for the binary
    int64 size = xla::ShapeUtil::ByteSizeOf(buffer->on_host_shape());
    ErlNifBinary binary;
    enif_alloc_binary(size, &binary);

    // Get the result buffer
    const stream_executor::DeviceMemoryBase mem_buffer =
      buffer->buffer()->root_buffer();

    // No need to copy, just move the underlying bytes in memory
    void* src_mem = const_cast<void *>(mem_buffer.opaque());
    std::memmove(binary.data, src_mem, size);

    return binary;
  }

  // Otherwise we have to do the transfer
  xla::TransferManager* transfer_manager =
    client()->backend().transfer_manager();

  EXLA_ASSIGN_OR_RETURN(xla::Literal literal,
    transfer_manager->TransferLiteralFromDevice(
      buffer->device()->device_to_host_stream(),
      *(buffer->buffer()),
      nullptr));

  // Allocate enough space for the binary
  int64 size = literal.size_bytes();
  ErlNifBinary binary;
  enif_alloc_binary(size, &binary);

  // No need to copy, just move to the underlying bytes in memory
  void *src_mem = const_cast<void*>(literal.untyped_data());
  std::memmove(binary.data, src_mem, size);

  return binary;
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

bool CanUseZeroCopy(ErlNifBinary bin,
                    const xla::Shape& shape,
                    const xla::Shape& compact_shape,
                    ExlaDevice* device) {
  // Only possible on CPUs
  bool is_cpu_platform =
    device->executor()->platform()->id() == se::host::kHostPlatformId;
  // With well-aligned data
  bool is_well_aligned =
    (absl::bit_cast<std::uintptr_t>(bin.data) &
      (xla::cpu_function_runtime::kMinAlign - 1)) == 0;
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
  xla::ScopedShapedBuffer* device_buffer =
    new xla::ScopedShapedBuffer(compact_shape, client->allocator(), device->id());
  // Tell it to point to the buffer we made above
  auto memory = se::OwningDeviceMemory(buffer, device->id(), client->allocator());
  device_buffer->set_buffer(std::move(memory), {});
  return device_buffer;
}

xla::StatusOr<xla::ScopedShapedBuffer*> TransferBinToBuffer(const ErlNifBinary bin,
                                                            const xla::Shape& shape,
                                                            const xla::Shape& compact_shape,
                                                            ExlaDevice* device,
                                                            ExlaClient* client) {
  // Allocate space on the device
  EXLA_ASSIGN_OR_RETURN(xla::ScopedShapedBuffer device_buffer,
    AllocateDestinationBuffer(compact_shape, device, client));
  // Read directly from binary data into a `BorrowingLiteral`, this is zero-copy again
  xla::BorrowingLiteral literal(const_cast<char*>(reinterpret_cast<char*>(bin.data)), shape);
  // Transfer literal to the device in the allocated buffer
  client->client()->backend().transfer_manager()->TransferLiteralToDevice(device->host_to_device_stream(), literal, device_buffer);
  return new xla::ScopedShapedBuffer(std::move(device_buffer));
}

xla::StatusOr<ExlaBuffer*> ExlaClient::BufferFromErlBin(const ErlNifBinary bin,
                                                        const xla::Shape& shape,
                                                        ExlaDevice* device,
                                                        bool transfer_for_run) {
  // Get the expected size of the given shape
  int64 size = xla::ShapeUtil::ByteSizeOf(shape);
  // Validate the expected size and actual data size are the same
  // If they are not, we need to return an error because otherwise we'll be trying to read
  // from invalid memory
  if (size != bin.size) {
    return xla::InvalidArgument("Expected %d bytes from binary but got %d.", size, bin.size);
  }
  // Transfer Manager will manager the "transfer" to the device
  xla::TransferManager* transfer_manager =
    client()->backend().transfer_manager();
  // Ask for shape which has a compact layout on the device, in other words the anticipated shape of the data
  // on the device
  EXLA_ASSIGN_OR_RETURN(xla::Shape compact_shape,
    transfer_manager->ChooseCompactLayoutForShape(shape));

  // Can we use a zero copy transfer?
  bool can_use_zero_copy = CanUseZeroCopy(bin, shape, compact_shape, device);
  if (can_use_zero_copy && transfer_for_run) {
    xla::ScopedShapedBuffer* device_buffer =
      ZeroCopyTransferBinToBuffer(bin, shape, compact_shape, device, this);
    return new ExlaBuffer(/*buffer=*/device_buffer,
                          /*device=*/device,
                          /*zero_copy=*/true);
  } else {
    EXLA_ASSIGN_OR_RETURN(xla::ScopedShapedBuffer* device_buffer,
      TransferBinToBuffer(bin, shape, compact_shape, device, this));
    return new ExlaBuffer(/*buffer=*/device_buffer,
                          /*device=*/device,
                          /*zero_copy=*/false);
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

    auto device = absl::make_unique<ExlaDevice>(i, executor, client);
    devices.emplace_back(std::move(device));
  }
  return new ExlaClient(client, /*host_id*/0,
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
    devices.emplace_back(absl::make_unique<ExlaDevice>(device_ordinal,
                                                    executor,
                                                    client));
  }

  // TODO(seanmor5): Allocator options should be a configuration option.
  EXLA_ASSIGN_OR_RETURN(std::unique_ptr<se::DeviceMemoryAllocator> allocator,
    allocator::GetGpuDeviceAllocator(devices, 0.9, true));

  std::unique_ptr<tensorflow::BFCAllocator> host_memory_allocator =
    allocator::GetGpuHostAllocator(devices.front()->executor());

  auto gpu_run_options = absl::make_unique<xla::GpuExecutableRunOptions>();

  return new ExlaClient(client, /*host_id*/0,
                        /*devices*/std::move(devices),
                        /*allocator*/std::move(allocator),
                        /*host_memory_allcoator*/std::move(host_memory_allocator),
                        /*gpu_run_options*/std::move(gpu_run_options));
}

}  // namespace exla
