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
    if(!empty()) {
      if(zero_copy_) {
        buffer_->release();
        buffer_ = nullptr;
      } else {
        delete buffer_;
        buffer_ = nullptr;
      }
      return tensorflow::Status::OK();
    }
    return tensorflow::errors::Aborted("Attempt to deallocate already deallocated buffer.");
  }

  xla::StatusOr<std::vector<ExlaBuffer*>> ExlaBuffer::DecomposeTuple() {
    if(!is_tuple()) {
      return tensorflow::errors::FailedPrecondition("Buffer is not a Tuple.");
    }

    std::vector<ExlaBuffer*> buffers;
    int64 tuple_elements = xla::ShapeUtil::TupleElementCount(on_device_shape());
    buffers.reserve(tuple_elements);
    for(int i=0;i<tuple_elements;i++) {
      xla::ScopedShapedBuffer* sub_buffer = new xla::ScopedShapedBuffer(std::move(buffer_->TakeSubTree({i})));
      buffers.push_back(new ExlaBuffer(sub_buffer, device_, false));
    }

    return buffers;
  }

  /*
   * ExlaClient Functions
   */
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
      host_memory_allocator_ = std::make_unique<allocator::ExlaErtsAllocator>();
    }

  }

  xla::StatusOr<ERL_NIF_TERM> ExlaClient::DecomposeBuffer(ErlNifEnv* env, ExlaBuffer* buffer) {
    if(!buffer->is_tuple()) {
      return make<ExlaBuffer*>(env, buffer);
    } else {
      EXLA_ASSIGN_OR_RETURN(std::vector<ExlaBuffer*> sub_buffers,
        buffer->DecomposeTuple());

      int num_elems = sub_buffers.size();
      ERL_NIF_TERM terms[num_elems];
      for(int i=0;i<num_elems;i++) {
        EXLA_ASSIGN_OR_RETURN(terms[i], DecomposeBuffer(env, sub_buffers.at(i)));
      }
      return enif_make_list_from_array(env, terms, num_elems);
    }
  }

  ERL_NIF_TERM ErlListFromLiteral(ErlNifEnv* env, xla::Literal& literal) {
    std::vector<xla::Literal> literals = literal.DecomposeTuple();
    int elems = literals.size();
    ERL_NIF_TERM data[elems];

    for(int i=0;i<elems;i++) {
      xla::Literal lit(std::move(literals.at(i)));
      if(lit.shape().IsTuple()) {
        ERL_NIF_TERM term = ErlListFromLiteral(env, lit);
        data[i] = term;
      } else {
        int64 size = lit.size_bytes();
        ErlNifBinary binary;
        enif_alloc_binary(size, &binary);

        // No need to copy, just move to the underlying bytes in memory
        void *src_mem = const_cast<void*>(lit.untyped_data());
        std::memmove(binary.data, src_mem, size);

        ERL_NIF_TERM term = enif_make_binary(env, &binary);
        data[i] = term;
      }
    }
    return enif_make_list_from_array(env, data, elems);
  }

  xla::StatusOr<ERL_NIF_TERM> ExlaClient::ErlListFromBuffer(ErlNifEnv* env, exla::ExlaBuffer* buffer) {
    if(buffer->empty()) {
      return tensorflow::errors::FailedPrecondition("Attempt to read from deallocated buffer.");
    }

    if(!buffer->is_tuple()) {
      return tensorflow::errors::FailedPrecondition("Attempt to extract tuple from non-tuple buffer.");
    }

    xla::TransferManager* transfer_manager = client()->backend().transfer_manager();

    EXLA_ASSIGN_OR_RETURN(xla::Literal literal,
      transfer_manager->TransferLiteralFromDevice(
        buffer->device()->device_to_host_stream(),
        *(buffer->buffer()),
        nullptr)
    );

    ERL_NIF_TERM list = ErlListFromLiteral(env, literal);

    return list;
  }

  xla::StatusOr<ErlNifBinary> ExlaClient::ErlBinFromBuffer(exla::ExlaBuffer* buffer) {
    if(buffer->empty()) {
      return tensorflow::errors::Aborted("Attempt to read from deallocated buffer.");
    }

    bool is_cpu_platform = buffer->device()->executor()->platform()->id() == stream_executor::host::kHostPlatformId;

    if(is_cpu_platform) {
      // Allocate enough space for the binary
      int64 size = xla::ShapeUtil::ByteSizeOf(buffer->on_host_shape());
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
    EXLA_ASSIGN_OR_RETURN(xla::Literal literal,
      transfer_manager->TransferLiteralFromDevice(
        buffer->device()->device_to_host_stream(),
        *(buffer->buffer()),
        nullptr)
    );

    // Allocate enough space for the binary
    long long int size = literal.size_bytes();
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
    xla::TransferManager* transfer_manager = client->client()->backend().transfer_manager();

    EXLA_ASSIGN_OR_RETURN(xla::ScopedShapedBuffer buffer,
      transfer_manager->AllocateScopedShapedBuffer(on_host_shape, client->allocator(), device->id()));

    return buffer;
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

  xla::StatusOr<xla::ScopedShapedBuffer*> TransferBinToBuffer(const ErlNifBinary bin,
                                                              const xla::Shape& shape,
                                                              const xla::Shape& compact_shape,
                                                              ExlaDevice* device,
                                                              ExlaClient* client) {
    // Allocate space on the device
    EXLA_ASSIGN_OR_RETURN(xla::ScopedShapedBuffer device_buffer,
      AllocateDestinationBuffer(compact_shape, device, client));
    // Read directly from binary data into a `BorrowingLiteral`, this is zero-copy again
    xla::BorrowingLiteral literal(const_cast<char*>((char*) bin.data), shape);
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
    if(size != bin.size) {
      return tensorflow::errors::InvalidArgument("Expected %d bytes from binary but got %d.", size, bin.size);
    }
    // Transfer Manager will manager the "transfer" to the device
    xla::TransferManager* transfer_manager = client()->backend().transfer_manager();
    // Ask for shape which has a compact layout on the device, in other words the anticipated shape of the data
    // on the device
    EXLA_ASSIGN_OR_RETURN(xla::Shape compact_shape,
      transfer_manager->ChooseCompactLayoutForShape(shape));

    // Can we use a zero copy transfer?
    bool can_use_zero_copy = CanUseZeroCopy(bin, shape, compact_shape, device);
    if(can_use_zero_copy && transfer_for_run) {
      xla::ScopedShapedBuffer* device_buffer = ZeroCopyTransferBinToBuffer(bin, shape, compact_shape, device, this);
      return new ExlaBuffer(/*buffer=*/device_buffer, /*device=*/device, /*zero_copy=*/true);
    } else {
      EXLA_ASSIGN_OR_RETURN(xla::ScopedShapedBuffer* device_buffer,
        TransferBinToBuffer(bin, shape, compact_shape, device, this));
      return new ExlaBuffer(/*buffer=*/device_buffer, /*device=*/device, /*zero_copy=*/false);
    }
  }

  xla::StatusOr<ERL_NIF_TERM> ExlaClient::Run(ErlNifEnv* env,
                                              xla::LocalExecutable* executable,
                                              ERL_NIF_TERM arguments,
                                              ExlaDevice* device,
                                              xla::ExecutableRunOptions& options,
                                              bool keep_on_device) {

    // Track buffers that need to be released after `Run`
    std::vector<ExlaBuffer**> buffers;
    std::vector<xla::ExecutionInput> inputs;

    ERL_NIF_TERM head, tail, list;
    list = arguments;

    while(enif_get_list_cell(env, list, &head, &tail)) {
      const ERL_NIF_TERM* tuple;
      int arity;
      exla::ExlaBuffer** buffer;

      if(enif_get_tuple(env, head, &arity, &tuple)) {
        ErlNifBinary data;
        xla::Shape* shape;

        if(!get_binary(env, tuple[0], data)) return tensorflow::errors::InvalidArgument("Unable to read binary data from input.");
        if(!get<xla::Shape>(env, tuple[1], shape)) return tensorflow::errors::InvalidArgument("Unable to read shape from input.");

        EXLA_ASSIGN_OR_RETURN(ExlaBuffer* buf, BufferFromErlBin(data, *shape, device, true));

        xla::ExecutionInput inp = xla::ExecutionInput(buf->on_device_shape());

        const xla::ShapeTree<se::DeviceMemoryBase> bufs = buf->buffer()->buffers();

        bufs.ForEachElement(
          [&](const xla::ShapeIndex& index, const se::DeviceMemoryBase& mem){
            inp.SetBuffer(index, xla::MaybeOwningDeviceMemory(mem));
          });

        inputs.push_back(std::move(inp));
        buffers.push_back(&buf);

      } else if(get<ExlaBuffer*>(env, head, buffer)) {

        if(*buffer == NULL) {
          return tensorflow::errors::Aborted("Attempt to re-use a previously deallocated device buffer.");
        }
        xla::ExecutionInput inp = xla::ExecutionInput((*buffer)->on_device_shape());

        const xla::ShapeTree<se::DeviceMemoryBase> bufs = (*buffer)->buffer()->buffers();

        bufs.ForEachElement(
          [&](const xla::ShapeIndex& index, const se::DeviceMemoryBase& mem){
            inp.SetBuffer(index, xla::MaybeOwningDeviceMemory(mem));
          });

        inputs.push_back(std::move(inp));

      } else {
        return tensorflow::errors::InvalidArgument("Invalid input passed to run.");
      }
      list = tail;
    }

    EXLA_ASSIGN_OR_RETURN(xla::ExecutionOutput exec_result, executable->Run(std::move(inputs), options));

    for(auto buf : buffers) {
      if(*buf != NULL) {
        delete *buf;
        *buf = NULL;
      }
    }

    xla::ScopedShapedBuffer result = exec_result.ConsumeResult();

    exla::ExlaBuffer* buffer_ref = new exla::ExlaBuffer(new xla::ScopedShapedBuffer(std::move(result)), device, false);

    if(keep_on_device && buffer_ref->is_tuple()) {
      EXLA_ASSIGN_OR_RETURN_NIF(ERL_NIF_TERM references, DecomposeBuffer(env, buffer_ref), env);
      return references;
    } else if(keep_on_device) {
      return make<exla::ExlaBuffer*>(env, buffer_ref);
    } else if(buffer_ref->is_tuple()) {
      EXLA_ASSIGN_OR_RETURN_NIF(ERL_NIF_TERM tuple, ErlListFromBuffer(env, buffer_ref), env);
      delete buffer_ref;
      return tuple;
    } else {
      EXLA_ASSIGN_OR_RETURN_NIF(ErlNifBinary binary, ErlBinFromBuffer(buffer_ref), env);
      delete buffer_ref;
      return make(env, binary);
    }
  }

  xla::StatusOr<ExlaClient*> getHostClient(int num_replicas, int intra_op_parallelism_threads) {
    EXLA_ASSIGN_OR_RETURN(se::Platform *platform,
      xla::PlatformUtil::GetPlatform("Host"));

    if(platform->VisibleDeviceCount() <= 0){
      return xla::FailedPrecondition("CPU platform has no visible devices.");
    }

    xla::LocalClientOptions options;
    options.set_platform(platform);
    options.set_number_of_replicas(num_replicas);
    options.set_intra_op_parallelism_threads(intra_op_parallelism_threads);

    EXLA_ASSIGN_OR_RETURN(xla::LocalClient* client,
      xla::ClientLibrary::GetOrCreateLocalClient(options));

    std::vector<std::unique_ptr<ExlaDevice>> devices;
    for(int i = 0; i < client->device_count(); ++i) {
      se::StreamExecutorConfig config;
      config.ordinal = i;
      config.device_options.non_portable_tags["host_thread_stack_size_in_bytes"] = absl::StrCat(8192*1024);

      EXLA_ASSIGN_OR_RETURN(se::StreamExecutor* executor,
        platform->GetExecutor(config));

      auto device = absl::make_unique<ExlaDevice>(i, executor, client);
      devices.push_back(std::move(device));
    }
    return new ExlaClient(client, /*host_id*/0,
                          /*devices*/std::move(devices),
                          /*allocator*/nullptr,
                          /*host_memory_allocator*/nullptr,
                          /*gpu_run_options*/nullptr);
  }

  xla::StatusOr<ExlaClient*> getCUDAClient(int num_replicas, int intra_op_parallelism_threads) {
    EXLA_ASSIGN_OR_RETURN(stream_executor::Platform *platform,
      xla::PlatformUtil::GetPlatform("CUDA"));

    if(platform->VisibleDeviceCount() <= 0){
      return xla::FailedPrecondition("CUDA Platform has no visible devices.");
    }

    xla::LocalClientOptions options;
    options.set_platform(platform);
    options.set_number_of_replicas(num_replicas);
    options.set_intra_op_parallelism_threads(intra_op_parallelism_threads);

    EXLA_ASSIGN_OR_RETURN(xla::LocalClient* client,
      xla::ClientLibrary::GetOrCreateLocalClient(options));

    std::vector<std::unique_ptr<ExlaDevice>> devices;
    for(int i = 0; i < client->device_count(); ++i) {
      EXLA_ASSIGN_OR_RETURN(se::StreamExecutor* executor,
        client->backend().stream_executor(i));

      int device_ordinal = executor->device_ordinal();
      devices.push_back(absl::make_unique<ExlaDevice>(device_ordinal, executor, client));
    }

    // TODO: Allocator options should be a configuration option.
    EXLA_ASSIGN_OR_RETURN(std::unique_ptr<se::DeviceMemoryAllocator> allocator,
      allocator::GetGpuDeviceAllocator(devices, 0.9, true));

    std::unique_ptr<tensorflow::BFCAllocator> host_memory_allocator = allocator::GetGpuHostAllocator(devices.front()->executor());

    std::unique_ptr<xla::GpuExecutableRunOptions> gpu_run_options = absl::make_unique<xla::GpuExecutableRunOptions>();

    return new ExlaClient(client, /*host_id*/0,
                          /*devices*/std::move(devices),
                          /*allocator*/std::move(allocator),
                          /*host_memory_allcoator*/std::move(host_memory_allocator),
                          /*gpu_run_options*/std::move(gpu_run_options));
  }
}
