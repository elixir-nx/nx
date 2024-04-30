#include "exla_client.h"

#include "exla_nif_util.h"
#include "xla/layout_util.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"

namespace exla {

ExlaBuffer::ExlaBuffer(std::unique_ptr<xla::PjRtBuffer> buffer) : buffer_(std::move(buffer)) {}

void CopyLiteralToBinary(xla::Literal* literal, ErlNifBinary* binary, exla::int64 size) {
  exla::int64 actual_size = literal->size_bytes();
  if (size < 0 or size > actual_size) size = actual_size;
  enif_alloc_binary(size, binary);
  std::memcpy(binary->data, literal->untyped_data(), size);
}

xla::StatusOr<ERL_NIF_TERM> ExlaBuffer::ToBinary(ErlNifEnv* env, exla::int64 size) {
  EXLA_ASSIGN_OR_RETURN(std::shared_ptr<xla::Literal> literal, buffer_->ToLiteralSync());
  ErlNifBinary binary;
  CopyLiteralToBinary(literal.get(), &binary, size);
  return nif::make(env, binary);
}

xla::Status ExlaBuffer::Deallocate() {
  if (buffer_->IsDeleted()) {
    return xla::FailedPrecondition("Attempt to deallocate already deallocated buffer.");
  } else {
    buffer_->Delete();
    return tsl::OkStatus();
  }
}

xla::StatusOr<ExlaBuffer*> ExlaBuffer::CopyToDevice(xla::PjRtDevice* dst_device) {
  EXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtBuffer> buf,
                        buffer_->CopyToDevice(dst_device));
  return new ExlaBuffer(std::move(buf));
}

ExlaExecutable::ExlaExecutable(std::unique_ptr<xla::PjRtLoadedExecutable> executable,
                               absl::optional<std::string> fingerprint,
                               ExlaClient* client) : executable_(std::move(executable)),
                                                     fingerprint_(std::move(fingerprint)),
                                                     client_(client) {}

xla::StatusOr<std::unique_ptr<xla::PjRtBuffer>> PjRtBufferFromBinary(xla::PjRtClient* client,
                                                                     ErlNifEnv* env,
                                                                     ERL_NIF_TERM source_term,
                                                                     const xla::Shape& shape,
                                                                     int device_id) {
  ErlNifEnv* copy_env = enif_alloc_env();
  ERL_NIF_TERM dest_term = enif_make_copy(copy_env, source_term);
  ErlNifBinary binary;

  if (!nif::get_binary(copy_env, dest_term, &binary)) {
    return xla::InvalidArgument("Expected buffer to be binary.");
  }

  xla::PjRtClient::HostBufferSemantics semantics = xla::PjRtClient::HostBufferSemantics::kZeroCopy;
  std::function<void()> on_done_with_host_buffer = [copy_env]() { enif_free_env(copy_env); };

  EXLA_ASSIGN_OR_RETURN(xla::PjRtDevice * device, client->LookupDevice(device_id));
  EXLA_ASSIGN_OR_RETURN(auto buffer, client->BufferFromHostBuffer(
                                         binary.data, shape.element_type(), shape.dimensions(), std::nullopt, semantics, on_done_with_host_buffer, device));

  return std::move(buffer);
}

xla::StatusOr<std::vector<xla::PjRtBuffer*>>
UnpackReplicaArguments(ErlNifEnv* env,
                       ERL_NIF_TERM replica_arguments,
                       ExlaClient* client,
                       int device) {
  unsigned int length;
  if (!enif_get_list_length(env, replica_arguments, &length)) {
    return xla::InvalidArgument("Argument is not a list.");
  }

  ERL_NIF_TERM head, tail;
  std::vector<xla::PjRtBuffer*> replica_buffers;
  replica_buffers.reserve(length);

  // for a single replica, the argument is a flat list of buffers where
  // each buffer can either be an erlang binary or a reference to another
  // EXLA buffer, it is not possible for any of the arguments to be nested
  // tuples because we handle normalization/flattening of tuples on the
  // Elixir side
  while (enif_get_list_cell(env, replica_arguments, &head, &tail)) {
    int arity;
    const ERL_NIF_TERM* tuple;
    ExlaBuffer** buffer;

    if (enif_get_tuple(env, head, &arity, &tuple)) {
      // if the term is a tuple, that means it represents a {shape, binary}
      // tuple which we must convert into an exla buffer for use in the computation
      xla::Shape* shape;

      if (!nif::get<xla::Shape>(env, tuple[1], shape)) {
        return xla::InvalidArgument("Expected argument to be shape reference.");
      }

      // we convert the binary into a buffer and transfer it to the correct device,
      // this buffer is not managed by the erlang vm so it must be deallocated explicitly
      // after use by the execution
      EXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtBuffer> buf,
                            PjRtBufferFromBinary(client->client(), env, tuple[0], *shape, device));
      replica_buffers.push_back(buf.release());
    } else if (nif::get<ExlaBuffer*>(env, head, buffer)) {
      // if the buffer is not a tuple it must be a reference to an exla buffer
      // which means the resource is already managed by the vm, and should already
      // be on the correct device, if it is not, we will not do any implicit transfers
      // and instead raise an error
      if ((*buffer)->device_id() != device) {
        return xla::InvalidArgument("Expected buffer to be placed on device %d", device);
      }
      replica_buffers.push_back((*buffer)->buffer());
    } else {
      return xla::InvalidArgument("Expected argument to be buffer reference.");
    }

    replica_arguments = tail;
  }

  return replica_buffers;
}

xla::StatusOr<std::vector<std::vector<xla::PjRtBuffer*>>>
UnpackRunArguments(ErlNifEnv* env,
                   ERL_NIF_TERM arguments,
                   ExlaClient* client,
                   xla::DeviceAssignment device_assignment,
                   int device_id) {
  unsigned int length;
  if (!enif_get_list_length(env, arguments, &length)) {
    return xla::InvalidArgument("Argument is not a list.");
  }

  ERL_NIF_TERM head, tail;
  std::vector<std::vector<xla::PjRtBuffer*>> arg_buffers;
  arg_buffers.reserve(length);

  int replica = 0;
  int device;

  while (enif_get_list_cell(env, arguments, &head, &tail)) {
    device = device_id >= 0 ? device_id : device_assignment(replica, 0);

    EXLA_ASSIGN_OR_RETURN(std::vector<xla::PjRtBuffer*> replica_buffers,
                          UnpackReplicaArguments(env, head, client, device));

    arg_buffers.push_back(replica_buffers);
    replica++;
    arguments = tail;
  }

  return arg_buffers;
}

xla::StatusOr<ERL_NIF_TERM> UnpackResult(ErlNifEnv* env,
                                         std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> result,
                                         xla::DeviceAssignment device_assignment,
                                         int device_id) {
  std::vector<ERL_NIF_TERM> per_replica_results;

  for (int i = 0; i < result.size(); i++) {
    std::vector<ERL_NIF_TERM> terms;
    terms.reserve(result.size());
    int device = device_id >= 0 ? device_id : device_assignment(i, 0);

    for (auto& pjrt_buf : result.at(i)) {
      pjrt_buf->BlockHostUntilReady();
      ExlaBuffer* buf = new ExlaBuffer(std::move(pjrt_buf));
      ERL_NIF_TERM term = nif::make<ExlaBuffer*>(env, buf);
      terms.push_back(term);
    }

    ERL_NIF_TERM replica_term = enif_make_int(env, device);
    ERL_NIF_TERM replica_results = enif_make_list_from_array(env, terms.data(), terms.size());
    per_replica_results.push_back(enif_make_tuple2(env, replica_results, replica_term));
  }

  ERL_NIF_TERM per_replica_term = enif_make_list_from_array(env, per_replica_results.data(), per_replica_results.size());

  return nif::ok(env, per_replica_term);
}

void FreeReplicaArguments(ErlNifEnv* env, ERL_NIF_TERM replica_arguments, std::vector<xla::PjRtBuffer*> buffers) {
  unsigned int length;
  if (!enif_get_list_length(env, replica_arguments, &length)) {
    return;
  }

  ERL_NIF_TERM head, tail;
  int arg = 0;

  while (enif_get_list_cell(env, replica_arguments, &head, &tail)) {
    xla::PjRtBuffer* buffer = buffers.at(arg);

    if (enif_is_tuple(env, head)) {
      delete buffer;
    }

    arg++;
    replica_arguments = tail;
  }
}

void FreeRunArguments(ErlNifEnv* env, ERL_NIF_TERM arguments, std::vector<std::vector<xla::PjRtBuffer*>> buffers) {
  unsigned int length;
  if (!enif_get_list_length(env, arguments, &length)) {
    return;
  }

  ERL_NIF_TERM head, tail;
  int replica = 0;

  while (enif_get_list_cell(env, arguments, &head, &tail)) {
    FreeReplicaArguments(env, head, buffers.at(replica));
    arguments = tail;
    replica++;
  }
}

xla::StatusOr<ERL_NIF_TERM> ExlaExecutable::Run(ErlNifEnv* env,
                                                ERL_NIF_TERM arguments,
                                                int device_id) {
  xla::ExecuteOptions options;
  // arguments are not passed as a single PjRt tuple buffer, but instead
  // as multiple pjrt buffers
  options.arguments_are_tupled = false;
  // result is a tuple, which pjrt decomposes into a vector of buffers for
  // us to handle ourselves
  options.untuple_result = true;
  // we do not handle multi-device launches at this time, so this must always
  // be set to 0
  options.launch_id = 0;
  // disable strict shape checking which ensures shapes of buffers match exact
  // shape (with layout) expected be compiled executable, we have mismatches
  // on gpu
  options.strict_shape_checking = false;
  // execution mode determines whether or not to launch the executable in the
  // calling thread or in a separate thread, default mode is either-or, here
  // we specify synchronous because the Elixir side ensures execution is always
  // synchronous
  options.execution_mode = xla::ExecuteOptions::ExecutionMode::kSynchronous;

  // the number of replicas will equal the number of devices involved in
  // a pmap, but in all other cases it will be equal to 1
  int num_replicas = executable_->num_replicas();

  // input buffers are a list of lists, where each list maps to the args
  // to pass to one of the replicas in a computation, e.g. [replica_args1, replica_args2, ...]
  std::vector<std::vector<xla::PjRtBuffer*>> input_buffers;

  // the device assignment is a 2d array which maps coordinates (replica, partition)
  // to a device; or in this case just maps a replica to a device
  xla::DeviceAssignment device_assignment;
  if (client_->client()->platform_name() == "METAL") {
    device_assignment = xla::DeviceAssignment(1, 1);
  } else {
    EXLA_ASSIGN_OR_RETURN(device_assignment,
                          client_->client()->GetDefaultDeviceAssignment(num_replicas, 1));
  }

  if (device_id >= 0 && num_replicas > 1) {
    // if the device id is greater than or equal to 1, that means we've specified
    // a portable executable which cannot be pmapped, this code path should never
    // be reached as it should be controlled from Elixir
    return xla::InvalidArgument("Cannot specify a device for replicated executable.");
  } else {
    // else we handle unpacking/validating the run arguments to the correct devices
    // according to the device id and the device assignment
    EXLA_ASSIGN_OR_RETURN(input_buffers, UnpackRunArguments(env, arguments, client_, device_assignment, device_id));
  }

  // at this point input buffers is a vector of arguments per replica
  // and the size of that vector should equal the number of replicas in the
  // executable, otherwise it is invalid
  if (num_replicas != input_buffers.size()) {
    return xla::InvalidArgument("Got %d replica arguments for %d replicas", input_buffers.size(), num_replicas);
  }

  std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> per_replica_results;

  if (device_id >= 0) {
    // if we specified a device id, then we need to execute the executable as a portable
    // executable, meaning we need to find the device corresponding to the specific device
    // id and execute on that device, we've already guaranteed this executable only has 1
    // replica
    EXLA_ASSIGN_OR_RETURN(xla::PjRtDevice * device, client_->client()->LookupDevice(device_id));
    // because this is a portable executable, it only has 1 replica and so we only need
    // to get the arguments at the first position of the input buffers
    std::vector<xla::PjRtBuffer*> portable_args = input_buffers.at(0);
    EXLA_ASSIGN_OR_RETURN(auto portable_result,
                          executable_->ExecutePortable(portable_args, device, options));
    // the logic for handling unpacking of results is shared between portable code path
    // and the replicated code-path, so we take ownership of the result buffers to unpack
    per_replica_results.push_back(std::move(portable_result));
  } else {
    // no device ID is present, so it may be a replicated executable which means we need
    // to use the replica execution path
    // TODO: This now exposes a `returned_futures` API, does this make sense for us?
    EXLA_ASSIGN_OR_RETURN(per_replica_results, executable_->Execute(input_buffers, options));
  }

  // EXLA_ASSIGN_OR_RETURN(per_replica_results, executable_->Execute(input_buffers, options));

  // sanity check
  if (per_replica_results.size() != num_replicas) {
    return xla::FailedPrecondition("Invalid execution.");
  }

  // we need to unpack the results into Erlang terms, the result is a vector
  // of vectors of unique ptrs to pjrt buffers, where the size of the output equals
  // the number of replicas and each individual replica is a vector of buffers, the
  // inner buffer represents a flattened output because we told PjRt we would always
  // return a tuple from the computation
  EXLA_ASSIGN_OR_RETURN(ERL_NIF_TERM ret,
                        UnpackResult(env, std::move(per_replica_results), device_assignment, device_id));

  // finally, we need to free any of the arguments we created for this computation
  FreeRunArguments(env, arguments, input_buffers);

  return ret;
}

ExlaClient::ExlaClient(std::shared_ptr<xla::PjRtClient> client) : client_(std::move(client)) {}

xla::StatusOr<ExlaBuffer*> ExlaClient::BufferFromBinary(ErlNifEnv* env,
                                                        ERL_NIF_TERM source_term,
                                                        xla::Shape& shape,
                                                        int device_id) {
  EXLA_ASSIGN_OR_RETURN(auto buffer, PjRtBufferFromBinary(client(), env, source_term, shape, device_id));
  ExlaBuffer* exla_buffer = new ExlaBuffer(std::move(buffer));
  return exla_buffer;
}

xla::StatusOr<std::optional<std::string>> ExecutableFingerprint(std::unique_ptr<xla::PjRtLoadedExecutable>& executable) {
  auto fingerprint = executable->FingerprintExecutable();

  if (fingerprint.ok()) {
    return {fingerprint.value()};
  } else if (fingerprint.status().code() == absl::StatusCode::kUnimplemented) {
    // Return nullopt in case of unimplemented error.
    return std::nullopt;
  } else {
    return fingerprint.status();
  }
}

xla::StatusOr<ExlaExecutable*> ExlaClient::DeserializeExecutable(std::string deserialized_executable) {
  EXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtLoadedExecutable> executable,
                        client_->DeserializeExecutable(deserialized_executable, std::nullopt));

  EXLA_ASSIGN_OR_RETURN(absl::optional<std::string> fingerprint,
                        ExecutableFingerprint(executable));

  return new ExlaExecutable(std::move(executable), std::move(fingerprint), this);
}

xla::StatusOr<ExlaExecutable*> ExlaClient::Compile(const mlir::OwningOpRef<mlir::ModuleOp>& module,
                                                   std::vector<xla::Shape*> argument_layouts,
                                                   xla::ExecutableBuildOptions& options,
                                                   bool compile_portable_executable) {
  std::vector<xla::Shape> layouts;
  layouts.reserve(argument_layouts.size());
  for (auto shape : argument_layouts) {
    xla::Shape cpy_shape = xla::ShapeUtil::MakeShape(shape->element_type(), shape->dimensions());
    xla::LayoutUtil::ClearLayout(&cpy_shape);
    layouts.push_back(cpy_shape);
  }

  xla::CompileOptions compile_opts;
  compile_opts.argument_layouts = layouts;
  compile_opts.parameter_is_tupled_arguments = false;
  compile_opts.executable_build_options = options;
  compile_opts.compile_portable_executable = compile_portable_executable;

  EXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtLoadedExecutable> executable,
                        client_->Compile(*module, std::move(compile_opts)));
  EXLA_ASSIGN_OR_RETURN(absl::optional<std::string> fingerprint,
                        ExecutableFingerprint(executable));

  return new ExlaExecutable(std::move(executable), std::move(fingerprint), this);
}

xla::Status ExlaClient::TransferToInfeed(ErlNifEnv* env,
                                         ERL_NIF_TERM data,
                                         const xla::Shape& shape,
                                         int device_id) {
  // Fast path to avoid any traversal when not sending Tuples
  ERL_NIF_TERM head, tail;
  if (!enif_get_list_cell(env, data, &head, &tail)) {
    return xla::InvalidArgument("infeed operation expects a list of binaries");
  }

  ErlNifBinary binary;
  if (!nif::get_binary(env, head, &binary)) {
    return xla::InvalidArgument("infeed operation expects a list of binaries");
  }

  const char* data_ptr = const_cast<char*>(reinterpret_cast<char*>(binary.data));
  xla::BorrowingLiteral literal(data_ptr, shape);

  EXLA_ASSIGN_OR_RETURN(xla::PjRtDevice * device, client_->LookupDevice(device_id));

  return device->TransferToInfeed(literal);
}

xla::StatusOr<ERL_NIF_TERM> ExlaClient::TransferFromOutfeed(ErlNifEnv* env, int device_id, xla::Shape& shape) {
  EXLA_ASSIGN_OR_RETURN(xla::PjRtDevice * device, client_->LookupDevice(device_id));

  auto literal = std::make_shared<xla::Literal>(shape);

  xla::Status transfer_status = device->TransferFromOutfeed(literal.get());

  if (!transfer_status.ok()) {
    return transfer_status;
  }

  ErlNifBinary binary;
  enif_alloc_binary(literal->size_bytes(), &binary);
  std::memcpy(binary.data, literal->untyped_data(), literal->size_bytes());

  return nif::make(env, binary);
}

xla::StatusOr<ExlaClient*> GetHostClient() {
  EXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                        xla::GetTfrtCpuClient(false));

  return new ExlaClient(std::move(client));
}

xla::StatusOr<ExlaClient*> GetGpuClient(double memory_fraction,
                                        bool preallocate,
                                        xla::GpuAllocatorConfig::Kind kind) {
  xla::GpuAllocatorConfig allocator_config = {
      .kind = kind,
      .memory_fraction = memory_fraction,
      .preallocate = preallocate};

  EXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                        xla::GetStreamExecutorGpuClient(false, allocator_config, 0));

  return new ExlaClient(std::move(client));
}

xla::StatusOr<ExlaClient*> GetTpuClient() {
  EXLA_EFFECT_OR_RETURN(pjrt::LoadPjrtPlugin("tpu", "libtpu.so"));

  xla::Status status = pjrt::InitializePjrtPlugin("tpu");

  if (!status.ok()) {
    return status;
  }

  EXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                        xla::GetCApiClient("TPU"));

  return new ExlaClient(std::move(client));
}

xla::StatusOr<ExlaClient*> GetCApiClient(std::string device_type) {
  EXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                        xla::GetCApiClient(device_type));

  return new ExlaClient(std::move(client));
}
}  // namespace exla
