#include "exla_client.h"

#include <fine.hpp>
#include <erl_nif.h>
#include "exla_nif_util.h"
#include "xla/layout_util.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "xla/shape_util.h"
#include <memory>

namespace exla {

ExlaBuffer::ExlaBuffer(std::unique_ptr<xla::PjRtBuffer> buffer) : buffer_(std::move(buffer)) {}

void CopyLiteralToBinary(xla::Literal* literal, ErlNifBinary* binary, exla::int64 size) {
  exla::int64 actual_size = literal->size_bytes();
  if (size < 0 or size > actual_size) size = actual_size;
  enif_alloc_binary(size, binary);
  std::memcpy(binary->data, literal->untyped_data(), size);
}

tsl::StatusOr<ERL_NIF_TERM> ExlaBuffer::ToBinary(ErlNifEnv* env, exla::int64 size) {
  EXLA_ASSIGN_OR_RETURN(std::shared_ptr<xla::Literal> literal, buffer_->ToLiteralSync());

  exla::int64 actual_size = literal->size_bytes();
  if (size < 0 or size > actual_size) size = actual_size;

  ERL_NIF_TERM binary_term;
  auto data = enif_make_new_binary(env, size, &binary_term);
  memcpy(data, literal->untyped_data(), size);

  return binary_term;
}

tsl::Status ExlaBuffer::Deallocate() {
  if (buffer_->IsDeleted()) {
    return xla::FailedPrecondition("Attempt to deallocate already deallocated buffer.");
  } else {
    buffer_->Delete();
    return tsl::OkStatus();
  }
}

tsl::StatusOr<fine::ResourcePtr<ExlaBuffer>> ExlaBuffer::CopyToDevice(xla::PjRtDevice* dst_device) {
  EXLA_ASSIGN_OR_RETURN(auto memory_space,
                        dst_device->default_memory_space());
  EXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtBuffer> buf,
                        buffer_->CopyToMemorySpace(memory_space));
  return fine::make_resource<ExlaBuffer>(std::move(buf));
}

ExlaExecutable::ExlaExecutable(std::unique_ptr<xla::PjRtLoadedExecutable> executable,
                               absl::optional<std::string> fingerprint,
                               ExlaClient* client) : executable_(std::move(executable)),
                                                     fingerprint_(std::move(fingerprint)),
                                                     client_(client) {}

tsl::StatusOr<std::unique_ptr<xla::PjRtBuffer>> PjRtBufferFromBinary(xla::PjRtClient* client,
                                                                     ERL_NIF_TERM source_term,
                                                                     const xla::Shape& shape,
                                                                     int device_id) {
  // We copy the binary term into a new env and point the buffer to
  // the binary content. Since larger binaries are shared and refcounted
  // this should be zero-copy.

  ErlNifEnv* copy_env = enif_alloc_env();
  ERL_NIF_TERM dest_term = enif_make_copy(copy_env, source_term);

  auto binary = fine::decode<ErlNifBinary>(copy_env, dest_term);

  xla::PjRtClient::HostBufferSemantics semantics = xla::PjRtClient::HostBufferSemantics::kImmutableZeroCopy;
  std::function<void()> on_done_with_host_buffer = [copy_env]() { enif_free_env(copy_env); };

  EXLA_ASSIGN_OR_RETURN(xla::PjRtDevice * device, client->LookupDevice(xla::PjRtGlobalDeviceId(device_id)));
  EXLA_ASSIGN_OR_RETURN(auto memory_space, device->default_memory_space());
  // Passing std::nullopt should work, but it fails for subbyte types,
  // so we build the default strides. See https://github.com/openxla/xla/issues/16795
  auto byte_strides = xla::ShapeUtil::ByteStrides(shape);
  EXLA_ASSIGN_OR_RETURN(auto buffer, client->BufferFromHostBuffer(
                                         binary.data, shape.element_type(), shape.dimensions(), byte_strides, semantics, on_done_with_host_buffer, memory_space, /*device_layout=*/nullptr));

  return std::move(buffer);
}

tsl::StatusOr<std::vector<std::vector<xla::PjRtBuffer*>>>
UnpackRunArguments(ErlNifEnv* env,
                   ExlaExecutable::RunArguments arguments,
                   std::vector<std::unique_ptr<xla::PjRtBuffer>> &transient_buffers,
                   ExlaClient* client,
                   xla::DeviceAssignment device_assignment,
                   int device_id) {
  std::vector<std::vector<xla::PjRtBuffer*>> arg_buffers;
  arg_buffers.reserve(arguments.size());

  int replica = 0;

  for (const auto & replica_arguments : arguments) {
    auto device = device_id >= 0 ? device_id : device_assignment(replica, 0);

    auto replica_buffers = std::vector<xla::PjRtBuffer*>();
    replica_buffers.reserve(replica_arguments.size());

    // For a single replica, the argument is a flat list of buffers where
    // each buffer can either be an erlang binary or a reference to another
    // EXLA buffer, it is not possible for any of the arguments to be nested
    // tuples because we handle normalization/flattening of tuples on the
    // Elixir side
    for (const auto & argument : replica_arguments) {
      if (auto value = std::get_if<std::tuple<fine::Term, xla::Shape>>(&argument)) {
        auto [term, shape] = *value;
        // We convert the binary into a buffer and transfer it to the
        // correct device, this buffer is not managed by the erlang vm
        // so it must be deallocated explicitly after use by the execution
        EXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtBuffer> buf,
                              PjRtBufferFromBinary(client->client(), term, shape, device));
        replica_buffers.push_back(buf.get());
        // Keep track of the buffer pointer, for automatic deallocation later
        transient_buffers.push_back(std::move(buf));
      } else if (auto value = std::get_if<fine::ResourcePtr<ExlaBuffer>>(&argument)) {
        auto buffer = *value;
        // if the buffer is not a tuple it must be a reference to an exla buffer
        // which means the resource is already managed by the vm, and should already
        // be on the correct device, if it is not, we will not do any implicit transfers
        // and instead raise an error
        if (buffer->device_id() != device) {
          return xla::InvalidArgument("Expected buffer to be placed on device %d", device);
        }
        replica_buffers.push_back(buffer->buffer());
      }
    }

    arg_buffers.push_back(std::move(replica_buffers));

    replica++;
  }

  return arg_buffers;
}

ExlaExecutable::RunResult UnpackResult(ErlNifEnv* env,
                                       std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> result,
                                       xla::DeviceAssignment device_assignment,
                                       int device_id) {
  auto per_replica_results = std::vector<std::tuple<std::vector<fine::ResourcePtr<ExlaBuffer>>, int64_t>>();

  for (int i = 0; i < result.size(); i++) {
    auto replica_results = std::vector<fine::ResourcePtr<ExlaBuffer>>();
    int64_t device = device_id >= 0 ? device_id : device_assignment(i, 0);

    for (auto& pjrt_buf : result.at(i)) {
      pjrt_buf->GetReadyFuture().Await();
      auto result = fine::make_resource<ExlaBuffer>(std::move(pjrt_buf));
      replica_results.push_back(result);
    }

    per_replica_results.push_back(std::make_tuple(std::move(replica_results), device));
  }

  return per_replica_results;
}

tsl::StatusOr<ExlaExecutable::RunResult> ExlaExecutable::Run(ErlNifEnv* env,
  ExlaExecutable::RunArguments arguments,
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

  // Buffers allocated from binaries for this specific run need to be
  // freed at the end. We store the corresponding pointers in this
  // vector, so they are all freed automatically when this function
  // finishes
  auto transient_buffers = std::vector<std::unique_ptr<xla::PjRtBuffer>>();

  if (device_id >= 0 && num_replicas > 1) {
    // if the device id is greater than or equal to 1, that means we've specified
    // a portable executable which cannot be pmapped, this code path should never
    // be reached as it should be controlled from Elixir
    return xla::InvalidArgument("Cannot specify a device for replicated executable.");
  } else {
    // else we handle unpacking/validating the run arguments to the correct devices
    // according to the device id and the device assignment
    EXLA_ASSIGN_OR_RETURN(input_buffers, UnpackRunArguments(env, arguments, transient_buffers, client_, device_assignment, device_id));
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
    EXLA_ASSIGN_OR_RETURN(xla::PjRtDevice * device, client_->client()->LookupDevice(xla::PjRtGlobalDeviceId(device_id)));
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
  auto ret = UnpackResult(env, std::move(per_replica_results), device_assignment, device_id);

  return ret;
}

ExlaClient::ExlaClient(std::shared_ptr<xla::PjRtClient> client) : client_(std::move(client)) {}

tsl::StatusOr<fine::ResourcePtr<ExlaBuffer>> ExlaClient::BufferFromBinary(ERL_NIF_TERM source_term,
                                                        xla::Shape& shape,
                                                        int device_id) {
  EXLA_ASSIGN_OR_RETURN(auto buffer, PjRtBufferFromBinary(client(), source_term, shape, device_id));
  return fine::make_resource<ExlaBuffer>(std::move(buffer));
}

tsl::StatusOr<std::optional<std::string>> ExecutableFingerprint(std::unique_ptr<xla::PjRtLoadedExecutable>& executable) {
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

tsl::StatusOr<fine::ResourcePtr<ExlaExecutable>> ExlaClient::DeserializeExecutable(std::string deserialized_executable) {
  EXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtLoadedExecutable> executable,
                        client_->LoadSerializedExecutable(deserialized_executable, std::nullopt, xla::LoadOptions()));

  EXLA_ASSIGN_OR_RETURN(absl::optional<std::string> fingerprint,
                        ExecutableFingerprint(executable));

  return fine::make_resource<ExlaExecutable>(std::move(executable), std::move(fingerprint), this);
}

tsl::StatusOr<fine::ResourcePtr<ExlaExecutable>> ExlaClient::Compile(mlir::ModuleOp module,
                                                   std::vector<xla::Shape> argument_layouts,
                                                   xla::ExecutableBuildOptions& options,
                                                   bool compile_portable_executable) {
  std::vector<xla::Shape> layouts;
  layouts.reserve(argument_layouts.size());
  for (auto shape : argument_layouts) {
    xla::Shape cpy_shape = xla::ShapeUtil::MakeShape(shape.element_type(), shape.dimensions());
    xla::LayoutUtil::ClearLayout(&cpy_shape);
    layouts.push_back(cpy_shape);
  }

  xla::CompileOptions compile_opts;
  compile_opts.argument_layouts = layouts;
  compile_opts.parameter_is_tupled_arguments = false;
  compile_opts.executable_build_options = options;
  compile_opts.compile_portable_executable = compile_portable_executable;

  EXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtLoadedExecutable> executable,
                        client_->CompileAndLoad(module, std::move(compile_opts)));
  EXLA_ASSIGN_OR_RETURN(absl::optional<std::string> fingerprint,
                        ExecutableFingerprint(executable));

  return fine::make_resource<ExlaExecutable>(std::move(executable), std::move(fingerprint), this);
}

tsl::Status ExlaClient::TransferToInfeed(ErlNifEnv* env,
                                         std::vector<ErlNifBinary> buffer_bins,
                                         std::vector<xla::Shape> shapes,
                                         int device_id) {
  std::vector<const char*> buf_ptrs;
  buf_ptrs.reserve(buffer_bins.size());

  for (const auto& buffer_bin : buffer_bins) {
    const char* data_ptr = const_cast<char*>(reinterpret_cast<char*>(buffer_bin.data));
    buf_ptrs.push_back(data_ptr);
  }

  auto shape = xla::ShapeUtil::MakeTupleShape(shapes);

  // Instead of pushing each buffer separately, we create a flat tuple
  // literal and push the whole group of buffers.
  //
  // On the CPU, XLA infeed reads buffers from a queue one at a time [1][2]
  // (or rather, the infeed operation is lowered to multiple queue reads),
  // hence pushing one at a time works fine. Pushing a flat tuple works
  // effectively the same, since it basically adds each element to the
  // queue [3].
  //
  // On the GPU, XLA infeed reads only a single "literal" from a queue [4]
  // and expects it to carry all buffers for the given infeed operation.
  // Consequently, we need to push all buffers as a single literal.
  //
  // Given that a flat tuple works in both cases, we just do that.
  //
  // [1]: https://github.com/openxla/xla/blob/fd58925adee147d38c25a085354e15427a12d00a/xla/service/cpu/ir_emitter.cc#L449-L450
  // [2]: https://github.com/openxla/xla/blob/fd58925adee147d38c25a085354e15427a12d00a/xla/service/cpu/cpu_runtime.cc#L222
  // [3]: https://github.com/openxla/xla/blob/fd58925adee147d38c25a085354e15427a12d00a/xla/service/cpu/cpu_xfeed.cc#L178
  // [4]: https://github.com/openxla/xla/blob/fd58925adee147d38c25a085354e15427a12d00a/xla/service/gpu/runtime/infeed_thunk.cc#L40-L41
  xla::BorrowingLiteral literal(buf_ptrs, shape);

  EXLA_ASSIGN_OR_RETURN(xla::PjRtDevice * device, client_->LookupDevice(xla::PjRtGlobalDeviceId(device_id)));

  tsl::Status status = device->TransferToInfeed(literal);

  return status;
}

tsl::StatusOr<ERL_NIF_TERM> ExlaClient::TransferFromOutfeed(ErlNifEnv* env, int device_id, xla::Shape& shape) {
  EXLA_ASSIGN_OR_RETURN(xla::PjRtDevice * device, client_->LookupDevice(xla::PjRtGlobalDeviceId(device_id)));

  auto literal = std::make_shared<xla::Literal>(shape);

  auto transfer_status = device->TransferFromOutfeed(literal.get());

  if (!transfer_status.ok()) {
    return transfer_status;
  }

  auto size = literal->size_bytes();

  ERL_NIF_TERM binary_term;
  auto data = enif_make_new_binary(env, size, &binary_term);
  memcpy(data, literal->untyped_data(), size);

  return binary_term;
}

tsl::StatusOr<fine::ResourcePtr<ExlaClient>> GetHostClient() {
  xla::CpuClientOptions options;
  options.asynchronous = false;
  EXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                        xla::GetXlaPjrtCpuClient(options));

  return fine::make_resource<ExlaClient>(std::move(client));
}

tsl::StatusOr<fine::ResourcePtr<ExlaClient>> GetGpuClient(double memory_fraction,
                                        bool preallocate,
                                        xla::GpuAllocatorConfig::Kind kind) {
  xla::GpuAllocatorConfig allocator_config = {
      .kind = kind,
      .memory_fraction = memory_fraction,
      .preallocate = preallocate};

  xla::GpuClientOptions client_options = {
      .allocator_config = allocator_config};

  EXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                        xla::GetStreamExecutorGpuClient(client_options));

  return fine::make_resource<ExlaClient>(std::move(client));
}

tsl::StatusOr<fine::ResourcePtr<ExlaClient>> GetTpuClient() {
  auto statusor = pjrt::LoadPjrtPlugin("tpu", "libtpu.so");
  if (!statusor.ok()) {
    return statusor.status();
  }

  tsl::Status status = pjrt::InitializePjrtPlugin("tpu");

  if (!status.ok()) {
    return status;
  }

  EXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                        xla::GetCApiClient("TPU"));

  return fine::make_resource<ExlaClient>(std::move(client));
}

tsl::StatusOr<fine::ResourcePtr<ExlaClient>> GetCApiClient(std::string device_type) {
  EXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
                        xla::GetCApiClient(device_type));

  return fine::make_resource<ExlaClient>(std::move(client));
}
}  // namespace exla
