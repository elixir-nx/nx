#include "tensorflow/compiler/xla/exla/exla_client.h"
#include "tensorflow/compiler/xla/exla/exla_nif_util.h"

namespace exla {

ExlaBuffer::ExlaBuffer(std::unique_ptr<xla::PjRtBuffer> buffer): buffer_(std::move(buffer)) {}

xla::StatusOr<ERL_NIF_TERM> ExlaBuffer::ToBinary(ErlNifEnv* env) {
  EXLA_ASSIGN_OR_RETURN(std::shared_ptr<xla::Literal> literal, buffer_->ToLiteral());

  ErlNifBinary binary;
  enif_alloc_binary(size, &binary);
  std::memcpy(binary.data, literal->untyped_data(), literal->size_bytes());

  return nif::make(env, binary);
}

xla::Status ExlaBuffer::Deallocate() {
  if (buffer_->IsDeleted()) {
    return xla::FailedPrecondition("Attempt to deallocate already deallocated buffer.");
  }
  else {
    buffer_->Delete();
    return xla::Status::OK();
  }
}

xla::StatusOr<std::vector<std::unique_ptr<xla::PjRtBuffer>>> UnpackRunArguments(ErlNifEnv* env,
                                                                                ERL_NIF_TERM arguments,
                                                                                xla::PjRtClient* client) {
  unsigned int length;
  if (!enif_get_list_length(env, list, &length)) {
    return xla::InvalidArgument("Argument is not a list.");
  }

  std::vector<std::unique_ptr<xla::PjRtBuffer>> arg_buffers;
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
        return xla::InvalidArgument("Expected argument to be binary.");
      }
      if (!nif::get<xla::Shape>(env, tuple[1], shape)) {
        return xla::InvalidArgument("Expected argument to be shape reference.");
      }

      EXLA_ASSIGN_OR_RETURN(ExlaBuffer* buf, client->BufferFromBinary(data, *shape, 0));
      arg_buffers.push_back(std::move(buf->buffer_));

    } else if (nif::get<ExlaBuffer*>(env, head, buffer)) {
      arg_buffers.push_back(std::move(buffer->buffer_));
    } else {
      return xla::InvalidArgument("Expected argument to be buffer reference.");
    }
    list = tail;
  }

  return std::move(arg_buffers);
}

xla::StatusOr<ERL_NIF_TERM> UnpackResult(ErlNifEnv* env, std::vector<std::unique_ptr<xla::PjRtBuffer>> result, bool keep_on_device) {
  if (result.size() > 1) {
    ExlaBuffer* buf = new ExlaBuffer(std::move(result.at(0)));
    if (keep_on_device) {
      return nif::ok(env, nif::make<ExlaBuffer*>(env, buf));
    } else {
      EXLA_ASSIGN_OR_RETURN_NIF(ERL_NIF_TERM term, buf->ToBinary(), env);
      delete buf;
      return nif::ok(env, term);
    }
  } else {
    return nif::ok(env);
  }
}

ExlaExecutable::ExlaExecutable(std::unique_ptr<xla::PjRtExecutable> executable) : executable_(std::move(executable)) {}

xla::StatusOr<ERL_NIF_TERM> ExlaExecutable::Run(ErlNifEnv* env,
                                                ERL_NIF_TERM arguments,
                                                bool keep_on_device) {
  xla::ExecuteOptions options;

  EXLA_ASSIGN_OR_RETURN_NIF(std::vector<std::unique_ptr<xla::PjRtBuffer>> arguments,
    UnpackRunArguments(env, arguments, client_.get()), env);

  auto inputs = absl::make_span<std::vector<std::unique_ptr<xla::PjRtBuffer>>>({arguments});

  EXLA_ASSIGN_OR_RETURN_NIF(auto result, executable_->Execute(inputs, options), env);

  EXLA_ASSIGN_OR_RETURN_NIF(ERL_NIF_TERM ret, UnpackResult(env, result.at(0), keep_on_device), env);

  return ret;
}

ExlaClient::ExlaClient(std::unique_ptr<xla::PjRtClient> client) : client_(std::move(client)) {}

ExlaClient::BufferFromBinary(const ErlNifBinary& binary, xla::Shape& shape, int device_id) {
  void * data = const_cast<void *>(binary.data);
  xla::PjRtClient::HostBufferSemantics semantics = PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall;

  // TODO(seanmor5): control this with an option
  EXLA_ASSIGN_OR_RETURN(xla::PjRtDevice* device, client_->LookupDevice(device_id));

  EXLA_ASSIGN_OR_RETURN(auto buffer, client_->BufferFromHostBuffer(data, shape, semantics, nullptr, device));

  return new ExlaBuffer(std::move(buffer));
}

xla::StatusOr<ExlaExecutable*> ExlaClient::Compile(const xla::XlaComputation& computation,
                                                   std::vector<xla::Shape*> argument_layouts,
                                                   xla::ExecutableBuildOptions& options,
                                                   bool compile_portable_executable) {
  xla::CompileOptions compile_opts = {
    .argument_layouts = std::move(argument_layouts),
    .parameter_is_tupled_arguments = false,
    .executable_build_options = options,
    .compile_portable_executable = compile_portable_executable
  };

  EXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtExecutable> executable,
    client_->Compile(computation, compile_opts));

  return new ExlaExecutable(std::move(executable));
}

xla::StatusOr<ExlaClient*> GetHostClient() {
  EXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
    xla::GetCpuClient(false));

  return new ExlaClient(client.release());
}

xla::StatusOr<ExlaClient*> GetGpuClient(double memory_fraction,
                                        bool preallocate,
                                        xla::GpuAllocatorConfig::Kind kind) {
  xla::GpuAllocatorConfig allocator_config = {
    .kind = kind,
    .memory_fraction = memory_fraction,
    .preallocate = preallocate
  };

  EXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
    xla::GetGpuClient(false, allocator_config, nullptr, 0));

  return new ExlaClient(client.release());
}

xla::StatusOr<ExlaClient*> GetTpuClient() {
  EXLA_ASSIGN_OR_RETURN(std::shared_ptr<xla::PjRtClient> client,
    xla::GetTpuClient(32));

  return new ExlaClient(client.get());
}
}  // namespace exla
