#include "tensorflow/compiler/xla/exla/exla_client.h"
#include "tensorflow/compiler/xla/exla/exla_nif_util.h"
#include "tensorflow/compiler/xla/pjrt/gpu_device.h"
#include "tensorflow/compiler/xla/pjrt/cpu_device.h"
#include "tensorflow/compiler/xla/pjrt/tpu_client.h"
#include "tensorflow/stream_executor/tpu/tpu_transfer_manager.h"

namespace exla {

ExlaBuffer::ExlaBuffer(std::unique_ptr<xla::PjRtBuffer> buffer): buffer_(std::move(buffer)) {}

xla::StatusOr<ERL_NIF_TERM> ExlaBuffer::ToBinary(ErlNifEnv* env) {
  buffer_->BlockHostUntilReady();
  EXLA_ASSIGN_OR_RETURN(std::shared_ptr<xla::Literal> literal, buffer_->ToLiteral());

  // TODO(seanmor5): Investigate ways around this, it should not be necessary
  // on CPU and GPU
  xla::Shape host_shape = xla::ShapeUtil::MakeShape(buffer_->on_device_shape().element_type(), buffer_->on_device_shape().dimensions());
  xla::Literal new_literal = literal->Relayout(host_shape);

  ErlNifBinary binary;
  enif_alloc_binary(new_literal.size_bytes(), &binary);
  std::memcpy(binary.data, new_literal.untyped_data(), new_literal.size_bytes());

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

xla::StatusOr<std::vector<xla::PjRtBuffer*>> UnpackRunArguments(ErlNifEnv* env,
                                                                ERL_NIF_TERM arguments,
                                                                ExlaClient* client) {
  unsigned int length;
  if (!enif_get_list_length(env, arguments, &length)) {
    return xla::InvalidArgument("Argument is not a list.");
  }

  std::vector<xla::PjRtBuffer*> arg_buffers;
  arg_buffers.reserve(length);

  ERL_NIF_TERM head, tail;
  while (enif_get_list_cell(env, arguments, &head, &tail)) {
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

      arg_buffers.push_back(buf->buffer());

    } else if (nif::get<ExlaBuffer*>(env, head, buffer)) {
      arg_buffers.push_back((*buffer)->buffer());
    } else {
      return xla::InvalidArgument("Expected argument to be buffer reference.");
    }
    arguments = tail;
  }

  return std::move(arg_buffers);
}

xla::StatusOr<ERL_NIF_TERM> UnpackResult(ErlNifEnv* env, std::vector<std::unique_ptr<xla::PjRtBuffer>> result, bool keep_on_device) {
  result.at(0)->BlockHostUntilReady();
  std::vector<ERL_NIF_TERM> terms;
  terms.reserve(result.size());
  for (auto& pjrt_buf : result) {
    ExlaBuffer* buf = new ExlaBuffer(std::move(pjrt_buf));
    ERL_NIF_TERM term;
    if (keep_on_device) {
      term = nif::make<ExlaBuffer*>(env, buf);
    } else {
      EXLA_ASSIGN_OR_RETURN_NIF(term, buf->ToBinary(env), env);
      delete buf;
    }
    terms.push_back(term);
  }
  return nif::ok(env, enif_make_tuple2(env, enif_make_list_from_array(env, &terms[0], terms.size()), enif_make_int(env, 0)));
}

ExlaExecutable::ExlaExecutable(std::unique_ptr<xla::PjRtExecutable> executable,
			       absl::optional<std::string> fingerprint,
			       ExlaClient* client) : executable_(std::move(executable)),
                                                     fingerprint_(std::move(fingerprint)),
                                                     client_(client) {}

xla::StatusOr<ERL_NIF_TERM> ExlaExecutable::Run(ErlNifEnv* env,
                                                ERL_NIF_TERM arguments,
                                                bool keep_on_device) {
  xla::ExecuteOptions options = {
    .untuple_result = true,
    .strict_shape_checking = false
  };

  EXLA_ASSIGN_OR_RETURN_NIF(std::vector<xla::PjRtBuffer*> input_buffers,
    UnpackRunArguments(env, arguments, client_), env);

  std::vector<std::vector<xla::PjRtBuffer*>> inputs = std::vector<std::vector<xla::PjRtBuffer*>>({input_buffers});

  EXLA_ASSIGN_OR_RETURN_NIF(auto result, executable_->Execute(inputs, options), env);

  EXLA_ASSIGN_OR_RETURN_NIF(ERL_NIF_TERM ret, UnpackResult(env, std::move(result.at(0)), keep_on_device), env);

  return ret;
}

ExlaClient::ExlaClient(std::shared_ptr<xla::PjRtClient> client) : client_(std::move(client)) {}

xla::StatusOr<ExlaBuffer*> ExlaClient::BufferFromBinary(const ErlNifBinary& binary, xla::Shape& shape, int device_id) {
  xla::PjRtClient::HostBufferSemantics semantics = xla::PjRtClient::HostBufferSemantics::kZeroCopy;

  EXLA_ASSIGN_OR_RETURN(xla::PjRtDevice* device, client_->LookupDevice(device_id));
  EXLA_ASSIGN_OR_RETURN(auto buffer, client_->BufferFromHostBuffer(binary.data, shape, semantics, nullptr, device));
  buffer->BlockHostUntilReady();

  return new ExlaBuffer(std::move(buffer));
}

xla::StatusOr<ExlaExecutable*> ExlaClient::Compile(const xla::XlaComputation& computation,
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

  xla::CompileOptions compile_opts = {
    .argument_layouts = layouts,
    .parameter_is_tupled_arguments = false,
    .executable_build_options = options,
    .compile_portable_executable = compile_portable_executable
  };

  EXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtExecutable> executable,
    client_->Compile(computation, std::move(compile_opts)));
  EXLA_ASSIGN_OR_RETURN(absl::optional<std::string> fingerprint,
    client_->ExecutableFingerprint(*executable));

  return new ExlaExecutable(std::move(executable), std::move(fingerprint), this);
}

std::vector<ExlaDevice*> ExlaClient::GetDevices() {
 absl::Span<xla::PjRtDevice* const> pjrt_devices = client_->devices();

  std::vector<ExlaDevice*> devices;
  devices.reserve(pjrt_devices.size());
  for (auto pjrt_device : pjrt_devices) {
    ExlaDevice* device = new ExlaDevice(pjrt_device, this);
    devices.push_back(device);
  }

  return devices;
}

xla::StatusOr<ExlaClient*> GetHostClient() {
  EXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtClient> client,
    xla::GetCpuClient(false));

  return new ExlaClient(std::move(client));
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

  return new ExlaClient(std::move(client));
}

xla::StatusOr<ExlaClient*> GetTpuClient() {
  EXLA_ASSIGN_OR_RETURN(std::shared_ptr<xla::PjRtClient> client,
    xla::GetTpuClient(32));

  return new ExlaClient(std::move(client));
}
}  // namespace exla
