#include "tensorflow/compiler/xla/exla/exla_client.h"
#include "tensorflow/compiler/xla/exla/exla_nif_util.h"
#include "tensorflow/compiler/xla/pjrt/gpu_device.h"
#include "tensorflow/compiler/xla/pjrt/cpu_device.h"
#include "tensorflow/compiler/xla/pjrt/tpu_client.h"
#include "tensorflow/stream_executor/tpu/tpu_transfer_manager.h"

namespace exla {

ExlaBuffer::ExlaBuffer(std::unique_ptr<xla::PjRtBuffer> buffer,
                       bool can_be_released_after_run): buffer_(std::move(buffer)),
                                                        can_be_released_after_run_(can_be_released_after_run) {}

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

xla::StatusOr<std::vector<ExlaBuffer*>> UnpackRunArguments(ErlNifEnv* env,
                                                           ERL_NIF_TERM arguments,
                                                           ExlaClient* client,
                                                           int device_id) {
  unsigned int length;
  if (!enif_get_list_length(env, arguments, &length)) {
    return xla::InvalidArgument("Argument is not a list.");
  }

  std::vector<ExlaBuffer*> arg_buffers;
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

      EXLA_ASSIGN_OR_RETURN(ExlaBuffer* buf, client->BufferFromBinary(data, *shape, device_id, true));

      arg_buffers.push_back(buf);

    } else if (nif::get<ExlaBuffer*>(env, head, buffer)) {
      arg_buffers.push_back(*buffer);
    } else {
      return xla::InvalidArgument("Expected argument to be buffer reference.");
    }
    arguments = tail;
  }

  return std::move(arg_buffers);
}

xla::StatusOr<ERL_NIF_TERM> UnpackResult(ErlNifEnv* env, std::vector<std::unique_ptr<xla::PjRtBuffer>> result, bool keep_on_device) {
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
                                                bool keep_on_device,
                                                int device_id) {
  xla::ExecuteOptions options;
  options.untuple_result = true;
  options.strict_shape_checking = false;

  std::vector<ExlaBuffer*> input_buffers;
  if (device_id >= 0) {
    EXLA_ASSIGN_OR_RETURN_NIF(input_buffers, UnpackRunArguments(env, arguments, client_, device_id), env);
  } else {
    // TODO(seanmor5): With pmap, this should unpack to all devices
    EXLA_ASSIGN_OR_RETURN_NIF(input_buffers, UnpackRunArguments(env, arguments, client_, 0), env);
  }

  std::vector<xla::PjRtBuffer*> pjrt_buffers;
  std::vector<ERL_NIF_TERM> terms;
  pjrt_buffers.reserve(input_buffers.size());
  terms.reserve(input_buffers.size());

  for (auto buf : input_buffers) {
    pjrt_buffers.push_back(buf->buffer());
    // If the buffer was not received as a resource (e.g. we converted
    // it from a binary to a buffer), we need to make it a resource so
    // it's tracked and GC'ed along with other buffers that are no longer
    // in use
    if (buf->release_after_run()) {
      terms.push_back(nif::make<ExlaBuffer*>(env, buf));
    }
  }

  ERL_NIF_TERM ret;

  if (device_id >= 0) {
    EXLA_ASSIGN_OR_RETURN_NIF(xla::PjRtDevice* device, client_->client()->LookupDevice(device_id), env);
    EXLA_ASSIGN_OR_RETURN_NIF(auto result, executable_->ExecutePortable(pjrt_buffers, device, options), env);
    EXLA_ASSIGN_OR_RETURN_NIF(ret, UnpackResult(env, std::move(result), keep_on_device), env);
  } else {
    std::vector<std::vector<xla::PjRtBuffer*>> inputs = std::vector<std::vector<xla::PjRtBuffer*>>({pjrt_buffers});
    EXLA_ASSIGN_OR_RETURN_NIF(auto result, executable_->Execute(inputs, options), env);
    EXLA_ASSIGN_OR_RETURN_NIF(ret, UnpackResult(env, std::move(result.at(0)), keep_on_device), env);
  }

  return ret;
}

ExlaClient::ExlaClient(std::shared_ptr<xla::PjRtClient> client) : client_(std::move(client)) {}

xla::StatusOr<ExlaBuffer*> ExlaClient::BufferFromBinary(const ErlNifBinary& binary,
                                                        xla::Shape& shape,
                                                        int device_id,
                                                        bool can_be_released_after_run) {
  xla::PjRtClient::HostBufferSemantics semantics = xla::PjRtClient::HostBufferSemantics::kZeroCopy;

  EXLA_ASSIGN_OR_RETURN(xla::PjRtDevice* device, client_->LookupDevice(device_id));
  EXLA_ASSIGN_OR_RETURN(auto buffer, client_->BufferFromHostBuffer(binary.data, shape, semantics, nullptr, device));

  return new ExlaBuffer(std::move(buffer), can_be_released_after_run);
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

  xla::CompileOptions compile_opts;
  compile_opts.argument_layouts = layouts;
  compile_opts.parameter_is_tupled_arguments = false;
  compile_opts.executable_build_options = options;
  compile_opts.compile_portable_executable = compile_portable_executable;

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

xla::Status ExlaClient::TransferToInfeed(int device_id, ErlNifBinary data, const xla::Shape& shape) {
  const char * data_ptr = const_cast<char *>(reinterpret_cast<char *>(data.data));
  xla::BorrowingLiteral literal(data_ptr, shape);

  EXLA_ASSIGN_OR_RETURN(xla::PjRtDevice* device, client_->LookupDevice(device_id));

  return device->TransferToInfeed(literal);
}

xla::StatusOr<ERL_NIF_TERM> ExlaClient::TransferFromOutfeed(ErlNifEnv* env, int device_id, xla::Shape& shape) {
  EXLA_ASSIGN_OR_RETURN(xla::PjRtDevice* device, client_->LookupDevice(device_id));

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
