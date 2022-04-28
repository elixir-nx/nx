#include "exla_client.h"
#include "exla_nif_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/pjrt/gpu_device.h"
#include "tensorflow/compiler/xla/pjrt/cpu_device.h"
#include "tensorflow/compiler/xla/pjrt/tpu_client.h"
#include "tensorflow/stream_executor/tpu/tpu_transfer_manager.h"

namespace exla {

ExlaBuffer::ExlaBuffer(std::unique_ptr<xla::PjRtBuffer> buffer,
                       bool can_be_released_after_run): buffer_(std::move(buffer)),
                                                        can_be_released_after_run_(can_be_released_after_run) {}

void CopyLiteralToBinary(xla::Literal* literal, ErlNifBinary* binary, exla::int64 size) {
  exla::int64 actual_size = literal->size_bytes();
  if(size < 0 or size > actual_size) size = actual_size;
  enif_alloc_binary(size, binary);
  std::memcpy(binary->data, literal->untyped_data(), size);
}

xla::StatusOr<ERL_NIF_TERM> ExlaBuffer::ToBinary(ErlNifEnv* env, exla::int64 size) {
  EXLA_EFFECT_OR_RETURN(buffer_->BlockHostUntilReady());
  EXLA_ASSIGN_OR_RETURN(std::shared_ptr<xla::Literal> literal, buffer_->ToLiteral());

  ErlNifBinary binary;

  xla::Shape host_shape = xla::ShapeUtil::MakeShape(buffer_->on_device_shape().element_type(), buffer_->on_device_shape().dimensions());

  if (xla::LayoutUtil::LayoutsInShapesEqual(host_shape, literal->shape())) {
    CopyLiteralToBinary(literal.get(), &binary, size);
  } else {
    xla::Literal new_literal = literal->Relayout(host_shape);
    CopyLiteralToBinary(&new_literal, &binary, size);
  }

  return nif::make(env, binary);
}

xla::Status ExlaBuffer::BlockHostUntilReady() {
  return buffer_->BlockHostUntilReady();
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

xla::StatusOr<ExlaBuffer *> ExlaBuffer::CopyToDevice(xla::PjRtDevice * dst_device) {
  // TODO: On TPUs buffers might reside on different hosts which requires
  // a different API
  EXLA_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtBuffer> buf,
      buffer_->CopyToDevice(dst_device));
  return new ExlaBuffer(std::move(buf));
}

xla::StatusOr<std::vector<std::vector<ExlaBuffer*>>>
UnpackRunArguments(ErlNifEnv* env,
                   ERL_NIF_TERM arguments,
                   ExlaClient* client,
                   xla::DeviceAssignment device_assignment,
                   int device_id) {
  unsigned int length;
  if (!enif_get_list_length(env, arguments, &length)) {
    return xla::InvalidArgument("Argument is not a list.");
  }

  std::vector<std::vector<ExlaBuffer*>> arg_buffers;
  arg_buffers.reserve(length);

  ERL_NIF_TERM head, tail;
  int replica = 0;

  while (enif_get_list_cell(env, arguments, &head, &tail)) {
    ERL_NIF_TERM inner_head, inner_tail;
    unsigned int inner_length;

    if (!enif_get_list_length(env, head, &inner_length)) {
      return xla::InvalidArgument("Argument is not a list.");
    }
    std::vector<ExlaBuffer*> device_buffers;
    device_buffers.reserve(inner_length);

    while (enif_get_list_cell(env, head, &inner_head, &inner_tail)) {
      const ERL_NIF_TERM* tuple;
      int arity;
      ExlaBuffer** buffer;

      if (enif_get_tuple(env, inner_head, &arity, &tuple)) {
        ErlNifBinary data;
        xla::Shape* shape;

        if (!nif::get_binary(env, tuple[0], &data)) {
          return xla::InvalidArgument("Expected argument to be binary.");
        }
        if (!nif::get<xla::Shape>(env, tuple[1], shape)) {
          return xla::InvalidArgument("Expected argument to be shape reference.");
        }

        int device = device_id >= 0 ? device_id : device_assignment(replica, 0);

        EXLA_ASSIGN_OR_RETURN(ExlaBuffer* buf, client->BufferFromBinary(data, *shape, device, true));

        device_buffers.push_back(buf);

      } else if (nif::get<ExlaBuffer*>(env, inner_head, buffer)) {
        // XLA already raises if the device_ids do not match, so we don't need to check it.
        device_buffers.push_back(*buffer);
      } else {
        return xla::InvalidArgument("Expected argument to be buffer reference.");
      }
      head = inner_tail;
    }

    arg_buffers.push_back(device_buffers);
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

ExlaExecutable::ExlaExecutable(std::unique_ptr<xla::PjRtExecutable> executable,
			                         absl::optional<std::string> fingerprint,
			                         ExlaClient* client) : executable_(std::move(executable)),
                                                     fingerprint_(std::move(fingerprint)),
                                                     client_(client) {}

xla::StatusOr<ERL_NIF_TERM> ExlaExecutable::Run(ErlNifEnv* env,
                                                ERL_NIF_TERM arguments,
                                                int device_id) {
  xla::ExecuteOptions options;
  options.untuple_result = true;
  options.strict_shape_checking = false;

  int num_replicas = executable_->num_replicas();

  std::vector<std::vector<ExlaBuffer*>> input_buffers;
  EXLA_ASSIGN_OR_RETURN(xla::DeviceAssignment device_assignment,
    client_->client()->GetDefaultDeviceAssignment(num_replicas, 1));

  if (device_id >= 0) {
    EXLA_ASSIGN_OR_RETURN(input_buffers, UnpackRunArguments(env, arguments, client_, device_assignment, device_id));
  } else {
    EXLA_ASSIGN_OR_RETURN(input_buffers, UnpackRunArguments(env, arguments, client_, device_assignment, -1));
  }

  std::vector<std::vector<xla::PjRtBuffer*>> pjrt_buffers;
  std::vector<ERL_NIF_TERM> terms;
  pjrt_buffers.reserve(input_buffers.size());
  terms.reserve(input_buffers.size());

  for (auto device_buf : input_buffers) {
    std::vector<xla::PjRtBuffer*> arg_buffers;

    for (auto buf : device_buf) {
      arg_buffers.push_back(buf->buffer());

      // If the buffer was not received as a resource (e.g. we converted
      // it from a binary to a buffer), we need to make sure it has been
      // fully transferred before we exit the NIF and make it a resource
      // so it's tracked and GC'ed along with other buffers that are no
      // longer in use
      if (buf->release_after_run()) {
        EXLA_EFFECT_OR_RETURN(buf->BlockHostUntilReady());
        terms.push_back(nif::make<ExlaBuffer*>(env, buf));
      }
    }

    pjrt_buffers.push_back(arg_buffers);
  }

  ERL_NIF_TERM ret;

  if (device_id >= 0) {
    EXLA_ASSIGN_OR_RETURN(xla::PjRtDevice* device, client_->client()->LookupDevice(device_id));
    EXLA_ASSIGN_OR_RETURN(auto result, executable_->ExecutePortable(pjrt_buffers.at(0), device, options));
    std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> per_replica_results;
    per_replica_results.push_back(std::move(result));
    EXLA_ASSIGN_OR_RETURN(ret, UnpackResult(env, std::move(per_replica_results), device_assignment, device_id));
  } else {
    EXLA_ASSIGN_OR_RETURN(auto result, executable_->Execute(pjrt_buffers, options));
    EXLA_ASSIGN_OR_RETURN(ret, UnpackResult(env, std::move(result), device_assignment, device_id));
  }

  return ret;
}

ExlaClient::ExlaClient(std::shared_ptr<xla::PjRtClient> client) : client_(std::move(client)) {}

xla::StatusOr<ExlaBuffer*> ExlaClient::BufferFromBinary(const ErlNifBinary& binary,
                                                        xla::Shape& shape,
                                                        int device_id,
                                                        bool can_be_released_after_run) {
  xla::PjRtClient::HostBufferSemantics semantics = xla::PjRtClient::HostBufferSemantics::kImmutableUntilTransferCompletes;

  EXLA_ASSIGN_OR_RETURN(xla::PjRtDevice* device, client_->LookupDevice(device_id));
  EXLA_ASSIGN_OR_RETURN(auto buffer, client_->BufferFromHostBuffer(
    binary.data, shape.element_type(), shape.dimensions(), absl::nullopt, semantics, nullptr, device));

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

xla::Status ExlaClient::TransferToInfeed(ErlNifEnv* env,
                                         ERL_NIF_TERM data,
                                         const xla::Shape& shape,
                                         int device_id) {
  // Tuples need to be decomposed a bit
  if (shape.IsTuple()) {
    // unsupported right now
    if (xla::ShapeUtil::IsNestedTuple(shape)) {
      return xla::InvalidArgument("nested tuples are not supported in infeed operation");
    }

    int num_elements = xla::ShapeUtil::TupleElementCount(shape);
    std::vector<const char*> buf_ptrs;
    buf_ptrs.reserve(num_elements);

    ERL_NIF_TERM head, tail;
    while (enif_get_list_cell(env, data, &head, &tail)) {
      ErlNifBinary tmp_bin;
      if (!nif::get_binary(env, head, &tmp_bin)) {
        return xla::InvalidArgument("infeed operation expects a list of binaries");
      }

      const char * data_ptr = const_cast<char *>(reinterpret_cast<char *>(tmp_bin.data));
      buf_ptrs.push_back(data_ptr);
      data = tail;
    }

    xla::BorrowingLiteral literal(buf_ptrs, shape);

    EXLA_ASSIGN_OR_RETURN(xla::PjRtDevice* device, client_->LookupDevice(device_id));

    xla::Status status = device->TransferToInfeed(literal);

    return status;
  }

  // Fast path to avoid any traversal when not sending Tuples
  ERL_NIF_TERM head, tail;
  if (!enif_get_list_cell(env, data, &head, &tail)) {
    return xla::InvalidArgument("infeed operation expects a list of binaries");
  }

  ErlNifBinary binary;
  if (!nif::get_binary(env, head, &binary)) {
    return xla::InvalidArgument("infeed operation expects a list of binaries");
  }

  const char * data_ptr = const_cast<char *>(reinterpret_cast<char *>(binary.data));
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
