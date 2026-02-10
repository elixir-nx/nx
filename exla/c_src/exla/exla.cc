#include <cstring>
#include <fine.hpp>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <tuple>
#include <unordered_map>

#include "custom_calls/runtime_callback_bridge.h"
#include "exla_client.h"
#include "exla_cuda.h"
#include "exla_log_sink.h"
#include "exla_mlir.h"
#include "exla_nif_util.h"
#include "ipc.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/service/platform_util.h"
#include "xla/tsl/platform/statusor.h"
#include "llvm/Support/ThreadPool.h"
#include "absl/log/log_sink_registry.h"
#include "absl/log/initialize.h"
#include "absl/log/globals.h"

namespace exla {

using callback_bridge::Pending;

FINE_RESOURCE(llvm::StdThreadPool);
FINE_RESOURCE(mlir::MLIRContext);
FINE_RESOURCE(mlir::Value);
FINE_RESOURCE(mlir::Region);
FINE_RESOURCE(exla::ExlaClient);
FINE_RESOURCE(exla::ExlaBuffer);
FINE_RESOURCE(exla::ExlaExecutable);
FINE_RESOURCE(exla::MLIRModule);
FINE_RESOURCE(exla::MLIRFunction);
FINE_RESOURCE(Pending);

// MLIR Functions

fine::ResourcePtr<ExlaBuffer> decode_exla_buffer(ErlNifEnv *env,
                                                 fine::Term buffer_term) {
  try {
    return fine::decode<fine::ResourcePtr<ExlaBuffer>>(env, buffer_term);
  } catch (std::invalid_argument) {
    throw std::invalid_argument(
        "unable to get buffer. It may belong to another node, "
        "consider using Nx.backend_transfer/1");
  }
}

fine::ResourcePtr<llvm::StdThreadPool>
mlir_new_thread_pool(ErlNifEnv *env, int64_t concurrency) {
  auto strategy = llvm::hardware_concurrency(concurrency);
  return fine::make_resource<llvm::StdThreadPool>(strategy);
}

FINE_NIF(mlir_new_thread_pool, 0);

fine::ResourcePtr<mlir::MLIRContext>
mlir_new_context(ErlNifEnv *env,
                 fine::ResourcePtr<llvm::StdThreadPool> thread_pool) {
  auto context = fine::make_resource<mlir::MLIRContext>(
      mlir::MLIRContext::Threading::DISABLED);

  context->setThreadPool(*thread_pool);
  context->getOrLoadDialect<mlir::func::FuncDialect>();
  context->getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
  context->getOrLoadDialect<mlir::chlo::ChloDialect>();
  context->getOrLoadDialect<mlir::sdy::SdyDialect>();
  mlir::sdy::registerAllDialects(
      const_cast<mlir::DialectRegistry &>(context->getDialectRegistry()));

  return context;
}

FINE_NIF(mlir_new_context, 0);

fine::ResourcePtr<MLIRModule>
mlir_new_module(ErlNifEnv *env, fine::ResourcePtr<mlir::MLIRContext> ctx) {
  return fine::make_resource<MLIRModule>(ctx);
}

FINE_NIF(mlir_new_module, 0);

fine::ResourcePtr<MLIRFunction> mlir_create_function(
    ErlNifEnv *env, fine::ResourcePtr<MLIRModule> module, std::string func_name,
    std::vector<std::string> arg_type_strings,
    std::vector<std::string> ret_type_strings, bool is_public) {
  auto arg_types = std::vector<mlir::Type>{};

  for (auto const &type_string : arg_type_strings) {
    auto type = module->ParseType(type_string);
    arg_types.push_back(type);
  }

  auto ret_types = std::vector<mlir::Type>{};

  for (auto const &type_string : ret_type_strings) {
    auto type = module->ParseType(type_string);
    ret_types.push_back(type);
  }

  auto func_op =
      module->CreateFunction(func_name, arg_types, ret_types, is_public);
  return fine::make_resource<MLIRFunction>(module, std::move(func_op));
}

FINE_NIF(mlir_create_function, 0);

std::vector<fine::ResourcePtr<mlir::Value>>
mlir_get_function_arguments(ErlNifEnv *env,
                            fine::ResourcePtr<MLIRFunction> function) {
  auto args = function->GetArguments();
  std::vector<fine::ResourcePtr<mlir::Value>> values;
  values.reserve(args.size());

  for (const auto &arg : args) {
    values.push_back(fine::make_resource<mlir::Value>(arg));
  }

  return values;
}

FINE_NIF(mlir_get_function_arguments, 0);

std::vector<fine::ResourcePtr<mlir::Value>>
mlir_op(ErlNifEnv *env, fine::ResourcePtr<MLIRFunction> function,
        std::string op_name,
        std::vector<fine::ResourcePtr<mlir::Value>> operands,
        std::vector<std::string> result_type_strings,
        std::vector<std::tuple<fine::Atom, std::string>> attributes_kwlist,
        std::vector<fine::ResourcePtr<mlir::Region>> regions) {
  auto result_types = std::vector<mlir::Type>{};

  for (auto const &type_string : result_type_strings) {
    auto type = function->module()->ParseType(type_string);
    result_types.push_back(type);
  }

  auto attributes = std::vector<std::tuple<std::string, mlir::Attribute>>{};

  for (auto const &[key, value] : attributes_kwlist) {
    auto attribute_value = function->module()->ParseAttribute(value);
    attributes.push_back(std::make_tuple(key.to_string(), attribute_value));
  }

  return function->Op(op_name, operands, result_types, attributes, regions);
}

FINE_NIF(mlir_op, 0);

std::tuple<fine::ResourcePtr<mlir::Region>,
           std::vector<fine::ResourcePtr<mlir::Value>>>
mlir_push_region(ErlNifEnv *env, fine::ResourcePtr<MLIRFunction> function,
                 std::vector<std::string> arg_types) {
  auto types = std::vector<mlir::Type>{};

  for (auto const &type_string : arg_types) {
    auto type = function->module()->ParseType(type_string);
    types.push_back(type);
  }

  return function->PushRegion(types);
}

FINE_NIF(mlir_push_region, 0);

fine::Ok<> mlir_pop_region(ErlNifEnv *env,
                           fine::ResourcePtr<MLIRFunction> function) {
  function->PopRegion();
  return fine::Ok();
}

FINE_NIF(mlir_pop_region, 0);

fine::Ok<> mlir_add_mesh(ErlNifEnv *env, fine::ResourcePtr<MLIRModule> module,
                         std::string mesh_name,
                         std::vector<std::tuple<std::string, int64_t>> axes) {
  auto builder = module->builder();
  auto context = module->module()->getContext();

  llvm::SmallVector<mlir::sdy::MeshAxisAttr> axis_attrs;
  for (auto [name, size] : axes) {
    axis_attrs.push_back(mlir::sdy::MeshAxisAttr::get(context, name, size));
  }

  auto mesh_attr = mlir::sdy::MeshAttr::get(context, axis_attrs);

  // Create the mesh op at the beginning of the module
  auto module_op = module->module();
  auto &body_region = module_op.getBodyRegion();
  mlir::OpBuilder::InsertionGuard guard(*builder);
  builder->setInsertionPointToStart(&body_region.front());

  mlir::OperationState state(builder->getUnknownLoc(), "sdy.mesh");
  mlir::sdy::MeshOp::build(*builder, state, mesh_name, mesh_attr);
  builder->create(state);

  return fine::Ok();
}

FINE_NIF(mlir_add_mesh, 0);

mlir::sdy::TensorShardingAttr mlir_create_tensor_sharding_attr(
    mlir::MLIRContext *context, std::string mesh_name,
    std::vector<std::vector<std::string>> dim_shardings) {
  llvm::SmallVector<mlir::sdy::DimensionShardingAttr> dim_sharding_attrs;
  for (const auto &dim : dim_shardings) {
    llvm::SmallVector<mlir::sdy::AxisRefAttr> axis_refs;
    for (const auto &axis : dim) {
      axis_refs.push_back(mlir::sdy::AxisRefAttr::get(context, axis));
    }
    dim_sharding_attrs.push_back(mlir::sdy::DimensionShardingAttr::get(
        context, axis_refs, /*is_closed=*/false, /*priority=*/0));
  }

  return mlir::sdy::TensorShardingAttr::get(
      context, mesh_name, dim_sharding_attrs,
      /*replicated_axes=*/llvm::ArrayRef<mlir::sdy::AxisRefAttr>(),
      /*unreduced_axes=*/llvm::ArrayRef<mlir::sdy::AxisRefAttr>());
}

fine::Ok<> mlir_set_function_argument_attribute(
    ErlNifEnv *env, fine::ResourcePtr<MLIRFunction> function, int64_t arg_index,
    std::string attribute_name, std::string mesh_name,
    std::vector<std::vector<std::string>> dim_shardings) {

  auto context = function->module()->module()->getContext();
  auto sharding_attr =
      mlir_create_tensor_sharding_attr(context, mesh_name, dim_shardings);

  function->function().setArgAttr(arg_index, attribute_name, sharding_attr);

  return fine::Ok();
}

FINE_NIF(mlir_set_function_argument_attribute, 0);

mlir::Type mlir_get_typespec(ErlNifEnv *env,
                             fine::ResourcePtr<mlir::Value> value) {
  return value->getType();
}

FINE_NIF(mlir_get_typespec, 0);

std::string mlir_module_to_string(ErlNifEnv *env,
                                  fine::ResourcePtr<MLIRModule> module) {
  return module->ToString();
}

FINE_NIF(mlir_module_to_string, 0);

template <typename T> T unwrap(tsl::StatusOr<T> status_or) {
  if (!status_or.ok()) {
    throw std::runtime_error(status_or.status().message().data());
  }

  return std::move(status_or.value());
}

void unwrap(tsl::Status status) {
  if (!status.ok()) {
    throw std::runtime_error(status.message().data());
  }
}

fine::ResourcePtr<ExlaExecutable>
mlir_compile(ErlNifEnv *env, fine::ResourcePtr<ExlaClient> client,
             fine::ResourcePtr<MLIRModule> module,
             std::vector<xla::Shape> argument_layouts, int64_t num_replicas,
             int64_t num_partitions, bool use_spmd, int64_t device_id,
             fine::Term callback_server_pid_term) {
  auto build_options = xla::ExecutableBuildOptions();

  build_options.set_num_replicas(num_replicas);
  build_options.set_num_partitions(num_partitions);
  build_options.set_use_spmd_partitioning(use_spmd);

  // Enable Shardy partitioner when using SPMD to support output sharding
  // Shardy runs propagation before the SPMD partitioner and properly handles
  // FuncResultSharding custom calls that would otherwise fail the side-effect
  // check
  if (use_spmd) {
    build_options.set_use_shardy_partitioner(true);
  }

  auto compile_portable_executable = false;
  if (device_id >= 0) {
    compile_portable_executable = true;
    build_options.set_device_ordinal(device_id);
  }

  // Decode the optional callback server pid. If the term is a pid, we convert
  // it to an ErlNifPid; otherwise we treat it as "no pid" (e.g. nil).
  absl::optional<ErlNifPid> pid_opt;
  ERL_NIF_TERM pid_term = callback_server_pid_term;

  if (enif_is_pid(env, pid_term)) {
    ErlNifPid pid;
    if (enif_get_local_pid(env, pid_term, &pid)) {
      pid_opt = pid;
    }
  }

  return unwrap(client->Compile(module->module(), argument_layouts,
                                build_options, compile_portable_executable,
                                pid_opt));
}

FINE_NIF(mlir_compile, ERL_NIF_DIRTY_JOB_CPU_BOUND);

// ExlaBuffer Functions

std::variant<std::tuple<fine::Atom, uint64_t, uint64_t>,
             std::tuple<fine::Atom, std::string, uint64_t, uint64_t>,
             std::tuple<fine::Atom, std::string, uint64_t>>
get_buffer_device_pointer(ErlNifEnv *env, fine::ResourcePtr<ExlaClient> client,
                          fine::Term buffer_term, fine::Atom pointer_kind) {
  auto buffer = decode_exla_buffer(env, buffer_term);

  uint64_t device_size = unwrap(buffer->GetOnDeviceSizeInBytes());
  uint64_t ptr = unwrap(buffer->GetDevicePointer(client->client()));

  if (pointer_kind == "local") {
    return std::make_tuple(pointer_kind, ptr, device_size);
  }

  if (pointer_kind == "host_ipc") {
    auto handle_name =
        "exla:ipc:" + std::to_string(device_size) + ":" + std::to_string(ptr);
    auto fd = get_ipc_handle(handle_name.c_str(), device_size);

    if (fd == -1) {
      throw std::runtime_error("unable to get IPC handle");
    }

    auto ipc_ptr = open_ipc_handle(fd, device_size);
    if (ipc_ptr == nullptr) {
      throw std::runtime_error("unable to open IPC handle");
    }

    memcpy(ipc_ptr, reinterpret_cast<void *>(ptr), device_size);

    return std::make_tuple(pointer_kind, handle_name, static_cast<uint64_t>(fd),
                           device_size);
  }

  if (pointer_kind == "cuda_ipc") {
    auto maybe_handle = get_cuda_ipc_handle(ptr);
    if (!maybe_handle) {
      throw std::runtime_error("unable to get cuda IPC handle");
    }

    return std::make_tuple(pointer_kind, maybe_handle.value(), device_size);
  }

  throw std::invalid_argument("unexpected pointer type");
}

FINE_NIF(get_buffer_device_pointer, 0);

fine::ResourcePtr<ExlaBuffer> create_buffer_from_device_pointer(
    ErlNifEnv *env, fine::ResourcePtr<ExlaClient> client,
    fine::Atom pointer_kind, fine::Term pointer_data, xla::Shape shape,
    int64_t device_id) {
  void *ptr = nullptr;
  std::function<void()> on_delete_callback = []() {};

  if (pointer_kind == "cuda_ipc") {
    auto cuda_ipc_handle_bin = fine::decode<ErlNifBinary>(env, pointer_data);
    auto maybe_pointer = get_pointer_for_ipc_handle(
        cuda_ipc_handle_bin.data, cuda_ipc_handle_bin.size, device_id);
    if (!maybe_pointer) {
      throw std::runtime_error("unable to get pointer for IPC handle");
    }
    ptr = maybe_pointer.value();
  } else if (pointer_kind == "host_ipc") {
    auto tuple =
        fine::decode<std::tuple<uint64_t, std::string>>(env, pointer_data);
    auto fd = std::get<0>(tuple);
    auto memname = std::get<1>(tuple);
    auto device_size = xla::ShapeUtil::ByteSizeOf(shape);
    ptr = open_ipc_handle(fd, device_size);
    if (ptr == nullptr) {
      throw std::runtime_error("unable to get pointer for IPC handle");
    }
    on_delete_callback = [fd, memname, ptr, device_size]() {
      close_ipc_handle(fd, ptr, memname.c_str(), device_size);
    };
  } else if (pointer_kind == "local") {
    auto ptr_int = fine::decode<int64_t>(env, pointer_data);
    ptr = reinterpret_cast<void *>(ptr_int);
  } else {
    throw std::invalid_argument("unexpected pointer type");
  }

  auto device = unwrap(
      client->client()->LookupDevice(xla::PjRtGlobalDeviceId(device_id)));
  auto memory_space = unwrap(device->default_memory_space());
  auto buffer = unwrap(client->client()->CreateViewOfDeviceBuffer(
      ptr, shape, memory_space, on_delete_callback));
  return fine::make_resource<ExlaBuffer>(std::move(buffer));
}

FINE_NIF(create_buffer_from_device_pointer, 0);

fine::ResourcePtr<ExlaBuffer>
binary_to_device_mem(ErlNifEnv *env, fine::ResourcePtr<ExlaClient> client,
                     fine::Term data, xla::Shape shape, int64_t device_id) {
  return unwrap(client->BufferFromBinary(data, shape, device_id));
}

FINE_NIF(binary_to_device_mem, ERL_NIF_DIRTY_JOB_IO_BOUND);

fine::Term read_device_mem(ErlNifEnv *env, fine::Term buffer_term,
                           int64_t size) {
  auto buffer = decode_exla_buffer(env, buffer_term);
  return unwrap(buffer->ToBinary(env, size));
}

FINE_NIF(read_device_mem, ERL_NIF_DIRTY_JOB_IO_BOUND);

xla::Shape get_buffer_typespec(ErlNifEnv *env, fine::Term buffer_term) {
  auto buffer = decode_exla_buffer(env, buffer_term);
  return unwrap(buffer->buffer()->logical_on_device_shape());
}

FINE_NIF(get_buffer_typespec, 0);

std::variant<fine::Ok<>, fine::Error<fine::Atom>>
deallocate_device_mem(ErlNifEnv *env, fine::Term buffer_term) {
  auto buffer = decode_exla_buffer(env, buffer_term);

  tsl::Status dealloc_status = buffer->Deallocate();

  if (!dealloc_status.ok()) {
    return fine::Error(atoms::already_deallocated);
  } else {
    return fine::Ok();
  }
}

FINE_NIF(deallocate_device_mem, ERL_NIF_DIRTY_JOB_IO_BOUND);

fine::Ok<> transfer_to_infeed(ErlNifEnv *env,
                              fine::ResourcePtr<ExlaClient> client,
                              int64_t device_id,
                              std::vector<ErlNifBinary> buffers,
                              std::vector<xla::Shape> shapes) {
  unwrap(client->TransferToInfeed(env, buffers, shapes, device_id));

  return fine::Ok();
}

FINE_NIF(transfer_to_infeed, ERL_NIF_DIRTY_JOB_IO_BOUND);

fine::Ok<> transfer_from_outfeed(ErlNifEnv *env,
                                 fine::ResourcePtr<ExlaClient> client,
                                 int64_t device_id,
                                 std::vector<xla::Shape> shapes, ErlNifPid pid,
                                 fine::Term ref) {
  for (auto &shape : shapes) {
    auto msg_env = enif_alloc_env();
    auto msg = unwrap(client->TransferFromOutfeed(msg_env, device_id, shape));
    enif_send(env, &pid, msg_env, enif_make_tuple(msg_env, 2, ref, msg));
    enif_free_env(msg_env);
  }

  return fine::Ok();
}

FINE_NIF(transfer_from_outfeed, ERL_NIF_DIRTY_JOB_IO_BOUND);

fine::ResourcePtr<ExlaBuffer>
copy_buffer_to_device(ErlNifEnv *env, fine::ResourcePtr<ExlaClient> client,
                      fine::Term buffer_term, int64_t device_id) {
  auto buffer = decode_exla_buffer(env, buffer_term);

  auto device = unwrap(
      client->client()->LookupDevice(xla::PjRtGlobalDeviceId(device_id)));

  return unwrap(buffer->CopyToDevice(device));
}

FINE_NIF(copy_buffer_to_device, ERL_NIF_DIRTY_JOB_IO_BOUND);

// ExlaClient Functions

fine::ResourcePtr<ExlaClient> get_host_client(ErlNifEnv *env) {
  return unwrap(GetHostClient());
}

FINE_NIF(get_host_client, 0);

fine::ResourcePtr<ExlaClient>
get_gpu_client(ErlNifEnv *env, double memory_fraction, bool preallocate) {
  return unwrap(GetGpuClient(memory_fraction, preallocate,
                             xla::GpuAllocatorConfig::Kind::kBFC));
}

FINE_NIF(get_gpu_client, 0);

fine::ResourcePtr<ExlaClient> get_tpu_client(ErlNifEnv *env) {
  return unwrap(GetTpuClient());
}

FINE_NIF(get_tpu_client, 0);

fine::ResourcePtr<ExlaClient> get_c_api_client(ErlNifEnv *env,
                                               std::string device_type) {
  return unwrap(GetCApiClient(device_type));
}

FINE_NIF(get_c_api_client, 0);

fine::Ok<> load_pjrt_plugin(ErlNifEnv *env, std::string device_type,
                            std::string library_path) {
  unwrap(pjrt::LoadPjrtPlugin(device_type, library_path));
  return fine::Ok();
}

FINE_NIF(load_pjrt_plugin, 0);

int64_t get_device_count(ErlNifEnv *env, fine::ResourcePtr<ExlaClient> client) {
  return client->client()->device_count();
}

FINE_NIF(get_device_count, 0);

std::map<fine::Atom, int64_t> get_supported_platforms(ErlNifEnv *env) {
  auto platforms = unwrap(xla::PlatformUtil::GetSupportedPlatforms());

  std::map<fine::Atom, int64_t> platform_info;

  for (auto &platform : platforms) {
    auto key = fine::Atom(absl::AsciiStrToLower(platform->Name()));
    auto device_count = platform->VisibleDeviceCount();
    platform_info.insert({key, device_count});
  }

  return platform_info;
}

FINE_NIF(get_supported_platforms, 0);

// ExlaExecutable Functions

ExlaExecutable::RunResult run(ErlNifEnv *env,
                              fine::ResourcePtr<ExlaExecutable> executable,
                              ExlaExecutable::RunArguments arguments,
                              int64_t device_id) {
  return unwrap(executable->Run(env, arguments, device_id));
}

ExlaExecutable::RunResult run_cpu(ErlNifEnv *env,
                                  fine::ResourcePtr<ExlaExecutable> executable,
                                  ExlaExecutable::RunArguments arguments,
                                  int64_t device_id) {
  return run(env, executable, arguments, device_id);
}

FINE_NIF(run_cpu, ERL_NIF_DIRTY_JOB_CPU_BOUND);

ExlaExecutable::RunResult run_io(ErlNifEnv *env,
                                 fine::ResourcePtr<ExlaExecutable> executable,
                                 ExlaExecutable::RunArguments arguments,
                                 int64_t device_id) {
  return run(env, executable, arguments, device_id);
}

FINE_NIF(run_io, ERL_NIF_DIRTY_JOB_IO_BOUND);

// Serialization Functions

std::string serialize_executable(ErlNifEnv *env,
                                 fine::ResourcePtr<ExlaExecutable> executable) {
  return unwrap(executable->SerializeExecutable());
}

FINE_NIF(serialize_executable, 0);

fine::ResourcePtr<ExlaExecutable>
deserialize_executable(ErlNifEnv *env, fine::ResourcePtr<ExlaClient> client,
                       std::string serialized) {
  return unwrap(client->DeserializeExecutable(serialized));
}

FINE_NIF(deserialize_executable, 0);

// Memory tracking functions

int64_t get_allocated_memory(ErlNifEnv *env,
                             fine::ResourcePtr<ExlaClient> client) {
  return client->GetAllocatedMemory();
}

FINE_NIF(get_allocated_memory, 0);

int64_t get_peak_memory(ErlNifEnv *env, fine::ResourcePtr<ExlaClient> client) {
  return client->GetPeakMemory();
}

FINE_NIF(get_peak_memory, 0);

fine::Ok<> reset_peak_memory(ErlNifEnv *env,
                             fine::ResourcePtr<ExlaClient> client) {
  client->ResetPeakMemory();
  return fine::Ok();
}

FINE_NIF(reset_peak_memory, 0);

std::map<int64_t, int64_t>
get_per_device_memory(ErlNifEnv *env, fine::ResourcePtr<ExlaClient> client) {
  auto device_memory = client->GetPerDeviceMemory();
  std::map<int64_t, int64_t> result;
  for (const auto &pair : device_memory) {
    result[pair.first] = pair.second;
  }
  return result;
}

FINE_NIF(get_per_device_memory, 0);

// Elixir callback bridge NIF registrations

using callback_bridge::clear_runtime_callback_bridge;
using callback_bridge::runtime_callback_reply;
using callback_bridge::start_runtime_callback_bridge;

FINE_NIF(start_runtime_callback_bridge, 0);
FINE_NIF(runtime_callback_reply, ERL_NIF_DIRTY_JOB_IO_BOUND);
FINE_NIF(clear_runtime_callback_bridge, 0);

// Logging

fine::Ok<> start_log_sink(ErlNifEnv *env, ErlNifPid logger_pid) {
  ExlaLogSink *sink = new ExlaLogSink(logger_pid);

  // In addition to sinks, logs go to stderr above the given threshold.
  // We could disable it entirely with kInfinity, but we keep errors
  // just to make sure they are logged, in case the process crashes
  // right after the log, without time for Elixir to print it.
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kError);
  absl::InitializeLog();
  absl::AddLogSink(sink);

  return fine::Ok();
}

FINE_NIF(start_log_sink, 0);

} // namespace exla

FINE_INIT("Elixir.EXLA.NIF");
