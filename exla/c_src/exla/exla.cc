#include <functional>
#include <iostream>
#include <map>
#include <string>

#include "exla_client.h"
#include "exla_cuda.h"
#include "exla_log_sink.h"
#include "exla_nif_util.h"
#include "mlir/ops.h"
#include "xla/client/client.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/primitive_util.h"
#include "xla/service/platform_util.h"
#include "xla/shape_util.h"

// All of these are created with calls to `new` and subsequently
// passed to the VM as pointers-to-pointers so we balance it out
// with calls to delete rather than just using the default destructor.

void free_exla_executable(ErlNifEnv* env, void* obj) {
  exla::ExlaExecutable** executable = reinterpret_cast<exla::ExlaExecutable**>(obj);
  if (*executable != nullptr) {
    delete *executable;
    *executable = nullptr;
  }
}

void free_xla_builder(ErlNifEnv* env, void* obj) {
  xla::XlaBuilder** builder = reinterpret_cast<xla::XlaBuilder**>(obj);
  if (*builder != nullptr) {
    delete *builder;
    *builder = nullptr;
  }
}

void free_exla_client(ErlNifEnv* env, void* obj) {
  exla::ExlaClient** client = reinterpret_cast<exla::ExlaClient**>(obj);
  if (*client != nullptr) {
    delete *client;
    *client = nullptr;
  }
}

void free_exla_buffer(ErlNifEnv* env, void* obj) {
  exla::ExlaBuffer** buffer = reinterpret_cast<exla::ExlaBuffer**>(obj);
  if (*buffer != nullptr) {
    delete *buffer;
    *buffer = nullptr;
  }
}

static int open_resources(ErlNifEnv* env) {
  const char* mod = "EXLA";

  if (!exla::nif::open_resource<xla::Shape>(env, mod, "Shape")) {
    return -1;
  }
  if (!exla::nif::open_resource<exla::ExlaExecutable*>(env, mod, "Executable", free_exla_executable)) {
    return -1;
  }
  if (!exla::nif::open_resource<xla::XlaBuilder*>(env, mod, "Builder", free_xla_builder)) {
    return -1;
  }
  if (!exla::nif::open_resource<exla::ExlaClient*>(env, mod, "ExlaClient", free_exla_client)) {
    return -1;
  }
  if (!exla::nif::open_resource<exla::ExlaBuffer*>(env, mod, "ExlaBuffer", free_exla_buffer)) {
    return -1;
  }
  // MLIR
  if (!exla::nif::open_resource<exla::MLIRFunction*>(env, mod, "MLIRBlock")) {
    return -1;
  }
  if (!exla::nif::open_resource<mlir::Value>(env, mod, "MLIRValue")) {
    return -1;
  }
  if (!exla::nif::open_resource<mlir::Region*>(env, mod, "MLIRRegion")) {
    return -1;
  }
  if (!exla::nif::open_resource<exla::MLIRModule*>(env, mod, "ExlaMLIRModule")) {
    return -1;
  }

  if (!exla::nif::open_resource<mlir::MLIRContext*>(env, mod, "MLIRContext")) {
    return -1;
  }
  return 1;
}

static int load(ErlNifEnv* env, void** priv, ERL_NIF_TERM load_info) {
  if (open_resources(env) == -1) return -1;

  return 0;
}

// XlaBuilder Functions

ERL_NIF_TERM new_builder(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  std::string name;
  if (!exla::nif::get(env, argv[0], name)) {
    return exla::nif::error(env, "Unable to get builder name.");
  }

  xla::XlaBuilder* builder = new xla::XlaBuilder(name);

  return exla::nif::ok(env, exla::nif::make<xla::XlaBuilder*>(env, builder));
}

ERL_NIF_TERM create_sub_builder(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaBuilder** builder;
  std::string name;

  if (!exla::nif::get<xla::XlaBuilder*>(env, argv[0], builder)) {
    return exla::nif::error(env, "Unable to get builder.");
  }
  if (!exla::nif::get(env, argv[1], name)) {
    return exla::nif::error(env, "Unable to get name.");
  }

  auto uniq_sub_builder = (*builder)->CreateSubBuilder(name);
  xla::XlaBuilder* sub_builder = uniq_sub_builder.release();
  return exla::nif::ok(env, exla::nif::make<xla::XlaBuilder*>(env, sub_builder));
}

// ExlaBuffer Functions

ERL_NIF_TERM get_buffer_device_pointer(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::ExlaClient** client;
  exla::ExlaBuffer** buffer;
  std::string pointer_kind;

  if (!exla::nif::get<exla::ExlaClient*>(env, argv[0], client)) {
    return exla::nif::error(env, "Unable to get client.");
  }
  if (!exla::nif::get<exla::ExlaBuffer*>(env, argv[1], buffer)) {
    return exla::nif::error(env, "Unable to get buffer.");
  }
  if (!exla::nif::get_atom(env, argv[2], pointer_kind)) {
    return exla::nif::error(env, "Unable to get device pointer kind.");
  }

  EXLA_ASSIGN_OR_RETURN_NIF(std::uintptr_t ptr,
                            (*buffer)->GetDevicePointer((*client)->client()), env);

  std::vector<unsigned char> pointer_vec;
  if (pointer_kind == "local") {
    unsigned char* bytePtr = reinterpret_cast<unsigned char*>(&ptr);
    for (size_t i = 0; i < sizeof(void*); i++) {
      pointer_vec.push_back(bytePtr[i]);
    }
  } else if (pointer_kind == "cuda_ipc") {
    auto result = get_cuda_ipc_handle(ptr);
    if (result.second) {
      return exla::nif::error(env, "Unable to get cuda IPC handle");
    }
    pointer_vec = result.first;
  }

  EXLA_ASSIGN_OR_RETURN_NIF(unsigned long device_size, (*buffer)->GetOnDeviceSizeInBytes(), env);

  ERL_NIF_TERM handle_list[pointer_vec.size()];
  for (int i = 0; i < pointer_vec.size(); i++) {
    handle_list[i] = enif_make_uint(env, pointer_vec[i]);
  }

  ERL_NIF_TERM handle_list_term = enif_make_list_from_array(env, handle_list, pointer_vec.size());
  ERL_NIF_TERM device_size_term = enif_make_uint64(env, device_size);

  return exla::nif::ok(env, enif_make_tuple2(env, handle_list_term, device_size_term));
}

ERL_NIF_TERM create_buffer_from_device_pointer(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 5) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::ExlaClient** client;
  std::vector<int64_t> pointer_vec;
  xla::Shape* shape;
  int device_id;
  std::string pointer_kind;

  if (!exla::nif::get<exla::ExlaClient*>(env, argv[0], client)) {
    return exla::nif::error(env, "Unable to get client.");
  }
  if (!exla::nif::get_list(env, argv[1], pointer_vec)) {
    return exla::nif::error(env, "Unable to get device pointer.");
  }
  if (!exla::nif::get_atom(env, argv[2], pointer_kind)) {
    return exla::nif::error(env, "Unable to get device pointer kind.");
  }
  if (!exla::nif::get<xla::Shape>(env, argv[3], shape)) {
    return exla::nif::error(env, "Unable to get shape.");
  }
  if (!exla::nif::get(env, argv[4], &device_id)) {
    return exla::nif::error(env, "Unable to get device ordinal.");
  }

  void* ptr;
  if (pointer_kind == "local") {
    if (pointer_vec.size() != sizeof(void*)) {
      // This helps prevent segfaults if someone passes an IPC handle instead of
      // a local pointer.
      return exla::nif::error(env, "Invalid pointer size for selected mode.");
    }
    unsigned char* bytePtr = reinterpret_cast<unsigned char*>(&ptr);
    for (size_t i = 0; i < sizeof(void*); i++) {
      bytePtr[i] = pointer_vec[i];
    }
  } else if (pointer_kind == "cuda_ipc") {
    auto result = get_pointer_for_ipc_handle(pointer_vec, device_id);
    if (result.second) {
      return exla::nif::error(env, "Unable to get pointer for IPC handle.");
    }
    ptr = result.first;
  }

  EXLA_ASSIGN_OR_RETURN_NIF(xla::PjRtDevice * device, (*client)->client()->LookupDevice(device_id), env);

  std::function<void()> on_delete_callback = []() {};
  EXLA_ASSIGN_OR_RETURN_NIF(std::unique_ptr<xla::PjRtBuffer> buffer, (*client)->client()->CreateViewOfDeviceBuffer(ptr, *shape, device, on_delete_callback), env);
  exla::ExlaBuffer* exla_buffer = new exla::ExlaBuffer(std::move(buffer));
  return exla::nif::ok(env, exla::nif::make<exla::ExlaBuffer*>(env, exla_buffer));
}

ERL_NIF_TERM binary_to_device_mem(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return exla::nif::error(env, "Bad argument count.");
  }

  ErlNifBinary bin;
  xla::Shape* shape;
  exla::ExlaClient** client;
  int device_id;

  if (!exla::nif::get<exla::ExlaClient*>(env, argv[0], client)) {
    return exla::nif::error(env, "Unable to get client.");
  }
  if (!exla::nif::get_binary(env, argv[1], &bin)) {
    return exla::nif::error(env, "Unable to get data.");
  }
  if (!exla::nif::get<xla::Shape>(env, argv[2], shape)) {
    return exla::nif::error(env, "Unable to get shape.");
  }
  if (!exla::nif::get(env, argv[3], &device_id)) {
    return exla::nif::error(env, "Unable to get device ordinal.");
  }

  EXLA_ASSIGN_OR_RETURN_NIF(exla::ExlaBuffer * buffer,
                            (*client)->BufferFromBinary(env, argv[1], *shape, device_id), env);
  return exla::nif::ok(env, exla::nif::make<exla::ExlaBuffer*>(env, buffer));
}

ERL_NIF_TERM read_device_mem(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::ExlaBuffer** buffer;
  exla::int64 size;

  if (!exla::nif::get<exla::ExlaBuffer*>(env, argv[0], buffer)) {
    return exla::nif::error(env, "Unable to get buffer.");
  }
  if (!exla::nif::get(env, argv[1], &size)) {
    return exla::nif::error(env, "Unable to get size.");
  }

  EXLA_ASSIGN_OR_RETURN_NIF(ERL_NIF_TERM binary, (*buffer)->ToBinary(env, size), env);

  return exla::nif::ok(env, binary);
}

ERL_NIF_TERM deallocate_device_mem(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::ExlaBuffer** buffer;

  if (!exla::nif::get<exla::ExlaBuffer*>(env, argv[0], buffer)) {
    return exla::nif::error(env, "Unable to get buffer.");
  }

  xla::Status dealloc_status = (*buffer)->Deallocate();

  if (!dealloc_status.ok()) {
    return exla::nif::atom(env, "already_deallocated");
  } else {
    return exla::nif::ok(env);
  }
}

// Shape Functions

ERL_NIF_TERM make_shape(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::PrimitiveType element_type;
  std::vector<exla::int64> dims;

  if (!exla::nif::get_primitive_type(env, argv[0], &element_type)) {
    return exla::nif::error(env, "Unable to get type.");
  }
  if (!exla::nif::get_tuple(env, argv[1], dims)) {
    return exla::nif::error(env, "Unable to get dimensions.");
  }

  xla::Shape shape = xla::ShapeUtil::MakeShape(element_type, dims);

  return exla::nif::ok(env, exla::nif::make<xla::Shape>(env, shape));
}

ERL_NIF_TERM make_token_shape(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 0) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::Shape shape = xla::ShapeUtil::MakeTokenShape();
  return exla::nif::ok(env, exla::nif::make<xla::Shape>(env, shape));
}

ERL_NIF_TERM get_shape_info(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::Shape* shape;

  if (!exla::nif::get<xla::Shape>(env, argv[0], shape)) {
    return exla::nif::error(env, "Unable to get shape.");
  }

  return exla::nif::ok(env, exla::nif::make_shape_info(env, *shape));
}

ERL_NIF_TERM transfer_to_infeed(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::ExlaClient** client;
  int device_id;
  ERL_NIF_TERM data = argv[2];

  if (!exla::nif::get<exla::ExlaClient*>(env, argv[0], client)) {
    return exla::nif::error(env, "Unable to get client.");
  }
  if (!exla::nif::get(env, argv[1], &device_id)) {
    return exla::nif::error(env, "Unable to get device ID.");
  }

  ERL_NIF_TERM head, tail;
  while (enif_get_list_cell(env, data, &head, &tail)) {
    const ERL_NIF_TERM* terms;
    int count;
    xla::Shape* shape;

    if (!enif_get_tuple(env, head, &count, &terms) && count != 2) {
      return exla::nif::error(env, "Unable to binary-shape tuple.");
    }

    if (!exla::nif::get<xla::Shape>(env, terms[1], shape)) {
      return exla::nif::error(env, "Unable to get shape.");
    }

    xla::Status transfer_status = (*client)->TransferToInfeed(env, terms[0], *shape, device_id);

    if (!transfer_status.ok()) {
      return exla::nif::error(env, transfer_status.message().data());
    }

    data = tail;
  }

  return exla::nif::ok(env);
}

ERL_NIF_TERM transfer_from_outfeed(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 5) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::ExlaClient** client;
  int device_id;
  ErlNifPid pid;

  if (!exla::nif::get<exla::ExlaClient*>(env, argv[0], client)) {
    return exla::nif::error(env, "Unable to get client.");
  }
  if (!exla::nif::get(env, argv[1], &device_id)) {
    return exla::nif::error(env, "Unable to get device ID.");
  }
  if (!enif_get_local_pid(env, argv[3], &pid)) {
    return exla::nif::error(env, "Unable to get pid.");
  }

  ERL_NIF_TERM data = argv[2];
  ERL_NIF_TERM head, tail;
  while (enif_get_list_cell(env, data, &head, &tail)) {
    xla::Shape* shape;

    if (!exla::nif::get<xla::Shape>(env, head, shape)) {
      return exla::nif::error(env, "Unable to get shape.");
    }

    ErlNifEnv* penv = enif_alloc_env();
    ERL_NIF_TERM ref = enif_make_copy(penv, argv[4]);
    auto statusor = (*client)->TransferFromOutfeed(penv, device_id, *shape);

    if (!statusor.ok()) {
      enif_clear_env(penv);
      return exla::nif::error(env, statusor.status().message().data());
    }

    ERL_NIF_TERM msg = std::move(statusor.value());

    if (!enif_send(env, &pid, penv, enif_make_tuple(penv, 2, ref, msg))) {
      enif_clear_env(penv);
    }

    data = tail;
  }

  return exla::nif::ok(env);
}

ERL_NIF_TERM copy_buffer_to_device(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::ExlaClient** client;
  exla::ExlaBuffer** buffer;
  int device_id;

  if (!exla::nif::get<exla::ExlaClient*>(env, argv[0], client)) {
    return exla::nif::error(env, "Unable to get client.");
  }
  if (!exla::nif::get<exla::ExlaBuffer*>(env, argv[1], buffer)) {
    return exla::nif::error(env, "Unable to get buffer.");
  }
  if (!exla::nif::get(env, argv[2], &device_id)) {
    return exla::nif::error(env, "Unable to get device ID.");
  }

  EXLA_ASSIGN_OR_RETURN_NIF(xla::PjRtDevice * device,
                            (*client)->client()->LookupDevice(device_id), env);
  EXLA_ASSIGN_OR_RETURN_NIF(exla::ExlaBuffer * buf,
                            (*buffer)->CopyToDevice(device), env);

  return exla::nif::ok(env, exla::nif::make<exla::ExlaBuffer*>(env, buf));
}

// ExlaClient Functions

ERL_NIF_TERM get_host_client(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 0) {
    return exla::nif::error(env, "Bad argument count.");
  }

  EXLA_ASSIGN_OR_RETURN_NIF(exla::ExlaClient * client, exla::GetHostClient(), env);

  return exla::nif::ok(env, exla::nif::make<exla::ExlaClient*>(env, client));
}

ERL_NIF_TERM get_gpu_client(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  double memory_fraction;
  bool preallocate;

  if (!exla::nif::get(env, argv[0], &memory_fraction)) {
    return exla::nif::error(env, "Unable to get memory fraction.");
  }
  if (!exla::nif::get(env, argv[1], &preallocate)) {
    return exla::nif::error(env, "Unable to get preallocate flag.");
  }
  EXLA_ASSIGN_OR_RETURN_NIF(exla::ExlaClient * client,
                            exla::GetGpuClient(memory_fraction, preallocate, xla::GpuAllocatorConfig::Kind::kBFC), env);

  return exla::nif::ok(env, exla::nif::make<exla::ExlaClient*>(env, client));
}

ERL_NIF_TERM get_tpu_client(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 0) {
    return exla::nif::error(env, "Bad argument count.");
  }

  EXLA_ASSIGN_OR_RETURN_NIF(exla::ExlaClient * client, exla::GetTpuClient(), env);

  return exla::nif::ok(env, exla::nif::make<exla::ExlaClient*>(env, client));
}

ERL_NIF_TERM get_c_api_client(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  std::string device_type;
  if (!exla::nif::get(env, argv[0], device_type)) {
    return exla::nif::error(env, "Unable to get device type.");
  }

  EXLA_ASSIGN_OR_RETURN_NIF(exla::ExlaClient * client, exla::GetCApiClient(device_type), env);

  return exla::nif::ok(env, exla::nif::make<exla::ExlaClient*>(env, client));
}

ERL_NIF_TERM load_pjrt_plugin(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  std::string device_type;
  std::string library_path;
  if (!exla::nif::get(env, argv[0], device_type)) {
    return exla::nif::error(env, "Unable to get device type.");
  }
  if (!exla::nif::get(env, argv[1], library_path)) {
    return exla::nif::error(env, "Unable to get library path.");
  }

  auto result = pjrt::LoadPjrtPlugin(device_type, library_path);

  if (!result.ok()) {
    return exla::nif::error(env, result.status().message().data());
  } else {
    return exla::nif::ok(env);
  }
}

ERL_NIF_TERM get_device_count(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::ExlaClient** client;

  if (!exla::nif::get<exla::ExlaClient*>(env, argv[0], client)) {
    return exla::nif::error(env, "Unable to get client.");
  }

  int device_count = (*client)->client()->device_count();

  return exla::nif::ok(env, exla::nif::make(env, device_count));
}

ERL_NIF_TERM get_supported_platforms(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 0) {
    return exla::nif::error(env, "Bad argument count.");
  }

  EXLA_ASSIGN_OR_RETURN_NIF(
      std::vector<stream_executor::Platform*> platforms,
      xla::PlatformUtil::GetSupportedPlatforms(),
      env);

  std::vector<std::string> platform_names;
  std::map<std::string, int> platform_info;

  for (auto& platform : platforms) {
    std::string key = platform->Name();
    int device_count = platform->VisibleDeviceCount();
    platform_info.insert({key, device_count});
  }

  return exla::nif::ok(env, exla::nif::make_map(env, platform_info));
}

// ExlaExecutable Functions

ERL_NIF_TERM run(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::ExlaClient** client;
  exla::ExlaExecutable** executable;
  int device_id;

  ERL_NIF_TERM arguments = argv[2];

  if (!exla::nif::get<exla::ExlaClient*>(env, argv[0], client)) {
    return exla::nif::error(env, "Unable to get client.");
  }
  if (!exla::nif::get<exla::ExlaExecutable*>(env, argv[1], executable)) {
    return exla::nif::error(env, "Unable to get executable.");
  }
  if (!exla::nif::get(env, argv[3], &device_id)) {
    return exla::nif::error(env, "Unable to get device ID.");
  }

  EXLA_ASSIGN_OR_RETURN_NIF(ERL_NIF_TERM term,
                            (*executable)->Run(env, arguments, device_id), env);

  return term;
}

// Serialization Functions

ERL_NIF_TERM serialize_executable(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::ExlaExecutable** executable;

  if (!exla::nif::get<exla::ExlaExecutable*>(env, argv[0], executable)) {
    return exla::nif::error(env, "Unable to get executable.");
  }

  EXLA_ASSIGN_OR_RETURN_NIF(std::string serialized, (*executable)->SerializeExecutable(), env);
  ErlNifBinary raw;
  enif_alloc_binary(serialized.size(), &raw);
  std::memcpy((&raw)->data, serialized.data(), serialized.size());

  return exla::nif::ok(env, exla::nif::make(env, raw));
}

ERL_NIF_TERM deserialize_executable(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::ExlaClient** client;
  std::string serialized;

  if (!exla::nif::get<exla::ExlaClient*>(env, argv[0], client)) {
    return exla::nif::error(env, "Unable to get client.");
  }
  if (!exla::nif::get(env, argv[1], serialized)) {
    return exla::nif::error(env, "Unable to get executable.");
  }

  EXLA_ASSIGN_OR_RETURN_NIF(exla::ExlaExecutable * executable,
                            (*client)->DeserializeExecutable(serialized), env);

  return exla::nif::ok(env, exla::nif::make<exla::ExlaExecutable*>(env, executable));
}

// Logging

ERL_NIF_TERM start_log_sink(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  ErlNifPid logger_pid;

  if (!enif_get_local_pid(env, argv[0], &logger_pid)) {
    return exla::nif::error(env, "Unable to get logger pid");
  }

  exla::ExlaLogSink* sink = new exla::ExlaLogSink(logger_pid);

  // NO_DEFAULT_LOGGER doesn't behave right
  for (auto* log_sink : tsl::TFGetLogSinks()) {
    tsl::TFRemoveLogSink(log_sink);
  }

  tsl::TFAddLogSink(sink);

  return exla::nif::ok(env);
}

static ErlNifFunc exla_funcs[] = {
    // MLIR Builder
    {"new_mlir_context", 0, new_mlir_context},
    {"new_mlir_module", 1, new_mlir_module},
    {"create_mlir_function", 5, create_mlir_function},
    {"get_mlir_function_arguments", 1, get_mlir_function_arguments},
    {"mlir_add", 3, mlir_add},
    {"mlir_subtract", 3, mlir_subtract},
    {"mlir_multiply", 3, mlir_multiply},
    {"mlir_min", 3, mlir_min},
    {"mlir_max", 3, mlir_max},
    {"mlir_remainder", 3, mlir_remainder},
    {"mlir_pow", 3, mlir_pow},
    {"mlir_divide", 3, mlir_divide},
    {"mlir_atan2", 3, mlir_atan2},
    {"mlir_equal", 3, mlir_equal},
    {"mlir_not_equal", 3, mlir_not_equal},
    {"mlir_less", 3, mlir_less},
    {"mlir_less_equal", 3, mlir_less_equal},
    {"mlir_greater", 3, mlir_greater},
    {"mlir_greater_equal", 3, mlir_greater_equal},
    {"dump_mlir_module", 1, dump_mlir_module},
    {"mlir_get_shape", 1, mlir_get_shape},
    {"mlir_convert", 3, mlir_convert},
    {"mlir_bitcast_convert", 4, mlir_bitcast_convert},
    {"mlir_abs", 2, mlir_abs},
    {"mlir_exp", 2, mlir_exp},
    {"mlir_expm1", 2, mlir_expm1},
    {"mlir_floor", 2, mlir_floor},
    {"mlir_ceil", 2, mlir_ceil},
    {"mlir_round", 2, mlir_round},
    {"mlir_log", 2, mlir_log},
    {"mlir_sigmoid", 2, mlir_sigmoid},
    {"mlir_log1p", 2, mlir_log1p},
    {"mlir_sign", 2, mlir_sign},
    {"mlir_cos", 2, mlir_cos},
    {"mlir_sin", 2, mlir_sin},
    {"mlir_tan", 2, mlir_tan},
    {"mlir_acos", 2, mlir_acos},
    {"mlir_asin", 2, mlir_asin},
    {"mlir_atan", 2, mlir_atan},
    {"mlir_cosh", 2, mlir_cosh},
    {"mlir_sinh", 2, mlir_sinh},
    {"mlir_tanh", 2, mlir_tanh},
    {"mlir_acosh", 2, mlir_acosh},
    {"mlir_asinh", 2, mlir_asinh},
    {"mlir_atanh", 2, mlir_atanh},
    {"mlir_sqrt", 2, mlir_sqrt},
    {"mlir_cbrt", 2, mlir_cbrt},
    {"mlir_iota", 3, mlir_iota},
    {"mlir_top_k", 3, mlir_top_k},
    {"mlir_sort", 5, mlir_sort},
    {"mlir_scatter", 9, mlir_scatter},
    {"mlir_select_and_scatter", 8, mlir_select_and_scatter},
    {"mlir_gather", 8, mlir_gather},
    {"mlir_reshape", 3, mlir_reshape},
    {"mlir_reverse", 3, mlir_reverse},
    {"mlir_transpose", 3, mlir_transpose},
    {"mlir_slice", 5, mlir_slice},
    {"mlir_dynamic_slice", 4, mlir_dynamic_slice},
    {"mlir_constant_r0", 3, mlir_constant_r0},
    {"mlir_constant_from_binary", 4, mlir_constant_from_binary},
    {"mlir_bitwise_and", 3, mlir_bitwise_and},
    {"mlir_bitwise_or", 3, mlir_bitwise_or},
    {"mlir_bitwise_xor", 3, mlir_bitwise_xor},
    {"mlir_bitwise_not", 2, mlir_bitwise_not},
    {"mlir_left_shift", 3, mlir_shift_left},
    {"mlir_right_shift_logical", 3, mlir_shift_right_logical},
    {"mlir_right_shift_arithmetic", 3, mlir_shift_right_arithmetic},
    {"mlir_negate", 2, mlir_negate},
    {"mlir_erf", 2, mlir_erf},
    {"mlir_erfc", 2, mlir_erfc},
    {"mlir_erf_inv", 2, mlir_erf_inv},
    {"mlir_is_infinity", 2, mlir_is_infinity},
    {"mlir_is_nan", 2, mlir_is_nan},
    {"mlir_rsqrt", 2, mlir_rsqrt},
    {"mlir_count_leading_zeros", 2, mlir_clz},
    {"mlir_real", 2, mlir_real},
    {"mlir_imag", 2, mlir_imag},
    {"mlir_conjugate", 2, mlir_conjugate},
    {"mlir_dot_general", 6, mlir_dot_general},
    {"mlir_clamp", 4, mlir_clamp},
    {"mlir_population_count", 2, mlir_population_count},
    {"mlir_broadcast_in_dim", 4, mlir_broadcast_in_dim},
    {"mlir_concatenate", 3, mlir_concatenate},
    {"mlir_optimization_barrier", 2, mlir_optimization_barrier},
    {"mlir_select", 4, mlir_select},
    {"mlir_pad", 6, mlir_pad},
    {"mlir_fft", 4, mlir_fft},
    {"mlir_convolution", 12, mlir_convolution},
    {"mlir_create_token", 1, mlir_create_token},
    {"mlir_triangular_solve", 6, mlir_triangular_solve},
    {"mlir_dynamic_update_slice", 4, mlir_dynamic_update_slice},
    {"mlir_infeed", 3, mlir_infeed},
    {"mlir_outfeed", 3, mlir_outfeed},
    {"mlir_call", 3, mlir_call},
    {"mlir_reduce", 5, mlir_reduce},
    {"mlir_window_reduce", 9, mlir_window_reduce},
    {"mlir_map", 4, mlir_map},
    {"mlir_if", 3, mlir_if},
    {"mlir_push_region", 2, mlir_push_region},
    {"mlir_pop_region", 1, mlir_pop_region},
    {"mlir_while", 2, mlir_while},
    {"mlir_return", 2, mlir_return},
    {"mlir_qr", 4, mlir_qr},
    // ExlaClient
    {"get_host_client", 0, get_host_client},
    {"get_gpu_client", 2, get_gpu_client},
    {"get_tpu_client", 0, get_tpu_client},
    {"get_c_api_client", 1, get_c_api_client},
    {"load_pjrt_plugin", 2, load_pjrt_plugin},
    {"get_device_count", 1, get_device_count},
    {"get_supported_platforms", 0, get_supported_platforms},
    {"mlir_compile", 7, mlir_compile, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    // ExlaBuffer
    {"get_buffer_device_pointer", 3, get_buffer_device_pointer},
    {"create_buffer_from_device_pointer", 5, create_buffer_from_device_pointer},
    {"binary_to_device_mem", 4, binary_to_device_mem, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"read_device_mem", 2, read_device_mem, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"deallocate_device_mem", 1, deallocate_device_mem, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"transfer_to_infeed", 3, transfer_to_infeed, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"transfer_from_outfeed", 5, transfer_from_outfeed, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"copy_buffer_to_device", 3, copy_buffer_to_device, ERL_NIF_DIRTY_JOB_IO_BOUND},
    // ExlaExecutable
    {"run_io", 4, run, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"run_cpu", 4, run, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    // Shape
    {"make_shape", 2, make_shape},
    {"make_token_shape", 0, make_token_shape},
    {"get_shape_info", 1, get_shape_info},
    // Log Sink
    {"start_log_sink", 1, start_log_sink},
    // Serialization
    {"serialize_executable", 1, serialize_executable},
    {"deserialize_executable", 2, deserialize_executable}};

ERL_NIF_INIT(Elixir.EXLA.NIF, exla_funcs, &load, NULL, NULL, NULL);
