#include <string>

#include "exla_client.h"
#include "exla_cuda.h"
#include "exla_log_sink.h"
#include "exla_mlir.h"
#include "exla_nif_util.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/service/platform_util.h"

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

  if (!exla::nif::open_resource<exla::ExlaExecutable*>(env, mod, "Executable", free_exla_executable)) {
    return -1;
  }
  if (!exla::nif::open_resource<exla::ExlaClient*>(env, mod, "ExlaClient", free_exla_client)) {
    return -1;
  }
  if (!exla::nif::open_resource<exla::ExlaBuffer*>(env, mod, "ExlaBuffer", free_exla_buffer)) {
    return -1;
  }
  // MLIR
  if (!exla::nif::open_resource<exla::MLIRFunction*>(env, mod, "MLIRFunction")) {
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

// MLIR Functions

ERL_NIF_TERM type_parsing_error(ErlNifEnv* env, std::string type_string) {
  return exla::nif::make(env, "Unable to parse MLIR type: " + type_string);
}

ERL_NIF_TERM attribute_parsing_error(ErlNifEnv* env, std::string attribute_string) {
  return exla::nif::make(env, "Unable to parse MLIR attribute: " + attribute_string);
}

ERL_NIF_TERM mlir_compile(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 7) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::ExlaClient** client;
  exla::MLIRModule** module;
  std::vector<xla::Shape> argument_layouts;
  xla::ExecutableBuildOptions build_options;
  int num_replicas;
  int num_partitions;
  bool use_spmd;
  int device_id;

  if (!exla::nif::get<exla::ExlaClient*>(env, argv[0], client)) {
    return exla::nif::error(env, "Unable to get client.");
  }
  if (!exla::nif::get<exla::MLIRModule*>(env, argv[1], module)) {
    return exla::nif::error(env, "Unable to get module.");
  }
  if (!exla::nif::get_list(env, argv[2], argument_layouts)) {
    return exla::nif::error(env, "Unable to get argument layouts.");
  }
  if (!exla::nif::get(env, argv[3], &num_replicas)) {
    return exla::nif::error(env, "Unable to get Number of Replicas.");
  }
  if (!exla::nif::get(env, argv[4], &num_partitions)) {
    return exla::nif::error(env, "Unable to get Number of Partitions.");
  }
  if (!exla::nif::get(env, argv[5], &use_spmd)) {
    return exla::nif::error(env, "Unable to get SPMD Partitioning Flag.");
  }
  if (!exla::nif::get(env, argv[6], &device_id)) {
    return exla::nif::error(env, "Unable to get device ID.");
  }

  build_options.set_num_replicas(num_replicas);
  build_options.set_num_partitions(num_partitions);
  build_options.set_use_spmd_partitioning(use_spmd);

  bool compile_portable_executable = false;
  if (device_id >= 0) {
    compile_portable_executable = true;
    build_options.set_device_ordinal(device_id);
  }

  EXLA_ASSIGN_OR_RETURN_NIF(exla::ExlaExecutable * executable,
                            (*client)->Compile((*module)->module(), argument_layouts, build_options, compile_portable_executable), env);

  return exla::nif::ok(env, exla::nif::make<exla::ExlaExecutable*>(env, executable));
}

ERL_NIF_TERM mlir_new_context(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 0) {
    return exla::nif::error(env, "Bad argument count.");
  }

  mlir::MLIRContext* context = new mlir::MLIRContext();
  context->getOrLoadDialect<mlir::func::FuncDialect>();
  context->getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
  context->getOrLoadDialect<mlir::mhlo::MhloDialect>();
  context->getOrLoadDialect<mlir::chlo::ChloDialect>();

  auto ret = exla::nif::make<mlir::MLIRContext*>(env, context);
  return exla::nif::ok(env, ret);
}

ERL_NIF_TERM mlir_new_module(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  mlir::MLIRContext** ctx;

  if (!exla::nif::get<mlir::MLIRContext*>(env, argv[0], ctx)) {
    return exla::nif::error(env, "Unable to get context.");
  }

  exla::MLIRModule* module = new exla::MLIRModule(*ctx);

  return exla::nif::ok(env, exla::nif::make<exla::MLIRModule*>(env, module));
}

ERL_NIF_TERM mlir_create_function(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 5) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRModule** module;
  std::string func_name;
  std::vector<std::string> arg_type_strings;
  std::vector<std::string> ret_type_strings;
  bool is_public;

  if (!exla::nif::get<exla::MLIRModule*>(env, argv[0], module)) {
    return exla::nif::error(env, "Unable to get module.");
  }
  if (!exla::nif::get(env, argv[1], func_name)) {
    return exla::nif::error(env, "Unable to get function name.");
  }
  if (!exla::nif::get_list(env, argv[2], arg_type_strings)) {
    return exla::nif::error(env, "Unable to get args.");
  }
  if (!exla::nif::get_list(env, argv[3], ret_type_strings)) {
    return exla::nif::error(env, "Unable to get return.");
  }
  if (!exla::nif::get(env, argv[4], &is_public)) {
    return exla::nif::error(env, "Unable to get is_public.");
  }

  auto arg_types = std::vector<mlir::Type>{};

  for (auto const& type_string : arg_type_strings) {
    auto type = (*module)->ParseType(type_string);
    if (type == nullptr) {
      return type_parsing_error(env, type_string);
    }
    arg_types.push_back(type);
  }

  auto ret_types = std::vector<mlir::Type>{};

  for (auto const& type_string : ret_type_strings) {
    auto type = (*module)->ParseType(type_string);
    if (type == nullptr) {
      return type_parsing_error(env, type_string);
    }
    ret_types.push_back(type);
  }

  exla::MLIRFunction* func = (*module)->CreateFunction(func_name, arg_types, ret_types, is_public);

  return exla::nif::ok(env, exla::nif::make<exla::MLIRFunction*>(env, func));
}

ERL_NIF_TERM mlir_get_function_arguments(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }

  llvm::MutableArrayRef<mlir::BlockArgument> args = (*function)->GetArguments();
  std::vector<ERL_NIF_TERM> terms;
  terms.reserve(args.size());

  for (auto arg : args) {
    ERL_NIF_TERM term = exla::nif::make<mlir::Value>(env, arg);
    terms.push_back(term);
  }

  return exla::nif::ok(env, enif_make_list_from_array(env, terms.data(), terms.size()));
}

ERL_NIF_TERM mlir_op(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 6) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  std::string op_name;
  std::vector<mlir::Value> operands;
  std::vector<std::string> result_type_strings;
  std::vector<std::pair<std::string, std::string>> attributes_kwlist;
  std::vector<mlir::Region*> regions;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get(env, argv[1], op_name)) {
    return exla::nif::error(env, "Unable to get op name.");
  }
  if (!exla::nif::get_list<mlir::Value>(env, argv[2], operands)) {
    return exla::nif::error(env, "Unable to get operands.");
  }
  if (!exla::nif::get_list(env, argv[3], result_type_strings)) {
    return exla::nif::error(env, "Unable to get result types.");
  }
  if (!exla::nif::get_keyword_list<std::string>(env, argv[4], attributes_kwlist)) {
    return exla::nif::error(env, "Unable to get attributes.");
  }
  if (!exla::nif::get_list<mlir::Region*>(env, argv[5], regions)) {
    return exla::nif::error(env, "Unable to get regions.");
  }

  auto result_types = std::vector<mlir::Type>{};

  for (auto const& type_string : result_type_strings) {
    auto type = (*function)->module()->ParseType(type_string);
    if (type == nullptr) {
      return type_parsing_error(env, type_string);
    }
    result_types.push_back(type);
  }

  auto attributes = std::vector<std::pair<std::string, mlir::Attribute>>{};

  for (auto const& pair : attributes_kwlist) {
    auto attribute_value = (*function)->module()->ParseAttribute(pair.second);
    if (attribute_value == nullptr) {
      return attribute_parsing_error(env, pair.second);
    }
    attributes.push_back(std::pair{pair.first, attribute_value});
  }

  auto results = (*function)->Op(op_name, operands, result_types, attributes, regions);

  return exla::nif::ok(env, exla::nif::make_list<mlir::Value>(env, results));
}

ERL_NIF_TERM mlir_push_region(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  std::vector<std::string> arg_types;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get_list(env, argv[1], arg_types)) {
    return exla::nif::error(env, "Unable to get arg types.");
  }

  auto types = std::vector<mlir::Type>{};

  for (auto const& type_string : arg_types) {
    auto type = (*function)->module()->ParseType(type_string);
    if (type == nullptr) {
      return type_parsing_error(env, type_string);
    }
    types.push_back(type);
  }

  mlir::Region* region;
  std::vector<mlir::Value> args;
  std::tie(region, args) = (*function)->PushRegion(types);

  return exla::nif::ok(env, enif_make_tuple2(env, exla::nif::make<mlir::Region*>(env, region), exla::nif::make_list<mlir::Value>(env, args)));
}

ERL_NIF_TERM mlir_pop_region(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }

  (*function)->PopRegion();
  return exla::nif::ok(env);
}

ERL_NIF_TERM mlir_get_typespec(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  mlir::Value* t;

  if (!exla::nif::get<mlir::Value>(env, argv[0], t)) {
    return exla::nif::error(env, "Unable to get tensor.");
  }

  mlir::Type type = t->getType();

  return exla::nif::ok(env, exla::nif::make_typespec(env, type));
}

ERL_NIF_TERM mlir_module_to_string(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRModule** module;

  if (!exla::nif::get<exla::MLIRModule*>(env, argv[0], module)) {
    return exla::nif::error(env, "Unable to get builder.");
  }

  std::string string = (*module)->ToString();

  ErlNifBinary bin;
  enif_alloc_binary(string.size(), &bin);
  memcpy(bin.data, string.c_str(), string.size());

  return exla::nif::ok(env, exla::nif::make(env, bin));
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
  xla::Shape shape;
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
  if (!exla::nif::get_typespec_as_xla_shape(env, argv[3], &shape)) {
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

  EXLA_ASSIGN_OR_RETURN_NIF(xla::PjRtDevice * device, (*client)->client()->LookupDevice(xla::PjRtGlobalDeviceId(device_id)), env);

  std::function<void()> on_delete_callback = []() {};
  EXLA_ASSIGN_OR_RETURN_NIF(std::unique_ptr<xla::PjRtBuffer> buffer, (*client)->client()->CreateViewOfDeviceBuffer(ptr, shape, device, on_delete_callback), env);
  exla::ExlaBuffer* exla_buffer = new exla::ExlaBuffer(std::move(buffer));
  return exla::nif::ok(env, exla::nif::make<exla::ExlaBuffer*>(env, exla_buffer));
}

ERL_NIF_TERM binary_to_device_mem(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return exla::nif::error(env, "Bad argument count.");
  }

  ErlNifBinary bin;
  xla::Shape shape;
  exla::ExlaClient** client;
  int device_id;

  if (!exla::nif::get<exla::ExlaClient*>(env, argv[0], client)) {
    return exla::nif::error(env, "Unable to get client.");
  }
  if (!exla::nif::get_binary(env, argv[1], &bin)) {
    return exla::nif::error(env, "Unable to get data.");
  }
  if (!exla::nif::get_typespec_as_xla_shape(env, argv[2], &shape)) {
    return exla::nif::error(env, "Unable to get shape.");
  }
  if (!exla::nif::get(env, argv[3], &device_id)) {
    return exla::nif::error(env, "Unable to get device ordinal.");
  }

  EXLA_ASSIGN_OR_RETURN_NIF(exla::ExlaBuffer * buffer,
                            (*client)->BufferFromBinary(env, argv[1], shape, device_id), env);
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

  std::vector<ErlNifBinary> buffer_bins;
  std::vector<xla::Shape> shapes;

  ERL_NIF_TERM head, tail;
  while (enif_get_list_cell(env, data, &head, &tail)) {
    const ERL_NIF_TERM* terms;
    int count;

    if (!enif_get_tuple(env, head, &count, &terms) && count != 2) {
      return exla::nif::error(env, "Unable to {binary, shape} tuple.");
    }

    ErlNifBinary buffer_bin;
    if (!exla::nif::get_binary(env, terms[0], &buffer_bin)) {
      return exla::nif::error(env, "Unable to binary.");
    }

    xla::Shape shape;
    if (!exla::nif::get_typespec_as_xla_shape(env, terms[1], &shape)) {
      return exla::nif::error(env, "Unable to get shape.");
    }

    buffer_bins.push_back(buffer_bin);
    shapes.push_back(shape);

    data = tail;
  }

  xla::Status transfer_status = (*client)->TransferToInfeed(env, buffer_bins, shapes, device_id);

  if (!transfer_status.ok()) {
    return exla::nif::error(env, transfer_status.message().data());
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
    xla::Shape shape;

    if (!exla::nif::get_typespec_as_xla_shape(env, head, &shape)) {
      return exla::nif::error(env, "Unable to get shape.");
    }

    ErlNifEnv* penv = enif_alloc_env();
    ERL_NIF_TERM ref = enif_make_copy(penv, argv[4]);
    auto statusor = (*client)->TransferFromOutfeed(penv, device_id, shape);

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
                            (*client)->client()->LookupDevice(xla::PjRtGlobalDeviceId(device_id)), env);
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
    {"mlir_new_context", 0, mlir_new_context},
    {"mlir_new_module", 1, mlir_new_module},
    {"mlir_create_function", 5, mlir_create_function},
    {"mlir_get_function_arguments", 1, mlir_get_function_arguments},
    {"mlir_op", 6, mlir_op},
    {"mlir_push_region", 2, mlir_push_region},
    {"mlir_get_typespec", 1, mlir_get_typespec},
    {"mlir_pop_region", 1, mlir_pop_region},
    {"mlir_module_to_string", 1, mlir_module_to_string},
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
    // Log Sink
    {"start_log_sink", 1, start_log_sink},
    // Serialization
    {"serialize_executable", 1, serialize_executable},
    {"deserialize_executable", 2, deserialize_executable}};

ERL_NIF_INIT(Elixir.EXLA.NIF, exla_funcs, &load, NULL, NULL, NULL);
