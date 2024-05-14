#include "runtime.h"

#include <iostream>
#include <sstream>

class IREEInput {
 public:
  void *data;
  size_t size;
  std::vector<iree_hal_dim_t> dims;
  iree_hal_element_type_t type;

  // Default constructor
  IREEInput(void *data, size_t size, std::vector<int64_t> in_dims, iree_hal_element_type_t type) : size(size), type(type) {
    dims.reserve(in_dims.size());

    for (auto dim : in_dims) {
      dims.push_back(static_cast<iree_hal_dim_t>(dim));
    }

    this->data = std::malloc(size);  // Allocate memory
    std::memcpy(this->data, data, size);
  }

  // Destructor
  ~IREEInput() {
    if (data) {
      std::free(data);
      data = nullptr;
    }
  }

  // Disable copy and move semantics for simplicity
  IREEInput(const IREEInput &) = delete;
  IREEInput &operator=(const IREEInput &) = delete;
  IREEInput(IREEInput &&) = delete;
  IREEInput &operator=(IREEInput &&) = delete;
};

bool primitive_type_to_iree_element_type(xla::PrimitiveType t, iree_hal_element_type_t *type) {
  using xla::PrimitiveType;
  using type_enum = iree_hal_element_types_t;

  switch (t) {
    case PrimitiveType::PRED:
      *type = type_enum::IREE_HAL_ELEMENT_TYPE_BOOL_8;
      return true;
    case PrimitiveType::S8:
      *type = type_enum::IREE_HAL_ELEMENT_TYPE_INT_8;
      return true;
    case PrimitiveType::S16:
      *type = type_enum::IREE_HAL_ELEMENT_TYPE_INT_16;
      return true;
    case PrimitiveType::S32:
      *type = type_enum::IREE_HAL_ELEMENT_TYPE_INT_32;
      return true;
    case PrimitiveType::S64:
      *type = type_enum::IREE_HAL_ELEMENT_TYPE_INT_32;
      return true;
    case PrimitiveType::U8:
      *type = type_enum::IREE_HAL_ELEMENT_TYPE_UINT_8;
      return true;
    case PrimitiveType::U16:
      *type = type_enum::IREE_HAL_ELEMENT_TYPE_UINT_16;
      return true;
    case PrimitiveType::U32:
      *type = type_enum::IREE_HAL_ELEMENT_TYPE_UINT_32;
      return true;
    case PrimitiveType::U64:
      *type = type_enum::IREE_HAL_ELEMENT_TYPE_UINT_32;
      return true;
    case PrimitiveType::BF16:
      *type = type_enum::IREE_HAL_ELEMENT_TYPE_BFLOAT_16;
      return true;
    case PrimitiveType::F16:
      *type = type_enum::IREE_HAL_ELEMENT_TYPE_FLOAT_16;
      return true;
    case PrimitiveType::F32:
      *type = type_enum::IREE_HAL_ELEMENT_TYPE_FLOAT_32;
      return true;
    case PrimitiveType::F64:
      *type = type_enum::IREE_HAL_ELEMENT_TYPE_FLOAT_64;
      return true;
    case PrimitiveType::C64:
      *type = type_enum::IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64;
      return true;
    case PrimitiveType::C128:
      *type = type_enum::IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128;
      return true;
    default:
      return false;
  }
}

bool iree_element_type_to_nx_type(iree_hal_element_type_t type, std::string &nx_type) {
  using type_enum = iree_hal_element_types_t;

  switch (type) {
    case type_enum::IREE_HAL_ELEMENT_TYPE_BOOL_8:
      nx_type = "pred";
      return true;
    case type_enum::IREE_HAL_ELEMENT_TYPE_INT_8:
      nx_type = "s8";
      return true;
    case type_enum::IREE_HAL_ELEMENT_TYPE_INT_16:
      nx_type = "s16";
      return true;
    case type_enum::IREE_HAL_ELEMENT_TYPE_INT_32:
      nx_type = "s32";
      return true;
    case type_enum::IREE_HAL_ELEMENT_TYPE_INT_64:
      nx_type = "s64";
      return true;
    case type_enum::IREE_HAL_ELEMENT_TYPE_UINT_8:
      nx_type = "u8";
      return true;
    case type_enum::IREE_HAL_ELEMENT_TYPE_UINT_16:
      nx_type = "u16";
      return true;
    case type_enum::IREE_HAL_ELEMENT_TYPE_UINT_32:
      nx_type = "u32";
      return true;
    case type_enum::IREE_HAL_ELEMENT_TYPE_UINT_64:
      nx_type = "u64";
      return true;
    case type_enum::IREE_HAL_ELEMENT_TYPE_BFLOAT_16:
      nx_type = "bf16";
      return true;
    case type_enum::IREE_HAL_ELEMENT_TYPE_FLOAT_16:
      nx_type = "f16";
      return true;
    case type_enum::IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      nx_type = "f32";
      return true;
    case type_enum::IREE_HAL_ELEMENT_TYPE_FLOAT_64:
      nx_type = "f32";
      return true;
    case type_enum::IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64:
      nx_type = "c64";
      return true;
    case type_enum::IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128:
      nx_type = "c64";
      return true;
    default:
      return false;
  }
}

int load_inputs(ErlNifEnv *env, std::vector<ERL_NIF_TERM> terms, std::vector<IREEInput *> &loaded) {
  const ERL_NIF_TERM *tuple, *typespec;
  int length;
  ErlNifBinary bin;

  loaded.clear();
  loaded.reserve(terms.size());

  for (auto term : terms) {
    std::vector<int64_t> dims;
    xla::PrimitiveType primitive_type;
    iree_hal_element_type_t type;

    if (!enif_get_tuple(env, term, &length, &tuple)) {
      return 0;
    }

    if (!enif_inspect_binary(env, tuple[0], &bin)) {
      return 0;
    }

    if (!enif_get_tuple(env, tuple[1], &length, &typespec)) {
      return 0;
    }

    if (!exla::nif::get_primitive_type(env, typespec[0], &primitive_type)) {
      return 0;
    }

    if (!primitive_type_to_iree_element_type(primitive_type, &type)) {
      return 0;
    }

    if (!exla::nif::get_tuple(env, typespec[1], dims)) {
      return 0;
    }

    loaded.push_back(std::move(new IREEInput(bin.data, bin.size, dims, type)));
  }

  return 1;
}

iree_status_t iree_input_to_hal_arg(iree_hal_buffer_view_t **arg, IREEInput *input, iree_hal_device_t *device, iree_hal_allocator_t *device_allocator) {
  const iree_const_byte_span_t data_span = iree_make_const_byte_span(input->data, input->size);

  iree_hal_buffer_params_t buffer_params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
  };

  return iree_hal_buffer_view_allocate_buffer_copy(
      device,
      device_allocator,
      input->dims.size(),
      input->dims.data(),
      input->type,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      buffer_params,
      data_span,
      arg);
}

iree_status_t call_module(exla::iree::runtime::Session *session, std::vector<IREEInput *> inputs, std::vector<std::pair<iree_hal_element_type_t, ErlNifBinary>> *result) {
  iree_runtime_call_t call;

  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(session->get(), iree_make_cstring_view("module.main"), &call));

  iree_hal_allocator_t *device_allocator = iree_runtime_session_device_allocator(session->get());

  for (size_t i = 0; i < inputs.size(); i++) {
    IREEInput *input = inputs[i];
    // iree_hal_buffer_view_t *buffer_view = nullptr;
    iree_hal_buffer_view_t *arg = nullptr;

    IREE_RETURN_IF_ERROR(iree_input_to_hal_arg(&arg, input, session->instance()->device(), device_allocator));
    IREE_RETURN_IF_ERROR(iree_runtime_call_inputs_push_back_buffer_view(&call, arg));
    iree_hal_buffer_view_release(arg);
  }

  IREE_RETURN_IF_ERROR(iree_runtime_call_invoke(&call, /*flags=*/0));

  iree_vm_list_t *outputs = iree_runtime_call_outputs(&call);

  ErlNifBinary binary;
  size_t size = iree_vm_list_size(outputs);

  result->clear();
  result->reserve(size);

  for (iree_vm_size_t i = 0; i < size; i++) {
    iree_hal_buffer_view_t *buffer_view = nullptr;
    iree_vm_ref_t ref = iree_vm_ref_null();
    IREE_RETURN_IF_ERROR(iree_vm_list_get_ref_assign(outputs, i, &ref));

    // iree_runtime_call_outputs_pop_front_buffer_view(&call, &buffer_view);
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_check_deref(ref, &buffer_view));
    iree_hal_element_type_t element_type = iree_hal_buffer_view_element_type(buffer_view);

    iree_hal_buffer_t *buffer = iree_hal_buffer_view_buffer(buffer_view);
    // size_t byte_size = iree_hal_buffer_view_byte_length(buffer_view);
    size_t byte_size = iree_hal_buffer_byte_length(buffer);
    enif_alloc_binary(byte_size, &binary);

    IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(
        iree_runtime_session_device(session->get()),
        buffer, 0, binary.data,
        byte_size, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout()));

    iree_hal_buffer_view_release(buffer_view);

    result->push_back({element_type, binary});
  }

  iree_runtime_call_reset(&call);

  return iree_make_status(IREE_STATUS_OK);
}

ERL_NIF_TERM return_results(ErlNifEnv *env, std::vector<std::pair<iree_hal_element_type_t, ErlNifBinary>> results) {
  size_t n = results.size();

  std::vector<ERL_NIF_TERM> nif_terms;
  nif_terms.clear();
  nif_terms.reserve(n);

  for (auto [iree_type, binary] : results) {
    std::string nx_type;
    if (!iree_element_type_to_nx_type(iree_type, nx_type)) {
      return exla::nif::error(env, "Unable to convert IREE type to NX type");
    }
    ERL_NIF_TERM type = exla::nif::make(env, nx_type);
    ERL_NIF_TERM bin_term = enif_make_binary(env, &binary);

    nif_terms.push_back(enif_make_tuple2(env, type, bin_term));
  }

  auto data = nif_terms.data();
  auto list = enif_make_list_from_array(env, &data[0], n);
  return exla::nif::ok(env, list);
}

ERL_NIF_TERM
run_module(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  std::vector<ERL_NIF_TERM> bytecode_vec = {};
  std::vector<ERL_NIF_TERM> input_terms = {};
  std::vector<IREEInput *> inputs = {};
  std::vector<uint8_t> bytecode = {};
  exla::iree::runtime::Instance **instance;
  iree_status_t status;

  if (!exla::nif::get<exla::iree::runtime::Instance *>(env, argv[0], instance)) {
    return exla::nif::error(env, "Unable to get instance");
  }

  if (!exla::nif::get_list(env, argv[1], bytecode_vec)) {
    return exla::nif::error(env, "Unable to load bytecode binary");
  }

  bytecode.clear();
  bytecode.resize(bytecode_vec.size());
  unsigned int byte;
  for (int i = 0; i < bytecode_vec.size(); i++) {
    enif_get_uint(env, bytecode_vec[i], &byte);
    bytecode[i] = static_cast<uint8_t>(byte);
  }

  if (!exla::nif::get_list(env, argv[2], input_terms)) {
    return exla::nif::error(env, "Unable to load input terms");
  }

  if (!load_inputs(env, input_terms, inputs)) {
    return exla::nif::error(env, "Unable to decode input terms");
  }

  exla::iree::runtime::Session *session = new exla::iree::runtime::Session(*instance);
  status = session->initialize(bytecode);

  if (!iree_status_is_ok(status)) {
    return exla::nif::error(env, "Failed to initialize IREE runtime session");
  }

  std::vector<std::pair<iree_hal_element_type_t, ErlNifBinary>> results;
  status = call_module(session, inputs, &results);

  if (!iree_status_is_ok(status)) {
    // Dump nice status messages to stderr on failure.
    // An application can route these through its own logging infrastructure as
    // needed. Note that the status is a handle and must be freed!

    char *status_string = NULL;
    size_t status_length = 0;

    auto system_allocator = iree_allocator_system();

    iree_status_to_string(status, &system_allocator, &status_string, &status_length);

    std::stringstream ss;
    ss << "Failed to execute IREE runtime due to error: ";
    ss << status_string;
    iree_status_ignore(status);

    return exla::nif::error(env, ss.str().c_str());
  }

  delete session;

  return return_results(env, results);
}

ERL_NIF_TERM runtime_create_instance(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 0) {
    return exla::nif::error(env, "Bad argument count.");
  }

  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t *instance_ptr = NULL;
  iree_status_t status = iree_runtime_instance_create(&instance_options, iree_allocator_system(), &instance_ptr);

  if (!iree_status_is_ok(status)) {
    iree_runtime_instance_release(instance_ptr);
    return exla::nif::error(env, "Failed to create IREE runtime instance");
  }

  iree_hal_device_t *device_ptr = NULL;
  char device_uri[] = "metal://0000000100000971";  // TO-DO: change this to an argument
  status = iree_hal_create_device(
      iree_runtime_instance_driver_registry(instance_ptr),
      iree_make_cstring_view(device_uri),
      iree_runtime_instance_host_allocator(instance_ptr), &device_ptr);

  if (!iree_status_is_ok(status)) {
    if (device_ptr) {
      iree_hal_device_release(device_ptr);
    }
    if (instance_ptr) {
      iree_runtime_instance_release(instance_ptr);
    }
    return exla::nif::error(env, "Failed to create IREE device instance");
  }

  exla::iree::runtime::Instance *instance = new exla::iree::runtime::Instance(instance_ptr, device_ptr);

  return exla::nif::ok(env, exla::nif::make(env, instance));
}