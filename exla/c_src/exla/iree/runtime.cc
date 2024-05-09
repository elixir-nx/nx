#include "runtime.h"

#include <iree/hal/api.h>
#include <iree/modules/hal/types.h>
#include <iree/runtime/api.h>

typedef struct {
  void *data;
  size_t size;
  std::vector<iree_hal_dim_t> dims;
  xla::PrimitiveType type;
} IREEInput;

int load_inputs(ErlNifEnv *env, std::vector<ERL_NIF_TERM> terms, std::vector<IREEInput> &loaded) {
  const ERL_NIF_TERM *tuple, *typespec;
  int length;
  ErlNifBinary bin;
  IREEInput item;
  std::vector<int64_t> dims;

  loaded.reserve(terms.size());

  for (ERL_NIF_TERM term : terms) {
    if (!enif_get_tuple(env, term, &length, &tuple)) {
      return 0;
    }

    if (!enif_inspect_binary(env, tuple[0], &bin)) {
      return 0;
    }

    item.data = bin.data;
    item.size = bin.size;

    if (!enif_get_tuple(env, tuple[1], &length, &typespec)) {
      return 0;
    }

    if (!exla::nif::get_primitive_type(env, typespec[0], &item.type)) {
      return 0;
    }

    if (!exla::nif::get_tuple(env, typespec[1], dims)) {
      return 0;
    }

    item.dims.reserve(dims.size());
    for (int64_t dim : dims) {
      item.dims.push_back(dim);
    }

    loaded.push_back(item);
  }

  return 1;
}

std::pair<iree_hal_element_type_t, bool> primitive_type_to_iree_element_type(xla::PrimitiveType t) {
  using xla::PrimitiveType;

  switch (t) {
    case PrimitiveType::PRED:
      return {IREE_HAL_ELEMENT_TYPE_BOOL_8, true};
    case PrimitiveType::S8:
      return {IREE_HAL_ELEMENT_TYPE_INT_8, true};
    case PrimitiveType::S16:
      return {IREE_HAL_ELEMENT_TYPE_INT_16, true};
    case PrimitiveType::S32:
      return {IREE_HAL_ELEMENT_TYPE_INT_32, true};
    case PrimitiveType::S64:
      return {IREE_HAL_ELEMENT_TYPE_INT_64, true};
    case PrimitiveType::U8:
      return {IREE_HAL_ELEMENT_TYPE_UINT_8, true};
    case PrimitiveType::U16:
      return {IREE_HAL_ELEMENT_TYPE_UINT_16, true};
    case PrimitiveType::U32:
      return {IREE_HAL_ELEMENT_TYPE_UINT_32, true};
    case PrimitiveType::U64:
      return {IREE_HAL_ELEMENT_TYPE_UINT_64, true};
    case PrimitiveType::BF16:
      return {IREE_HAL_ELEMENT_TYPE_BFLOAT_16, true};
    case PrimitiveType::F16:
      return {IREE_HAL_ELEMENT_TYPE_FLOAT_16, true};
    case PrimitiveType::F32:
      return {IREE_HAL_ELEMENT_TYPE_FLOAT_32, true};
    case PrimitiveType::F64:
      return {IREE_HAL_ELEMENT_TYPE_FLOAT_64, true};
    case PrimitiveType::C64:
      return {IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64, true};
    case PrimitiveType::C128:
      return {IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128, true};
    default:
      return {IREE_HAL_ELEMENT_TYPE_BOOL_8, false};
  }
}

iree_status_t iree_input_to_hal_arg(iree_hal_buffer_view_t **arg, IREEInput &input, iree_hal_device_t *device, iree_hal_allocator_t *device_allocator) {
  auto result = primitive_type_to_iree_element_type(input.type);
  if (!result.second) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }

  auto type = result.first;
  const iree_const_byte_span_t data_span = iree_make_const_byte_span(input.data, input.size);

  iree_hal_buffer_params_t buffer_params = {
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
  };

  return iree_hal_buffer_view_allocate_buffer_copy(
      device,
      device_allocator,
      input.dims.size(),
      input.dims.data(),
      type,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      buffer_params,
      data_span,
      arg);
}

iree_status_t call_module(iree_runtime_session_t *session, std::vector<IREEInput> inputs, std::vector<ErlNifBinary> *result) {
  iree_runtime_call_t call;

  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.main"), &call));

  // Append the function inputs with the HAL device allocator in use by the
  // session. The buffers will be usable within the session and _may_ be usable
  // in other sessions depending on whether they share a compatible device.
  iree_hal_device_t *device = iree_runtime_session_device(session);
  iree_hal_allocator_t *device_allocator =
      iree_runtime_session_device_allocator(session);
  iree_hal_buffer_view_t *arg;
  for (IREEInput input : inputs) {
    IREE_RETURN_IF_ERROR(iree_input_to_hal_arg(&arg, input, device, device_allocator));
    IREE_RETURN_IF_ERROR(iree_runtime_call_inputs_push_back_buffer_view(&call, arg));
    iree_hal_buffer_view_release(arg);
  }

  std::cout << "before call\n";
  IREE_RETURN_IF_ERROR(iree_runtime_call_invoke(&call, /*flags=*/0));
  std::cout << "after call\n";

  iree_vm_list_t *outputs = iree_runtime_call_outputs(&call);

  std::cout << "size: " << iree_vm_list_size(outputs) << "\n";

  ErlNifBinary binary;
  size_t size = iree_vm_list_size(outputs);

  for (iree_vm_size_t i = 0; i < size; i++) {
    iree_hal_buffer_view_t *buffer_view = nullptr;

    iree_runtime_call_outputs_pop_front_buffer_view(&call, &buffer_view);

    size_t byte_size = iree_hal_buffer_view_byte_length(buffer_view);

    enif_alloc_binary(byte_size, &binary);

    IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(
        iree_runtime_session_device(session),
        iree_hal_buffer_view_buffer(buffer_view), 0, binary.data,
        byte_size, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout()));

    iree_hal_buffer_view_release(buffer_view);

    result->push_back(binary);
  }

  return iree_make_status(IREE_STATUS_OK);
}

ERL_NIF_TERM
run_module(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return enif_make_badarg(env);
  }

  std::vector<ERL_NIF_TERM> bytecode_vec;
  std::vector<ERL_NIF_TERM> input_terms;
  std::vector<IREEInput> inputs;
  std::vector<uint8_t> bytecode;

  if (!exla::nif::get_list(env, argv[0], bytecode_vec)) {
    return exla::nif::error(env, "Unable to load bytecode binary");
  }

  bytecode.resize(bytecode_vec.size());
  unsigned int byte;
  for (int i = 0; i < bytecode_vec.size(); i++) {
    enif_get_uint(env, bytecode_vec[i], &byte);
    bytecode[i] = static_cast<uint8_t>(byte);
  }

  if (!exla::nif::get_list(env, argv[1], input_terms)) {
    return exla::nif::error(env, "Unable to load input terms");
  }

  if (!load_inputs(env, input_terms, inputs)) {
    return exla::nif::error(env, "Unable to decode input terms");
  }

  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t *instance = NULL;
  iree_status_t status = iree_runtime_instance_create(&instance_options, iree_allocator_system(), &instance);

  iree_hal_device_t *device = NULL;
  char device_uri[] = "metal://0000000100000971";  // TO-DO: change this to an argument
  if (iree_status_is_ok(status)) {
    status = iree_hal_create_device(
        iree_runtime_instance_driver_registry(instance),
        iree_make_cstring_view(device_uri),
        iree_runtime_instance_host_allocator(instance), &device);
  }

  iree_runtime_session_t *session = NULL;
  if (iree_status_is_ok(status)) {
    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);
    status = iree_runtime_session_create_with_device(
        instance, &session_options, device,
        iree_runtime_instance_host_allocator(instance), &session);
  }

  iree_const_byte_span_t span{.data = bytecode.data(), .data_length = bytecode.size()};

  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_append_bytecode_module_from_memory(session, span, iree_runtime_instance_host_allocator(instance));
  }

  std::vector<ErlNifBinary> results;
  if (iree_status_is_ok(status)) {
    // this is where we actually call code
    // status = iree_runtime_demo_perform_mul(session);
    status = call_module(session, inputs, &results);
  }

  if (session) {
    // Release the session and free all cached resources.
    iree_runtime_session_release(session);
  }

  if (device) {
    // Release shared device once all sessions using it have been released.
    iree_hal_device_release(device);
  }

  if (instance) {
    // Release the shared instance - it will be deallocated when all sessions
    // using it have been released (here it is deallocated immediately).
    iree_runtime_instance_release(instance);
  }

  if (!iree_status_is_ok(status)) {
    // Dump nice status messages to stderr on failure.
    // An application can route these through its own logging infrastructure as
    // needed. Note that the status is a handle and must be freed!
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    return exla::nif::error(env, "Failed to execute IREE runtime");
  }

  return exla::nif::ok(env, exla::nif::make_list(env, results));
}