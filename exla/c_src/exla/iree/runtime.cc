#include "runtime.h"

#include <iree/hal/api.h>
#include <iree/hal/drivers/init.h>

#include <iostream>
#include <sstream>

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

int load_inputs(ErlNifEnv *env, std::vector<ERL_NIF_TERM> terms, std::vector<exla::iree::runtime::IREEInput *> &loaded) {
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

    loaded.push_back(std::move(new exla::iree::runtime::IREEInput(bin.data, bin.size, dims, type)));
  }

  return 1;
}

ERL_NIF_TERM return_results(ErlNifEnv *env, std::vector<std::pair<iree_hal_element_type_t, ErlNifBinary>> results) {
  size_t n = results.size();

  std::vector<ERL_NIF_TERM> nif_terms;
  nif_terms.clear();
  nif_terms.reserve(n);

  for (auto [iree_type, binary] : results) {
    std::string nx_type;
    if (!iree_element_type_to_nx_type(iree_type, nx_type)) {
      return exla::nif::error(env, "Unable to convert IREE type to Nx type");
    }
    ERL_NIF_TERM type = exla::nif::make(env, nx_type);
    ERL_NIF_TERM bin_term = enif_make_binary(env, &binary);

    nif_terms.push_back(enif_make_tuple2(env, type, bin_term));
  }

  auto data = nif_terms.data();
  auto list = enif_make_list_from_array(env, &data[0], n);
  return exla::nif::ok(env, list);
}

iree_status_t call(iree_hal_device_t *device, std::vector<uint8_t> bytecode, std::vector<exla::iree::runtime::IREEInput *> exla_inputs, std::vector<std::pair<iree_hal_element_type_t, ErlNifBinary>> &results) {
  iree_vm_instance_t *instance = nullptr;
  iree_vm_module_t *hal_module = nullptr;
  iree_vm_module_t *bytecode_module = nullptr;
  iree_vm_context_t *context = nullptr;
  const char kMainFunctionName[] = "module.main";
  iree_vm_function_t main_function;
  iree_vm_list_t *inputs = nullptr;
  iree_vm_list_t *outputs = nullptr;

  IREE_RETURN_IF_ERROR(iree_vm_instance_create(
      IREE_VM_TYPE_CAPACITY_DEFAULT, iree_allocator_system(), &instance));
  IREE_RETURN_IF_ERROR(iree_hal_module_register_all_types(instance));

  IREE_RETURN_IF_ERROR(iree_hal_module_create(
      instance, /*device_count=*/1, &device, IREE_HAL_MODULE_FLAG_SYNCHRONOUS,
      iree_allocator_system(), &hal_module));

  // (kFloat4, sizeof(kFloat4))
  const iree_const_byte_span_t module_data = iree_make_const_byte_span(bytecode.data(), bytecode.size());

  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
      instance, module_data, iree_allocator_null(), iree_allocator_system(),
      &bytecode_module));

  iree_vm_module_t *modules[] = {hal_module, bytecode_module};
  IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(
      instance, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules), &modules[0],
      iree_allocator_system(), &context));
  iree_vm_module_release(hal_module);
  iree_vm_module_release(bytecode_module);

  IREE_RETURN_IF_ERROR(iree_vm_context_resolve_function(
      context, iree_make_cstring_view(kMainFunctionName), &main_function));

  IREE_RETURN_IF_ERROR(
      iree_vm_list_create(iree_vm_make_undefined_type_def(), exla_inputs.size(), iree_allocator_system(), &inputs),
      "can't allocate input vm list");

  for (auto input : exla_inputs) {
    iree_hal_buffer_view_t *arg_buffer_view = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
        device, iree_hal_device_allocator(device), input->dims.size(), input->dims.data(),
        input->type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        },
        input->data_byte_span(), &arg_buffer_view));

    iree_vm_ref_t arg_buffer_view_ref = iree_hal_buffer_view_move_ref(arg_buffer_view);
    IREE_RETURN_IF_ERROR(iree_vm_list_push_ref_move(inputs, &arg_buffer_view_ref));
  }

  iree_vm_function_signature_t signature =
      iree_vm_function_signature(&main_function);
  iree_string_view_t input_signature;
  iree_string_view_t output_signature;

  IREE_RETURN_IF_ERROR(iree_vm_function_call_get_cconv_fragments(
      &signature, &input_signature, &output_signature));

  IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(), output_signature.size, iree_allocator_system(), &outputs), "can't allocate output vm list");

  // Synchronously invoke the function.
  IREE_RETURN_IF_ERROR(iree_vm_invoke(
      context, main_function, IREE_VM_INVOCATION_FLAG_NONE,
      /*policy=*/NULL, inputs, outputs, iree_allocator_system()));

  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);
  return iree_ok_status();
}

ERL_NIF_TERM
run_module(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  std::vector<ERL_NIF_TERM> bytecode_vec = {};
  std::vector<ERL_NIF_TERM> input_terms = {};
  std::vector<exla::iree::runtime::IREEInput *> inputs = {};
  std::vector<uint8_t> bytecode = {};
  iree_hal_device_t **device;

  if (!exla::nif::get<iree_hal_device_t *>(env, argv[0], device)) {
    return exla::nif::error(env, "Unable to load device");
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

  std::vector<std::pair<iree_hal_element_type_t, ErlNifBinary>> results;

  iree_status_t status = call(*device, bytecode, inputs, results);

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
    iree_status_free(status);

    return exla::nif::error(env, ss.str().c_str());
  }

  iree_status_free(status);
  return return_results(env, results);
}

ERL_NIF_TERM setup_runtime(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  iree_hal_device_t *device = nullptr;

  iree_status_t status = iree_hal_register_all_available_drivers(iree_hal_driver_registry_default());

  char device_uri[] = "metal://0000000100000971";  // TO-DO: change this to an argument

  if (iree_status_is_ok(status)) {
    status = iree_hal_create_device(
        iree_hal_driver_registry_default(),
        iree_make_cstring_view(device_uri),
        iree_allocator_system(), &device);
  }

  return iree_status_is_ok(status) ? exla::nif::ok(env, exla::nif::make<iree_hal_device_t *>(env, device)) : exla::nif::error(env, "Failed to setup IREE runtime");
}