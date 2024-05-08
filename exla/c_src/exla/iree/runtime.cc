#include "runtime.h"

#include <iree/runtime/api.h>

typedef struct {
  void *data;
  size_t size;
  std::vector<int64_t> dims;
  xla::PrimitiveType type;
} IREEInput;

int load_inputs(ErlNifEnv *env, std::vector<ERL_NIF_TERM> terms, std::vector<IREEInput> &loaded) {
  const ERL_NIF_TERM *tuple, *typespec;
  int length;
  ErlNifBinary bin;
  xla::PrimitiveType type;
  std::vector<int64_t> dims;
  IREEInput item;

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

    if (!exla::nif::get_tuple(env, typespec[1], item.dims)) {
      return 0;
    }

    loaded.push_back(item);
  }

  return 1;
}

iree_status_t call_module(iree_runtime_session_t *session, std::vector<IREEInput> inputs) {
  iree_runtime_call_t call;

  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.main"), &call));

  // Append the function inputs with the HAL device allocator in use by the
  // session. The buffers will be usable within the session and _may_ be usable
  // in other sessions depending on whether they share a compatible device.
  iree_hal_device_t *device = iree_runtime_session_device(session);
  iree_hal_allocator_t *device_allocator =
      iree_runtime_session_device_allocator(session);
  iree_allocator_t host_allocator =
      iree_runtime_session_host_allocator(session);

  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    // TO-DO: make this process inputs vector
    // // %lhs: tensor<4xf32>
    // iree_hal_buffer_view_t *lhs = NULL;
    // if (iree_status_is_ok(status)) {
    //   static const iree_hal_dim_t lhs_shape[1] = {4};
    //   static const float lhs_data[4] = {1.0f, 1.1f, 1.2f, 1.3f};
    //   status = iree_hal_buffer_view_allocate_buffer_copy(
    //       device, device_allocator,
    //       // Shape rank and dimensions:
    //       IREE_ARRAYSIZE(lhs_shape), lhs_shape,
    //       // Element type:
    //       IREE_HAL_ELEMENT_TYPE_FLOAT_32,
    //       // Encoding type:
    //       IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
    //       (iree_hal_buffer_params_t){
    //           // Where to allocate (host or device):
    //           .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
    //           // Access to allow to this memory:
    //           .access = IREE_HAL_MEMORY_ACCESS_ALL,
    //           // Intended usage of the buffer (transfers, dispatches, etc):
    //           .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
    //       },
    //       // The actual heap buffer to wrap or clone and its allocator:
    //       iree_make_const_byte_span(lhs_data, sizeof(lhs_data)),
    //       // Buffer view + storage are returned and owned by the caller:
    //       &lhs);
    // if (iree_status_is_ok(status)) {
    //   IREE_IGNORE_ERROR(iree_hal_buffer_view_fprint(
    //       stdout, lhs, /*max_element_count=*/4096, host_allocator));
    //   // Add to the call inputs list (which retains the buffer view).
    //   status = iree_runtime_call_inputs_push_back_buffer_view(&call, lhs);
    // }
    // // Since the call retains the buffer view we can release it here.
    // iree_hal_buffer_view_release(lhs);
  }

  return 0;
}

ERL_NIF_TERM
run_module(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return enif_make_badarg(env);
  }

  ErlNifBinary bytecode_binary;
  std::vector<ERL_NIF_TERM> input_terms;
  std::vector<IREEInput> inputs;

  if (!enif_inspect_binary(env, argv[0], &bytecode_binary)) {
    return exla::nif::error(env, "Unable to load bytecode binary");
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
  char *device_uri = "metal";  // TO-DO: change this to an argument
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

  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_append_bytecode_module_from_memory(session, reinterpret_cast<iree_const_byte_span_t>(bytecode_binary.data), iree_runtime_instance_host_allocator(instance));
  }

  if (iree_status_is_ok(status)) {
    // this is where we actually call code
    // status = iree_runtime_demo_perform_mul(session);
    status = call_module(session, inputs)
  }

  // Release the session and free all cached resources.
  iree_runtime_session_release(session);

  // Release shared device once all sessions using it have been released.
  iree_hal_device_release(device);

  // Release the shared instance - it will be deallocated when all sessions
  // using it have been released (here it is deallocated immediately).
  iree_runtime_instance_release(instance);

  int ret = (int)iree_status_code(status);
  if (!iree_status_is_ok(status)) {
    // Dump nice status messages to stderr on failure.
    // An application can route these through its own logging infrastructure as
    // needed. Note that the status is a handle and must be freed!
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
  }

  if (!ret) {
    exla::nif::error(env, "Fail to execute IREE runtime");
  }

  // TO-DO: we want to get output values too
  return exla::nif::ok(env);
}