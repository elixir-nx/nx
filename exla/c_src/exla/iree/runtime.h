#pragma once
#include <iree/hal/api.h>
#include <iree/modules/hal/module.h>
#include <iree/modules/hal/types.h>
#include <iree/runtime/api.h>
#include <iree/vm/api.h>
#include <iree/vm/bytecode/module.h>

#include <memory>

#include "../exla_nif_util.h"

ERL_NIF_TERM run_module(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM runtime_create_instance(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM runtime_create_session(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);

namespace exla {
namespace iree {
namespace runtime {

class IREEInput {
 public:
  void* data;
  size_t size;
  std::vector<iree_hal_dim_t> dims;
  iree_hal_element_type_t type;

  // Default constructor
  IREEInput(void* data, size_t size, std::vector<int64_t> in_dims, iree_hal_element_type_t type) : size(size), type(type) {
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
  IREEInput(const IREEInput&) = delete;
  IREEInput& operator=(const IREEInput&) = delete;
  IREEInput(IREEInput&&) = delete;
  IREEInput& operator=(IREEInput&&) = delete;
};

class Instance {
 public:
  // Constructor
  explicit Instance(iree_runtime_instance_t* instance, iree_hal_device_t* device)
      : instance_(instance), device_(device) {}

  // Default destructor is fine, unique_ptr will handle the resource release
  ~Instance() {
    iree_hal_device_release(device_);
    iree_runtime_instance_release(instance_);
  }

  // Copy and move operations are disabled to maintain unique ownership semantics
  Instance(const Instance&) = delete;
  Instance& operator=(const Instance&) = delete;
  Instance(Instance&&) noexcept = default;
  Instance& operator=(Instance&&) noexcept = default;

  iree_runtime_instance_t* get() const {
    return instance_;
  }

  iree_runtime_instance_t* operator->() const {
    return instance_;
  }

  iree_hal_device_t* device() const {
    return device_;
  }

 private:
  iree_runtime_instance_t* instance_;
  iree_hal_device_t* device_;
};

class Session {
 public:
  // Constructor
  explicit Session(Instance* instance) : instance_(instance) {}

  iree_status_t initialize(std::vector<uint8_t> bytecode) {
    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);

    iree_vm_instance_t* vm_instance = iree_runtime_instance_vm_instance(instance_->get());
    iree_hal_device_t* device = instance_->device();

    iree_vm_module_t* hal_module = NULL;
    IREE_RETURN_IF_ERROR(iree_hal_module_create(
        vm_instance, /*device_count=*/1, &device, IREE_HAL_MODULE_FLAG_SYNCHRONOUS,
        iree_allocator_system(), &hal_module));

    iree_const_byte_span_t module_data{.data = bytecode.data(), .data_length = bytecode.size()};

    iree_vm_module_t* bytecode_module = NULL;
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
        vm_instance, module_data, iree_allocator_null(), iree_allocator_system(),
        &bytecode_module));

    iree_vm_module_t* modules[] = {hal_module, bytecode_module};
    IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(
        vm_instance, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules), &modules[0],
        iree_allocator_system(), &context_));
    iree_vm_module_release(hal_module);
    iree_vm_module_release(bytecode_module);

    // Lookup the entry point function.
    // Note that we use the synchronous variant which operates on pure type/shape
    // erased buffers.
    const char kMainFunctionName[] = "module.main";
    IREE_RETURN_IF_ERROR(iree_vm_context_resolve_function(
        context_, iree_make_cstring_view(kMainFunctionName), &main_function_));

    return iree_ok_status();
  }

  iree_status_t iree_input_to_hal_arg(iree_hal_buffer_view_t** arg, IREEInput* input, iree_hal_device_t* device, iree_hal_allocator_t* device_allocator) {
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

  iree_status_t init_inputs_and_outputs(std::vector<IREEInput*> inputs) {
    iree_hal_allocator_t* device_allocator = iree_hal_device_allocator(instance_->device());

    iree_vm_function_signature_t signature =
        iree_vm_function_signature(&main_function_);
    iree_string_view_t arguments;
    iree_string_view_t results;

    IREE_RETURN_IF_ERROR(iree_vm_function_call_get_cconv_fragments(
        &signature, &arguments, &results));

    inputs_ = NULL;
    IREE_RETURN_IF_ERROR(
        iree_vm_list_create(iree_vm_make_undefined_type_def(),
                            inputs.size(), iree_allocator_system(), &inputs_),
        "can't allocate input vm list");

    outputs_ = NULL;
    IREE_RETURN_IF_ERROR(
        iree_vm_list_create(iree_vm_make_undefined_type_def(), results.size, iree_allocator_system(), &outputs_),
        "can't allocate output vm list");

    for (size_t i = 0; i < inputs.size(); i++) {
      IREEInput* input = inputs[i];
      // iree_hal_buffer_view_t *buffer_view = nullptr;
      iree_hal_buffer_view_t* arg = nullptr;
      IREE_RETURN_IF_ERROR(iree_input_to_hal_arg(&arg, input, instance()->device(), device_allocator));
      iree_vm_ref_t arg_ref = iree_hal_buffer_view_move_ref(arg);
      IREE_RETURN_IF_ERROR(iree_vm_list_push_ref_move(inputs_, &arg_ref));
    }
  }

  iree_status_t call(std::vector<std::pair<iree_hal_element_type_t, ErlNifBinary>>* result) {
    // Synchronously invoke the function.
    IREE_RETURN_IF_ERROR(iree_vm_invoke(
        context_, main_function_, IREE_VM_INVOCATION_FLAG_NONE,
        /*policy=*/NULL, inputs_, outputs_, iree_allocator_system()));

    ErlNifBinary binary;
    size_t size = iree_vm_list_size(outputs_);

    result->resize(size);

    for (iree_vm_size_t i = 0; i < size; i++) {
      iree_hal_buffer_view_t* buffer_view = nullptr;
      iree_vm_ref_t ref = iree_vm_ref_null();
      IREE_RETURN_IF_ERROR(iree_vm_list_get_ref_assign(outputs_, i, &ref));

      // iree_runtime_call_outputs_pop_front_buffer_view(&call, &buffer_view);
      IREE_RETURN_IF_ERROR(iree_hal_buffer_view_check_deref(ref, &buffer_view));
      iree_hal_element_type_t element_type = iree_hal_buffer_view_element_type(buffer_view);

      iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(buffer_view);
      // size_t byte_size = iree_hal_buffer_view_byte_length(buffer_view);
      size_t byte_size = iree_hal_buffer_byte_length(buffer);
      enif_alloc_binary(byte_size, &binary);

      iree_status_t status = iree_hal_device_transfer_d2h(
          instance_->device(),
          buffer, 0, binary.data,
          byte_size, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
          iree_infinite_timeout());

      if (!iree_status_is_ok(status)) {
        enif_release_binary(&binary);
        return status;
      }

      iree_hal_buffer_view_release(buffer_view);

      (*result)[i] = {element_type, binary};
    }

    return iree_ok_status();
  }

  ~Session() {
    instance_ = nullptr;
    iree_vm_list_release(inputs_);
    iree_vm_list_release(outputs_);
    iree_vm_context_release(context_);
  }

  // Copy and move operations are disabled to maintain unique ownership semantics
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  Session(Session&&) noexcept = default;
  Session& operator=(Session&&) noexcept = default;

  Instance* instance() const {
    return instance_;
  }

 private:
  Instance* instance_;
  iree_vm_context_t* context_;
  iree_vm_list_t* inputs_;
  iree_vm_list_t* outputs_;
  iree_vm_function_t main_function_;
};

}  // namespace runtime
}  // namespace iree
};  // namespace exla