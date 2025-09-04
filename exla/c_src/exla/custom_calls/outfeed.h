#pragma once

#include <cstring>
#include <erl_nif.h>

#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace exla_outfeed {
namespace ffi = xla::ffi;

inline size_t product(const ffi::Span<const int64_t> dims) {
  size_t p = 1;
  for (auto d : dims)
    p *= static_cast<size_t>(d);
  return p;
}

// Template for outfeed implementation that sends data to processes
template <typename DataType, typename BufferType>
static inline ffi::Error
outfeed_cpu_custom_call_impl_typed(BufferType tensor_data,
                                   ffi::Buffer<ffi::U8> pid_buffer) {
  ErlNifEnv *env = enif_alloc_env();
  if (env == nullptr) {
    return ffi::Error::Internal("enomem");
  }

  // Decode PID from binary
  const size_t pid_bytes = product(pid_buffer.dimensions());
  ERL_NIF_TERM pid_term;
  if (!enif_binary_to_term(
          env,
          reinterpret_cast<const unsigned char *>(pid_buffer.untyped_data()),
          pid_bytes, &pid_term, 0)) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("invalid_pid");
  }

  ErlNifPid pid;
  if (!enif_get_local_pid(env, pid_term, &pid)) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("not_local_pid");
  }

  // Get tensor data
  const void *data_ptr = tensor_data.untyped_data();
  size_t data_bytes = tensor_data.size_bytes();

  // Create binary term for the tensor
  ERL_NIF_TERM tensor_term;
  unsigned char *msg_ptr = enif_make_new_binary(env, data_bytes, &tensor_term);
  std::memcpy(msg_ptr, data_ptr, data_bytes);

  // Create list with single tensor
  ERL_NIF_TERM tensor_list = enif_make_list1(env, tensor_term);

  // Send the list to the process
  enif_send(env, &pid, env, tensor_list);

  enif_free_env(env);
  return ffi::Error::Success();
}

// Single tensor outfeed for testing the pattern
static inline ffi::Error
outfeed_cpu_custom_call_s32_impl(ffi::Buffer<ffi::S32> tensor_data,
                                 ffi::Buffer<ffi::U8> pid_buffer) {
  return outfeed_cpu_custom_call_impl_typed<int32_t>(tensor_data, pid_buffer);
}

// Generic outfeed implementation using RemainingArgs (fallback)
static inline ffi::Error
outfeed_cpu_custom_call_impl(ffi::RemainingArgs remaining_args) {
  // Fallback implementation - try to handle multiple tensors
  if (remaining_args.size() < 2) {
    return ffi::Error::InvalidArgument("insufficient_args");
  }

  // Last argument should be the PID
  auto pid_arg_or_error =
      remaining_args.get<ffi::Buffer<ffi::U8>>(remaining_args.size() - 1);
  if (!pid_arg_or_error.has_value()) {
    return ffi::Error::InvalidArgument("invalid_pid_arg");
  }
  ffi::Buffer<ffi::U8> pid_buffer = pid_arg_or_error.value();

  ErlNifEnv *env = enif_alloc_env();
  if (env == nullptr) {
    return ffi::Error::Internal("enomem");
  }

  // Decode PID from binary
  const size_t pid_bytes = product(pid_buffer.dimensions());
  ERL_NIF_TERM pid_term;
  if (!enif_binary_to_term(
          env,
          reinterpret_cast<const unsigned char *>(pid_buffer.untyped_data()),
          pid_bytes, &pid_term, 0)) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("invalid_pid");
  }

  ErlNifPid pid;
  if (!enif_get_local_pid(env, pid_term, &pid)) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("not_local_pid");
  }

  // Create a list of binaries for all tensor data (excluding PID)
  size_t num_tensors = remaining_args.size() - 1;
  ERL_NIF_TERM *tensor_terms = new ERL_NIF_TERM[num_tensors];

  for (size_t i = 0; i < num_tensors; ++i) {
    auto arg_or_error = remaining_args.get<ffi::AnyBuffer>(i);
    if (!arg_or_error.has_value()) {
      delete[] tensor_terms;
      enif_free_env(env);
      return ffi::Error::InvalidArgument("invalid_tensor_arg");
    }

    auto arg = arg_or_error.value();

    // Get the buffer data regardless of type
    const void *data_ptr = arg.untyped_data();
    size_t data_bytes = arg.size_bytes();

    // Create binary term
    unsigned char *msg_ptr =
        enif_make_new_binary(env, data_bytes, &tensor_terms[i]);
    std::memcpy(msg_ptr, data_ptr, data_bytes);
  }

  // Create list of tensor binaries
  ERL_NIF_TERM tensor_list =
      enif_make_list_from_array(env, tensor_terms, num_tensors);

  // Send the list to the process
  enif_send(env, &pid, env, tensor_list);

  delete[] tensor_terms;
  enif_free_env(env);
  return ffi::Error::Success();
}

// Main outfeed implementation for token-based calls
// Sends tensor payloads directly to the registered outfeed process for the
// device.
static inline ffi::Error
outfeed_main_custom_call_impl(ffi::RemainingArgs remaining_args) {
  if (remaining_args.size() < 2) {
    return ffi::Error::InvalidArgument("insufficient_args");
  }

  // Execution context gives us the device id; the Elixir side registers
  // an outfeed process named :"exla_feed_process_<device_id>" that receives
  // binaries or lists of binaries.
  auto *execution_context = ffi::GetExecutionContext();
  if (!execution_context) {
    return ffi::Error::Success();
  }

  auto *device = execution_context->device();
  if (!device) {
    return ffi::Error::Success();
  }

  // The last argument is the token. All prior args are tensors to send.
  const size_t num_payloads = remaining_args.size() - 1;

  // Build the registered process name: "exla_feed_process_<id>"
  int device_id = device->device_ordinal();

  ErlNifEnv *env = enif_alloc_env();
  if (env == nullptr) {
    return ffi::Error::Internal("enomem");
  }

  // Construct the atom name dynamically
  char name_buf[64];
  int name_len =
      snprintf(name_buf, sizeof(name_buf), "exla_feed_process_%d", device_id);
  if (name_len < 0 || name_len >= (int)sizeof(name_buf)) {
    enif_free_env(env);
    return ffi::Error::Internal("name_overflow");
  }

  ERL_NIF_TERM reg_name = enif_make_atom(env, name_buf);
  ErlNifPid pid;
  if (!enif_whereis_pid(env, reg_name, &pid)) {
    // If the process is not registered, just succeed (no receiver)
    enif_free_env(env);
    return ffi::Error::Success();
  }

  // Prepare tensor binary list
  ERL_NIF_TERM *tensor_terms = new ERL_NIF_TERM[num_payloads];
  for (size_t i = 0; i < num_payloads; ++i) {
    auto arg_or_error = remaining_args.get<ffi::AnyBuffer>(i);
    if (!arg_or_error.has_value()) {
      delete[] tensor_terms;
      enif_free_env(env);
      return ffi::Error::InvalidArgument("invalid_tensor_arg");
    }

    auto arg = arg_or_error.value();
    const void *data_ptr = arg.untyped_data();
    size_t data_bytes = arg.size_bytes();

    unsigned char *msg_ptr =
        enif_make_new_binary(env, data_bytes, &tensor_terms[i]);
    std::memcpy(msg_ptr, data_ptr, data_bytes);
  }

  ERL_NIF_TERM payload;
  if (num_payloads == 1) {
    payload = tensor_terms[0];
  } else {
    payload = enif_make_list_from_array(env, tensor_terms, num_payloads);
  }

  enif_send(env, &pid, env, payload);

  delete[] tensor_terms;
  enif_free_env(env);
  return ffi::Error::Success();
}

} // namespace exla_outfeed
