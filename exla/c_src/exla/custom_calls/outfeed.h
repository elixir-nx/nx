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

template <ffi::DataType dtype>
ffi::Error outfeed_cpu_custom_call_impl(ffi::Buffer<dtype> data,
                                        ffi::Buffer<ffi::U8> pid_tag,
                                        ffi::Result<ffi::Token> /*token*/) {
  ErlNifEnv *env = enif_alloc_env();
  if (env == nullptr) {
    return ffi::Error::Internal("enomem");
  }

  // Decode pid term from binary
  const size_t pid_bytes = product(pid_tag.dimensions());
  ERL_NIF_TERM pid_term;
  if (!enif_binary_to_term(
          env, reinterpret_cast<const unsigned char *>(pid_tag.untyped_data()),
          pid_bytes, &pid_term, 0)) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("invalid_pid_tag");
  }

  ErlNifPid pid;
  if (!enif_get_local_pid(env, pid_term, &pid)) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("not_local_pid");
  }

  // Build message binary with data contents
  const size_t out_bytes =
      product(data.dimensions()) * xla::ffi::ByteWidth(dtype);
  ERL_NIF_TERM msg_bin;
  unsigned char *msg_ptr = enif_make_new_binary(env, out_bytes, &msg_bin);
  std::memcpy(msg_ptr, data.untyped_data(), out_bytes);

  enif_send(env, &pid, env, msg_bin);
  enif_free_env(env);
  return ffi::Error::Success();
}

// Variadic outfeed implementation for multiple tensors
static inline ffi::Error
outfeed_variadic_cpu_custom_call_impl(ffi::RemainingArgs remaining_args,
                                      ffi::Result<ffi::Token> /*token*/) {
  // The last argument should be the pid_tag
  if (remaining_args.size() < 2) {
    return ffi::Error::InvalidArgument("insufficient_args");
  }

  // Get the pid_tag (last argument)
  auto pid_tag_arg_or_error =
      remaining_args.get<ffi::Buffer<ffi::U8>>(remaining_args.size() - 1);
  if (!pid_tag_arg_or_error.has_value()) {
    return ffi::Error::InvalidArgument("invalid_pid_tag_arg");
  }
  ffi::Buffer<ffi::U8> pid_tag = pid_tag_arg_or_error.value();

  ErlNifEnv *env = enif_alloc_env();
  if (env == nullptr) {
    return ffi::Error::Internal("enomem");
  }

  // Decode pid term from binary
  const size_t pid_bytes = product(pid_tag.dimensions());
  ERL_NIF_TERM pid_term;
  if (!enif_binary_to_term(
          env, reinterpret_cast<const unsigned char *>(pid_tag.untyped_data()),
          pid_bytes, &pid_term, 0)) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("invalid_pid_tag");
  }

  ErlNifPid pid;
  if (!enif_get_local_pid(env, pid_term, &pid)) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("not_local_pid");
  }

  // Create a list of binaries for all tensor data (excluding pid_tag)
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

} // namespace exla_outfeed
