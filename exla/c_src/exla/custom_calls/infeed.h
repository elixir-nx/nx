#pragma once

#include <cstring>
#include <erl_nif.h>

#include "../exla_nif_call.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace exla_infeed {
namespace ffi = xla::ffi;

inline size_t product(const ffi::Span<const int64_t> dims) {
  size_t p = 1;
  for (auto d : dims)
    p *= static_cast<size_t>(d);
  return p;
}

template <ffi::DataType dtype>
ffi::Error
infeed_cpu_custom_call_impl(ffi::Buffer<ffi::U8> tag,
                            ffi::ResultBuffer<dtype> out,
                            ffi::ResultBuffer<ffi::U8> next_tag_out) {
  const size_t tag_bytes = product(tag.dimensions());
  const size_t out_bytes =
      product(out->dimensions()) * xla::ffi::ByteWidth(dtype);
  const size_t next_tag_bytes = product(next_tag_out->dimensions());

  ErlNifEnv *env = enif_alloc_env();
  if (env == nullptr) {
    return ffi::Error::Internal("enomem");
  }

  ERL_NIF_TERM tag_term;
  if (!enif_binary_to_term(
          env, reinterpret_cast<const unsigned char *>(tag.untyped_data()),
          tag_bytes, &tag_term, 0)) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("invalid_tag");
  }

  ERL_NIF_TERM arg = enif_make_atom(env, "next");
  ERL_NIF_TERM res_val;
  if (!exla_nif_call_make(env, tag_term, arg, &res_val)) {
    enif_free_env(env);
    return ffi::Error::Internal("nif_call_failed");
  }

  int arity = 0;
  const ERL_NIF_TERM *tuple = nullptr;
  if (!enif_get_tuple(env, res_val, &arity, &tuple) || arity != 2) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("expected_tuple");
  }

  ErlNifBinary data_bin;
  if (!enif_inspect_binary(env, tuple[0], &data_bin) ||
      data_bin.size != out_bytes) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("bad_data_binary");
  }
  std::memcpy(out->untyped_data(), data_bin.data, out_bytes);

  ErlNifBinary next_bin;
  if (!enif_inspect_binary(env, tuple[1], &next_bin)) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("bad_next_tag_binary");
  }
  size_t copy = next_bin.size < next_tag_bytes ? next_bin.size : next_tag_bytes;
  if (next_tag_bytes > 0) {
    std::memset(next_tag_out->untyped_data(), 0, next_tag_bytes);
    std::memcpy(next_tag_out->untyped_data(), next_bin.data, copy);
  }
  enif_free_env(env);
  return ffi::Error::Success();
}

// Variadic infeed implementation for multiple tensors
static inline ffi::Error
infeed_variadic_cpu_custom_call_impl(ffi::Buffer<ffi::U8> tag,
                                     ffi::RemainingArgs remaining_args,
                                     ffi::RemainingRets remaining_results) {
  const size_t tag_bytes = product(tag.dimensions());

  ErlNifEnv *env = enif_alloc_env();
  if (env == nullptr) {
    return ffi::Error::Internal("enomem");
  }

  ERL_NIF_TERM tag_term;
  if (!enif_binary_to_term(
          env, reinterpret_cast<const unsigned char *>(tag.untyped_data()),
          tag_bytes, &tag_term, 0)) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("invalid_tag");
  }

  // Request data for all tensors
  ERL_NIF_TERM arg = enif_make_atom(env, "next_variadic");
  ERL_NIF_TERM res_val;
  if (!exla_nif_call_make(env, tag_term, arg, &res_val)) {
    enif_free_env(env);
    return ffi::Error::Internal("nif_call_failed");
  }

  int arity = 0;
  const ERL_NIF_TERM *tuple = nullptr;
  if (!enif_get_tuple(env, res_val, &arity, &tuple) || arity != 2) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("expected_tuple");
  }

  // tuple[0] should be a list of binaries for each tensor
  // tuple[1] should be the next tag binary
  ERL_NIF_TERM data_list = tuple[0];
  ERL_NIF_TERM next_tag_term = tuple[1];

  // Get the list length and validate it matches the number of result buffers
  unsigned int list_length = 0;
  if (!enif_get_list_length(env, data_list, &list_length)) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("expected_data_list");
  }

  if (list_length != remaining_results.size() - 1) { // -1 for next_tag_out
    enif_free_env(env);
    return ffi::Error::InvalidArgument("data_list_length_mismatch");
  }

  // Process each tensor data
  ERL_NIF_TERM head, tail = data_list;
  for (size_t i = 0; i < list_length; ++i) {
    if (!enif_get_list_cell(env, tail, &head, &tail)) {
      enif_free_env(env);
      return ffi::Error::InvalidArgument("invalid_data_list");
    }

    ErlNifBinary data_bin;
    if (!enif_inspect_binary(env, head, &data_bin)) {
      enif_free_env(env);
      return ffi::Error::InvalidArgument("bad_data_binary");
    }

    // Get the result buffer using AnyBuffer
    auto result_or_error = remaining_results.get<ffi::AnyBuffer>(i);
    if (!result_or_error.has_value()) {
      enif_free_env(env);
      return ffi::Error::InvalidArgument("invalid_result_buffer");
    }

    auto result_buffer = result_or_error.value();
    void *result_data = result_buffer->untyped_data();
    size_t expected_bytes = result_buffer->size_bytes();

    if (data_bin.size != expected_bytes) {
      enif_free_env(env);
      return ffi::Error::InvalidArgument("data_size_mismatch");
    }

    std::memcpy(result_data, data_bin.data, data_bin.size);
  }

  // Handle next_tag (last result)
  ErlNifBinary next_bin;
  if (!enif_inspect_binary(env, next_tag_term, &next_bin)) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("bad_next_tag_binary");
  }

  auto next_tag_result_or_error =
      remaining_results.get<ffi::AnyBuffer>(remaining_results.size() - 1);
  if (!next_tag_result_or_error.has_value()) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("invalid_next_tag_buffer");
  }

  auto next_tag_buffer = next_tag_result_or_error.value();
  size_t next_tag_bytes = next_tag_buffer->size_bytes();
  size_t copy = next_bin.size < next_tag_bytes ? next_bin.size : next_tag_bytes;
  void *next_tag_data = next_tag_buffer->untyped_data();

  if (next_tag_bytes > 0) {
    std::memset(next_tag_data, 0, next_tag_bytes);
    std::memcpy(next_tag_data, next_bin.data, copy);
  }

  enif_free_env(env);
  return ffi::Error::Success();
}

} // namespace exla_infeed
