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

// Infeed implementation using NIF calls and process communication
static inline ffi::Error
infeed_cpu_custom_call_impl(ffi::Buffer<ffi::U8> token,
                            ffi::RemainingRets remaining_results) {
  const size_t token_bytes = product(token.dimensions());

  ErlNifEnv *env = enif_alloc_env();
  if (env == nullptr) {
    return ffi::Error::Internal("enomem");
  }

  ERL_NIF_TERM tag_term;
  if (!enif_binary_to_term(
          env, reinterpret_cast<const unsigned char *>(token.untyped_data()),
          token_bytes, &tag_term, 0)) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("invalid_tag");
  }

  // Try NIF call with :next_variadic to get data from infeed_callback
  ERL_NIF_TERM arg = enif_make_atom(env, "next_variadic");
  ERL_NIF_TERM res_val;
  bool is_nif_call = exla_nif_call_make(env, tag_term, arg, &res_val);

  if (is_nif_call) {
    // NIF call successful - extract data and next tag
    int arity = 0;
    const ERL_NIF_TERM *res_tuple = nullptr;
    if (!enif_get_tuple(env, res_val, &arity, &res_tuple) || arity != 2) {
      enif_free_env(env);
      return ffi::Error::InvalidArgument("expected_result_tuple");
    }

    // res_tuple[0] should be a list of binaries for each tensor
    // res_tuple[1] should be the next nif_call tag
    ERL_NIF_TERM data_list = res_tuple[0];
    ERL_NIF_TERM next_tag_term = res_tuple[1];

    // Get the list length and validate it matches the number of result buffers
    unsigned int list_length = 0;
    if (!enif_get_list_length(env, data_list, &list_length)) {
      enif_free_env(env);
      return ffi::Error::InvalidArgument("expected_data_list");
    }

    // Check if we have a next_tag buffer (infeed_custom) or not (regular
    // infeed)
    bool has_next_tag = remaining_results.size() > list_length;
    size_t expected_results = has_next_tag ? list_length + 1 : list_length;

    if (remaining_results.size() != expected_results) {
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

    // Handle next_tag if it exists (for infeed_custom)
    if (has_next_tag) {
      ErlNifBinary next_tag_bin;
      if (!enif_term_to_binary(env, next_tag_term, &next_tag_bin)) {
        enif_free_env(env);
        return ffi::Error::Internal("failed_to_encode_next_tag");
      }

      auto next_tag_result_or_error =
          remaining_results.get<ffi::AnyBuffer>(remaining_results.size() - 1);
      if (!next_tag_result_or_error.has_value()) {
        enif_free_env(env);
        return ffi::Error::InvalidArgument("invalid_next_tag_buffer");
      }

      auto next_tag_buffer = next_tag_result_or_error.value();
      size_t next_tag_bytes = next_tag_buffer->size_bytes();
      size_t copy = next_tag_bin.size < next_tag_bytes ? next_tag_bin.size
                                                       : next_tag_bytes;
      void *next_tag_data = next_tag_buffer->untyped_data();

      if (next_tag_bytes > 0) {
        std::memset(next_tag_data, 0, next_tag_bytes);
        std::memcpy(next_tag_data, next_tag_bin.data, copy);
      }
    }
  } else {
    // NIF call failed - fall back to zero-filled buffers for non-NIF scenarios
    for (size_t i = 0; i < remaining_results.size(); ++i) {
      auto result_or_error = remaining_results.get<ffi::AnyBuffer>(i);
      if (!result_or_error.has_value()) {
        enif_free_env(env);
        return ffi::Error::InvalidArgument("invalid_result_buffer");
      }

      auto result_buffer = result_or_error.value();
      void *result_data = result_buffer->untyped_data();
      size_t result_bytes = result_buffer->size_bytes();

      // Zero-fill the result buffer for non-NIF scenarios
      std::memset(result_data, 0, result_bytes);
    }
  }

  enif_free_env(env);
  return ffi::Error::Success();
}

// Main infeed implementation for token-based calls
// Pulls from BEAM via NifCall (no native queues)
static inline ffi::Error
infeed_main_custom_call_impl(ffi::Buffer<ffi::U8> /*token*/,
                             ffi::RemainingRets remaining_results) {
  // Ask BEAM (EXLA.NifCall.Runner) for the next tuple
  ErlNifEnv *env = enif_alloc_env();
  if (env == nullptr) {
    return ffi::Error::Internal("enomem");
  }

  ERL_NIF_TERM arg = enif_make_atom(env, "next_variadic");
  ERL_NIF_TERM res_val;
  // We rely on the session-specific callback registered by
  // Outfeed.start_child/4
  if (!exla_nif_call_make(env, enif_make_atom(env, "nif"), arg, &res_val)) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("infeed_queue_error");
  }

  int arity = 0;
  const ERL_NIF_TERM *res_tuple = nullptr;
  if (!enif_get_tuple(env, res_val, &arity, &res_tuple) || arity != 2) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("expected_result_tuple");
  }

  ERL_NIF_TERM data_list = res_tuple[0];
  unsigned int list_length = 0;
  if (!enif_get_list_length(env, data_list, &list_length)) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("expected_data_list");
  }

  if (remaining_results.size() != list_length) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("data_list_length_mismatch");
  }

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

  enif_free_env(env);
  return ffi::Error::Success();
}

} // namespace exla_infeed