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
ffi::Error infeed_cpu_custom_call_impl(ffi::Buffer<ffi::U8> tag,
                                       ffi::ResultBuffer<dtype> out) {
  const size_t tag_bytes = product(tag.dimensions());
  const size_t out_bytes =
      product(out->dimensions()) * xla::ffi::ByteWidth(dtype);

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

  ErlNifBinary bin;
  if (!enif_inspect_binary(env, res_val, &bin)) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("expected_binary");
  }

  if (bin.size != out_bytes) {
    enif_free_env(env);
    return ffi::Error::InvalidArgument("size_mismatch");
  }

  std::memcpy(out->untyped_data(), bin.data, out_bytes);
  enif_free_env(env);
  return ffi::Error::Success();
}

} // namespace exla_infeed
