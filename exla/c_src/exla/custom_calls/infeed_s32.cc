#include "../exla_nif_call.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"
#include <erl_nif.h>

namespace ffi = xla::ffi;

static size_t product(const ffi::Span<const int64_t> dims) {
  size_t p = 1;
  for (auto d : dims)
    p *= static_cast<size_t>(d);
  return p;
}

// infeed via nif_call for s32 tensors on Host
static ffi::Error
infeed_cpu_custom_call_s32_impl(ffi::Buffer<ffi::U8> tag,
                                ffi::ResultBuffer<ffi::S32> out) {
  // Compute sizes
  const size_t tag_bytes = product(tag.dimensions());
  const size_t out_bytes = product(out->dimensions()) * sizeof(int32_t);

  // Build a temporary env to interact with the VM
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

  // Arg is ignored for now; the Elixir callback can decide what to do
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    infeed_cpu_custom_call_s32, infeed_cpu_custom_call_s32_impl,
    ffi::Ffi::Bind().Arg<ffi::Buffer<ffi::U8>>().Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "infeed_cpu_custom_call_s32",
                         "Host", infeed_cpu_custom_call_s32);