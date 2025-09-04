#include "infeed.h"

namespace ffi = xla::ffi;

static ffi::Error
infeed_cpu_custom_call_impl_wrapper(ffi::Buffer<ffi::U8> token,
                                    ffi::RemainingRets remaining_results) {
  return exla_infeed::infeed_cpu_custom_call_impl(token, remaining_results);
}

// Main infeed custom call (token-based)
static ffi::Error
infeed_main_custom_call_impl_wrapper(ffi::Buffer<ffi::U8> token,
                                     ffi::RemainingRets remaining_results) {
  return exla_infeed::infeed_main_custom_call_impl(token, remaining_results);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    infeed_main_custom_call, infeed_main_custom_call_impl_wrapper,
    ffi::Ffi::Bind().Arg<ffi::Buffer<ffi::U8>>().RemainingRets());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "infeed_main_custom_call", "Host",
                         infeed_main_custom_call);

// Original variadic infeed custom call (NIF-based)
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    infeed_cpu_custom_call, infeed_cpu_custom_call_impl_wrapper,
    ffi::Ffi::Bind().Arg<ffi::Buffer<ffi::U8>>().RemainingRets());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "infeed_cpu_custom_call", "Host",
                         infeed_cpu_custom_call);
