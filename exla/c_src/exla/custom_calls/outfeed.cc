#include "outfeed.h"

namespace ffi = xla::ffi;

static ffi::Error
outfeed_cpu_custom_call_impl_wrapper(ffi::RemainingArgs remaining_args) {
  return exla_outfeed::outfeed_cpu_custom_call_impl(remaining_args);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(outfeed_cpu_custom_call,
                              outfeed_cpu_custom_call_impl_wrapper,
                              ffi::Ffi::Bind().RemainingArgs());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "outfeed_cpu_custom_call", "Host",
                         outfeed_cpu_custom_call);
