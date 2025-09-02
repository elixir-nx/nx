#include "outfeed.h"

namespace ffi = xla::ffi;

static ffi::Error
outfeed_variadic_cpu_custom_call_impl(ffi::RemainingArgs remaining_args,
                                      ffi::Result<ffi::Token> token) {
  return exla_outfeed::outfeed_variadic_cpu_custom_call_impl(remaining_args,
                                                             token);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    outfeed_variadic_cpu_custom_call, outfeed_variadic_cpu_custom_call_impl,
    ffi::Ffi::Bind().RemainingArgs().Ret<ffi::Token>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         "outfeed_variadic_cpu_custom_call", "Host",
                         outfeed_variadic_cpu_custom_call);
