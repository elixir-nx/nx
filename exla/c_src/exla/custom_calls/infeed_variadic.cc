#include "infeed.h"

namespace ffi = xla::ffi;

static ffi::Error
infeed_variadic_cpu_custom_call_impl(ffi::Buffer<ffi::U8> tag,
                                     ffi::RemainingArgs remaining_args,
                                     ffi::RemainingRets remaining_results) {
  return exla_infeed::infeed_variadic_cpu_custom_call_impl(tag, remaining_args,
                                                           remaining_results);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(infeed_variadic_cpu_custom_call,
                              infeed_variadic_cpu_custom_call_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::U8>>()
                                  .RemainingArgs()
                                  .RemainingRets());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "infeed_variadic_cpu_custom_call",
                         "Host", infeed_variadic_cpu_custom_call);
