#include "infeed.h"

namespace ffi = xla::ffi;

static ffi::Error
infeed_cpu_custom_call_s64_impl(ffi::Buffer<ffi::U8> tag,
                                ffi::ResultBuffer<ffi::S64> out) {
  return exla_infeed::infeed_cpu_custom_call_impl<ffi::S64>(tag, out);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    infeed_cpu_custom_call_s64, infeed_cpu_custom_call_s64_impl,
    ffi::Ffi::Bind().Arg<ffi::Buffer<ffi::U8>>().Ret<ffi::Buffer<ffi::S64>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "infeed_cpu_custom_call_s64",
                         "Host", infeed_cpu_custom_call_s64);
