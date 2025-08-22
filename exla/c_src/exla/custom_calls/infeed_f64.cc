#include "infeed.h"

namespace ffi = xla::ffi;

static ffi::Error
infeed_cpu_custom_call_f64_impl(ffi::Buffer<ffi::U8> tag,
                                ffi::ResultBuffer<ffi::F64> out) {
  return exla_infeed::infeed_cpu_custom_call_impl<ffi::F64>(tag, out);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    infeed_cpu_custom_call_f64, infeed_cpu_custom_call_f64_impl,
    ffi::Ffi::Bind().Arg<ffi::Buffer<ffi::U8>>().Ret<ffi::Buffer<ffi::F64>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "infeed_cpu_custom_call_f64",
                         "Host", infeed_cpu_custom_call_f64);
