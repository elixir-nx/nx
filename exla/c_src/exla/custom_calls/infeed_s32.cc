#include "infeed.h"

namespace ffi = xla::ffi;

static ffi::Error
infeed_cpu_custom_call_s32_impl(ffi::Buffer<ffi::U8> tag,
                                ffi::ResultBuffer<ffi::S32> out) {
  return exla_infeed::infeed_cpu_custom_call_impl<ffi::S32>(tag, out);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    infeed_cpu_custom_call_s32, infeed_cpu_custom_call_s32_impl,
    ffi::Ffi::Bind().Arg<ffi::Buffer<ffi::U8>>().Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "infeed_cpu_custom_call_s32",
                         "Host", infeed_cpu_custom_call_s32);