#include "infeed.h"

namespace ffi = xla::ffi;

static ffi::Error
infeed_cpu_custom_call_c64_impl(ffi::Buffer<ffi::U8> tag,
                                ffi::ResultBuffer<ffi::C64> out,
                                ffi::ResultBuffer<ffi::U8> next_tag_out) {
  return exla_infeed::infeed_cpu_custom_call_impl<ffi::C64>(tag, out,
                                                            next_tag_out);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(infeed_cpu_custom_call_c64,
                              infeed_cpu_custom_call_c64_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::U8>>()
                                  .Ret<ffi::Buffer<ffi::C64>>()
                                  .Ret<ffi::Buffer<ffi::U8>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "infeed_cpu_custom_call_c64",
                         "Host", infeed_cpu_custom_call_c64);
