#include "outfeed.h"

namespace ffi = xla::ffi;

static ffi::Error
outfeed_cpu_custom_call_u32_impl(ffi::Buffer<ffi::U32> data,
                                 ffi::Buffer<ffi::U8> pid,
                                 ffi::Result<ffi::Token> tok) {
  return exla_outfeed::outfeed_cpu_custom_call_impl<ffi::U32>(data, pid, tok);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(outfeed_cpu_custom_call_u32,
                              outfeed_cpu_custom_call_u32_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::U32>>()
                                  .Arg<ffi::Buffer<ffi::U8>>()
                                  .Ret<ffi::Token>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "outfeed_cpu_custom_call_u32",
                         "Host", outfeed_cpu_custom_call_u32);
