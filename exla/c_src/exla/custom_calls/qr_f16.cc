#include "../exla_types.h"
#include "qr.h"

ffi::Error qr_cpu_custom_call_f16_impl(ffi::Buffer<ffi::F16> operand,
                                       ffi::ResultBuffer<ffi::F16> q,
                                       ffi::ResultBuffer<ffi::F16> r) {
  return qr_cpu_custom_call_impl<exla::float16, ffi::Buffer<ffi::F16>>(operand,
                                                                       q, r);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(qr_cpu_custom_call_f16,
                              qr_cpu_custom_call_f16_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F16>>()
                                  .Ret<ffi::Buffer<ffi::F16>>()
                                  .Ret<ffi::Buffer<ffi::F16>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "qr_cpu_custom_call_f16", "Host",
                         qr_cpu_custom_call_f16);
