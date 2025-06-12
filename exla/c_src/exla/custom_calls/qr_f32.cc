#include "qr.h"

ffi::Error qr_cpu_custom_call_f32_impl(ffi::Buffer<ffi::F32> operand,
                                       ffi::ResultBuffer<ffi::F32> q,
                                       ffi::ResultBuffer<ffi::F32> r) {
  return qr_cpu_custom_call_impl<float, ffi::Buffer<ffi::F32>>(operand, q, r);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(qr_cpu_custom_call_f32,
                              qr_cpu_custom_call_f32_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "qr_cpu_custom_call_f32", "Host",
                         qr_cpu_custom_call_f32);
