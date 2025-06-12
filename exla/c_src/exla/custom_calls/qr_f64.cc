#include "qr.h"

ffi::Error qr_cpu_custom_call_f64_impl(ffi::Buffer<ffi::F64> operand,
                                       ffi::ResultBuffer<ffi::F64> q,
                                       ffi::ResultBuffer<ffi::F64> r) {
  return qr_cpu_custom_call_impl<double, ffi::Buffer<ffi::F64>>(operand, q, r);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(qr_cpu_custom_call_f64,
                              qr_cpu_custom_call_f64_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "qr_cpu_custom_call_f64", "Host",
                         qr_cpu_custom_call_f64);
