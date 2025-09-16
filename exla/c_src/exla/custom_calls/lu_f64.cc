#include "lu.h"

ffi::Error lu_cpu_custom_call_f64_impl(ffi::Buffer<ffi::F64> operand,
                                       ffi::ResultBuffer<ffi::U8> p,
                                       ffi::ResultBuffer<ffi::F64> l,
                                       ffi::ResultBuffer<ffi::F64> u) {
  return lu_cpu_custom_call_impl<double, ffi::Buffer<ffi::F64>>(operand, p, l,
                                                                u);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(lu_cpu_custom_call_f64,
                              lu_cpu_custom_call_f64_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::U8>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "lu_cpu_custom_call_f64", "Host",
                         lu_cpu_custom_call_f64);
