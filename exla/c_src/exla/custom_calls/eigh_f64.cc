#include "eigh.h"

ffi::Error
eigh_cpu_custom_call_f64_impl(ffi::Buffer<ffi::F64> operand,
                              ffi::ResultBuffer<ffi::F64> eigenvalues,
                              ffi::ResultBuffer<ffi::F64> eigenvectors) {
  return eigh_cpu_custom_call_impl<double, ffi::Buffer<ffi::F64>>(
      operand, eigenvalues, eigenvectors);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(eigh_cpu_custom_call_f64,
                              eigh_cpu_custom_call_f64_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>()
                                  .Ret<ffi::Buffer<ffi::F64>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "eigh_cpu_custom_call_f64",
                         "Host", eigh_cpu_custom_call_f64);
