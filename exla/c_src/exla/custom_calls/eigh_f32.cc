#include "eigh.h"

ffi::Error
eigh_cpu_custom_call_f32_impl(ffi::Buffer<ffi::F32> operand,
                              ffi::ResultBuffer<ffi::F32> eigenvalues,
                              ffi::ResultBuffer<ffi::F32> eigenvectors) {
  return eigh_cpu_custom_call_impl<float, ffi::Buffer<ffi::F32>>(
      operand, eigenvalues, eigenvectors);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(eigh_cpu_custom_call_f32,
                              eigh_cpu_custom_call_f32_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "eigh_cpu_custom_call_f32",
                         "Host", eigh_cpu_custom_call_f32);
