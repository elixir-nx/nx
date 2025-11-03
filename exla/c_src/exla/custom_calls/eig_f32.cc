#include "eig.h"

ffi::Error
eig_cpu_custom_call_f32_impl(ffi::Buffer<ffi::F32> operand,
                             ffi::ResultBuffer<ffi::C64> eigenvalues,
                             ffi::ResultBuffer<ffi::C64> eigenvectors) {
  return eig_cpu_custom_call_impl_real<
      float, std::complex<float>, ffi::Buffer<ffi::F32>, ffi::Buffer<ffi::C64>>(
      operand, eigenvalues, eigenvectors);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(eig_cpu_custom_call_f32,
                              eig_cpu_custom_call_f32_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::C64>>()
                                  .Ret<ffi::Buffer<ffi::C64>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "eig_cpu_custom_call_f32", "Host",
                         eig_cpu_custom_call_f32);
