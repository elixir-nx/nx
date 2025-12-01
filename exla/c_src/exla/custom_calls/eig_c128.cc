#include "eig.h"

ffi::Error
eig_cpu_custom_call_c128_impl(ffi::Buffer<ffi::C128> operand,
                              ffi::ResultBuffer<ffi::C128> eigenvalues,
                              ffi::ResultBuffer<ffi::C128> eigenvectors) {
  return eig_cpu_custom_call_impl_complex<std::complex<double>,
                                          ffi::Buffer<ffi::C128>>(
      operand, eigenvalues, eigenvectors);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(eig_cpu_custom_call_c128,
                              eig_cpu_custom_call_c128_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::C128>>()
                                  .Ret<ffi::Buffer<ffi::C128>>()
                                  .Ret<ffi::Buffer<ffi::C128>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "eig_cpu_custom_call_c128",
                         "Host", eig_cpu_custom_call_c128);
