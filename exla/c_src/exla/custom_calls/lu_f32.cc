#include "lu.h"

ffi::Error lu_cpu_custom_call_f32_impl(ffi::Buffer<ffi::F32> operand,
                                       ffi::ResultBuffer<ffi::U8> p,
                                       ffi::ResultBuffer<ffi::F32> l,
                                       ffi::ResultBuffer<ffi::F32> u) {
  return lu_cpu_custom_call_impl<float, ffi::Buffer<ffi::F32>>(operand, p, l,
                                                               u);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(lu_cpu_custom_call_f32,
                              lu_cpu_custom_call_f32_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::U8>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "lu_cpu_custom_call_f32", "Host",
                         lu_cpu_custom_call_f32);
