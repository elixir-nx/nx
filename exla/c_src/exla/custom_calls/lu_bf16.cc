#include "../exla_types.h"
#include "lu.h"

ffi::Error lu_cpu_custom_call_bf16_impl(ffi::Buffer<ffi::BF16> operand,
                                        ffi::ResultBuffer<ffi::U8> p,
                                        ffi::ResultBuffer<ffi::BF16> l,
                                        ffi::ResultBuffer<ffi::BF16> u) {
  return lu_cpu_custom_call_impl<exla::bfloat16, ffi::Buffer<ffi::BF16>>(
      operand, p, l, u);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(lu_cpu_custom_call_bf16,
                              lu_cpu_custom_call_bf16_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Ret<ffi::Buffer<ffi::U8>>()
                                  .Ret<ffi::Buffer<ffi::BF16>>()
                                  .Ret<ffi::Buffer<ffi::BF16>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "lu_cpu_custom_call_bf16", "Host",
                         lu_cpu_custom_call_bf16);
