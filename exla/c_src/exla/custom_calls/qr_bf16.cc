#include "../exla_types.h"
#include "qr.h"

ffi::Error qr_cpu_custom_call_bf16_impl(ffi::Buffer<ffi::BF16> operand,
                                        ffi::ResultBuffer<ffi::BF16> q,
                                        ffi::ResultBuffer<ffi::BF16> r) {
  return qr_cpu_custom_call_impl<exla::bfloat16, ffi::Buffer<ffi::BF16>>(
      operand, q, r);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(qr_cpu_custom_call_bf16,
                              qr_cpu_custom_call_bf16_impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Ret<ffi::Buffer<ffi::BF16>>()
                                  .Ret<ffi::Buffer<ffi::BF16>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "qr_cpu_custom_call_bf16", "Host",
                         qr_cpu_custom_call_bf16);
