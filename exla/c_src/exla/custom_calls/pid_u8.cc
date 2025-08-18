#include "pid.h"

ffi::Error pid_cpu_custom_call_u8_impl(ffi::Buffer<ffi::U8> operand,
                                       ffi::ResultBuffer<ffi::U8> result) {
  return pid_cpu_custom_call_impl<uint8_t, ffi::Buffer<ffi::U8>>(operand,
                                                                 result);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    pid_cpu_custom_call_u8, pid_cpu_custom_call_u8_impl,
    ffi::Ffi::Bind().Arg<ffi::Buffer<ffi::U8>>().Ret<ffi::Buffer<ffi::U8>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "pid_cpu_custom_call_u8", "Host",
                         pid_cpu_custom_call_u8);