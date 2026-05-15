// Test-only shared library: registers an alias FFI name that reuses the
// existing qr_cpu_custom_call_f32 handler symbol from libxla_extension.so.
#ifndef EXLA_PROD

#include "xla/ffi/api/api.h"
#include "xla/ffi/ffi_api.h"

namespace ffi = xla::ffi;

extern "C" XLA_FFI_Error *qr_cpu_custom_call_f32(XLA_FFI_CallFrame *call_frame);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "qr_cpu_custom_call_f32_exla_alias",
                         "Host", qr_cpu_custom_call_f32);

#endif
