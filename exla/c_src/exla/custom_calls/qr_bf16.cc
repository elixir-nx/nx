#include "../custom_calls.h"
#include "../exla_types.h"
#include "xla/service/custom_call_target_registry.h"

void qr_cpu_custom_call_bf16(void *out[], const void *in[]) {
  qr_cpu_custom_call<exla::bfloat16>(out, in);
}

XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("qr_cpu_custom_call_bf16", qr_cpu_custom_call_bf16);