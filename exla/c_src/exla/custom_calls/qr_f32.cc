#include "../custom_calls.h"
#include "xla/service/custom_call_target_registry.h"

void qr_cpu_custom_call_f32(void *out[], const void *in[]) {
  qr_cpu_custom_call<float>(out, in);
}

XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("qr_cpu_custom_call_f32", qr_cpu_custom_call_f32);