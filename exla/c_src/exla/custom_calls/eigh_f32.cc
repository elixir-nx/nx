#include "../custom_calls.h"

void eigh_cpu_custom_call_f32(void *out[], const void *in[]) {
  eigh_cpu_custom_call<float>(out, in);
}

XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("eigh_cpu_custom_call_f32", eigh_cpu_custom_call_f32);