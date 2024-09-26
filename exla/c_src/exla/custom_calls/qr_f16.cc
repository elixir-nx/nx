#include "../custom_calls.h"

void qr_cpu_custom_call_f16(void *out[], const void *in[]) {
  qr_cpu_custom_call<exla::float16>(out, in);
}

XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("qr_cpu_custom_call_f16", qr_cpu_custom_call_f16);