#include "qr.h"
#include "../exla_types.h"

void qr_cpu_custom_call_bf16(void *out[], const void *in[]) {
  qr_cpu_custom_call<exla::bfloat16>(out, in);
}
