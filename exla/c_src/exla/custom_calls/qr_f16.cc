#include "qr.h"
#include "../exla_types.h"

void qr_cpu_custom_call_f16(void *out[], const void *in[]) {
  qr_cpu_custom_call<exla::float16>(out, in);
}
