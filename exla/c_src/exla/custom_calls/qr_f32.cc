#include "qr.h"

void qr_cpu_custom_call_f32(void *out[], const void *in[]) {
  qr_cpu_custom_call<float>(out, in);
}
