#include "qr.h"

void qr_cpu_custom_call_f64(void *out[], const void *in[]) {
  qr_cpu_custom_call<double>(out, in);
}
