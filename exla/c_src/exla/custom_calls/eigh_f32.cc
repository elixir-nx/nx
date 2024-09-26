#include "eigh.h"

void eigh_cpu_custom_call_f32(void *out[], const void *in[]) {
  eigh_cpu_custom_call<float>(out, in);
}
