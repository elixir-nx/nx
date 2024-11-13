#include "lu.h"

void lu_cpu_custom_call_f32(void *out[], const void *in[]) {
  lu_cpu_custom_call<float>(out, in);
}
