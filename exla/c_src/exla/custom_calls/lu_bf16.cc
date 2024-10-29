#include "lu.h"
#include "../exla_types.h"

void lu_cpu_custom_call_bf16(void *out[], const void *in[]) {
  lu_cpu_custom_call<exla::bfloat16>(out, in);
}
