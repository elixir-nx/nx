#include "lu.h"
#include "../exla_types.h"

void lu_cpu_custom_call_f16(void *out[], const void *in[]) {
  lu_cpu_custom_call<exla::float16>(out, in);
}
