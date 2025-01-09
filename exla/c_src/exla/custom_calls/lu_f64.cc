#include "lu.h"

void lu_cpu_custom_call_f64(void *out[], const void *in[]) {
  lu_cpu_custom_call<double>(out, in);
}
