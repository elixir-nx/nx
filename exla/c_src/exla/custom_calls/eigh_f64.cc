#include "../custom_calls.h"

void eigh_cpu_custom_call_f64(void *out[], const void *in[]) {
  eigh_cpu_custom_call<double>(out, in);
}
