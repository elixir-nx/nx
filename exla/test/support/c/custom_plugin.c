#include <cstdint>
#include <stddef.h>

typedef void (*ExlaCustomCallFunction)(void *out[], const void *in[]);

typedef struct {
  const char* name;
  ExlaCustomCallFunction func;
} ExlaPluginCustomCall;

void custom_increment(void *out[], const void *in[]) {
  int64_t *operand = (int64_t *)in[0];
  int64_t *dim_sizes = (int64_t *)in[1];

  int64_t *out_buffer = (int64_t *)out[0];

  int64_t n = dim_sizes[0];

  for (int64_t i = 0; i < n; i++) {
    out_buffer[i] = operand[i] + 1;
  }
}

extern "C" ExlaPluginCustomCall exla_custom_calls[] = {
  {"custom_increment", custom_increment},
  {NULL, NULL}
};