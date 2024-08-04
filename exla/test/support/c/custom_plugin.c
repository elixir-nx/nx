#include <cstdint>
#include <stddef.h>

typedef void (*ExlaCustomCallFunction)(void *out[], const void *in[], int **dims);

typedef struct {
  const char* name;
  ExlaCustomCallFunction func;
} ExlaPluginCustomCall;

extern "C" void custom_increment(void *out[], const void *in[], int **dims) {
  int64_t *operand = (int64_t *)in[0];
  int64_t *dim_sizes = (int64_t *)dims[0];

  int64_t *out_buffer = (int64_t *)out[0];

  int64_t n = dim_sizes[0];

  for (int64_t i = 0; i < n; i++) {
    out_buffer[i] = operand[i] + 1;
  }
}