#ifndef EXLA_MLIR_CUSTOM_CALLS_H_
#define EXLA_MLIR_CUSTOM_CALLS_H_

void qr_cpu_custom_call_bf16(void *out[], const void *in[]);
void qr_cpu_custom_call_f16(void *out[], const void *in[]);
void qr_cpu_custom_call_f32(void *out[], const void *in[]);
void qr_cpu_custom_call_f64(void *out[], const void *in[]);

void eigh_cpu_custom_call_f32(void *out[], const void *in[]);
void eigh_cpu_custom_call_f64(void *out[], const void *in[]);

#endif
