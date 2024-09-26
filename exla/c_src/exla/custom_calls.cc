#include "xla/service/custom_call_target_registry.h"

void qr_cpu_custom_call_f32(void *out[], const void *in[]);
void qr_cpu_custom_call_f64(void *out[], const void *in[]);
void qr_cpu_custom_call_f16(void *out[], const void *in[]);
void qr_cpu_custom_call_bf16(void *out[], const void *in[]);
void eigh_cpu_custom_call_f32(void *out[], const void *in[]);
void eigh_cpu_custom_call_f64(void *out[], const void *in[]);

XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("qr_cpu_custom_call_f64", qr_cpu_custom_call_f64);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("qr_cpu_custom_call_f32", qr_cpu_custom_call_f32);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("qr_cpu_custom_call_f16", qr_cpu_custom_call_f16);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("qr_cpu_custom_call_bf16", qr_cpu_custom_call_bf16);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("eigh_cpu_custom_call_f64", eigh_cpu_custom_call_f64);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("eigh_cpu_custom_call_f32", eigh_cpu_custom_call_f32);