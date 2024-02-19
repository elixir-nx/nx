#pragma once
#include "../exla_nif_util.h"
#include "builder.h"

#define DEFINE_NIF(FUNCTION_NAME) ERL_NIF_TERM FUNCTION_NAME(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])

DEFINE_NIF(mlir_compile);
DEFINE_NIF(new_mlir_module);
DEFINE_NIF(create_mlir_function);
DEFINE_NIF(get_mlir_function_arguments);
DEFINE_NIF(mlir_tuple);
DEFINE_NIF(mlir_get_tuple_element);

// Binary Ops
DEFINE_NIF(mlir_add);
DEFINE_NIF(mlir_subtract);
DEFINE_NIF(mlir_multiply);
DEFINE_NIF(mlir_min);
DEFINE_NIF(mlir_max);
DEFINE_NIF(mlir_remainder);
DEFINE_NIF(mlir_pow);
DEFINE_NIF(mlir_divide);
DEFINE_NIF(mlir_atan2);
DEFINE_NIF(mlir_equal);
DEFINE_NIF(mlir_not_equal);
DEFINE_NIF(mlir_less);
DEFINE_NIF(mlir_less_equal);
DEFINE_NIF(mlir_greater);
DEFINE_NIF(mlir_greater_equal);
DEFINE_NIF(mlir_bitwise_and);
DEFINE_NIF(mlir_bitwise_or);
DEFINE_NIF(mlir_bitwise_xor);
DEFINE_NIF(mlir_shift_left);
DEFINE_NIF(mlir_shift_right_logical);
DEFINE_NIF(mlir_shift_right_arithmetic);

// Unary Ops
DEFINE_NIF(mlir_abs);
DEFINE_NIF(mlir_exp);
DEFINE_NIF(mlir_expm1);
DEFINE_NIF(mlir_floor);
DEFINE_NIF(mlir_ceil);
DEFINE_NIF(mlir_round);
DEFINE_NIF(mlir_log);
DEFINE_NIF(mlir_sigmoid);
DEFINE_NIF(mlir_log1p);
DEFINE_NIF(mlir_sign);
DEFINE_NIF(mlir_cos);
DEFINE_NIF(mlir_sin);
DEFINE_NIF(mlir_tan);
DEFINE_NIF(mlir_acos);
DEFINE_NIF(mlir_asin);
DEFINE_NIF(mlir_atan);
DEFINE_NIF(mlir_cosh);
DEFINE_NIF(mlir_sinh);
DEFINE_NIF(mlir_tanh);
DEFINE_NIF(mlir_acosh);
DEFINE_NIF(mlir_asinh);
DEFINE_NIF(mlir_atanh);
DEFINE_NIF(mlir_sqrt);
DEFINE_NIF(mlir_cbrt);
DEFINE_NIF(mlir_bitwise_not);
DEFINE_NIF(mlir_negate);
DEFINE_NIF(mlir_erf);
DEFINE_NIF(mlir_erfc);
DEFINE_NIF(mlir_erf_inv);
DEFINE_NIF(mlir_is_infinity);
DEFINE_NIF(mlir_is_nan);
DEFINE_NIF(mlir_rsqrt);
DEFINE_NIF(mlir_clz);
DEFINE_NIF(mlir_real);
DEFINE_NIF(mlir_imag);
DEFINE_NIF(mlir_conjugate);
DEFINE_NIF(mlir_population_count);
DEFINE_NIF(mlir_convolution);

//
DEFINE_NIF(mlir_iota);
DEFINE_NIF(mlir_reshape);
DEFINE_NIF(mlir_reverse);
DEFINE_NIF(mlir_transpose);
DEFINE_NIF(mlir_slice);
DEFINE_NIF(mlir_dynamic_slice);
DEFINE_NIF(mlir_constant_r0);
DEFINE_NIF(mlir_constant_from_binary);
DEFINE_NIF(mlir_dot_general);
DEFINE_NIF(mlir_select);
DEFINE_NIF(mlir_convert);
DEFINE_NIF(mlir_top_k);
DEFINE_NIF(mlir_sort);
DEFINE_NIF(mlir_bitcast_convert);
DEFINE_NIF(mlir_pad);
DEFINE_NIF(mlir_optimization_barrier);
DEFINE_NIF(mlir_clamp);
DEFINE_NIF(mlir_get_shape);
DEFINE_NIF(mlir_broadcast_in_dim);
DEFINE_NIF(mlir_concatenate);
DEFINE_NIF(dump_mlir_module);
DEFINE_NIF(mlir_scatter);
DEFINE_NIF(mlir_select_and_scatter);
DEFINE_NIF(mlir_gather);
DEFINE_NIF(mlir_fft);
DEFINE_NIF(mlir_create_token);
DEFINE_NIF(mlir_triangular_solve);
DEFINE_NIF(mlir_dynamic_update_slice);
DEFINE_NIF(mlir_reduce);
DEFINE_NIF(mlir_window_reduce);
DEFINE_NIF(mlir_map);
DEFINE_NIF(mlir_if);
DEFINE_NIF(mlir_infeed);
DEFINE_NIF(mlir_outfeed);
DEFINE_NIF(mlir_call);
DEFINE_NIF(mlir_while);
DEFINE_NIF(mlir_return);
