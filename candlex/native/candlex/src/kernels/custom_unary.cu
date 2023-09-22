#define _USE_MATH_DEFINES
#include<math.h>
#include<stdint.h>
#include<cmath>
#include "strides.cuh"

__device__ __forceinline__ float atang(float a) { return atanf(a); }
__device__ __forceinline__ double atang(double a) { return atan(a); }
__device__ __forceinline__ float erfinvg(float a) { return erfinvf(a); }
__device__ __forceinline__ double erfinvg(double a) { return erfinv(a); }
__device__ __forceinline__ float tang(float a) { return tanf(a); }
__device__ __forceinline__ double tang(double a) { return tan(a); }

#define CUSTOM_UNARY_OP_OUT(TYPENAME, OUT_TYPENAME, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const TYPENAME *inp, \
    OUT_TYPENAME *out \
) { \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    if (is_contiguous(num_dims, dims, strides)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            TYPENAME x = inp ? inp[i] : out[i]; \
            out[i] = FUNC; \
        } \
    } \
    else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides); \
            TYPENAME x = inp ? inp[strided_i] : out[i]; \
            out[i] = FUNC; \
        } \
    } \
} \

#define CUSTOM_UNARY_OP(TYPENAME, FN_NAME, FUNC) \
  CUSTOM_UNARY_OP_OUT(TYPENAME, TYPENAME, FN_NAME, FUNC)

CUSTOM_UNARY_OP(float, acos_f32, acos(x))
CUSTOM_UNARY_OP(double, acos_f64, acos(x))
CUSTOM_UNARY_OP(float, asin_f32, asin(x))
CUSTOM_UNARY_OP(double, asin_f64, asin(x))
CUSTOM_UNARY_OP(float, atan_f32, atang(x))
CUSTOM_UNARY_OP(double, atan_f64, atang(x))
CUSTOM_UNARY_OP(uint8_t, bit_not_u8, ~x)
CUSTOM_UNARY_OP(uint32_t, bit_not_u32, ~x)
CUSTOM_UNARY_OP(int64_t, bit_not_i64, ~x)
CUSTOM_UNARY_OP(float, cbrt_f32, cbrt(x))
CUSTOM_UNARY_OP(double, cbrt_f64, cbrt(x))
CUSTOM_UNARY_OP(float, ceil_f32, ceil(x))
CUSTOM_UNARY_OP(double, ceil_f64, ceil(x))
CUSTOM_UNARY_OP(float, erf_inv_f32, erfinvg(x))
CUSTOM_UNARY_OP(double, erf_inv_f64, erfinvg(x))
CUSTOM_UNARY_OP(float, floor_f32, floor(x))
CUSTOM_UNARY_OP(double, floor_f64, floor(x))
CUSTOM_UNARY_OP(float, ln_1p_f32, log1p(x))
CUSTOM_UNARY_OP(double, ln_1p_f64, log1p(x))
CUSTOM_UNARY_OP(float, round_f32, round(x))
CUSTOM_UNARY_OP(double, round_f64, round(x))
CUSTOM_UNARY_OP(float, sigmoid_f32, 1.0 / (1.0 + exp(-x)))
CUSTOM_UNARY_OP(double, sigmoid_f64, 1.0 / (1.0 + exp(-x)))
CUSTOM_UNARY_OP(float, tan_f32, tang(x))
CUSTOM_UNARY_OP(double, tan_f64, tang(x))

CUSTOM_UNARY_OP_OUT(float, uint8_t, is_inf_f32, isinf(x) ? 1 : 0)
CUSTOM_UNARY_OP_OUT(double, uint8_t, is_inf_f64, isinf(x) ? 1 : 0)
