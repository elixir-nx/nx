#include<stdint.h>
#include<cmath>
#include "strides.cuh"

#define DEVICE_FN_FLOAT_WRAPPER(FN_NAME) \
  __device__ __forceinline__ float FN_NAME##g(float a, float b) { return FN_NAME##f(a, b); }

#define DEVICE_FN_DOUBLE_WRAPPER(FN_NAME) \
  __device__ __forceinline__ double FN_NAME##g(double a, double b) { return FN_NAME(a, b); }

DEVICE_FN_FLOAT_WRAPPER(atan2)
DEVICE_FN_DOUBLE_WRAPPER(atan2)
DEVICE_FN_FLOAT_WRAPPER(fmod)
DEVICE_FN_DOUBLE_WRAPPER(fmod)
DEVICE_FN_FLOAT_WRAPPER(pow)
DEVICE_FN_DOUBLE_WRAPPER(pow)

#define CUSTOM_BINARY_OP_OUT(TYPENAME, OUT_TYPENAME, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims_and_strides, \
    const TYPENAME *lhs, \
    const TYPENAME *rhs, \
    OUT_TYPENAME *out \
) { \
    const size_t *dims = dims_and_strides; \
    const size_t *lhs_strides = dims_and_strides + 1 * num_dims; \
    const size_t *rhs_strides = dims_and_strides + 2 * num_dims; \
    bool lhs_cont = is_contiguous(num_dims, dims, lhs_strides); \
    bool rhs_cont = is_contiguous(num_dims, dims, rhs_strides); \
    if (lhs_cont && rhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            TYPENAME x = lhs[i]; \
            TYPENAME y = rhs[i]; \
            out[i] = FUNC; \
        } \
    } else if (lhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int i_dim = tmp_i % dims[d]; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dims[d]; \
            } \
            TYPENAME x = lhs[i]; \
            TYPENAME y = rhs[rhs_i]; \
            out[i] = FUNC; \
        } \
    } else if (rhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int lhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int i_dim = tmp_i % dims[d]; \
                lhs_i += i_dim * lhs_strides[d]; \
                tmp_i /= dims[d]; \
            } \
            TYPENAME x = lhs[lhs_i]; \
            TYPENAME y = rhs[i]; \
            out[i] = FUNC; \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int lhs_i = 0; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int i_dim = tmp_i % dims[d]; \
                lhs_i += i_dim * lhs_strides[d]; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dims[d]; \
            } \
            TYPENAME x = lhs[lhs_i]; \
            TYPENAME y = rhs[rhs_i]; \
            out[i] = FUNC; \
        } \
    } \
} \

#define CUSTOM_BINARY_OP(TYPENAME, FN_NAME, FUNC) \
  CUSTOM_BINARY_OP_OUT(TYPENAME, TYPENAME, FN_NAME, FUNC)

CUSTOM_BINARY_OP(float, atan2_f32, atan2g(x, y))
CUSTOM_BINARY_OP(double, atan2_f64, atan2g(x, y))
CUSTOM_BINARY_OP(uint32_t, bit_and_u32, x & y)
CUSTOM_BINARY_OP(int64_t, bit_and_i64, x & y)
CUSTOM_BINARY_OP(uint32_t, bit_or_u32, x | y)
CUSTOM_BINARY_OP(int64_t, bit_or_i64, x | y)
CUSTOM_BINARY_OP(uint32_t, bit_xor_u32, x ^ y)
CUSTOM_BINARY_OP(int64_t, bit_xor_i64, x ^ y)
CUSTOM_BINARY_OP(float, pow_f32, powg(x, y))
CUSTOM_BINARY_OP(double, pow_f64, powg(x, y))
CUSTOM_BINARY_OP(uint8_t, remainder_u8, x % y)
CUSTOM_BINARY_OP(int64_t, remainder_i64, x % y)
CUSTOM_BINARY_OP(float, remainder_f32, fmodg(x, y))
CUSTOM_BINARY_OP(double, remainder_f64, fmodg(x, y))
CUSTOM_BINARY_OP(uint32_t, shl_u32, x << y)
CUSTOM_BINARY_OP(int64_t, shl_i64, x << y)
CUSTOM_BINARY_OP(uint32_t, shr_u32, x >> y)
CUSTOM_BINARY_OP(int64_t, shr_i64, x >> y)

CUSTOM_BINARY_OP_OUT(uint8_t, uint8_t, logical_and_u8, x && y)
CUSTOM_BINARY_OP_OUT(int64_t, uint8_t, logical_and_i64, x && y)
CUSTOM_BINARY_OP_OUT(float, uint8_t, logical_and_f32, x && y)
CUSTOM_BINARY_OP_OUT(uint8_t, uint8_t, logical_or_u8, x || y)
CUSTOM_BINARY_OP_OUT(int64_t, uint8_t, logical_or_i64, x || y)
CUSTOM_BINARY_OP_OUT(float, uint8_t, logical_or_f32, x || y)
CUSTOM_BINARY_OP_OUT(int64_t, uint8_t, logical_xor_i64, !x != !y)
CUSTOM_BINARY_OP_OUT(float, uint8_t, logical_xor_f32, !x != !y)
