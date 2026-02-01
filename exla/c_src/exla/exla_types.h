#pragma once

#include <complex>

#include "tsl/platform/bfloat16.h"
#include "Eigen/Core"  // from @eigen_archive

namespace exla {
// We standardize numeric types with tensorflow to ensure we are always
// getting an input with the correct width and to ensure that tensorflow
// is happy with what we're giving it.
//
// Most of these types will only ever be used when creating scalar constants;
// however, some methods require a 64-bit integer. You should prefer standard
// types over these unless you are (1) working with computations, or
// (2) require a non-standard width numeric type (like 64-bit integer).
using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;
using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;
using float16 = Eigen::half;
using bfloat16 = Eigen::bfloat16;
using float32 = float;
using float64 = double;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace exla
