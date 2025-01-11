#pragma once

#include <complex>

#include "tsl/platform/types.h"
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
using int8 = tsl::int8;
using int16 = tsl::int16;
using int32 = tsl::int32;
using int64 = tsl::int64;
using uint8 = tsl::uint8;
using uint16 = tsl::uint16;
using uint32 = tsl::uint32;
using uint64 = tsl::uint64;
using float16 = Eigen::half;
using bfloat16 = tsl::bfloat16;
using float32 = float;
using float64 = double;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;
}  // namespace exla
