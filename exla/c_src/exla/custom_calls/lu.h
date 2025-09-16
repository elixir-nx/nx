#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include "Eigen/LU"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace ffi = xla::ffi;

template <typename DataType>
void single_matrix_lu_cpu_custom_call(uint8_t *p_out, DataType *l_out,
                                      DataType *u_out, DataType *in,
                                      uint64_t n) {
  typedef Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>
      RowMajorMatrix;

  Eigen::Map<RowMajorMatrix> input(in, n, n);
  Eigen::PartialPivLU<RowMajorMatrix> lu = input.partialPivLu();

  // Get the permutation matrix P and convert to indices
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P =
      lu.permutationP();
  for (uint64_t i = 0; i < n; i++) {
    for (uint64_t j = 0; j < n; j++) {
      p_out[i * n + j] = static_cast<uint8_t>(P.indices()[i] == j ? 1 : 0);
    }
  }

  // Get L and U matrices
  RowMajorMatrix L = lu.matrixLU().template triangularView<Eigen::UnitLower>();
  RowMajorMatrix U = lu.matrixLU().template triangularView<Eigen::Upper>();

  // Copy L matrix
  for (uint64_t i = 0; i < n; i++) {
    for (uint64_t j = 0; j < n; j++) {
      if (j < i) {
        l_out[i * n + j] = static_cast<DataType>(L(i, j));
      } else if (j == i) {
        l_out[i * n + j] = static_cast<DataType>(1.0);
      } else {
        l_out[i * n + j] = static_cast<DataType>(0.0);
      }
    }
  }

  // Copy U matrix
  for (uint64_t i = 0; i < n; i++) {
    for (uint64_t j = 0; j < n; j++) {
      if (j >= i) {
        u_out[i * n + j] = static_cast<DataType>(U(i, j));
      } else {
        u_out[i * n + j] = static_cast<DataType>(0.0);
      }
    }
  }
}

template <typename DataType, typename BufferType>
ffi::Error
lu_cpu_custom_call_impl(BufferType operand, ffi::Result<ffi::Buffer<ffi::U8>> p,
                        ffi::Result<BufferType> l, ffi::Result<BufferType> u) {
  auto operand_dims = operand.dimensions();
  auto l_dims = l->dimensions();
  uint64_t n = l_dims[l_dims.size() - 1];

  uint64_t batch_items = 1;
  for (auto it = operand_dims.begin(); it != operand_dims.end() - 2; it++) {
    batch_items *= *it;
  }

  uint64_t stride = n * n;

  for (uint64_t i = 0; i < batch_items; i++) {
    single_matrix_lu_cpu_custom_call<DataType>(
        p->typed_data() + i * stride,
        (DataType *)l->untyped_data() + i * stride,
        (DataType *)u->untyped_data() + i * stride,
        (DataType *)operand.untyped_data() + i * stride, n);
  }

  return ffi::Error::Success();
}
