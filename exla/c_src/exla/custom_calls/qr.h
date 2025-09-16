#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include "Eigen/QR"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace ffi = xla::ffi;

template <typename DataType>
void single_matrix_qr_cpu_custom_call(DataType *q_out, DataType *r_out,
                                      DataType *in, uint64_t m, uint64_t k,
                                      uint64_t n, bool complete) {
  typedef Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>
      RowMajorMatrix;

  Eigen::Map<RowMajorMatrix> input(in, m, n);
  Eigen::HouseholderQR<RowMajorMatrix> qr = input.householderQr();

  RowMajorMatrix Q, R;
  size_t num_bytes_q;

  if (complete) {
    Q = qr.householderQ() * RowMajorMatrix::Identity(m, m);
    R = qr.matrixQR();

    num_bytes_q = m * m * sizeof(DataType);

    for (uint64_t i = 0; i < m; ++i) {
      for (uint64_t j = 0; j < n; ++j) {
        r_out[i * n + j] = (j >= i) ? R(i, j) : static_cast<DataType>(0.0);
      }
    }
  } else {
    Q = qr.householderQ() * RowMajorMatrix::Identity(m, k);
    R = qr.matrixQR().topRows(k);

    num_bytes_q = m * k * sizeof(DataType);

    for (uint64_t i = 0; i < k; ++i) {
      for (uint64_t j = 0; j < n; ++j) {
        r_out[i * n + j] = (j >= i) ? R(i, j) : static_cast<DataType>(0.0);
      }
    }
  }

  memcpy(q_out, Q.data(), num_bytes_q);
}

template <typename DataType, typename BufferType>
ffi::Error qr_cpu_custom_call_impl(BufferType operand,
                                   ffi::Result<BufferType> q,
                                   ffi::Result<BufferType> r) {
  auto operand_dims = operand.dimensions();
  auto q_dims = q->dimensions();
  auto r_dims = r->dimensions();

  uint64_t m = q_dims[q_dims.size() - 2];
  uint64_t k = q_dims[q_dims.size() - 1];
  uint64_t n = r_dims[r_dims.size() - 1];
  uint64_t l = r_dims[r_dims.size() - 2];

  bool complete = l == m;

  uint64_t batch_items = 1;
  for (auto it = operand_dims.begin(); it != operand_dims.end() - 2; it++) {
    batch_items *= *it;
  }

  uint64_t q_stride = m * k;
  uint64_t r_stride = n * l;
  uint64_t inner_stride = m * n;

  for (uint64_t i = 0; i < batch_items; i++) {
    single_matrix_qr_cpu_custom_call<DataType>(
        (DataType *)q->untyped_data() + i * q_stride,
        (DataType *)r->untyped_data() + i * r_stride,
        (DataType *)operand.untyped_data() + i * inner_stride, m, k, n,
        complete);
  }

  return ffi::Error::Success();
}
