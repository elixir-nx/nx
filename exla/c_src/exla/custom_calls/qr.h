
#pragma once

#include "Eigen/QR"

template <typename DataType>
void single_matrix_qr_cpu_custom_call(DataType *q_out, DataType *r_out, DataType *in, uint64_t m, uint64_t k, uint64_t n, bool complete) {
  typedef Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorMatrix;

  Eigen::Map<RowMajorMatrix> input(in, m, n);
  Eigen::HouseholderQR<RowMajorMatrix> qr = input.householderQr();

  RowMajorMatrix Q, R;
  size_t num_bytes_q, num_bytes_r;

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

template <typename DataType>
void qr_cpu_custom_call(void *out[], const void *in[]) {
  DataType *operand = (DataType *)in[0];

  uint64_t *dim_sizes = (uint64_t *)in[1];
  uint64_t num_operand_dims = dim_sizes[0];
  uint64_t num_q_dims = dim_sizes[1];
  uint64_t num_r_dims = dim_sizes[2];

  uint64_t *operand_dims_ptr = (uint64_t *)in[2];
  std::vector<uint64_t> operand_dims(operand_dims_ptr, operand_dims_ptr + num_operand_dims);

  uint64_t *q_dims_ptr = (uint64_t *)in[3];
  std::vector<uint64_t> q_dims(q_dims_ptr, q_dims_ptr + num_q_dims);

  uint64_t *r_dims_ptr = (uint64_t *)in[4];
  std::vector<uint64_t> r_dims(r_dims_ptr, r_dims_ptr + num_r_dims);

  uint64_t m = q_dims[q_dims.size() - 2];
  uint64_t k = q_dims[q_dims.size() - 1];
  uint64_t n = r_dims[r_dims.size() - 1];
  bool complete = r_dims[r_dims.size() - 2] == m;

  auto leading_dimensions = std::vector<uint64_t>(operand_dims.begin(), operand_dims.end() - 2);

  uint64_t batch_items = 1;
  for (uint64_t i = 0; i < leading_dimensions.size(); i++) {
    batch_items *= leading_dimensions[i];
  }

  DataType *q = (DataType *)out[0];
  DataType *r = (DataType *)out[1];

  uint64_t r_stride = r_dims[r_dims.size() - 1] * r_dims[r_dims.size() - 2];
  uint64_t q_stride = q_dims[q_dims.size() - 1] * q_dims[q_dims.size() - 2];
  uint64_t inner_stride = m * n;

  for (uint64_t i = 0; i < batch_items; i++) {
    single_matrix_qr_cpu_custom_call<DataType>(
        (DataType *)out[0] + i * q_stride,
        (DataType *)out[1] + i * r_stride,
        operand + i * inner_stride,
        m, k, n, complete);
  }
}