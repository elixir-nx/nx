#include "custom_calls.h"

#include <Eigen/Dense>
#include <Eigen/QR>

#include "builder.h"

template <typename DataType>
void single_matrix_qr_cpu_custom_call(DataType *q_out, DataType *r_out, DataType *in, int64_t m, int64_t k, int64_t n, bool complete) {
  typedef Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorMatrix;

  Eigen::Map<RowMajorMatrix> input(in, m, n);
  Eigen::HouseholderQR<RowMajorMatrix> qr = input.householderQr();

  RowMajorMatrix Q, R;
  size_t num_bytes_q, num_bytes_r;

  if (complete) {
    Q = qr.householderQ() * RowMajorMatrix::Identity(m, m);
    R = qr.matrixQR();

    num_bytes_q = m * m * sizeof(DataType);

    for (int64_t i = 0; i < m; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        r_out[i * n + j] = (j >= i) ? R(i, j) : static_cast<DataType>(0.0);
      }
    }
  } else {
    Q = qr.householderQ() * RowMajorMatrix::Identity(m, k);
    R = qr.matrixQR().topRows(k);

    num_bytes_q = m * k * sizeof(DataType);

    for (int64_t i = 0; i < k; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        r_out[i * n + j] = (j >= i) ? R(i, j) : static_cast<DataType>(0.0);
      }
    }
  }

  memcpy(q_out, Q.data(), num_bytes_q);
}

template <typename DataType>
void qr_cpu_custom_call(void *out[], const void *in[]) {
  DataType *operand = (DataType *)in[0];

  int64_t *dim_sizes = (int64_t *)in[1];
  int64_t num_operand_dims = dim_sizes[0];
  int64_t num_q_dims = dim_sizes[1];
  int64_t num_r_dims = dim_sizes[2];

  int64_t *operand_dims_ptr = (int64_t *)in[2];
  std::vector<int64_t> operand_dims(operand_dims_ptr, operand_dims_ptr + num_operand_dims);

  int64_t *q_dims_ptr = (int64_t *)in[3];
  std::vector<int64_t> q_dims(q_dims_ptr, q_dims_ptr + num_q_dims);

  int64_t *r_dims_ptr = (int64_t *)in[4];
  std::vector<int64_t> r_dims(r_dims_ptr, r_dims_ptr + num_r_dims);

  int64_t m = q_dims[q_dims.size() - 2];
  int64_t k = q_dims[q_dims.size() - 1];
  int64_t n = r_dims[r_dims.size() - 1];
  bool complete = r_dims[r_dims.size() - 2] == m;

  auto leading_dimensions = std::vector<int64_t>(operand_dims.begin(), operand_dims.end() - 2);

  int64_t batch_items = 1;
  for (int64_t i = 0; i < leading_dimensions.size(); i++) {
    batch_items *= leading_dimensions[i];
  }

  DataType *q = (DataType *)out[0];
  DataType *r = (DataType *)out[1];

  int64_t r_stride = r_dims[r_dims.size() - 1] * r_dims[r_dims.size() - 2] * sizeof(DataType);
  int64_t q_stride = q_dims[q_dims.size() - 1] * q_dims[q_dims.size() - 2] * sizeof(DataType);
  int64_t inner_stride = m * n * sizeof(DataType);

  for (int64_t i = 0; i < batch_items; i++) {
    single_matrix_qr_cpu_custom_call<DataType>(
        (DataType *)out[0] + i * q_stride,
        (DataType *)out[1] + i * r_stride,
        operand + i * inner_stride * sizeof(DataType),
        m, k, n, complete);
  }
}

void qr_cpu_custom_call_bf16(void *out[], const void *in[]) {
  qr_cpu_custom_call<exla::bfloat16>(out, in);
}

void qr_cpu_custom_call_f16(void *out[], const void *in[]) {
  qr_cpu_custom_call<exla::float16>(out, in);
}

void qr_cpu_custom_call_f32(void *out[], const void *in[]) {
  qr_cpu_custom_call<float>(out, in);
}

void qr_cpu_custom_call_f64(void *out[], const void *in[]) {
  qr_cpu_custom_call<double>(out, in);
}