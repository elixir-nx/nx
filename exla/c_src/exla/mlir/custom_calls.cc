#include "custom_calls.h"

#include <Eigen/Dense>
#include <Eigen/QR>

#include "builder.h"

template <typename DataType>
void qr_cpu_custom_call(void *out[], const void *in[]) {
  typedef Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorMatrix;
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

  // TO-DO: support batched matrices
  Eigen::Map<RowMajorMatrix> input(operand, operand_dims[operand_dims.size() - 2], operand_dims[operand_dims.size() - 1]);
  Eigen::HouseholderQR<RowMajorMatrix> qr = input.householderQr();
  RowMajorMatrix Q = qr.householderQ();
  RowMajorMatrix R = qr.matrixQR().template triangularView<Eigen::Upper>();

  int64_t total_q_size = 1;
  for (int64_t i = 0; i < num_q_dims; i++) {
    total_q_size *= q_dims[i];
  }
  int64_t total_r_size = 1;
  for (int64_t i = 0; i < num_r_dims; i++) {
    total_r_size *= r_dims[i];
  }

  DataType *q = (DataType *)out[0];
  memcpy(q, Q.data(), total_q_size * sizeof(DataType));

  DataType *r = (DataType *)out[1];
  memcpy(r, R.data(), total_r_size * sizeof(DataType));

  return;
}

void qr_cpu_custom_call_f32(void *out[], const void *in[]) {
  qr_cpu_custom_call<float>(out, in);
}

void qr_cpu_custom_call_f64(void *out[], const void *in[]) {
  qr_cpu_custom_call<double>(out, in);
}