#include "custom_calls.h"

#include "Eigen/Dense"
#include "Eigen/Eigenvalues"
#include "Eigen/QR"
#include "exla_nif_util.h"
#include "xla/service/custom_call_target_registry.h"

template <typename DataType>
void single_matrix_eigh_cpu_custom_call(DataType *eigenvalues_out, DataType *eigenvectors_out, DataType *in, uint64_t m, uint64_t n) {
  typedef Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorMatrix;

  // Map the input matrix
  Eigen::Map<RowMajorMatrix> input(in, m, n);

  // Compute the Eigenvalue decomposition
  Eigen::SelfAdjointEigenSolver<RowMajorMatrix> eigensolver(input);

  if (eigensolver.info() != Eigen::Success) {
    std::cerr << "Eigenvalue decomposition failed!" << std::endl;
    return;
  }

  // Get the eigenvalues and eigenvectors
  Eigen::Matrix<DataType, Eigen::Dynamic, 1> eigenvalues = eigensolver.eigenvalues();
  RowMajorMatrix eigenvectors = eigensolver.eigenvectors();

  // Copy the eigenvalues to the output
  std::memcpy(eigenvalues_out, eigenvalues.data(), m * sizeof(DataType));

  // Copy the eigenvectors to the output
  std::memcpy(eigenvectors_out, eigenvectors.data(), m * n * sizeof(DataType));
}

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

  uint64_t r_stride = r_dims[r_dims.size() - 1] * r_dims[r_dims.size() - 2] * sizeof(DataType);
  uint64_t q_stride = q_dims[q_dims.size() - 1] * q_dims[q_dims.size() - 2] * sizeof(DataType);
  uint64_t inner_stride = m * n * sizeof(DataType);

  for (uint64_t i = 0; i < batch_items; i++) {
    single_matrix_qr_cpu_custom_call<DataType>(
        (DataType *)out[0] + i * q_stride,
        (DataType *)out[1] + i * r_stride,
        operand + i * inner_stride * sizeof(DataType),
        m, k, n, complete);
  }
}

template <typename DataType>
void eigh_cpu_custom_call(void *out[], const void *in[]) {
  DataType *operand = (DataType *)in[0];

  uint64_t *dim_sizes = (uint64_t *)in[1];
  uint64_t num_operand_dims = dim_sizes[0];
  uint64_t num_eigenvalues_dims = dim_sizes[1];
  uint64_t num_eigenvectors_dims = dim_sizes[2];

  uint64_t *operand_dims_ptr = (uint64_t *)in[2];
  std::vector<uint64_t> operand_dims(operand_dims_ptr, operand_dims_ptr + num_operand_dims);

  uint64_t *eigenvalues_dims_ptr = (uint64_t *)in[3];
  std::vector<uint64_t> eigenvalues_dims(eigenvalues_dims_ptr, eigenvalues_dims_ptr + num_eigenvalues_dims);

  uint64_t *eigenvectors_dims_ptr = (uint64_t *)in[4];
  std::vector<uint64_t> eigenvectors_dims(eigenvectors_dims_ptr, eigenvectors_dims_ptr + num_eigenvectors_dims);

  uint64_t m = eigenvectors_dims[eigenvectors_dims.size() - 2];
  uint64_t n = eigenvectors_dims[eigenvectors_dims.size() - 1];

  auto leading_dimensions = std::vector<uint64_t>(operand_dims.begin(), operand_dims.end() - 2);

  uint64_t batch_items = 1;
  for (uint64_t i = 0; i < leading_dimensions.size(); i++) {
    batch_items *= leading_dimensions[i];
  }

  DataType *eigenvalues = (DataType *)out[0];
  DataType *eigenvectors = (DataType *)out[1];

  uint64_t eigenvalues_stride = eigenvalues_dims[eigenvalues_dims.size() - 1] * sizeof(DataType);
  uint64_t eigenvectors_stride = eigenvectors_dims[eigenvectors_dims.size() - 1] * eigenvectors_dims[eigenvectors_dims.size() - 2] * sizeof(DataType);
  uint64_t inner_stride = m * n * sizeof(DataType);

  for (uint64_t i = 0; i < batch_items; i++) {
    single_matrix_eigh_cpu_custom_call<DataType>(
        eigenvalues + i * eigenvalues_stride,
        eigenvectors + i * eigenvectors_stride,
        operand + i * inner_stride / sizeof(DataType),
        m, n);
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

void eigh_cpu_custom_call_f32(void *out[], const void *in[]) {
  eigh_cpu_custom_call<float>(out, in);
}

void eigh_cpu_custom_call_f64(void *out[], const void *in[]) {
  eigh_cpu_custom_call<double>(out, in);
}

XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("qr_cpu_custom_call_f32", qr_cpu_custom_call_f32);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("qr_cpu_custom_call_f64", qr_cpu_custom_call_f64);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("qr_cpu_custom_call_f16", qr_cpu_custom_call_f16);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("qr_cpu_custom_call_bf16", qr_cpu_custom_call_bf16);


XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("eigh_cpu_custom_call_f32", eigh_cpu_custom_call_f32);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("eigh_cpu_custom_call_f64", eigh_cpu_custom_call_f64);
