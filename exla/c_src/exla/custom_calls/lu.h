#pragma once

#include "Eigen/LU";

template <typename DataType>
void single_matrix_lu_cpu_custom_call(uint8_t *p_out, DataType *l_out, DataType *u_out, DataType *in, uint64_t n) {
  typedef Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMajorMatrix;

  Eigen::Map<RowMajorMatrix> input(in, n, n);
  Eigen::PartialPivLU<RowMajorMatrix> lu = input.partialPivLu();

  // Get the permutation matrix P and convert to indices
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P = lu.permutationP();
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

template <typename DataType>
void lu_cpu_custom_call(void *out[], const void *in[]) {
  DataType *operand = (DataType *)in[0];

  uint64_t *dim_sizes = (uint64_t *)in[1];
  uint64_t num_operand_dims = dim_sizes[0];
  uint64_t num_p_dims = dim_sizes[1];
  uint64_t num_l_dims = dim_sizes[2];
  uint64_t num_u_dims = dim_sizes[3];

  uint64_t *operand_dims_ptr = (uint64_t *)in[2];
  std::vector<uint64_t> operand_dims(operand_dims_ptr, operand_dims_ptr + num_operand_dims);

  uint64_t *p_dims_ptr = (uint64_t *)in[3];
  std::vector<uint64_t> p_dims(p_dims_ptr, p_dims_ptr + num_p_dims);

  uint64_t *l_dims_ptr = (uint64_t *)in[4];
  std::vector<uint64_t> l_dims(l_dims_ptr, l_dims_ptr + num_l_dims);

  uint64_t *u_dims_ptr = (uint64_t *)in[5];
  std::vector<uint64_t> u_dims(u_dims_ptr, u_dims_ptr + num_u_dims);

  uint64_t n = l_dims[l_dims.size() - 1];

  auto leading_dimensions = std::vector<uint64_t>(operand_dims.begin(), operand_dims.end() - 2);

  uint64_t batch_items = 1;
  for (uint64_t i = 0; i < leading_dimensions.size(); i++) {
    batch_items *= leading_dimensions[i];
  }

  uint8_t *p = (uint8_t *)out[0];
  DataType *l = (DataType *)out[1];
  DataType *u = (DataType *)out[2];

  uint64_t stride = n * n;

  for (uint64_t i = 0; i < batch_items; i++) {
    single_matrix_lu_cpu_custom_call<DataType>(
        p + i * stride,
        l + i * stride,
        u + i * stride,
        operand + i * stride,
        n);
  }
}