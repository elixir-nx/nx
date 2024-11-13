#pragma once

#include "Eigen/Eigenvalues"

#include <iostream>

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