#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include "Eigen/Eigenvalues"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace ffi = xla::ffi;

template <typename DataType>
void single_matrix_eigh_cpu_custom_call(DataType *eigenvalues_out,
                                        DataType *eigenvectors_out,
                                        DataType *in, uint64_t m, uint64_t n) {
  typedef Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>
      RowMajorMatrix;

  // Map the input matrix
  Eigen::Map<RowMajorMatrix> input(in, m, n);

  // Compute the Eigenvalue decomposition
  Eigen::SelfAdjointEigenSolver<RowMajorMatrix> eigensolver(input);

  if (eigensolver.info() != Eigen::Success) {
    std::cerr << "Eigenvalue decomposition failed!" << std::endl;
    return;
  }

  // Get the eigenvalues and eigenvectors
  Eigen::Matrix<DataType, Eigen::Dynamic, 1> eigenvalues =
      eigensolver.eigenvalues();
  RowMajorMatrix eigenvectors = eigensolver.eigenvectors();

  // Create a vector of indices and sort it based on eigenvalues in decreasing
  // order
  std::vector<int> indices(m);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&eigenvalues](int i, int j) {
    return std::abs(eigenvalues(i)) > std::abs(eigenvalues(j));
  });

  // Sort eigenvalues and rearrange eigenvectors
  Eigen::Matrix<DataType, Eigen::Dynamic, 1> sorted_eigenvalues(m);
  RowMajorMatrix sorted_eigenvectors(m, n);
  for (int i = 0; i < m; ++i) {
    sorted_eigenvalues(i) = eigenvalues(indices[i]);
    sorted_eigenvectors.col(i) = eigenvectors.col(indices[i]);
  }

  // Copy the sorted eigenvalues to the output
  std::memcpy(eigenvalues_out, sorted_eigenvalues.data(), m * sizeof(DataType));

  // Copy the sorted eigenvectors to the output
  std::memcpy(eigenvectors_out, sorted_eigenvectors.data(),
              m * n * sizeof(DataType));
}

template <typename DataType, typename BufferType>
ffi::Error eigh_cpu_custom_call_impl(BufferType operand,
                                     ffi::Result<BufferType> eigenvalues,
                                     ffi::Result<BufferType> eigenvectors) {
  auto operand_dims = operand.dimensions();
  auto eigenvalues_dims = eigenvalues->dimensions();
  auto eigenvectors_dims = eigenvectors->dimensions();

  uint64_t m = eigenvectors_dims[eigenvectors_dims.size() - 2];
  uint64_t n = eigenvectors_dims[eigenvectors_dims.size() - 1];

  uint64_t batch_items = 1;
  for (auto it = operand_dims.begin(); it != operand_dims.end() - 2; it++) {
    batch_items *= *it;
  }

  uint64_t eigenvalues_stride = eigenvalues_dims[eigenvalues_dims.size() - 1];
  uint64_t eigenvectors_stride = m * n;
  uint64_t inner_stride = m * n;

  for (uint64_t i = 0; i < batch_items; i++) {
    single_matrix_eigh_cpu_custom_call<DataType>(
        eigenvalues->typed_data() + i * eigenvalues_stride,
        eigenvectors->typed_data() + i * eigenvectors_stride,
        operand.typed_data() + i * inner_stride, m, n);
  }

  return ffi::Error::Success();
}
