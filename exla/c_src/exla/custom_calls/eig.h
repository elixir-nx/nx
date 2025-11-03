#pragma once

#include <algorithm>
#include <complex>
#include <iostream>
#include <numeric>
#include <vector>

#include "Eigen/Eigenvalues"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace ffi = xla::ffi;

// For real input types, compute complex eigenvalues/eigenvectors
template <typename DataType, typename ComplexType>
void single_matrix_eig_cpu_custom_call_real(ComplexType *eigenvalues_out,
                                            ComplexType *eigenvectors_out,
                                            DataType *in, uint64_t m,
                                            uint64_t n) {
  typedef Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>
      RowMajorMatrix;
  typedef Eigen::Matrix<ComplexType, Eigen::Dynamic, 1> ComplexVector;
  typedef Eigen::Matrix<ComplexType, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>
      ComplexRowMajorMatrix;

  // Map the input matrix
  Eigen::Map<RowMajorMatrix> input(in, m, n);

  // Compute the Eigenvalue decomposition for general (non-symmetric) matrices
  Eigen::EigenSolver<RowMajorMatrix> eigensolver(input);

  if (eigensolver.info() != Eigen::Success) {
    std::cerr << "Eigenvalue decomposition failed!" << std::endl;
    return;
  }

  // Get the eigenvalues and eigenvectors (both are complex)
  ComplexVector eigenvalues = eigensolver.eigenvalues();
  ComplexRowMajorMatrix eigenvectors = eigensolver.eigenvectors();

  // Create a vector of indices and sort it based on eigenvalues magnitude in
  // decreasing order
  std::vector<int> indices(m);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&eigenvalues](int i, int j) {
    return std::abs(eigenvalues(i)) > std::abs(eigenvalues(j));
  });

  // Sort eigenvalues and rearrange eigenvectors
  ComplexVector sorted_eigenvalues(m);
  ComplexRowMajorMatrix sorted_eigenvectors(m, n);
  for (int i = 0; i < m; ++i) {
    sorted_eigenvalues(i) = eigenvalues(indices[i]);
    sorted_eigenvectors.col(i) = eigenvectors.col(indices[i]);
  }

  // Copy the sorted eigenvalues to the output
  std::memcpy(eigenvalues_out, sorted_eigenvalues.data(),
              m * sizeof(ComplexType));

  // Copy the sorted eigenvectors to the output
  std::memcpy(eigenvectors_out, sorted_eigenvectors.data(),
              m * n * sizeof(ComplexType));
}

// For complex input types
template <typename ComplexType>
void single_matrix_eig_cpu_custom_call_complex(ComplexType *eigenvalues_out,
                                               ComplexType *eigenvectors_out,
                                               ComplexType *in, uint64_t m,
                                               uint64_t n) {
  typedef Eigen::Matrix<ComplexType, Eigen::Dynamic, Eigen::Dynamic,
                        Eigen::RowMajor>
      ComplexRowMajorMatrix;
  typedef Eigen::Matrix<ComplexType, Eigen::Dynamic, 1> ComplexVector;

  // Map the input matrix
  Eigen::Map<ComplexRowMajorMatrix> input(in, m, n);

  // Compute the Eigenvalue decomposition for complex matrices
  Eigen::ComplexEigenSolver<ComplexRowMajorMatrix> eigensolver(input);

  if (eigensolver.info() != Eigen::Success) {
    std::cerr << "Eigenvalue decomposition failed!" << std::endl;
    return;
  }

  // Get the eigenvalues and eigenvectors
  ComplexVector eigenvalues = eigensolver.eigenvalues();
  ComplexRowMajorMatrix eigenvectors = eigensolver.eigenvectors();

  // Create a vector of indices and sort it based on eigenvalues magnitude in
  // decreasing order
  std::vector<int> indices(m);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&eigenvalues](int i, int j) {
    return std::abs(eigenvalues(i)) > std::abs(eigenvalues(j));
  });

  // Sort eigenvalues and rearrange eigenvectors
  ComplexVector sorted_eigenvalues(m);
  ComplexRowMajorMatrix sorted_eigenvectors(m, n);
  for (int i = 0; i < m; ++i) {
    sorted_eigenvalues(i) = eigenvalues(indices[i]);
    sorted_eigenvectors.col(i) = eigenvectors.col(indices[i]);
  }

  // Copy the sorted eigenvalues to the output
  std::memcpy(eigenvalues_out, sorted_eigenvalues.data(),
              m * sizeof(ComplexType));

  // Copy the sorted eigenvectors to the output
  std::memcpy(eigenvectors_out, sorted_eigenvectors.data(),
              m * n * sizeof(ComplexType));
}

// For real types (f32, f64)
template <typename DataType, typename ComplexType, typename BufferType,
          typename ComplexBufferType>
ffi::Error
eig_cpu_custom_call_impl_real(BufferType operand,
                              ffi::Result<ComplexBufferType> eigenvalues,
                              ffi::Result<ComplexBufferType> eigenvectors) {
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
    single_matrix_eig_cpu_custom_call_real<DataType, ComplexType>(
        eigenvalues->typed_data() + i * eigenvalues_stride,
        eigenvectors->typed_data() + i * eigenvectors_stride,
        operand.typed_data() + i * inner_stride, m, n);
  }

  return ffi::Error::Success();
}

// For complex types (c64, c128)
template <typename ComplexType, typename BufferType>
ffi::Error
eig_cpu_custom_call_impl_complex(BufferType operand,
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
    single_matrix_eig_cpu_custom_call_complex<ComplexType>(
        eigenvalues->typed_data() + i * eigenvalues_stride,
        eigenvectors->typed_data() + i * eigenvectors_stride,
        operand.typed_data() + i * inner_stride, m, n);
  }

  return ffi::Error::Success();
}
