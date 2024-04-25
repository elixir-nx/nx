#include "exla_cuda.h"

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>

#include <cstring>
#include <iostream>

std::pair<std::vector<unsigned char>, int> get_cuda_ipc_handle(std::uintptr_t ptr) {
  cudaIpcMemHandle_t ipc_handle;
  cudaError_t status = cudaIpcGetMemHandle(&ipc_handle, reinterpret_cast<void*>(ptr));

  // Assuming sizeof(cudaIpcMemHandle_t) is constant
  const size_t size = sizeof(cudaIpcMemHandle_t);

  // Copy the memory handle to a byte array
  std::vector<unsigned char> result(size);
  memcpy(result.data(), &ipc_handle, size);

  return std::make_pair(result, status != cudaSuccess);
}

std::pair<void*, int> get_pointer_for_ipc_handle(std::vector<int64_t> handle_list) {
  unsigned char ipc_handle_data[sizeof(cudaIpcMemHandle_t)];
  for (int i = 0; i < sizeof(cudaIpcMemHandle_t); i++) {
    ipc_handle_data[i] = (uint8_t)handle_list[i];
  }

  cudaIpcMemHandle_t ipc_handle;
  memcpy(&ipc_handle, ipc_handle_data, sizeof(cudaIpcMemHandle_t));

  int* ptr;
  cudaError_t cuda_status = cudaSetDevice(0);  // Assuming device 0, change as needed
  if (cuda_status != cudaSuccess) {
    printf("Error setting CUDA device: %s\n", cudaGetErrorString(cuda_status));
    return std::make_pair(nullptr, 1);  // Return with error status
  }

  cuda_status = cudaIpcOpenMemHandle((void**)&ptr, ipc_handle, cudaIpcMemLazyEnablePeerAccess);
  if (cuda_status != cudaSuccess) {
    printf("Error opening CUDA IPC memory handle: %s\n", cudaGetErrorString(cuda_status));
    return std::make_pair(nullptr, 1);  // Return with error status
  }

  return std::make_pair(ptr, cuda_status != cudaSuccess);
}
#else
std::pair<std::vector<unsigned char>, int> get_cuda_ipc_handle(std::uintptr_t ptr) {
  return std::make_pair(std::vector<unsigned char>(0), 1);
}

std::pair<void*, int> get_pointer_for_ipc_handle(std::vector<int64_t> handle_list) {
  return std::make_pair(nullptr, 1);
}
#endif