#include "exla_cuda.h"

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>

#include <cstring>
#include <iostream>
#include <optional>
#include <string>

std::optional<std::string> get_cuda_ipc_handle(std::uintptr_t ptr) {
  cudaIpcMemHandle_t ipc_handle;
  cudaError_t status = cudaIpcGetMemHandle(&ipc_handle, reinterpret_cast<void*>(ptr));

  if (status != cudaSuccess) {
    return std::nullopt;
  }

  // Assuming sizeof(cudaIpcMemHandle_t) is constant
  const size_t size = sizeof(cudaIpcMemHandle_t);

  // Copy the memory handle to a buffer
  auto buffer = std::string(reinterpret_cast<const char*>(&ipc_handle), size);

  return buffer;
}

std::optional<void*> get_pointer_for_ipc_handle(uint8_t* handle_bin, size_t handle_size, int device_id) {
  if (handle_size != sizeof(cudaIpcMemHandle_t)) {
    return std::nullopt;
  }

  unsigned char ipc_handle_data[sizeof(cudaIpcMemHandle_t)];
  for (int i = 0; i < sizeof(cudaIpcMemHandle_t); i++) {
    ipc_handle_data[i] = handle_bin[i];
  }

  cudaIpcMemHandle_t ipc_handle;
  memcpy(&ipc_handle, ipc_handle_data, sizeof(cudaIpcMemHandle_t));

  int* ptr;
  cudaError_t cuda_status = cudaSetDevice(device_id);  // Assuming device 0, change as needed
  if (cuda_status != cudaSuccess) {
    printf("Error setting CUDA device: %s\n", cudaGetErrorString(cuda_status));
    return std::nullopt;
  }

  cuda_status = cudaIpcOpenMemHandle((void**)&ptr, ipc_handle, cudaIpcMemLazyEnablePeerAccess);
  if (cuda_status != cudaSuccess) {
    printf("Error opening CUDA IPC memory handle: %s\n", cudaGetErrorString(cuda_status));
    return std::nullopt;
  }

  return ptr;
}
#else
std::optional<std::string> get_cuda_ipc_handle(std::uintptr_t ptr) {
  return std::nullopt;
}

std::optional<void*> get_pointer_for_ipc_handle(uint8_t* handle_bin, size_t handle_size, int device_id) {
  return std::nullopt;
}
#endif
