#include "torchx_cuda.hpp"

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>

#include <cstring>

namespace torchx {

std::optional<std::string> get_cuda_ipc_handle(std::uintptr_t ptr) {
  cudaIpcMemHandle_t ipc_handle;
  cudaError_t status =
      cudaIpcGetMemHandle(&ipc_handle, reinterpret_cast<void *>(ptr));

  if (status != cudaSuccess) {
    return std::nullopt;
  }

  const size_t size = sizeof(cudaIpcMemHandle_t);
  return std::string(reinterpret_cast<const char *>(&ipc_handle), size);
}

std::optional<void *> get_pointer_for_ipc_handle(uint8_t *handle_bin,
                                                 size_t handle_size,
                                                 int device_id) {
  if (handle_size != sizeof(cudaIpcMemHandle_t)) {
    return std::nullopt;
  }

  cudaIpcMemHandle_t ipc_handle;
  std::memcpy(&ipc_handle, handle_bin, sizeof(cudaIpcMemHandle_t));

  cudaError_t cuda_status = cudaSetDevice(device_id);
  if (cuda_status != cudaSuccess) {
    return std::nullopt;
  }

  void *ptr = nullptr;
  cuda_status =
      cudaIpcOpenMemHandle(&ptr, ipc_handle, cudaIpcMemLazyEnablePeerAccess);
  if (cuda_status != cudaSuccess) {
    return std::nullopt;
  }

  return ptr;
}

void close_cuda_ipc_handle(void *ptr) {
  if (ptr != nullptr) {
    cudaIpcCloseMemHandle(ptr);
  }
}

} // namespace torchx

#else

namespace torchx {

std::optional<std::string> get_cuda_ipc_handle(std::uintptr_t) {
  return std::nullopt;
}

std::optional<void *> get_pointer_for_ipc_handle(uint8_t *, size_t, int) {
  return std::nullopt;
}

void close_cuda_ipc_handle(void *) {}

} // namespace torchx

#endif
