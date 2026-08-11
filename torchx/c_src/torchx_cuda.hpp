#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace torchx {

std::optional<std::string> get_cuda_ipc_handle(std::uintptr_t ptr);
std::optional<void *> get_pointer_for_ipc_handle(uint8_t *handle_bin,
                                                 size_t handle_size,
                                                 int device_id);
void close_cuda_ipc_handle(void *ptr);

} // namespace torchx
