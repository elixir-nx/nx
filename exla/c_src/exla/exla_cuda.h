#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

std::optional<std::string> get_cuda_ipc_handle(std::uintptr_t);
std::optional<void *> get_pointer_for_ipc_handle(uint8_t *, size_t, int);
