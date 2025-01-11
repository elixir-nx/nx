#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

std::pair<std::vector<unsigned char>, int> get_cuda_ipc_handle(std::uintptr_t);
std::pair<void*, int> get_pointer_for_ipc_handle(uint8_t*, size_t, int);