#pragma once

#include <cstddef>

int get_ipc_handle(const char* memname, size_t memsize);
void* open_ipc_handle(int fd, size_t memsize);
int close_ipc_handle(int fd, void* ptr, char* memname, size_t memsize);
