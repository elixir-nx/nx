#pragma once

#include <cstddef>
#include <sys/types.h>

int get_ipc_handle(const char* memname, size_t memsize, mode_t mode);
void* open_ipc_handle(int fd, size_t memsize);
int close_ipc_handle(int fd, void* ptr, const char* memname, size_t memsize);
