#pragma once

#include <cstddef>
#include <sys/types.h>

// Create a new shm segment, set its size, and return a writable fd.
// `mode` is the file permission bits (e.g. 0o400, 0o600).
int get_ipc_handle(const char* memname, size_t memsize, mode_t mode);

// Open an existing shm segment.  Tries O_RDWR first; if EACCES, falls back to
// O_RDONLY.  Sets *out_writable to 1 if write access was granted, 0 otherwise.
// Returns -1 on error.
int open_existing_ipc_handle(const char* memname, int* out_writable);

// Map a shm fd with MAP_SHARED.  `writable` controls whether PROT_WRITE is
// included; the fd must have been opened with the matching access mode.
void* open_ipc_handle(int fd, size_t memsize, int writable);

// Sender cleanup: munmap + close + shm_unlink.
int close_ipc_handle(int fd, void* ptr, const char* memname, size_t memsize);

// Receiver cleanup: munmap + close only (no unlink — sender owns the name).
int close_imported_ipc_handle(int fd, void* ptr, size_t memsize);
