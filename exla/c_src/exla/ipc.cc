#include "ipc.h"

#include <cstdio>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Create or open a shared memory object and set its size. `mode` is the
// file mode bits forwarded to shm_open(3). The default is chosen in the
// Elixir caller (EXLA.Backend.to_pointer/2).
int get_ipc_handle(const char* memname, size_t memsize, mode_t mode) {
  int fd = shm_open(memname, O_CREAT | O_RDWR, mode);
  if (fd == -1) {
    return -1;
  }

  if (ftruncate(fd, memsize) == -1) {
    close(fd);
    return -1;
  }

  return fd;
}

// Try O_RDWR first (permissions allow write); fall back to O_RDONLY on EACCES.
int open_existing_ipc_handle(const char* memname, int* out_writable) {
  int fd = shm_open(memname, O_RDWR, 0);
  if (fd != -1) {
    *out_writable = 1;
    return fd;
  }
  if (errno == EACCES) {
    fd = shm_open(memname, O_RDONLY, 0);
    if (fd != -1) {
      *out_writable = 0;
      return fd;
    }
  }
  return -1;
}

// MAP_SHARED zero-copy mapping. `writable` adds PROT_WRITE; fd access mode
// must match (O_RDWR for writable, O_RDONLY for read-only).
void* open_ipc_handle(int fd, size_t memsize, int writable) {
  int prot = writable ? (PROT_READ | PROT_WRITE) : PROT_READ;
  void* ptr = mmap(NULL, memsize, prot, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    perror("mmap");
    return nullptr;
  }
  return ptr;
}

// Sender cleanup: remove the shm name so the object is freed once all fds close.
int close_ipc_handle(int fd, void* ptr, const char* memname, size_t memsize) {
  if (munmap(ptr, memsize) == -1) {
    return -1;
  }

  if (close(fd) == -1) {
    return -1;
  }

  shm_unlink(memname);

  return 0;
}

// Receiver cleanup: only munmap + close; the sender owns the shm name/lifetime.
int close_imported_ipc_handle(int fd, void* ptr, size_t memsize) {
  munmap(ptr, memsize);
  close(fd);
  return 0;
}
