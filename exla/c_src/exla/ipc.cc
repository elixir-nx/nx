#include "ipc.h"

#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Function to create or open a shared memory object and set its size
int get_ipc_handle(const char* memname, size_t memsize) {
  int fd = shm_open(memname, O_CREAT | O_RDWR, 0666);
  if (fd == -1) {
    return -1;
  }

  if (ftruncate(fd, memsize) == -1) {
    close(fd);
    return -1;
  }

  return fd;
}

// Function to map the shared memory in this process
void* open_ipc_handle(int fd, size_t memsize) {
  void* ptr = mmap(NULL, memsize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (ptr == MAP_FAILED) {
    perror("mmap");
    return nullptr;
  }
  return ptr;
}

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
