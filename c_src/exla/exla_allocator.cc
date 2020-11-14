#include "tensorflow/compiler/xla/exla/exla_allocator.h"

namespace xla {

  ExlaAllocator::ExlaAllocator(se::Platform* platform)
      : se::DeviceMemoryAllocator(platform) {}

  ExlaAllocator::~ExlaAllocator() {
    if (!allocations_.empty()) {
      LOG(FATAL) << "Some allocations not freed!";
    }
  }

  StatusOr<se::OwningDeviceMemory> ExlaAllocator::Allocate(int device_ordinal, uint64 size,
                                            bool /*retry_on_failure*/,
                                            int64 /*memory_space*/){
    // By contract, we must return null if size == 0.
    if (size == 0) {
      return se::OwningDeviceMemory();
    }
    void *buf = malloc(size);
    allocations_.insert({device_ordinal, buf});
    return se::OwningDeviceMemory(se::DeviceMemoryBase(buf, size),
                                  device_ordinal, this);
  }

  Status ExlaAllocator::Deallocate(int device_ordinal, se::DeviceMemoryBase mem){
    if (mem.is_null()) {
      return Status::OK();
    }

    auto it = allocations_.find({device_ordinal, mem.opaque()});
    if (it == allocations_.end()) {
      LOG(FATAL) << "Allocation not found (double free?)";
    } else {
      free(mem.opaque());
      allocations_.erase(it);
    }
    return Status::OK();
  }
} // namespace xla