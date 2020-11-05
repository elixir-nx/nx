#ifndef EXLA_ALLOCATOR_H_
#define EXLA_ALLOCATOR_H_

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"

namespace xla {

  namespace se = tensorflow::se;
  using uint64 = tensorflow::uint64;
  using int64 = tensorflow::int64;

  class ExlaAllocator : public se::DeviceMemoryAllocator {
    public:

      explicit ExlaAllocator(se::Platform* platform);
      virtual ~ExlaAllocator();

      // Pull in two-arg overload of Allocate.
      using se::DeviceMemoryAllocator::Allocate;
      StatusOr<se::OwningDeviceMemory> Allocate(int device_ordinal, uint64 size,
                                          bool /*retry_on_failure*/,
                                          int64 /*memory_space*/) override;

      Status Deallocate(int device_ordinal, se::DeviceMemoryBase mem) override;

      bool AllowsAsynchronousDeallocation() const override { return false; };

      StatusOr<se::Stream*> GetStream(int device_ordinal) override { LOG(FATAL) << "Not implemented"; };

    private:
      std::set<std::pair</*device_ordinal*/ int64, void *>> allocations_;
  };
} // namespace xla

#endif