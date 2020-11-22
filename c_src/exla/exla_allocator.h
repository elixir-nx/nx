#ifndef EXLA_ALLOCATOR_H_
#define EXLA_ALLOCATOR_H_

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/compiler/xla/exla/erts/erl_nif.h"
#include "tensorflow/core/platform/mem.h"

namespace exla {

  /*
   * Allocator which allocates/deallocates directly on ERTS.
   */
  class ExlaErtsAllocator : public tensorflow::Allocator {
    public:

      ExlaErtsAllocator() = default;

      std::string Name() override { return "erts"; }

      void* AllocateRaw(size_t alignment, size_t num_bytes) override {
        return enif_alloc(num_bytes);
      }

      void DeallocateRaw(void* ptr) override {
        return enif_free(ptr);
      }
  };

} // namespace exla

#endif
