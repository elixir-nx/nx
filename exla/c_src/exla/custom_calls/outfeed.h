#pragma once

#include <cstring>
#include <erl_nif.h>

#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace exla_outfeed {
namespace ffi = xla::ffi;

inline size_t product(const ffi::Span<const int64_t> dims) {
  size_t p = 1;
  for (auto d : dims)
    p *= static_cast<size_t>(d);
  return p;
}

// Outfeed implementation - simple no-op for now
static inline ffi::Error
outfeed_cpu_custom_call_impl(ffi::RemainingArgs remaining_args) {
  // Simple no-op implementation for now
  // The actual outfeed will be handled by the existing XLA outfeed queue
  // mechanism
  return ffi::Error::Success();
}

} // namespace exla_outfeed
