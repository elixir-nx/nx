#pragma once

#include <cstring>
#include <erl_nif.h>

#include "../exla_nif_call.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"

namespace exla_infeed {
namespace ffi = xla::ffi;

inline size_t product(const ffi::Span<const int64_t> dims) {
  size_t p = 1;
  for (auto d : dims)
    p *= static_cast<size_t>(d);
  return p;
}

// Infeed implementation - simple test version
static inline ffi::Error
infeed_cpu_custom_call_impl(ffi::Buffer<ffi::U8> token,
                            ffi::RemainingRets remaining_results) {
  fprintf(stderr,
          "DEBUG: infeed_cpu_custom_call_impl called with %zu results\n",
          remaining_results.size());
  fflush(stderr);

  // For now, just fill with test data to verify the custom call is working
  for (size_t i = 0; i < remaining_results.size(); ++i) {
    auto result_or_error = remaining_results.get<ffi::AnyBuffer>(i);
    if (!result_or_error.has_value()) {
      return ffi::Error::InvalidArgument("invalid_result_buffer");
    }

    auto result_buffer = result_or_error.value();
    void *result_data = result_buffer->untyped_data();
    size_t result_bytes = result_buffer->size_bytes();

    // Fill with a test pattern instead of zeros
    memset(result_data, 0x42,
           result_bytes); // Fill with 0x42 (which is 42 in decimal)
  }

  fprintf(stderr, "DEBUG: infeed_cpu_custom_call_impl completed\n");
  fflush(stderr);
  return ffi::Error::Success();
}

} // namespace exla_infeed