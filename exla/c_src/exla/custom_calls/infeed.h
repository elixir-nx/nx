#pragma once

#include <cstring>
#include <erl_nif.h>

#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape_util.h"

namespace exla_infeed {
namespace ffi = xla::ffi;

inline size_t product(const ffi::Span<const int64_t> dims) {
  size_t p = 1;
  for (auto d : dims)
    p *= static_cast<size_t>(d);
  return p;
}

// Infeed implementation using XLA's built-in infeed queues
static inline ffi::Error
infeed_cpu_custom_call_impl(ffi::Buffer<ffi::U8> token,
                            ffi::RemainingRets remaining_results) {
  // Simple implementation that just zero-fills the buffers for now
  // This will be a placeholder until we can properly access XLA's infeed queue
  for (size_t i = 0; i < remaining_results.size(); ++i) {
    auto result_or_error = remaining_results.get<ffi::AnyBuffer>(i);
    if (!result_or_error.has_value()) {
      return ffi::Error::InvalidArgument("invalid_result_buffer");
    }

    auto result_buffer = result_or_error.value();
    void *result_data = result_buffer->untyped_data();
    size_t result_bytes = result_buffer->size_bytes();

    // Zero-fill the result buffer for now
    std::memset(result_data, 0, result_bytes);
  }

  return ffi::Error::Success();
}

} // namespace exla_infeed
