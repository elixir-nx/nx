#ifndef EXLA_LOG_SINK_H_
#define EXLA_LOG_SINK_H_

// Note: TFLogSink API was removed in XLA 0.9.1
// Log sink functionality is disabled in this version
// Logging now uses absl::log directly

#include <string>

#include "absl/base/log_severity.h"
#include "exla_nif_util.h"
#include <fine.hpp>

namespace exla {

// Placeholder class - TFLogSink API no longer available in XLA 0.9.1+
// This class is kept for compatibility but does nothing
class ExlaLogSink {
public:
  explicit ExlaLogSink(ErlNifPid sink_pid) : sink_pid_(sink_pid) {}

private:
  ErlNifPid sink_pid_;
};

} // namespace exla

#endif
