#ifndef EXLA_LOG_SINK_H_
#define EXLA_LOG_SINK_H_

#include <string>

#include "exla_nif_util.h"
#include <fine.hpp>
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/base/log_severity.h"

namespace exla {

// Redirects calls to logging to the Elixir Logger. `sink_pid`
// is the PID for a GenServer in Elixir which receives messages
// with logging information on every call to `LOG(severity)`.
class ExlaLogSink : public absl::LogSink {
public:
  explicit ExlaLogSink(ErlNifPid sink_pid) : sink_pid_(sink_pid) {}

  void Send(const absl::LogEntry &entry) {
    auto string = std::string(entry.text_message());
    auto filename = std::string(entry.source_filename());
    int64_t line = entry.source_line();
    auto severity = entry.log_severity();

    if (severity == absl::LogSeverity::kFatal) {
      // LOG(FATAL) aborts the program before we are able to send and
      // log the information from Elixir, so we need to get it out
      // there for debugging before everything crashes
      std::cerr << "[FATAL] " << filename << ":" << line << " " << string << "\n";
    }

    auto env = enif_alloc_env();

    auto message = fine::encode(
        env, std::make_tuple(severity_to_atom(severity), string, filename, line));

    enif_send(NULL, &sink_pid_, env, message);

    enif_free_env(env);
  }

private:
  fine::Atom severity_to_atom(absl::LogSeverity severity) {
    switch (severity) {
    case absl::LogSeverity::kInfo:
      return atoms::info;
    case absl::LogSeverity::kWarning:
      return atoms::warning;
    case absl::LogSeverity::kError:
      return atoms::error;
    case absl::LogSeverity::kFatal:
      return atoms::error;
    default:
      return atoms::info;
    }
  }

  ErlNifPid sink_pid_;
};

} // namespace exla

#endif
