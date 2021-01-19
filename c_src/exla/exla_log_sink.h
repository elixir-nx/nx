#ifndef EXLA_LOG_SINK_H_
#define EXLA_LOG_SINK_H_

#include <string>

#include "tensorflow/compiler/xla/exla/exla_nif_util.h"
#include "tensorflow/core/platform/logging.h"
#include "absl/base/log_severity.h"

namespace exla {

/*
 * Redirect TF Logs to Elixir Logger.
 */
class ExlaLogSink : public tensorflow::TFLogSink {
 public:
  explicit ExlaLogSink(ErlNifPid sink_pid) : sink_pid_(sink_pid) {
    // Logger Env
    env_ = enif_alloc_env();
  }

  ~ExlaLogSink() { enif_free_env(env_); }

  ERL_NIF_TERM info(std::string str, std::string fname, int32 line) {
    ERL_NIF_TERM status = nif::atom(env_, "info");
    ERL_NIF_TERM msg = nif::make(env_, str);
    ERL_NIF_TERM file = nif::make(env_, fname);
    ERL_NIF_TERM line_term = nif::make(env_, line);
    return enif_make_tuple4(env_, status, msg, file, line_term);
  }

  ERL_NIF_TERM warning(std::string str, std::string fname, int32 line) {
    ERL_NIF_TERM status = nif::atom(env_, "warning");
    ERL_NIF_TERM msg = nif::make(env_, str);
    ERL_NIF_TERM file = nif::make(env_, fname);
    ERL_NIF_TERM line_term = nif::make(env_, line);
    return enif_make_tuple4(env_, status, msg, file, line_term);
  }

  ERL_NIF_TERM error(std::string str, std::string fname, int32 line) {
    ERL_NIF_TERM status = nif::atom(env_, "error");
    ERL_NIF_TERM msg = nif::make(env_, str);
    ERL_NIF_TERM file = nif::make(env_, fname);
    ERL_NIF_TERM line_term = nif::make(env_, line);
    return enif_make_tuple4(env_, status, msg, file, line_term);
  }

  ERL_NIF_TERM fatal(std::string str, std::string fname, int32 line) {
    ERL_NIF_TERM status = nif::atom(env_, "fatal");
    ERL_NIF_TERM msg = nif::make(env_, str);
    ERL_NIF_TERM file = nif::make(env_, fname);
    ERL_NIF_TERM line_term = nif::make(env_, line);
    return enif_make_tuple4(env_, status, msg, file, line_term);
  }

  void Send(const tensorflow::TFLogEntry& entry) {
    ERL_NIF_TERM msg;
    std::string msg_str = entry.ToString();
    std::string fname = entry.FName();
    int32 line = entry.Line();
    switch (entry.log_severity()) {
      case absl::LogSeverity::kInfo:
        msg = info(msg_str, fname, line);
        break;
      case absl::LogSeverity::kWarning:
        msg = warning(msg_str, fname, line);
        break;
      case absl::LogSeverity::kError:
        msg = error(msg_str, fname, line);
        break;
      case absl::LogSeverity::kFatal:
        msg = fatal(msg_str, fname, line);
        break;
      default:
        msg = info(msg_str, fname, line);
        break;
    }
    enif_send(env_, &sink_pid_, NULL, msg);
  }

 private:
  ErlNifPid sink_pid_;
  ErlNifEnv* env_;
};

}  // namespace exla

#endif
