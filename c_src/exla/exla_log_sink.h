#ifndef EXLA_LOG_SINK_H_
#define EXLA_LOG_SINK_H_

#include "tensorflow/core/platform/logging.h"
#include "absl/base/log_severity.h"
#include "tensorflow/compiler/xla/exla/exla_nif_util.h"

namespace exla {

  /*
   * Redirect TF Logs to Elixir Logger.
   */
  class ExlaLogSink : public tensorflow::TFLogSink {
  public:
    ExlaLogSink(ErlNifPid sink_pid) : sink_pid_(sink_pid) {
      // Logger Env
      env_ = enif_alloc_env();
    }

    ERL_NIF_TERM info(std::string& str, std::string& fname, int32 line) {
      return enif_make_tuple4(env_, atom(env_, "info"), make(env_, str), make(env_, fname), make(env_, line));
    }

    ERL_NIF_TERM warning(std::string& str, std::string& fname, int32 line) {
      return enif_make_tuple4(env_, atom(env_, "warning"), make(env_, str), make(env_, fname), make(env_, line));
    }

    ERL_NIF_TERM error(std::string& str, std::string& fname, int32 line) {
      return enif_make_tuple4(env_, atom(env_, "error"), make(env_, str), make(env_, fname), make(env_, line));
    }

    ERL_NIF_TERM fatal(std::string& str, std::string& fname, int32 line) {
      return enif_make_tuple4(env_, atom(env_, "fatal"), make(env_, str), make(env_, fname), make(env_, line));
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
} // namespace exla

#endif