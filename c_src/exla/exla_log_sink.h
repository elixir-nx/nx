#ifndef EXLA_LOG_SINK_H_
#define EXLA_LOG_SINK_H_

#include "tensorflow/core/platform/logging.h"
#include "absl/base/log_severity.h"
#include "tensorflow/compiler/xla/exla/erts/erl_nif.h"

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

    ERL_NIF_TERM info(std::string& str) {
      return enif_make_tuple2(env_, enif_make_atom(env_, "info"), enif_make_string(env_, str.c_str(), ERL_NIF_LATIN1));
    }

    ERL_NIF_TERM warning(std::string& str) {
      return enif_make_tuple2(env_, enif_make_atom(env_, "warning"), enif_make_string(env_, str.c_str(), ERL_NIF_LATIN1));
    }

    ERL_NIF_TERM error(std::string& str) {
      return enif_make_tuple2(env_, enif_make_atom(env_, "error"), enif_make_string(env_, str.c_str(), ERL_NIF_LATIN1));
    }

    ERL_NIF_TERM fatal(std::string& str) {
      return enif_make_tuple2(env_, enif_make_atom(env_, "fatal"), enif_make_string(env_, str.c_str(), ERL_NIF_LATIN1));
    }

    void Send(const tensorflow::TFLogEntry& entry) {
      ERL_NIF_TERM msg;
      std::string msg_str = entry.ToString();
      switch (entry.log_severity()) {
        case absl::LogSeverity::kInfo:
          msg = info(msg_str);
          break;
        case absl::LogSeverity::kWarning:
          msg = warning(msg_str);
          break;
        case absl::LogSeverity::kError:
          msg = error(msg_str);
          break;
        case absl::LogSeverity::kFatal:
          msg = fatal(msg_str);
          break;
        default:
          msg = info(msg_str);
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