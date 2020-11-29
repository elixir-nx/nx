// TODO: Combine this file with `exla_nif_util.h`
/*
 * See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/stream_executor/lib/statusor.h
 */
#ifndef EXLA_MACROS_H_
#define EXLA_MACROS_H_

#include "tensorflow/compiler/xla/exla/exla_nif_util.h"

#define EXLA_STATUS_MACROS_CONCAT_NAME(x, y) EXLA_STATUS_MACROS_CONCAT_NAME_IMPL(x, y)
#define EXLA_STATUS_MACROS_CONCAT_NAME_IMPL(x, y) x##y

#define EXLA_ASSIGN_OR_RETURN_NIF(lhs, rexpr, env) \
  EXLA_ASSIGN_OR_RETURN_NIF_IMPL(                  \
    EXLA_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, rexpr, env)

#define EXLA_ASSIGN_OR_RETURN_NIF_IMPL(statusor, lhs, rexpr, env)       \
  auto statusor = (rexpr);                                              \
  if(!statusor.ok()){                                                   \
    return exla::error(env, statusor.status().error_message().c_str()); \
  }                                                                     \
  lhs = std::move(statusor.ValueOrDie());

#define EXLA_ASSIGN_OR_RETURN(lhs, rexpr) \
  EXLA_ASSIGN_OR_RETURN_IMPL(                  \
    EXLA_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, rexpr)

#define EXLA_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr)                \
  auto statusor = (rexpr);                                              \
  if(!statusor.ok()){                                                   \
    return statusor.status();                                           \
  }                                                                     \
  lhs = std::move(statusor.ValueOrDie());

#endif
