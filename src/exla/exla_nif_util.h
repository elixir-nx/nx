#ifndef EXLA_NIF_UTIL_H_
#define EXLA_NIF_UTIL_H_

#include "absl/types/span.h"

#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/service/platform_util.h"

#include <erl_nif.h>
#include <string>
#include <algorithm>

namespace exla {
  /*
   * Helper for returning `{:error, msg}` from NIF.
   */
  ERL_NIF_TERM error(ErlNifEnv* env, const char* msg);

  /*
   * Getters for standard types.
   */
  int get(ErlNifEnv* env, ERL_NIF_TERM term, int &var);
  int get(ErlNifEnv* env, ERL_NIF_TERM term, long int &var);
  int get(ErlNifEnv* env, ERL_NIF_TERM term, long long int &var);
  int get(ErlNifEnv* env, ERL_NIF_TERM term, std::string &var);

  /*
   * Getters for non-standard types. Suffix to be explicit.
   */
  int get_platform(ErlNifEnv* env, ERL_NIF_TERM term, stream_executor::Platform* platform);
  int get_type(ErlNifEnv* env, ERL_NIF_TERM term, xla::PrimitiveType &type);
  int get_atom(ErlNifEnv* env, ERL_NIF_TERM term, std::string &var);

  /*
   * Getters for option structs.
   */
  int get_options(ErlNifEnv* env, ERL_NIF_TERM term, xla::ExecutableRunOptions& options);
  int get_options(ErlNifEnv* env, ERL_NIF_TERM term, xla::ExecutableBuildOptions& options);
  int get_options(ErlNifEnv* env, ERL_NIF_TERM term, xla::LocalClientOptions& options);

  /*
   * Makers for standard types.
   */
  ERL_NIF_TERM make(ErlNifEnv* env, std::string &var);

} // namespace exla

#endif