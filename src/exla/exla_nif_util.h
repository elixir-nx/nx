#ifndef EXLA_NIF_UTIL_H_
#define EXLA_NIF_UTIL_H_

#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/literal_util.h"

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/client_library.h"

#include <erl_nif.h>
#include <string>

class ExlaNifUtil {
  public:

    static int get(ErlNifEnv* env, ERL_NIF_TERM term, int &var);
    static int get(ErlNifEnv* env, ERL_NIF_TERM term, long int &var);
    static int get(ErlNifEnv* env, ERL_NIF_TERM term, long long int &var);

    static int get(ErlNifEnv* env, ERL_NIF_TERM term, std::string &var);
    static int get_atom(ErlNifEnv* env, ERL_NIF_TERM term, std::string &var);

    static int get_platform(ErlNifEnv* env, ERL_NIF_TERM term, stream_executor::Platform* platform);
    static int get_primitive_type(ErlNifEnv* env, ERL_NIF_TERM term, xla::PrimitiveType &type);
};

#endif