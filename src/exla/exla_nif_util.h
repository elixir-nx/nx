#ifndef EXLA_NIF_UTIL_H_
#define EXLA_NIF_UTIL_H_

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/service/platform_util.h"

#include <erl_nif.h>
#include <string>
#include <algorithm>

class ExlaNifUtil {
  public:

    static ERL_NIF_TERM error(ErlNifEnv* env, const char* msg);

    /*
     * Getters for standard types.
     */
    static int get(ErlNifEnv* env, ERL_NIF_TERM term, int &var);
    static int get(ErlNifEnv* env, ERL_NIF_TERM term, long int &var);
    static int get(ErlNifEnv* env, ERL_NIF_TERM term, long long int &var);
    static int get(ErlNifEnv* env, ERL_NIF_TERM term, std::string &var);

    /*
     * Getters for non-standard types. Suffix to be explicit.
     */
    static int get_platform(ErlNifEnv* env, ERL_NIF_TERM term, stream_executor::Platform* platform);
    static int get_type(ErlNifEnv* env, ERL_NIF_TERM term, xla::PrimitiveType &type);
    static int get_atom(ErlNifEnv* env, ERL_NIF_TERM term, std::string &var);

    template <typename T>
    static int get_span(ErlNifEnv* env, ERL_NIF_TERM term, absl::Span<T> &var);

    template <typename T>
    static int get(ErlNifEnv* env, ERL_NIF_TERM term, T* var);

    static int get_options(ErlNifEnv* env, ERL_NIF_TERM term, xla::ExecutableRunOptions& options);
    static int get_options(ErlNifEnv* env, ERL_NIF_TERM term, xla::ExecutableBuildOptions& options);
    static int get_options(ErlNifEnv* env, ERL_NIF_TERM term, xla::LocalClientOptions& options);

    /*
     * Makers for standard types.
     */
    static ERL_NIF_TERM make(ErlNifEnv* env, std::string &var);

    template <typename T>
    static ERL_NIF_TERM make(ErlNifEnv* env, T &var);

    template <typename T>
    static ERL_NIF_TERM make(ErlNifEnv* env, std::unique_ptr<T> &var);

    template <typename T>
    static int open_resource(ErlNifEnv* env, const char* mod, const char* name, ErlNifResourceDtor* dtor);

    template <typename T>
    struct resource_object {
      static ErlNifResourceType* type;
    };

};

#endif