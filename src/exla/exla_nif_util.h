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
   * Helper for returning `{:ok, term}` from NIF.
   */
  ERL_NIF_TERM ok(ErlNifEnv* env, ERL_NIF_TERM term);

  /*
   * Helper for returning `:ok` from NIF.
   */
  ERL_NIF_TERM ok(ErlNifEnv* env);

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
  int get_options(ErlNifEnv* env, const ERL_NIF_TERM terms[], xla::ExecutableRunOptions& options);
  int get_options(ErlNifEnv* env, const ERL_NIF_TERM terms[], xla::ExecutableBuildOptions& options);
  int get_options(ErlNifEnv* env, const ERL_NIF_TERM terms[] , xla::LocalClientOptions& options);

  /*
   * Makers for standard types.
   */
  ERL_NIF_TERM make(ErlNifEnv* env, std::string &var);

  ERL_NIF_TERM make(ErlNifEnv* env, const char* string);

  /*
   * Templates for resources.
   */
  template <typename T>
  struct resource_object {
    static ErlNifResourceType *type;
  };
  template<typename T> ErlNifResourceType* resource_object<T>::type=0;

  template <typename T>
  int open_resource(ErlNifEnv* env, const char* mod, const char* name, ErlNifResourceDtor* dtor){
    ErlNifResourceType *type;
    ErlNifResourceFlags flags = ErlNifResourceFlags(ERL_NIF_RT_CREATE|ERL_NIF_RT_TAKEOVER);
    type = enif_open_resource_type(env, mod, name, dtor, flags, NULL);
    if(type == NULL){
      resource_object<T>::type = 0;
      return -1;
    } else {
      resource_object<T>::type = type;
    }
    return 1;
  }

  template <typename T>
  ERL_NIF_TERM get(ErlNifEnv* env, ERL_NIF_TERM term, T* &var){
    return enif_get_resource(env, term, resource_object<T>::type, (void **) &var);
  }

  template <typename T>
  ERL_NIF_TERM make(ErlNifEnv* env, T &var){
    void* ptr = enif_alloc_resource(resource_object<T>::type, sizeof(T));
    new(ptr) T(std::move(var));
    ERL_NIF_TERM ret = enif_make_resource(env, ptr);
    enif_release_resource(ptr);
    return ret;
  }

  template <typename T>
  ERL_NIF_TERM make(ErlNifEnv* env, std::unique_ptr<T> &var){
    void* ptr = enif_alloc_resource(resource_object<T>::type, sizeof(T));
    T* value = var.release();
    new(ptr) T(std::move(*value));
    ERL_NIF_TERM ret = enif_make_resource(env, ptr);
    enif_release_resource(ptr);
    return ret;
  }

  template <typename T>
  int get_span(ErlNifEnv* env, ERL_NIF_TERM tuple, absl::Span<T> &span){
    const ERL_NIF_TERM* elems;
    int num_elems;
    if(!enif_get_tuple(env, tuple, &num_elems, &elems)) return 0;
    T data[num_elems];
    for(int i=0;i<num_elems;i++){
      T elem;
      if(!get(env, elems[i], elem)) return 0;
      data[i] = elem;
    }
    span = absl::Span<T>(data, num_elems);
    return 1;
  }

} // namespace exla

#endif
