#ifndef EXLA_NIF_UTIL_H_
#define EXLA_NIF_UTIL_H_

#include <complex>

#include "tensorflow/compiler/xla/exla/erts/erl_nif.h"

#include "absl/types/span.h"

#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xla/types.h"

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
   * Helper for returning a status from NIF.
   */
  ERL_NIF_TERM status(ErlNifEnv* env, const char* status);

  /*
   * Getters for standard types.
   */
  int get(ErlNifEnv* env, ERL_NIF_TERM term, ErlNifBinary &var);
  int get(ErlNifEnv* env, ERL_NIF_TERM term, tensorflow::int8 &var);
  int get(ErlNifEnv* env, ERL_NIF_TERM term, short &var);
  int get(ErlNifEnv* env, ERL_NIF_TERM term, int &var);
  int get(ErlNifEnv* env, ERL_NIF_TERM term, long int &var);
  int get(ErlNifEnv* env, ERL_NIF_TERM term, long long int &var);
  int get(ErlNifEnv* env, ERL_NIF_TERM term, tensorflow::uint8 &var);
  int get(ErlNifEnv* env, ERL_NIF_TERM term, unsigned short &var);
  int get(ErlNifEnv* env, ERL_NIF_TERM term, unsigned int &var);
  int get(ErlNifEnv* env, ERL_NIF_TERM term, unsigned long int &var);
  int get(ErlNifEnv* env, ERL_NIF_TERM term, unsigned long long int &var);
  int get(ErlNifEnv* env, ERL_NIF_TERM term, float &var);
  int get(ErlNifEnv* env, ERL_NIF_TERM term, double &var);
  int get(ErlNifEnv* env, ERL_NIF_TERM term, bool &var);
  int get(ErlNifEnv* env, ERL_NIF_TERM term, std::string &var);
  /*
   * Getters for non-standard types. Suffix to be explicit.
   */
  int get_type(ErlNifEnv* env, ERL_NIF_TERM term, xla::PrimitiveType &type);
  int get_atom(ErlNifEnv* env, ERL_NIF_TERM term, std::string &var);

  /*
   * Makers for standard types.
   */
  ERL_NIF_TERM make(ErlNifEnv* env, int &var);
  ERL_NIF_TERM make(ErlNifEnv* env, std::string &var);
  ERL_NIF_TERM make(ErlNifEnv* env, ErlNifBinary &var);
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
  ERL_NIF_TERM make_buffer(ErlNifEnv* env, std::unique_ptr<xla::ScopedShapedBuffer> buffer){
    void* ptr = (void*) enif_alloc_resource(exla::resource_object<T>::type, sizeof(T));
    new(ptr) T(std::move(buffer));
    ERL_NIF_TERM ret = enif_make_resource(env, ptr);
    enif_release_resource(ptr);
    return ret;
  }

  template <typename T>
  ERL_NIF_TERM make(ErlNifEnv* env, T &var){
    // TODO: Split this into two different functions: one that uses the copy constructor
    // and one that uses the move constructor, and then update which resources use which
    // constructor. This should enable easier handling of memory leaks: http://www.github.com/seanmor5/exla/pull/12
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

  int get_vector_tuple(ErlNifEnv* env, ERL_NIF_TERM tuple, std::vector<long long int> &var);

  int get_vector_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<long long int> &var);
  int get_vector_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<ErlNifBinary> &var);

  template <typename T>
  int get_vector_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<T*> &var){
    ERL_NIF_TERM head, tail;
    int i = 0;
    while(enif_get_list_cell(env, list, &head, &tail)){
      T* elem;
      if(!get<T>(env, head, elem)) return 0;
      var.insert(var.begin() + (i++), elem);
      list = tail;
    }
    return 1;
  }

  template <typename T>
  int get_vector_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<T> &var){
    ERL_NIF_TERM head, tail;
    int i = 0;
    while(enif_get_list_cell(env, list, &head, &tail)){
      T* elem;
      if(!get<T>(env, head, elem)) return 0;
      var.insert(var.begin() + (i++), *elem);
      list = tail;
    }
    return 1;
  }

  template <
      xla::PrimitiveType type,
      typename T = typename xla::primitive_util::PrimitiveTypeToNative<type>::type>
  T get_value(ErlNifEnv* env, ERL_NIF_TERM &term) {
    T value;
    exla::get(env, term, value);
    return value;
  }

  xla::StatusOr<xla::XlaOp> get_constant(ErlNifEnv* env, ERL_NIF_TERM term, xla::XlaBuilder* builder, xla::PrimitiveType type);

} // namespace exla

#endif
