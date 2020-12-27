#ifndef EXLA_NIF_UTIL_H_
#define EXLA_NIF_UTIL_H_

#include <complex>
#include <vector>
#include <string>
#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/exla/erts/erl_nif.h"

#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/errors.h"

namespace exla {

/*
 * We standardize numeric types to guarantee everything is platform-independent and
 * compatible with what the TF/XLA API wants.
 */
using int8 = tensorflow::int8;
using int16 = tensorflow::int16;
using int32 = tensorflow::int32;
using int64 = tensorflow::int64;
using uint8 = tensorflow::uint8;
using uint16 = tensorflow::uint16;
using uint32 = tensorflow::uint32;
using uint64 = tensorflow::uint64;
using bfloat16 = tensorflow::bfloat16;
using float32 = float;
using float64 = double;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

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
 * Helper for returning an atom from NIF.
 */
ERL_NIF_TERM atom(ErlNifEnv* env, const char* status);

/*
 * Getters for numeric types.
 */
int get(ErlNifEnv* env, ERL_NIF_TERM term, int8* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, int16* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, int32* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, int64* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, uint8* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, uint16* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, uint32* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, uint64* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, bfloat16* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, float32* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, float64* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, complex64* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, complex128* var);

/*
 * Getters for standard types.
 */
int get(ErlNifEnv* env, ERL_NIF_TERM term, std::string &var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, bool* var);

/*
 * Getters for non-standard types. Suffix to be explicit.
 */
int get_binary(ErlNifEnv* env, ERL_NIF_TERM term, ErlNifBinary* var);
int get_type(ErlNifEnv* env, ERL_NIF_TERM term, xla::PrimitiveType &type);
int get_atom(ErlNifEnv* env, ERL_NIF_TERM term, std::string* var);

/*
 * Getter for native type from term. Needed to avoid templates in NIF.
 */
template <
  xla::PrimitiveType type,
  typename T = typename xla::primitive_util::PrimitiveTypeToNative<type>::type>
T get_value(ErlNifEnv* env, ERL_NIF_TERM &term) {
  T value;
  exla::get(env, term, &value);
  return value;
}

/*
 * Templates for resources.
 */
template <typename T>
struct resource_object {
  static ErlNifResourceType *type;
};
template<typename T> ErlNifResourceType* resource_object<T>::type = 0;

/*
 * Opens a resource and stores it in a `resource_object`.
 */
template <typename T>
int open_resource(ErlNifEnv* env,
                  const char* mod,
                  const char* name,
                  ErlNifResourceDtor* dtor) {
  ErlNifResourceType *type;
  ErlNifResourceFlags flags = ErlNifResourceFlags(ERL_NIF_RT_CREATE|ERL_NIF_RT_TAKEOVER);
  type = enif_open_resource_type(env, mod, name, dtor, flags, NULL);
  if (type == NULL) {
    resource_object<T>::type = 0;
    return -1;
  } else {
    resource_object<T>::type = type;
  }
  return 1;
}

/*
 * Getter for resource object.
 */
template <typename T>
ERL_NIF_TERM get(ErlNifEnv* env, ERL_NIF_TERM term, T* &var) {
  return enif_get_resource(env, term,
                           resource_object<T>::type,
                           reinterpret_cast<void**>(&var));
}

/*
 * Getters for containers.
 */
int get_tuple(ErlNifEnv* env,
              ERL_NIF_TERM tuple,
              std::vector<int64> &var);
int get_list(ErlNifEnv* env,
             ERL_NIF_TERM list,
             std::vector<int64> &var);
int get_list(ErlNifEnv* env,
             ERL_NIF_TERM list,
             std::vector<ErlNifBinary> &var);

template <typename T>
int get_tuple(ErlNifEnv* env, ERL_NIF_TERM tuple, std::vector<T> &var) {
  const ERL_NIF_TERM* terms;
  int length;
  if (!enif_get_tuple(env, tuple, &length, &terms)) return 0;
  for (int i=0; i < length; i++) {
    T* elem;
    if (!get<T>(env, terms[i], elem)) return 0;
    var.push_back(*elem);
  }
  return 1;
}

template <typename T>
int get_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<T*> &var) {
  ERL_NIF_TERM head, tail;
  int i = 0;
  while (enif_get_list_cell(env, list, &head, &tail)) {
    T* elem;
    if (!get<T>(env, head, elem)) return 0;
    var.insert(var.begin() + (i++), elem);
    list = tail;
  }
  return 1;
}

template <typename T>
int get_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<T> &var) {
  ERL_NIF_TERM head, tail;
  int i = 0;
  while (enif_get_list_cell(env, list, &head, &tail)) {
    T* elem;
    if (!get<T>(env, head, elem)) return 0;
    var.insert(var.begin() + (i++), *elem);
    list = tail;
  }
  return 1;
}

/*
 * Getters for XLA Protobuf Types
 */
int get_padding_config(ErlNifEnv* env,
                       ERL_NIF_TERM list,
                       xla::PaddingConfig& padding_config);

/*
 * Makers for standard types.
 */
ERL_NIF_TERM make(ErlNifEnv* env, int var);
ERL_NIF_TERM make(ErlNifEnv* env, std::string var);
ERL_NIF_TERM make(ErlNifEnv* env, ErlNifBinary var);
ERL_NIF_TERM make(ErlNifEnv* env, const char* string);

/*
 * Maker for resource from `std::unique_ptr`.
 */
template <typename T>
ERL_NIF_TERM make(ErlNifEnv* env, std::unique_ptr<T> var) {
  void* ptr = enif_alloc_resource(resource_object<T>::type, sizeof(T));
  T* value = var.release();
  new(ptr) T(std::move(*value));
  ERL_NIF_TERM ret = enif_make_resource(env, ptr);
  enif_release_resource(ptr);
  return ret;
}

/*
 * Maker for resource objects.
 */
template <typename T>
ERL_NIF_TERM make(ErlNifEnv* env, T &var) {
  void* ptr = enif_alloc_resource(resource_object<T>::type, sizeof(T));
  new(ptr) T(std::move(var));
  ERL_NIF_TERM ret = enif_make_resource(env, ptr);
  enif_release_resource(ptr);
  return ret;
}

/*
 * Helper for extracting information from `GetShape` and sending it back as a Tuple.
 */
ERL_NIF_TERM make_shape_info(ErlNifEnv* env, xla::Shape shape);

}  // namespace exla

/*
 * Helper Macros
 * See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/stream_executor/lib/statusor.h
 */
#define EXLA_STATUS_MACROS_CONCAT_NAME(x, y)  \
  EXLA_STATUS_MACROS_CONCAT_NAME_IMPL(x, y)

#define EXLA_STATUS_MACROS_CONCAT_NAME_IMPL(x, y) x##y

#define EXLA_ASSIGN_OR_RETURN_NIF(lhs, rexpr, env)                      \
  EXLA_ASSIGN_OR_RETURN_NIF_IMPL(                                       \
    EXLA_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__),      \
                                    lhs, rexpr, env)

#define EXLA_ASSIGN_OR_RETURN_NIF_IMPL(statusor, lhs, rexpr, env)       \
  auto statusor = (rexpr);                                              \
  if (!statusor.ok()) {                                                 \
    return exla::error(env, statusor.status().error_message().c_str()); \
  }                                                                     \
  lhs = std::move(statusor.ValueOrDie());

#define EXLA_ASSIGN_OR_RETURN(lhs, rexpr)                               \
  EXLA_ASSIGN_OR_RETURN_IMPL(                                           \
    EXLA_STATUS_MACROS_CONCAT_NAME(                                     \
      _status_or_value, __COUNTER__),                                   \
  lhs, rexpr)

#define EXLA_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr)                \
  auto statusor = (rexpr);                                              \
  if (!statusor.ok()) {                                                 \
    return statusor.status();                                           \
  }                                                                     \
  lhs = std::move(statusor.ValueOrDie());

#endif
