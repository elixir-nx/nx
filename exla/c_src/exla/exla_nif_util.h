#ifndef EXLA_NIF_UTIL_H_
#define EXLA_NIF_UTIL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "erl_nif.h"
#include "xla/shape.h"
#include "exla_types.h"

#if !defined(__GNUC__) && (defined(__WIN32__) || defined(_WIN32) || defined(_WIN32_))
typedef unsigned __int64 nif_uint64_t;
typedef signed __int64 nif_int64_t;
#else
typedef unsigned long nif_uint64_t;
typedef signed long nif_int64_t;
#endif

// Implementation Notes:
//
// In most of these implementations you'll find we prefer output parameters
// over returning values. This follows the convention of the Erlang NIF
// API in which functions for retrieving terms from the VM return an
// integer status and populate an output parameter.
//
// We also follow the naming convention set forth in the the Erlang NIF
// API. Numeric, standard, and resource types use the polymorphic/template
// `get` or `make`.
//
// We mostly use vectors for containers (lists and tuples), and maps for
// returning maps back to the VM. These have suffixes to avoid conflicting
// signatures for retrieving/returning different signatures.
//
// We create separate methods for each XLA protobuf type, so we can guarantee
// the format we receive the protobuf in is correct.

namespace exla {

namespace nif {

// Status helpers

// Helper for returning `{:error, msg}` from NIF.
ERL_NIF_TERM error(ErlNifEnv* env, const char* msg);

// Helper for returning `{:ok, term}` from NIF.
ERL_NIF_TERM ok(ErlNifEnv* env, ERL_NIF_TERM term);

// Helper for returning `:ok` from NIF.
ERL_NIF_TERM ok(ErlNifEnv* env);

// Numeric types
//
// Floating/Complex types will never get used, except
// when defining scalar-constants with `constant`.

int get(ErlNifEnv* env, ERL_NIF_TERM term, int8* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, int16* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, int32* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, int64* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, uint8* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, uint16* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, uint32* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, uint64* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, bfloat16* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, float16* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, float32* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, float64* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, complex64* var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, complex128* var);

ERL_NIF_TERM make(ErlNifEnv* env, int32 var);

// Standard types
//
// We only define implementations for types we use in the
// NIF.

int get(ErlNifEnv* env, ERL_NIF_TERM term, std::string& var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, bool* var);

ERL_NIF_TERM make(ErlNifEnv* env, std::string var);
ERL_NIF_TERM make(ErlNifEnv* env, ErlNifBinary var);
ERL_NIF_TERM make(ErlNifEnv* env, const char* string);

// Atoms
//
// We have to be explicit in naming these functions because
// their signatures are the same for retrieving/returning
// regular strings.

int get_atom(ErlNifEnv* env, ERL_NIF_TERM term, std::string& var);

ERL_NIF_TERM atom(ErlNifEnv* env, const char* status);

// Template struct for resources. The struct lets us use templates
// to store and retrieve open resources later on. This implementation
// is the same as the approach taken in the goertzenator/nifpp
// C++11 wrapper around the Erlang NIF API.
template <typename T>
struct resource_object {
  static ErlNifResourceType* type;
};
template <typename T>
ErlNifResourceType* resource_object<T>::type = 0;

// Default destructor passed when opening a resource. The default
// behavior is to invoke the underlying objects destructor and
// set the resource pointer to NULL.
template <typename T>
void default_dtor(ErlNifEnv* env, void* obj) {
  T* resource = reinterpret_cast<T*>(obj);
  resource->~T();
  resource = nullptr;
}

// Opens a resource for the given template type T. If no
// destructor is given, uses the default destructor defined
// above.
template <typename T>
int open_resource(ErlNifEnv* env,
                  const char* mod,
                  const char* name,
                  ErlNifResourceDtor* dtor = nullptr) {
  if (dtor == nullptr) {
    dtor = &default_dtor<T>;
  }
  ErlNifResourceType* type;
  ErlNifResourceFlags flags = ErlNifResourceFlags(ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER);
  type = enif_open_resource_type(env, mod, name, dtor, flags, NULL);
  if (type == NULL) {
    resource_object<T>::type = 0;
    return -1;
  } else {
    resource_object<T>::type = type;
  }
  return 1;
}

// Returns a resource of the given template type T.
template <typename T>
ERL_NIF_TERM get(ErlNifEnv* env, ERL_NIF_TERM term, T*& var) {
  return enif_get_resource(env, term,
                           resource_object<T>::type,
                           reinterpret_cast<void**>(&var));
}

// Creates a reference to the given resource of type T. We
// use the move constructor by default because some XLA
// objects delete the copy-constructor. The move is intended
// to represent a transfer of ownership of the object to
// the VM.

template <typename T>
ERL_NIF_TERM make(ErlNifEnv* env, T& var) {
  void* ptr = enif_alloc_resource(resource_object<T>::type, sizeof(T));
  new (ptr) T(std::move(var));
  ERL_NIF_TERM ret = enif_make_resource(env, ptr);
  enif_release_resource(ptr);
  return ret;
}

template <typename T>
ERL_NIF_TERM make_list(ErlNifEnv* env, std::vector<T> result) {
  size_t n = result.size();

  std::vector<ERL_NIF_TERM> nif_terms;
  nif_terms.reserve(n);

  for (size_t i = 0; i < n; i++) {
    nif_terms[i] = exla::nif::make<T>(env, result[i]);
  }

  auto data = nif_terms.data();
  auto list = enif_make_list_from_array(env, &data[0], n);
  return list;
}
// Containers
//
// Both tuples and lists are treated as vectors, but extracting
// terms from both is slightly different, so we have to be
// explicit in the naming convention in order to differentiate.
//
// We also support reading resources into vectors from both tuples
// and lists. Once again, implementation is slightly different
// for resources, so we need to be explicit.
//
// Similar to standard types, we only define implementations for
// types used.

int get_tuple(ErlNifEnv* env,
              ERL_NIF_TERM tuple,
              std::vector<int64>& var);

template <typename T>
int get_tuple(ErlNifEnv* env, ERL_NIF_TERM tuple, std::vector<T>& var) {
  const ERL_NIF_TERM* terms;
  int length;
  if (!enif_get_tuple(env, tuple, &length, &terms)) return 0;
  var.reserve(length);

  for (int i = 0; i < length; i++) {
    T* elem;
    if (!get<T>(env, terms[i], elem)) return 0;
    var.push_back(*elem);
  }
  return 1;
}

int get_list(ErlNifEnv* env,
             ERL_NIF_TERM list,
             std::vector<int64>& var);
int get_list(ErlNifEnv* env,
             ERL_NIF_TERM list,
             std::vector<ErlNifBinary>& var);

int get_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<std::string>& var);

int get_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<xla::Shape>& var);

template <typename T>
int get_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<T*>& var) {
  unsigned int length;
  if (!enif_get_list_length(env, list, &length)) return 0;
  var.reserve(length);
  ERL_NIF_TERM head, tail;

  while (enif_get_list_cell(env, list, &head, &tail)) {
    T* elem;
    if (!get<T>(env, head, elem)) return 0;
    var.push_back(elem);
    list = tail;
  }
  return 1;
}

template <typename T>
int get_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<T>& var) {
  unsigned int length;
  if (!enif_get_list_length(env, list, &length)) return 0;
  var.reserve(length);
  ERL_NIF_TERM head, tail;

  while (enif_get_list_cell(env, list, &head, &tail)) {
    T* elem;
    if (!get<T>(env, head, elem)) return 0;
    var.push_back(*elem);
    list = tail;
  }
  return 1;
}

template <typename T>
int get_keyword_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<std::pair<std::string, T>>& var) {
  unsigned int length;
  if (!enif_get_list_length(env, list, &length)) return 0;
  var.reserve(length);
  ERL_NIF_TERM head, tail;

  while (enif_get_list_cell(env, list, &head, &tail)) {
    const ERL_NIF_TERM* terms;
    int count;

    if (!enif_get_tuple(env, head, &count, &terms)) return 0;
    if (count != 2) return 0;

    std::string lo;
    T hi;
    if (!get_atom(env, terms[0], lo)) return 0;
    if (!get(env, terms[1], hi)) return 0;

    var.push_back(std::pair<std::string, T>(lo, hi));

    list = tail;
  }
  return 1;
}

int get_binary(ErlNifEnv* env, ERL_NIF_TERM term, ErlNifBinary* var);

ERL_NIF_TERM make_map(ErlNifEnv* env, std::map<std::string, int>& map);

// XLA Protobuf Types
//
// See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/xla_data.proto
// for more details on each type and additional types not listed here.

// Gets encoded EXLA.Typespec as xla::Shape.
int get_typespec_as_xla_shape(ErlNifEnv* env, ERL_NIF_TERM term, xla::Shape* shape);

}  // namespace nif
}  // namespace exla

// Helper Macros
//
// See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/stream_executor/lib/statusor.h

#define EXLA_STATUS_MACROS_CONCAT_NAME(x, y) \
  EXLA_STATUS_MACROS_CONCAT_NAME_IMPL(x, y)

#define EXLA_STATUS_MACROS_CONCAT_NAME_IMPL(x, y) x##y

// Macro to be used to consume StatusOr from within a NIF. Will
// bind lhs to value if the status is OK, otherwise will return
// `{:error, msg}`.
#define EXLA_ASSIGN_OR_RETURN_NIF(lhs, rexpr, env)                   \
  EXLA_ASSIGN_OR_RETURN_NIF_IMPL(                                    \
      EXLA_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), \
      lhs, rexpr, env)

#define EXLA_ASSIGN_OR_RETURN_NIF_IMPL(statusor, lhs, rexpr, env)     \
  auto statusor = (rexpr);                                            \
  if (!statusor.ok()) {                                               \
    return exla::nif::error(env, statusor.status().message().data()); \
  }                                                                   \
  lhs = std::move(statusor.value());

// Macro to be used to consume StatusOr. Will bind lhs
// to value if the status is OK, otherwise will return
// the status.
#define EXLA_ASSIGN_OR_RETURN(lhs, rexpr) \
  EXLA_ASSIGN_OR_RETURN_IMPL(             \
      EXLA_STATUS_MACROS_CONCAT_NAME(     \
          _status_or_value, __COUNTER__), \
      lhs, rexpr)

#define EXLA_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                               \
  if (!statusor.ok()) {                                  \
    return statusor.status();                            \
  }                                                      \
  lhs = std::move(statusor.value());

#endif
