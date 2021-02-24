#ifndef EXLA_NIF_UTIL_H_
#define EXLA_NIF_UTIL_H_

#include <complex>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <map>

#include "tensorflow/compiler/xla/exla/erts/erl_nif.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/shape.h"

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

// We standardize numeric types with tensorflow to ensure we are always
// getting an input with the correct width and to ensure that tensorflow
// is happy with what we're giving it.
//
// Most of these types will only ever be used when creating scalar constants;
// however, some methods require a 64-bit integer. You should prefer standard
// types over these unless you are (1) working with computations, or
// (2) require a non-standard width numeric type (like 64-bit integer).
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
// when defining scalar-constants with `constant_r0`.

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

ERL_NIF_TERM make(ErlNifEnv* env, int32 var);

// Standard types
//
// We only define implementations for types we use in the
// NIF.

int get(ErlNifEnv* env, ERL_NIF_TERM term, std::string &var);
int get(ErlNifEnv* env, ERL_NIF_TERM term, bool* var);

ERL_NIF_TERM make(ErlNifEnv* env, std::string var);
ERL_NIF_TERM make(ErlNifEnv* env, ErlNifBinary var);
ERL_NIF_TERM make(ErlNifEnv* env, const char* string);

// Atoms
//
// We have to be explicit in naming these functions because
// their signatures are the same for retrieving/returning
// regular strings.

int get_atom(ErlNifEnv* env, ERL_NIF_TERM term, std::string* var);

ERL_NIF_TERM atom(ErlNifEnv* env, const char* status);

// Template struct for resources. The struct lets us use templates
// to store and retrieve open resources later on. This implementation
// is the same as the approach taken in the goertzenator/nifpp
// C++11 wrapper around the Erlang NIF API.
template <typename T>
struct resource_object {
  static ErlNifResourceType *type;
};
template<typename T> ErlNifResourceType* resource_object<T>::type = 0;

// Default destructor passed when opening a resource. The default
// behavior is to invoke the underlying objects destructor and
// set the resource pointer to NULL.
template <typename T>
void default_dtor(ErlNifEnv* env, void * obj) {
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

// Returns a resource of the given template type T.
template <typename T>
ERL_NIF_TERM get(ErlNifEnv* env, ERL_NIF_TERM term, T* &var) {
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
ERL_NIF_TERM make(ErlNifEnv* env, T &var) {
  void* ptr = enif_alloc_resource(resource_object<T>::type, sizeof(T));
  new(ptr) T(std::move(var));
  ERL_NIF_TERM ret = enif_make_resource(env, ptr);
  enif_release_resource(ptr);
  return ret;
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
              std::vector<int64> &var);

template <typename T>
int get_tuple(ErlNifEnv* env, ERL_NIF_TERM tuple, std::vector<T> &var) {
  const ERL_NIF_TERM* terms;
  int length;
  if (!enif_get_tuple(env, tuple, &length, &terms)) return 0;
  var.reserve(length);

  for (int i=0; i < length; i++) {
    T* elem;
    if (!get<T>(env, terms[i], elem)) return 0;
    var.push_back(*elem);
  }
  return 1;
}

int get_list(ErlNifEnv* env,
             ERL_NIF_TERM list,
             std::vector<int64> &var);
int get_list(ErlNifEnv* env,
             ERL_NIF_TERM list,
             std::vector<ErlNifBinary> &var);

template <typename T>
int get_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<T*> &var) {
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
int get_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<T> &var) {
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

int get_binary(ErlNifEnv* env, ERL_NIF_TERM term, ErlNifBinary* var);

ERL_NIF_TERM make_map(ErlNifEnv* env, std::map<std::string, int>& map);

// XLA Protobuf Types
//
// See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/xla_data.proto
// for more details on each type and additional types not listed here.

// Gets a padding configuration from `list`. A padding configuration
// is a list of 3-tuples representing edge high, edge low, and interior
// padding.
int get_padding_config(ErlNifEnv* env,
                       ERL_NIF_TERM list,
                       xla::PaddingConfig* padding_config);

// Gets dimension numbers for usage in the XLA DotGeneral operation.
// Dot dimension numbers are a 2-tuple of lists. The first list
// represents the lhs contraction dimensions. The second list
// represents the rhs contraction dimensions. We do not match
// on the batch dimensions for now.
int get_dot_dimension_numbers(ErlNifEnv* env,
                              ERL_NIF_TERM tuple,
                              xla::DotDimensionNumbers* dims);

// Gets a precision configuration from the configuration term.
// The term should be an integer `0`, `1`, or `2` corresponding
// to default, high, or highest precision respectively. Precision
// configuration is set for each term in an operation.
int get_precision_config(ErlNifEnv* env,
                         ERL_NIF_TERM config_term,
                         int num_operands,
                         xla::PrecisionConfig* precision_config);

// Gets the convolution dimension numbers. Convolutions are determined
// based on input, kernel, and output batch and feature dimensions.
// We receive the dimension numbers as a 3-tuple of tuples. Each tuple
// corresponds to input batch/feature dimensions, kernel input/output
// feature dimensions, and output batch/feature dimensions respectively.
int get_conv_dimension_numbers(ErlNifEnv* env,
                               ERL_NIF_TERM tuple,
                               xla::ConvolutionDimensionNumbers* dimension_numbers);

// Gets a general padding configuration. This is slightly different from
// get_padding_config for usage in a convolution. The convolution only
// supports passing padding as a vector of pairs of edge high, edge low padding
// values. We receive the padding configuration as a list of 2-tuples.
int get_general_padding(ErlNifEnv* env,
                        ERL_NIF_TERM padding_term,
                        std::vector<std::pair<int64, int64>>& padding);

// Gets the primitive type from the given term. The term is a string
// encoding one of the XLA primitive types.
int get_primitive_type(ErlNifEnv* env, ERL_NIF_TERM term, xla::PrimitiveType* type);

// Template for retrieving a value from a scalar. This is
// necessary to avoid having to use templates in the NIF.
template <
  xla::PrimitiveType type,
  typename T = typename xla::primitive_util::PrimitiveTypeToNative<type>::type>
T get_value(ErlNifEnv* env, ERL_NIF_TERM term) {
  T value;
  get(env, term, &value);
  return value;
}

// Extracts information from `GetShape` into a useable term.
ERL_NIF_TERM make_shape_info(ErlNifEnv* env, xla::Shape shape);

}  // namespace nif
}  // namespace exla

// Helper Macros
//
// See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/stream_executor/lib/statusor.h

#define EXLA_STATUS_MACROS_CONCAT_NAME(x, y)                                 \
  EXLA_STATUS_MACROS_CONCAT_NAME_IMPL(x, y)

#define EXLA_STATUS_MACROS_CONCAT_NAME_IMPL(x, y) x##y

// Macro to be used to consume StatusOr from within a NIF. Will
// bind lhs to value if the status is OK, otherwise will return
// `{:error, msg}`.
#define EXLA_ASSIGN_OR_RETURN_NIF(lhs, rexpr, env)                           \
  EXLA_ASSIGN_OR_RETURN_NIF_IMPL(                                            \
    EXLA_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__),           \
                                    lhs, rexpr, env)

#define EXLA_ASSIGN_OR_RETURN_NIF_IMPL(statusor, lhs, rexpr, env)            \
  auto statusor = (rexpr);                                                   \
  if (!statusor.ok()) {                                                      \
    return exla::nif::error(env, statusor.status().error_message().c_str()); \
  }                                                                          \
  lhs = std::move(statusor.ValueOrDie());

// Macro to be used to consume StatusOr. Will bind lhs
// to value if the status is OK, otherwise will return
// the status.
#define EXLA_ASSIGN_OR_RETURN(lhs, rexpr)                                    \
  EXLA_ASSIGN_OR_RETURN_IMPL(                                                \
    EXLA_STATUS_MACROS_CONCAT_NAME(                                          \
      _status_or_value, __COUNTER__),                                        \
  lhs, rexpr)

#define EXLA_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr)                     \
  auto statusor = (rexpr);                                                   \
  if (!statusor.ok()) {                                                      \
    return statusor.status();                                                \
  }                                                                          \
  lhs = std::move(statusor.ValueOrDie());

#endif
