#include "exla_nif_util.h"

#include "mlir/IR/BuiltinTypes.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "mlir/IR/Builders.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace exla {
namespace nif {

// Status helpers

ERL_NIF_TERM error(ErlNifEnv* env, const char* msg) {
  ERL_NIF_TERM atom = enif_make_atom(env, "error");
  ERL_NIF_TERM msg_term = enif_make_string(env, msg, ERL_NIF_LATIN1);
  return enif_make_tuple2(env, atom, msg_term);
}

ERL_NIF_TERM ok(ErlNifEnv* env, ERL_NIF_TERM term) {
  return enif_make_tuple2(env, ok(env), term);
}

ERL_NIF_TERM ok(ErlNifEnv* env) {
  return enif_make_atom(env, "ok");
}

// Numeric types

int get(ErlNifEnv* env, ERL_NIF_TERM term, int8* var) {
  int value;
  if (!enif_get_int(env, term, &value)) return 0;
  *var = static_cast<int8>(value);
  return 1;
}

int get(ErlNifEnv* env, ERL_NIF_TERM term, int16* var) {
  int value;
  if (!enif_get_int(env, term, &value)) return 0;
  *var = static_cast<int16>(value);
  return 1;
}

int get(ErlNifEnv* env, ERL_NIF_TERM term, int32* var) {
  return enif_get_int(env, term,
                      reinterpret_cast<int*>(var));
}

int get(ErlNifEnv* env, ERL_NIF_TERM term, int64* var) {
  return enif_get_int64(env, term,
                        reinterpret_cast<nif_int64_t*>(var));
}

int get(ErlNifEnv* env, ERL_NIF_TERM term, uint8* var) {
  unsigned int value;
  if (!enif_get_uint(env, term, &value)) return 0;
  *var = static_cast<uint8>(value);
  return 1;
}

int get(ErlNifEnv* env, ERL_NIF_TERM term, uint16* var) {
  unsigned int value;
  if (!enif_get_uint(env, term, &value)) return 0;
  *var = static_cast<uint16>(value);
  return 1;
}

int get(ErlNifEnv* env, ERL_NIF_TERM term, uint32* var) {
  return enif_get_uint(env, term,
                       reinterpret_cast<unsigned int*>(var));
}

int get(ErlNifEnv* env, ERL_NIF_TERM term, uint64* var) {
  return enif_get_uint64(env, term,
                         reinterpret_cast<nif_uint64_t*>(var));
}

int get(ErlNifEnv* env, ERL_NIF_TERM term, float16* var) {
  double value;
  if (!enif_get_double(env, term, &value)) return 0;
  *var = static_cast<float16>(value);
  return 1;
}

int get(ErlNifEnv* env, ERL_NIF_TERM term, bfloat16* var) {
  double value;
  if (!enif_get_double(env, term, &value)) return 0;
  *var = static_cast<bfloat16>(value);
  return 1;
}

int get(ErlNifEnv* env, ERL_NIF_TERM term, float32* var) {
  double value;
  if (!enif_get_double(env, term, &value)) return 0;
  *var = static_cast<float32>(value);
  return 1;
}

int get(ErlNifEnv* env, ERL_NIF_TERM term, float64* var) {
  return enif_get_double(env, term, var);
}

int get(ErlNifEnv* env, ERL_NIF_TERM term, complex64* var) {
  return 0;
}

int get(ErlNifEnv* env, ERL_NIF_TERM term, complex128* var) {
  return 0;
}

ERL_NIF_TERM make(ErlNifEnv* env, int32 var) {
  return enif_make_int(env, var);
}

// Standard types

int get(ErlNifEnv* env, ERL_NIF_TERM term, std::string& var) {
  unsigned len;
  int ret = enif_get_list_length(env, term, &len);

  if (!ret) {
    ErlNifBinary bin;
    ret = enif_inspect_binary(env, term, &bin);
    if (!ret) {
      return 0;
    }
    var = std::string((const char*)bin.data, bin.size);
    return ret;
  }

  var.resize(len + 1);
  ret = enif_get_string(env, term, &*(var.begin()), var.size(), ERL_NIF_LATIN1);

  if (ret > 0) {
    var.resize(ret - 1);
  } else if (ret == 0) {
    var.resize(0);
  } else {
  }

  return ret;
}

int get(ErlNifEnv* env, ERL_NIF_TERM term, bool* var) {
  int value;
  if (!enif_get_int(env, term, &value)) return 0;
  *var = static_cast<bool>(value);
  return 1;
}

ERL_NIF_TERM make(ErlNifEnv* env, ErlNifBinary var) {
  return enif_make_binary(env, &var);
}

ERL_NIF_TERM make(ErlNifEnv* env, std::string var) {
  return enif_make_string(env, var.c_str(), ERL_NIF_LATIN1);
}

ERL_NIF_TERM make(ErlNifEnv* env, const char* string) {
  return enif_make_string(env, string, ERL_NIF_LATIN1);
}

// Atoms

int get_atom(ErlNifEnv* env, ERL_NIF_TERM term, std::string& var) {
  unsigned atom_length;
  if (!enif_get_atom_length(env, term, &atom_length, ERL_NIF_LATIN1)) {
    return 0;
  }

  var.resize(atom_length + 1);

  if (!enif_get_atom(env, term, &(*(var.begin())), var.size(), ERL_NIF_LATIN1)) return 0;

  var.resize(atom_length);

  return 1;
}

ERL_NIF_TERM atom(ErlNifEnv* env, const char* msg) {
  return enif_make_atom(env, msg);
}

// Containers

int get_tuple(ErlNifEnv* env, ERL_NIF_TERM tuple, std::vector<int64>& var) {
  const ERL_NIF_TERM* terms;
  int length;
  if (!enif_get_tuple(env, tuple, &length, &terms)) return 0;
  var.reserve(length);

  for (int i = 0; i < length; i++) {
    int data;
    if (!get(env, terms[i], &data)) return 0;
    var.push_back(data);
  }
  return 1;
}

int get_list(ErlNifEnv* env,
             ERL_NIF_TERM list,
             std::vector<ErlNifBinary>& var) {
  unsigned int length;
  if (!enif_get_list_length(env, list, &length)) return 0;
  var.reserve(length);
  ERL_NIF_TERM head, tail;

  while (enif_get_list_cell(env, list, &head, &tail)) {
    ErlNifBinary elem;
    if (!get_binary(env, head, &elem)) return 0;
    var.push_back(elem);
    list = tail;
  }
  return 1;
}

int get_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<int64>& var) {
  unsigned int length;
  if (!enif_get_list_length(env, list, &length)) return 0;
  var.reserve(length);
  ERL_NIF_TERM head, tail;

  while (enif_get_list_cell(env, list, &head, &tail)) {
    int64 elem;
    if (!get(env, head, &elem)) return 0;
    var.push_back(elem);
    list = tail;
  }
  return 1;
}

int get_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<std::string>& var) {
  unsigned int length;
  if (!enif_get_list_length(env, list, &length)) {
    return 0;
  }
  var.reserve(length);
  ERL_NIF_TERM head, tail;

  while (enif_get_list_cell(env, list, &head, &tail)) {
    std::string elem;
    if (!get(env, head, elem)) {
      return 0;
    }
    var.push_back(elem);
    list = tail;
  }
  return 1;
}

int get_binary(ErlNifEnv* env, ERL_NIF_TERM term, ErlNifBinary* var) {
  return enif_inspect_binary(env, term, var);
}

ERL_NIF_TERM make_map(ErlNifEnv* env, std::map<std::string, int>& map) {
  ERL_NIF_TERM term = enif_make_new_map(env);
  std::map<std::string, int>::iterator itr;
  for (itr = map.begin(); itr != map.end(); ++itr) {
    ERL_NIF_TERM key = make(env, itr->first);
    ERL_NIF_TERM value = make(env, itr->second);
    enif_make_map_put(env, term, key, value, &term);
  }
  return term;
}

int get_primitive_type(ErlNifEnv* env, ERL_NIF_TERM term, xla::PrimitiveType* type) {
  std::string type_str;
  if (!get(env, term, type_str)) return 0;

  xla::StatusOr<xla::PrimitiveType> type_status =
      xla::primitive_util::StringToPrimitiveType(type_str);

  if (!type_status.ok()) {
    return 0;
  }
  *type = type_status.value();
  return 1;
}

int get_typespec_as_xla_shape(ErlNifEnv* env, ERL_NIF_TERM term, xla::Shape* shape) {
  int arity;
  const ERL_NIF_TERM* tuple;

  if (!enif_get_tuple(env, term, &arity, &tuple)) return 0;

  xla::PrimitiveType element_type;
  std::vector<exla::int64> dims;

  if (!get_primitive_type(env, tuple[0], &element_type)) return 0;
  if (!get_tuple(env, tuple[1], dims)) return 0;

  *shape = xla::ShapeUtil::MakeShape(element_type, dims);

  return 1;
}

int get_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<xla::Shape>& var) {
  unsigned int length;
  if (!enif_get_list_length(env, list, &length)) {
    return 0;
  }
  var.reserve(length);
  ERL_NIF_TERM head, tail;

  while (enif_get_list_cell(env, list, &head, &tail)) {
    xla::Shape elem;
    if (!get_typespec_as_xla_shape(env, head, &elem)) {
      return 0;
    }
    var.push_back(elem);
    list = tail;
  }
  return 1;
}

std::string mlir_numeric_type_to_string(mlir::Type type) {
  if (type.isSignlessInteger(1)) {
    return "pred";
  }
  if (auto integer_type = type.dyn_cast<mlir::IntegerType>()) {
    if (integer_type.isUnsigned()) {
      return "u" + std::to_string(integer_type.getWidth());
    } else {
      return "s" + std::to_string(integer_type.getWidth());
    }
  }
  if (type.isBF16()) {
    return "bf16";
  }
  if (auto float_type = type.dyn_cast<mlir::FloatType>()) {
    return "f" + std::to_string(float_type.getWidth());
  }
  if (auto complex_type = type.dyn_cast<mlir::ComplexType>()) {
    auto element_type = complex_type.getElementType();
    return "c" + std::to_string(element_type.cast<mlir::FloatType>().getWidth() * 2);
  }

  std::cerr << "Unexpected mlir type" << std::endl;
  exit(1);
}

ERL_NIF_TERM make_typespec(ErlNifEnv* env, mlir::Type type) {
  if (type.isa<mlir::stablehlo::TokenType>()) {
    auto type_term = make(env, "token");
    auto shape_term = enif_make_tuple(env, 0);

    return enif_make_tuple(env, 2, type_term, shape_term);
  }

  if (type.isa<mlir::RankedTensorType>()) {
    auto tensor_type = type.cast<mlir::RankedTensorType>();
    auto dims = tensor_type.getShape();
    auto element_type = tensor_type.getElementType();

    auto dims_array = std::vector<ERL_NIF_TERM>{};
    dims_array.reserve(dims.size());

    for (auto dim : dims) {
      dims_array.push_back(enif_make_int(env, dim));
    }

    auto type_term = make(env, mlir_numeric_type_to_string(element_type));
    auto shape_term = enif_make_tuple_from_array(env, dims_array.data(), dims_array.size());

    return enif_make_tuple(env, 2, type_term, shape_term);
  }

  std::cerr << "Unexpected mlir type" << std::endl;
  exit(1);
}

}  // namespace nif
}  // namespace exla
