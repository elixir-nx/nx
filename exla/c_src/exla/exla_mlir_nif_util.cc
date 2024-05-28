#include "exla_mlir_nif_util.h"

#include "mlir/IR/Builders.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace exla {
namespace nif {

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