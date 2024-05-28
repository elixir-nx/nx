#pragma once
#include "exla_nif_util.h"
#include "mlir/IR/BuiltinTypes.h"

namespace exla {
namespace nif {
// Extracts information from `GetShape` into a usable term.
ERL_NIF_TERM make_typespec(ErlNifEnv* env, mlir::Type type);
}  // namespace nif
}  // namespace exla