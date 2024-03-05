#pragma once
#include "../exla_nif_util.h"

#define IREE_COMPILER_EXPECTED_API_MAJOR 1  // At most this major version
#define IREE_COMPILER_EXPECTED_API_MINOR 2  // At least this minor version

#define DEFINE_NIF(FUNCTION_NAME) ERL_NIF_TERM FUNCTION_NAME(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])

DEFINE_NIF(iree_compile_mlir_module);