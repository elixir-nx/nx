#include <iree/compiler/embedding_api.h>
#include <iree/compiler/loader.h>

#include "../exla_mlir.h"
#include "../exla_nif_util.h"
#include "compiler.h"
#include "runtime.h"

ERL_NIF_TERM global_initialize(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  ireeCompilerLoadLibrary("libIREECompiler.dylib");
  ireeCompilerGlobalInitialize();
  return exla::nif::ok(env);
}

static ErlNifFunc iree_funcs[] = {
    // MLIR Builder
    {"global_initialize", 0, global_initialize},
    {"compile", 2, compile, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"run_module", 2, run_module, ERL_NIF_DIRTY_JOB_IO_BOUND}};

static int open_resources(ErlNifEnv *env) {
  const char *mod = "EXLA";

  if (!exla::nif::open_resource<exla::MLIRModule *>(env, mod, "ExlaMLIRModule")) {
    return -1;
  }
  return 1;
}

static int load(ErlNifEnv *env, void **priv, ERL_NIF_TERM load_info) {
  if (open_resources(env) == -1) return -1;

  return 0;
}

ERL_NIF_INIT(Elixir.EXLA.MLIR.IREE, iree_funcs, &load, NULL, NULL, NULL);