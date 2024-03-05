#include "iree_compiler.h"

#include <inttypes.h>
#include <iree/compiler/embedding_api.h>
#include <iree/compiler/loader.h>
#include <iree/compiler/mlir_interop.h>
#include <stdio.h>
#include <string.h>

#include "builder.h"

typedef struct compiler_state_t {
  iree_compiler_session_t *session;
  iree_compiler_source_t *source;
  iree_compiler_output_t *output;
  iree_compiler_invocation_t *invocation;
  MlirContext context;
} compiler_state_t;

void handle_compiler_error(iree_compiler_error_t *error) {
  const char *msg = ireeCompilerErrorGetMessage(error);
  fprintf(stderr, "Error from compiler API:\n%s\n", msg);
  ireeCompilerErrorDestroy(error);
}

void cleanup_compiler_state(compiler_state_t s) {
  if (s.invocation)
    ireeCompilerInvocationDestroy(s.invocation);
  if (s.output)
    ireeCompilerOutputDestroy(s.output);
  if (s.source)
    ireeCompilerSourceDestroy(s.source);
  if (s.session)
    ireeCompilerSessionDestroy(s.session);
  ireeCompilerGlobalShutdown();
}

static void initializeCompiler(struct compiler_state_t *state) {
  ireeCompilerGlobalInitialize();
  state->session = ireeCompilerSessionCreate();
  state->context = ireeCompilerSessionBorrowContext(state->session);
}

static void shutdownCompiler(struct compiler_state_t *state) {
  ireeCompilerSessionDestroy(state->session);
  ireeCompilerGlobalShutdown();
}

ERL_NIF_TERM iree_compile_mlir_module(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRModule **module;

  if (!exla::nif::get<exla::MLIRModule *>(env, argv[0], module)) {
    return exla::nif::error(env, "Unable to get module.");
  }

  compiler_state_t state;
  state.session = NULL;
  state.source = NULL;
  state.output = NULL;
  state.invocation = NULL;
  iree_compiler_error_t *error = NULL;

  initializeCompiler(&state);

  std::string module_str = (*module)->toMLIRString();
  MlirOperation module_op = mlirOperationCreateParse(
      state.context,
      mlirStringRefCreateFromCString(module_str.c_str()),
      mlirStringRefCreateFromCString("source.mlir"));
  if (mlirOperationIsNull(module_op)) {
    return exla::nif::error(env, "Unable to create MlirOperation module.");
  }

  // Set flags.
  iree_compiler_error_t *err;
  const char *flags[] = {
      "--iree-hal-target-backends=vmvx",
  };
  err = ireeCompilerSessionSetFlags(state.session, 1, flags);
  if (err) {
    cleanup_compiler_state(state);
    return exla::nif::error(env, "Unable to set flags.");
  }

  state.invocation = ireeCompilerInvocationCreate(state.session);
  ireeCompilerInvocationEnableConsoleDiagnostics(state.invocation);

  if (!ireeCompilerInvocationImportStealModule(state.invocation, module_op)) {
    cleanup_compiler_state(state);
    return exla::nif::error(env, "Unable to import module.");
  }

  // Compile.
  if (!ireeCompilerInvocationPipeline(state.invocation, iree_compiler_pipeline_t::IREE_COMPILER_PIPELINE_PRECOMPILE)) {
    cleanup_compiler_state(state);
    return exla::nif::error(env, "Unable to compile module.");
  }

  shutdownCompiler(&state);
  return exla::nif::ok(env);
}