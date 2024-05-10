#include "compiler.h"

#include <fcntl.h>  // For O_WRONLY, O_CREAT, O_TRUNC
#include <inttypes.h>
#include <iree/compiler/embedding_api.h>
#include <iree/compiler/loader.h>
#include <iree/compiler/mlir_interop.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>  // For mode constants
#include <unistd.h>    // For open, close

#include <iostream>
#include <sstream>

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
  // ireeCompilerGlobalShutdown();
}

static void initializeCompiler(struct compiler_state_t *state) {
  // ireeCompilerGlobalInitialize();
  state->session = ireeCompilerSessionCreate();
  state->context = ireeCompilerSessionBorrowContext(state->session);
}

static void shutdownCompiler(struct compiler_state_t *state) {
  ireeCompilerSessionDestroy(state->session);
  // ireeCompilerGlobalShutdown();
}

ERL_NIF_TERM compile(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  std::string module_str;

  if (!exla::nif::get(env, argv[0], module_str)) {
    return exla::nif::error(env, "Unable to get module.");
  }

  compiler_state_t state;
  state.session = NULL;
  state.source = NULL;
  state.output = NULL;
  state.invocation = NULL;
  iree_compiler_error_t *error = NULL;

  initializeCompiler(&state);

  MlirOperation module_op = mlirOperationCreateParse(
      state.context,
      mlirStringRefCreate(module_str.c_str(), module_str.size()),
      mlirStringRefCreateFromCString("source.stablehlo"));
  if (mlirOperationIsNull(module_op)) {
    return exla::nif::error(env, "Unable to create MlirOperation module.");
  }

  // Set flags.
  iree_compiler_error_t *err;
  const char *flags[] = {
      "--iree-hal-target-backends=metal-spirv",
      "--iree-input-type=stablehlo_xla",
      "--iree-execution-model=async-internal",
      "--output-format=vm-bytecode",
      "--iree-opt-demote-f64-to-f32=true",
      "--iree-opt-demote-i64-to-i32=true"};
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
  if (!ireeCompilerInvocationPipeline(state.invocation, iree_compiler_pipeline_t::IREE_COMPILER_PIPELINE_STD)) {
    cleanup_compiler_state(state);
    return exla::nif::error(env, "Unable to compile module.");
  }

  fflush(stdout);

  error = ireeCompilerOutputOpenMembuffer(&state.output);
  if (error) {
    handle_compiler_error(error);
    cleanup_compiler_state(state);
    return exla::nif::error(env, "Error opening output membuffer");
  }

  error = ireeCompilerInvocationOutputVMBytecode(state.invocation, state.output);
  if (error) {
    handle_compiler_error(error);
    cleanup_compiler_state(state);
    return exla::nif::error(env, "Failed to output VM Bytecode");
  }

  uint8_t *contents;
  uint64_t size;

  error = ireeCompilerOutputMapMemory(state.output, (void **)&contents, &size);

  std::vector<ERL_NIF_TERM> bytes_term;
  bytes_term.resize(size);
  for (size_t i = 0; i < size; i++) {
    bytes_term[i] = enif_make_uint(env, static_cast<unsigned int>(contents[i]));
  }

  if (error) {
    handle_compiler_error(error);
    cleanup_compiler_state(state);
    return exla::nif::error(env, "Failed to map output to output binary");
  }

  cleanup_compiler_state(state);

  return exla::nif::ok(env, enif_make_list_from_array(env, bytes_term.data(), bytes_term.size()));
}