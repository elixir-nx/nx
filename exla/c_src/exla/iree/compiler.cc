#include "compiler.h"

#include <fcntl.h>  // For O_WRONLY, O_CREAT, O_TRUNC
#include <inttypes.h>
#include <iree/base/tracing.h>
#include <iree/compiler/embedding_api.h>
#include <iree/compiler/loader.h>
#include <iree/compiler/mlir_interop.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>  // For mode constants
#include <unistd.h>    // For open, close

#include <iostream>
#include <sstream>
#include <tracy/Tracy.hpp>

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

ERL_NIF_TERM compile(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  ZoneScopedN("compile main");
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  std::string module_str;
  std::vector<std::string> flags_str;
  std::vector<const char *> flags;

  {
    ZoneScopedN("compiler get arguments");
    if (!exla::nif::get(env, argv[0], module_str)) {
      return exla::nif::error(env, "Unable to get module.");
    }

    if (!exla::nif::get_list(env, argv[1], flags_str)) {
      return exla::nif::error(env, "Unable to get list.");
    }

    for (auto &flag : flags_str) {
      flags.push_back(flag.c_str());
    }
  }

  compiler_state_t state;
  state.session = NULL;
  state.source = NULL;
  state.output = NULL;
  state.invocation = NULL;
  iree_compiler_error_t *error = NULL;

  initializeCompiler(&state);

  MlirOperation module_op;

  {
    ZoneScopedN("Parse module");
    module_op = mlirOperationCreateParse(
        state.context,
        mlirStringRefCreate(module_str.c_str(), module_str.size()),
        mlirStringRefCreateFromCString("source.stablehlo"));
    if (mlirOperationIsNull(module_op)) {
      return exla::nif::error(env, "Unable to create MlirOperation module.");
    }
  }

  // Set flags.
  {
    ZoneScopedN("Set flags");
    error = ireeCompilerSessionSetFlags(state.session, flags.size(), flags.data());
    if (error) {
      const char *msg = ireeCompilerErrorGetMessage(error);

      cleanup_compiler_state(state);

      std::stringstream ss;
      ss << "Unable to set flags due to error: ";
      ss << msg;

      return exla::nif::error(env, ss.str().c_str());
    }
  }

  {
    ZoneScopedN("Create invocation");
    state.invocation = ireeCompilerInvocationCreate(state.session);
    ireeCompilerInvocationEnableConsoleDiagnostics(state.invocation);
  }

  {
    ZoneScopedN("Import module");
    if (!ireeCompilerInvocationImportStealModule(state.invocation, module_op)) {
      cleanup_compiler_state(state);
      return exla::nif::error(env, "Unable to import module.");
    }
  }

  // Compile.
  {
    ZoneScopedN("Invocation Pipeline");
    if (!ireeCompilerInvocationPipeline(state.invocation, iree_compiler_pipeline_t::IREE_COMPILER_PIPELINE_STD)) {
      cleanup_compiler_state(state);
      return exla::nif::error(env, "Unable to compile module.");
    }
  }

  {
    ZoneScopedN("Open output membuffer");
    error = ireeCompilerOutputOpenMembuffer(&state.output);
    if (error) {
      handle_compiler_error(error);
      cleanup_compiler_state(state);
      return exla::nif::error(env, "Error opening output membuffer");
    }
  }

  {
    ZoneScopedN("Output VM Bytecode");
    error = ireeCompilerInvocationOutputVMBytecode(state.invocation, state.output);
    if (error) {
      handle_compiler_error(error);
      cleanup_compiler_state(state);
      return exla::nif::error(env, "Failed to output VM Bytecode");
    }
  }

  uint8_t *contents;
  uint64_t size;
  ErlNifBinary output_binary;

  {
    ZoneScopedN("Map and copy output to binary");
    error = ireeCompilerOutputMapMemory(state.output, (void **)&contents, &size);

    enif_alloc_binary(size, &output_binary);
    memcpy(output_binary.data, contents, size);
  }

  if (error) {
    handle_compiler_error(error);
    cleanup_compiler_state(state);
    return exla::nif::error(env, "Failed to map output to output binary");
  }

  cleanup_compiler_state(state);

  IREE_TRACE_ZONE_END(compile);
  return exla::nif::ok(env, enif_make_binary(env, &output_binary));
}