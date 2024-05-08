#include "runtime.h"

ERL_NIF_TERM run_module(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return enif_make_badarg(env);
  }

  return exla::nif::error(env, "runtime not implemented yet");
}