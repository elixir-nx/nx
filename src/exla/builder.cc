#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include <erl_nif.h>

ERL_NIF_TERM create_builder(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  auto builder = absl::make_unique<xla::XlaBuilder>("elixir");
  return enif_make_binary(env, (ErlNifBinary*) &builder);
}

ERL_NIF_TERM build(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  ErlNifBinary* builder_ptr;
  enif_inspect_binary(env, argv[0], builder_ptr);

  xla::XlaBuilder& builder = (xla::XlaBuilder&) *builder_ptr;
  builder.Build();

  return enif_make_atom(env, "ok");
}

static ErlNifFunc builder_funcs[] = {
  {"create_builder", 1, create_builder},
  {"_build", 1, build}
};


ERL_NIF_INIT(Elixir.Exla.Builder, builder_funcs, NULL, NULL, NULL, NULL);