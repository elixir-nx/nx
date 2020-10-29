/*
 * XLA Extension NIFs for use in Elixir/Erlang.
 */
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include <erl_nif.h>

static ERL_NIF_TERM create_builder(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
  xla::XlaBuilder builder("elixir xla builder");
  return enif_make_int(env, 0);
}

static ErlNifFunc exla_funcs[] = {
  {"create_builder", 0, create_builder},
};

ERL_NIF_INIT(Elixir.Exla, exla_funcs, NULL, NULL, NULL, NULL);