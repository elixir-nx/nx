#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include <erl_nif.h>

ErlNifResourceType* BUILDER_RES_TYPE;
ErlNifResourceType* OP_RES_TYPE;
ERL_NIF_TERM ok;

typedef struct {
  xla::XlaBuilder* builder;
} Builder;

typedef struct {
  xla::XlaOp op;
} Op;

void free_builder(ErlNifEnv* env, void* obj){
  // enif_release_resource(obj);
}

void free_op(ErlNifEnv* env, void* obj){
  enif_release_resource(obj);
}

static int load(ErlNifEnv* env, void** priv, ERL_NIF_TERM load_info){
  const char* mod = "builder";
  const char* name_builder = "Builder";
  const char* name_op = "Op";

  int flags = ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER;

  Builder* builder;

  BUILDER_RES_TYPE = enif_open_resource_type(env, mod, name_builder, free_builder, (ErlNifResourceFlags) flags, NULL);
  OP_RES_TYPE = enif_open_resource_type(env, mod, name_op, free_op, (ErlNifResourceFlags) flags, NULL);
  if(BUILDER_RES_TYPE == NULL || OP_RES_TYPE == NULL) return -1;

  builder = (Builder*) enif_alloc(sizeof(Builder));
  ok = enif_make_atom(env, "ok");

  builder->builder = new xla::XlaBuilder("Elixir");

  *priv = (void*) builder;

  return 0;
}

ERL_NIF_TERM enif_make_op(ErlNifEnv* env, xla::XlaOp value){
  Op* op = (Op*) enif_alloc_resource(OP_RES_TYPE, sizeof(Op*));
  op->op = value;
  ERL_NIF_TERM ret = enif_make_resource(env, op);
  enif_keep_resource(op);

  return ret;
}

ERL_NIF_TERM constant_from_array(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
  Builder* builder = (Builder*) enif_priv_data(env);

  xla::Array<uint> arr = xla::Array<uint>({1, 2, 3});
  xla::XlaOp op = xla::ConstantFromArray(builder->builder, arr);

  return enif_make_op(env, op);
}

ERL_NIF_TERM add(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
  Op *lhs, *rhs;
  enif_get_resource(env, argv[0], OP_RES_TYPE, (void **) &lhs);
  enif_get_resource(env, argv[1], OP_RES_TYPE, (void **) &rhs);

  absl::Span<const long long int> broadcast = absl::Span<const long long int>({1});

  xla::XlaOp result = xla::Add(lhs->op, rhs->op, broadcast);
  return enif_make_op(env, result);
}

ERL_NIF_TERM build(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
  Builder* builder = (Builder*) enif_priv_data(env);

  builder->builder->Build();

  return ok;
}

static ErlNifFunc builder_funcs[] = {
  {"add", 2, add},
  {"constant_from_array", 0, constant_from_array},
  {"build", 0, build}
};

ERL_NIF_INIT(Elixir.Exla.Builder, builder_funcs, &load, NULL, NULL, NULL);