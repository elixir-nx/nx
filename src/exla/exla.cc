#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include <erl_nif.h>

ErlNifResourceType* OP_RES_TYPE;

ERL_NIF_TERM ok, bad;

/* These are global instances of the main XLA API. My understanding is that it's correct
 * only to have and maintain one instance of each of these, so I figured it's best to keep them
 * as private data members in the environment. It's convenient not to have to pass references
 * between functions.
 *
 * I think we need to synchronize access to these resources, but I also
 * can't really think of a use case where you'd run in to problems if we didn't.
 */
typedef struct {
  xla::XlaBuilder* builder;
  xla::LocalClient* client;
} XLA;

/* xla::XlaOp resource wrapper */
typedef struct {
  xla::XlaOp op;
} Op;

/* Destructor for Op. We just release the resource. I'm not actually sure if this is correct? */
void free_op(ErlNifEnv* env, void* obj){
  enif_release_resource(obj);
}

static int load(ErlNifEnv* env, void** priv, ERL_NIF_TERM load_info){
  const char* mod = "XLA";
  const char* name_op = "Op";

  int flags = ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER;

  XLA* xla_objects;

  OP_RES_TYPE = enif_open_resource_type(env, mod, name_op, free_op, (ErlNifResourceFlags) flags, NULL);

  if(OP_RES_TYPE == NULL) return -1;

  xla_objects = (XLA*) enif_alloc(sizeof(XLA));

  ok = enif_make_atom(env, "ok");
  bad = enif_make_atom(env, "error");

  xla_objects->builder = new xla::XlaBuilder("Elixir");
  xla_objects->client = NULL;

  *priv = (void*) xla_objects;

  return 0;
}

ERL_NIF_TERM enif_make_op(ErlNifEnv* env, xla::XlaOp value){
  Op* op = (Op*) enif_alloc_resource(OP_RES_TYPE, sizeof(Op*));
  op->op = value;
  ERL_NIF_TERM ret = enif_make_resource(env, op);
  enif_release_resource(op);
  return ret;
}

ERL_NIF_TERM add(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  Op *lhs, *rhs;
  enif_get_resource(env, argv[0], OP_RES_TYPE, (void **) &lhs);
  enif_get_resource(env, argv[1], OP_RES_TYPE, (void **) &rhs);
  xla::XlaOp result = xla::Add(lhs->op, rhs->op);
  return enif_make_op(env, result);
}

ERL_NIF_TERM constant_r1(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  XLA* xla_objects = (XLA*) enif_priv_data(env);
  int length, value;
  enif_get_int(env, argv[0], &length);
  enif_get_int(env, argv[1], &value);
  xla::XlaOp op = xla::ConstantR1(xla_objects->builder, length, value);
  return enif_make_op(env, op);
}

/*
 * This creates the local client which interfaces with the underlying XLA service.
 * It usually takes config ops, but I haven't handled those yet.
 */
ERL_NIF_TERM get_or_create_local_client(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  XLA* xla_objects = (XLA*) enif_priv_data(env);
  // StatusOr matches really nicely to Elixir's {:ok, ...}/{:error, ...} pattern, haven't handled it yet
  xla::StatusOr<xla::LocalClient*> client_status = xla::ClientLibrary::GetOrCreateLocalClient();
  // This matches really nicely with the ! pattern
  xla::LocalClient* client = client_status.ConsumeValueOrDie();
  xla_objects->client = client;
  return ok;
}

/*
 * Running into some strange memory issues trying to give more fine grain control over what happens with
 * built up computations. The normal process is Compile -> Execute -> Transfer, but I'm having issues
 * passing instances of xla::XlaComputation, xla::LocalExecutable, etc. between NIFs. It may be okay to
 * do away with having to pass these references altogether. For now, the process is combined into this
 * function to run the built up computation.
 */
ERL_NIF_TERM run(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  XLA* xla_objects = (XLA*) enif_priv_data(env);
  xla::StatusOr<xla::XlaComputation> computation_status = xla_objects->builder->Build();
  xla::XlaComputation computation = computation_status.ConsumeValueOrDie();

  xla::StatusOr<xla::Literal> result = xla_objects->client->ExecuteAndTransfer(computation, absl::Span<xla::GlobalData* const>());
  xla::Literal s = result.ConsumeValueOrDie();

  std::string result_str = s.ToString();

  return enif_make_string(env, result_str.c_str(), ERL_NIF_LATIN1);
}

static ErlNifFunc exla_funcs[] = {
  {"get_or_create_local_client", 0, get_or_create_local_client},
  {"add", 2, add},
  {"constant_r1", 2, constant_r1},
  {"run", 0, run}
};

ERL_NIF_INIT(Elixir.Exla, exla_funcs, &load, NULL, NULL, NULL);