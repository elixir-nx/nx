#ifndef NIF_CALL_H
#define NIF_CALL_H

#pragma once

#include <erl_nif.h>

#ifdef NIF_CALL_NAMESPACE
#define NIF_CALL_CAT(A, B) A##B
#define NIF_CALL_SYMBOL(A, B) NIF_CALL_CAT(A, B)

#define NifCallCallbackNifRes NIF_CALL_SYMBOL(NIF_CALL_NAMESPACE, NifCallCallbackNifRes)
#define nif_call_onload NIF_CALL_SYMBOL(NIF_CALL_NAMESPACE, nif_call_onload)
#define prepare_nif_call NIF_CALL_SYMBOL(NIF_CALL_NAMESPACE, prepare_nif_call)
#define make_nif_call NIF_CALL_SYMBOL(NIF_CALL_NAMESPACE, make_nif_call)
#define nif_call_evaluated NIF_CALL_SYMBOL(NIF_CALL_NAMESPACE, nif_call_evaluated)
#define destruct_nif_call_res NIF_CALL_SYMBOL(NIF_CALL_NAMESPACE, destruct_nif_call_res)
#endif

#define NIF_CALL_NIF_FUNC(name) \
  {#name, 2, nif_call_evaluated, 0}

#ifndef NIF_CALL_IMPLEMENTATION

struct NifCallCallbackNifRes;
static int nif_call_onload(ErlNifEnv *env);
static NifCallCallbackNifRes * prepare_nif_call(ErlNifEnv* env);
static ERL_NIF_TERM make_nif_call(ErlNifEnv* caller_env, ERL_NIF_TERM tag, ERL_NIF_TERM args);
static ERL_NIF_TERM nif_call_evaluated(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]);
static void destruct_nif_call_res(ErlNifEnv *, void *obj);

#else

struct NifCallCallbackNifRes {
  static ErlNifResourceType *type;
  static ERL_NIF_TERM kAtomOK;
  static ERL_NIF_TERM kAtomError;
  static ERL_NIF_TERM kAtomNil;
  static ERL_NIF_TERM kAtomENOMEM;
  static ERL_NIF_TERM kAtomExecute;
  static ERL_NIF_TERM kAtomCallerEnv;
  static ERL_NIF_TERM kAtomNotTag;
  static ERL_NIF_TERM kAtomInvalidRunner;
  static ERL_NIF_TERM kAtomInvalidRunnerReply;
  static ERL_NIF_TERM kAtomRunnerIsDown;

  ErlNifEnv * msg_env;
  ErlNifMutex *mtx = NULL;
  ErlNifCond *cond = NULL;
  
  ERL_NIF_TERM return_value;
  bool return_value_set;
};

struct NifCallResult {
  bool err;
  
  // `kind` can be one of these atoms:
  // - `ok`
  // - `error`
  // - `exit`
  // - `throw`
  ERL_NIF_TERM kind;
  ERL_NIF_TERM value;

  bool is_ok() {
    return !err;
  }

  ERL_NIF_TERM get_kind() {
    return kind;
  }

  ERL_NIF_TERM get_value() {
    return value;
  }

  ERL_NIF_TERM get_err() {
    return value;
  }

  static NifCallResult ok(ERL_NIF_TERM value) {
    NifCallResult res;
    res.err = false;
    res.value = value;
    return res;
  }

  static NifCallResult error(ERL_NIF_TERM value) {
    NifCallResult res;
    res.err = true;
    res.kind = NifCallCallbackNifRes::kAtomError;
    res.value = value;
    return res;
  }

  static NifCallResult error(ERL_NIF_TERM kind, ERL_NIF_TERM value) {
    NifCallResult res;
    res.err = true;
    res.kind = kind;
    res.value = value;
    return res;
  }
};

ErlNifResourceType * NifCallCallbackNifRes::type = NULL;
ERL_NIF_TERM NifCallCallbackNifRes::kAtomOK;
ERL_NIF_TERM NifCallCallbackNifRes::kAtomError;
ERL_NIF_TERM NifCallCallbackNifRes::kAtomNil;
ERL_NIF_TERM NifCallCallbackNifRes::kAtomENOMEM;
ERL_NIF_TERM NifCallCallbackNifRes::kAtomExecute;
ERL_NIF_TERM NifCallCallbackNifRes::kAtomCallerEnv;
ERL_NIF_TERM NifCallCallbackNifRes::kAtomNotTag;
ERL_NIF_TERM NifCallCallbackNifRes::kAtomInvalidRunner;
ERL_NIF_TERM NifCallCallbackNifRes::kAtomInvalidRunnerReply;
ERL_NIF_TERM NifCallCallbackNifRes::kAtomRunnerIsDown;

NifCallCallbackNifRes * prepare_nif_call(ErlNifEnv* env) {
  NifCallCallbackNifRes *res = (NifCallCallbackNifRes *)enif_alloc_resource(NifCallCallbackNifRes::type, sizeof(NifCallCallbackNifRes));
  if (!res) return NULL;
  memset(res, 0, sizeof(NifCallCallbackNifRes));

  res->msg_env = enif_alloc_env();
  if (!res->msg_env) {
    enif_release_resource(res);
    return NULL;
  }

  res->mtx = enif_mutex_create((char *)"nif_call_mutex");
  if (!res->mtx) {
    enif_free_env(res->msg_env);
    enif_release_resource(res);
    return NULL;
  }

  res->cond = enif_cond_create((char *)"nif_call_cond");
  if (!res->cond) {
    enif_free_env(res->msg_env);
    enif_mutex_destroy(res->mtx);
    enif_release_resource(res);
    return NULL;
  }

  res->return_value_set = false;
  res->return_value = NifCallCallbackNifRes::kAtomNil;

  return res;
}

static NifCallResult make_nif_call(ErlNifEnv* caller_env, ERL_NIF_TERM tag, ERL_NIF_TERM args) {
  NifCallCallbackNifRes *callback_res = prepare_nif_call(caller_env);
  if (!callback_res) {
    return NifCallResult::error(NifCallCallbackNifRes::kAtomENOMEM);
  }

  int arity = 0;
  const ERL_NIF_TERM * tag_container = NULL;
  if (!enif_get_tuple(caller_env, tag, &arity, &tag_container) || arity != 2 || !enif_is_pid(caller_env, tag_container[0])) {
    return NifCallResult::error(NifCallCallbackNifRes::kAtomNotTag);
  }

  ErlNifPid evaluator;
  if (!enif_get_local_pid(caller_env, tag_container[0], &evaluator)) {
    return NifCallResult::error(NifCallCallbackNifRes::kAtomInvalidRunner);
  }

  ERL_NIF_TERM callback_term = enif_make_resource(caller_env, (void *)callback_res);
  enif_send(caller_env, &evaluator, callback_res->msg_env, enif_make_copy(callback_res->msg_env, enif_make_tuple4(caller_env,
    NifCallCallbackNifRes::kAtomExecute,
    callback_term,
    tag_container[1],
    args
  )));

  enif_mutex_lock(callback_res->mtx);
  while (!callback_res->return_value_set) {
    enif_cond_wait(callback_res->cond, callback_res->mtx);
    if (enif_is_process_alive(caller_env, &evaluator) == 0) {
      enif_mutex_unlock(callback_res->mtx);
      return NifCallResult::error(NifCallCallbackNifRes::kAtomRunnerIsDown);
    }
  }
  enif_mutex_unlock(callback_res->mtx);

  ERL_NIF_TERM return_value = enif_make_copy(caller_env, callback_res->return_value);
  enif_release_resource(callback_res);

  arity = 0;
  const ERL_NIF_TERM * val_container = NULL;
  if (!enif_get_tuple(caller_env, return_value, &arity, &val_container) || arity != 2) {
    return NifCallResult::error(NifCallCallbackNifRes::kAtomInvalidRunnerReply);
  }

  if (enif_compare(val_container[0], NifCallCallbackNifRes::kAtomOK) == 0) {
    return NifCallResult::ok(val_container[1]);
  }
  
  return NifCallResult::error(val_container[0], val_container[1]);
}

static ERL_NIF_TERM nif_call_evaluated(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  NifCallCallbackNifRes *res = NULL;
  if (!enif_get_resource(env, argv[0], NifCallCallbackNifRes::type, (void **)&res)) return enif_make_badarg(env);

  res->return_value = enif_make_copy(res->msg_env, argv[1]);
  res->return_value_set = true;
  enif_cond_signal(res->cond);

  return NifCallCallbackNifRes::kAtomOK;
}

static void destruct_nif_call_res(ErlNifEnv *, void *obj) {
  NifCallCallbackNifRes *res = (NifCallCallbackNifRes *)obj;
  if (res->cond) {
    enif_cond_destroy(res->cond);
    res->cond = NULL;
  }
  if (res->mtx) {
    enif_mutex_destroy(res->mtx);
    res->mtx = NULL;
  }
  if (res->msg_env) {
    enif_free_env(res->msg_env);
    res->msg_env = NULL;
  }
}

static int nif_call_onload(ErlNifEnv *env) {
  static int loaded = 0;
  if (loaded) return 0;

  ErlNifResourceType *rt;
  rt = enif_open_resource_type(env, "Elixir.NifCall.NIF", "NifCallCallbackNifRes", destruct_nif_call_res, ERL_NIF_RT_CREATE, NULL);
  if (!rt) return -1;
  NifCallCallbackNifRes::type = rt;

  NifCallCallbackNifRes::kAtomOK = enif_make_atom(env, "ok");
  NifCallCallbackNifRes::kAtomError = enif_make_atom(env, "error");
  NifCallCallbackNifRes::kAtomNil = enif_make_atom(env, "nil");
  NifCallCallbackNifRes::kAtomENOMEM = enif_make_atom(env, "enomem");
  NifCallCallbackNifRes::kAtomExecute = enif_make_atom(env, "execute");

  // https://www.erlang.org/doc/apps/erts/erl_nif.html#enif_self
  // https://www.erlang.org/doc/apps/erts/erl_nif#proc_bound_env
  NifCallCallbackNifRes::kAtomCallerEnv = enif_make_atom(env, "not_in_process_bound_env");
  
  NifCallCallbackNifRes::kAtomNotTag = enif_make_atom(env, "not_tag");
  NifCallCallbackNifRes::kAtomInvalidRunner = enif_make_atom(env, "invalid_runner");
  NifCallCallbackNifRes::kAtomInvalidRunnerReply = enif_make_atom(env, "invalid_runner_reply");
  NifCallCallbackNifRes::kAtomRunnerIsDown = enif_make_atom(env, "runner_is_down");
  
  loaded = 1;
  return 0;
}

#endif

#endif  // NIF_CALL_H
