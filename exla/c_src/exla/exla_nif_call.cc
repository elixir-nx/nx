#include "exla_nif_call.h"

#define NIF_CALL_IMPLEMENTATION
#include "../nif_call.h"

extern "C" bool exla_nif_call_make(ErlNifEnv *env, ERL_NIF_TERM tag,
                                   ERL_NIF_TERM arg, ERL_NIF_TERM *out_value) {
  NifCallResult res = make_nif_call(env, tag, arg);
  if (!res.is_ok()) {
    return false;
  }
  *out_value = res.get_value();
  return true;
}

extern "C" ERL_NIF_TERM exla_nif_call_evaluated(ErlNifEnv *env, int argc,
                                                const ERL_NIF_TERM argv[]) {
  return nif_call_evaluated(env, argc, argv);
}

extern "C" int exla_nif_call_onload(ErlNifEnv *env) {
  return nif_call_onload(env);
}
