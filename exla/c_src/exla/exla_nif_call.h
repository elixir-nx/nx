#pragma once

#include <erl_nif.h>

#ifdef __cplusplus
extern "C" {
#endif

// Wrapper around nif_call's make_nif_call to avoid including its implementation
// in multiple translation units. Returns true on success and writes the value
// to out_value. Returns false on error.
bool exla_nif_call_make(ErlNifEnv *env, ERL_NIF_TERM tag, ERL_NIF_TERM arg,
                        ERL_NIF_TERM *out_value);

// Expose the evaluated callback and onload through stable, exported wrappers.
ERL_NIF_TERM exla_nif_call_evaluated(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]);

int exla_nif_call_onload(ErlNifEnv *env);

#ifdef __cplusplus
}
#endif
