#include "exla/exla_nif_util.h"

int ExlaNifUtil::get(ErlNifEnv* env, ERL_NIF_TERM term, int &var){ return enif_get_int(env, term, &var); }
int ExlaNifUtil::get(ErlNifEnv* env, ERL_NIF_TERM term, long int &var){ return enif_get_int64(env, term, &var); }
int ExlaNifUtil::get(ErlNifEnv* env, ERL_NIF_TERM term, long long int &var){ return enif_get_int64(env, term, (long int*) &var); }

int ExlaNifUtil::get(ErlNifEnv* env, ERL_NIF_TERM term, std::string &var){
  unsigned len;
  int ret = enif_get_list_length(env, term, &len);

  if(!ret){
    ErlNifBinary bin;
    ret = enif_inspect_binary(env, term, &bin);
    if(!ret){
      return 0;
    }
    var = std::string((const char*)bin.data, bin.size);
    return ret;
  }

  var.resize(len+1);
  ret = enif_get_string(env, term, &*(var.begin()), var.size(), ERL_NIF_LATIN1);

  if(ret > 0){var.resize(ret-1);}
  else if(ret==0){var.resize(0);}
  else{}

  return ret;
}

int ExlaNifUtil::get_atom(ErlNifEnv* env, ERL_NIF_TERM term, std::string &var){
  unsigned atom_length;
  if(!enif_get_atom_length(env, term, &atom_length, ERL_NIF_LATIN1)) return 0;

  var.resize(atom_length+1);

  if(!enif_get_atom(env, term, &(*(var.begin())), var.size(), ERL_NIF_LATIN1)) return 0;

  var.resize(atom_length);

  return 1;
}