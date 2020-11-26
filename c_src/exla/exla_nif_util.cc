#include "tensorflow/compiler/xla/exla/exla_nif_util.h"

// TODO: Map all to corresponding TF type

namespace exla {

  ERL_NIF_TERM error(ErlNifEnv* env, const char* msg){
    return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_string(env, msg, ERL_NIF_LATIN1));
  }

  ERL_NIF_TERM ok(ErlNifEnv* env){
    return enif_make_atom(env, "ok");
  }

  ERL_NIF_TERM ok(ErlNifEnv* env, ERL_NIF_TERM term){
    return enif_make_tuple2(env, ok(env), term);
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, bool &var) {
    int value;
    if(!enif_get_int(env, term, &value)) return 0;
    var = (bool) value;
    return 1;
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, ErlNifBinary &var) {
    return enif_inspect_binary(env, term, &var);
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, uint8_t &var) {
    unsigned int value;
    if(!enif_get_uint(env, term, &value)) return 0;
    var = (uint8_t) value;
    return 1;
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, uint16_t &var) {
    unsigned int value;
    if(!enif_get_uint(env, term, &value)) return 0;
    var = (uint16_t) value;
    return 1;
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, uint32_t &var) {
    return enif_get_uint(env, term, &var);
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, long long unsigned int &var) {
    return enif_get_uint64(env, term, (uint64_t *) &var);
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, long long int &var) {
    return enif_get_int64(env, term, (long int*) &var);
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, int8_t &var) {
    int value;
    if(!enif_get_int(env, term, &value)) return 0;
    var = (int8_t) value;
    return 1;
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, int16_t &var) {
    int value;
    if(!enif_get_int(env, term, &value)) return 0;
    var = (int16_t) value;
    return 1;
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, int32_t &var) {
    return enif_get_int(env, term, &var);
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, int64_t &var) {
    return enif_get_int64(env, term, &var);
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, float &var) {
    double value;
    if(!enif_get_double(env, term, &value)) return 0;
    var = (float) value;
    return 1;
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, double &var) {
    return enif_get_double(env, term, &var);
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, std::string &var){
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

  int get_vector_tuple(ErlNifEnv* env, ERL_NIF_TERM tuple, std::vector<long long int> &var) {
    const ERL_NIF_TERM *terms;
    int length;
    if(!enif_get_tuple(env, tuple, &length, &terms)) return 0;
    for(int i=0;i<length;i++){
      int data;
      if(!get(env, terms[i], data)) return 0;
      var.push_back(data);
    }
    return 1;
  }

  int get_vector_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<ErlNifBinary> &var){
    ERL_NIF_TERM head, tail;
    while(enif_get_list_cell(env, list, &head, &tail)){
      ErlNifBinary elem;
      if(!get(env, head, elem)) return 0;
      var.push_back(elem);
      list = tail;
    }
    return 1;
  }

  int get_vector_list(ErlNifEnv* env, ERL_NIF_TERM list, std::vector<long long int> &var){
    ERL_NIF_TERM head, tail;
    while(enif_get_list_cell(env, list, &head, &tail)){
      long long int elem;
      if(!get(env, head, elem)) return 0;
      var.push_back(elem);
      list = tail;
    }
    return 1;
  }

  int get_atom(ErlNifEnv* env, ERL_NIF_TERM term, std::string &var){
    unsigned atom_length;
    if(!enif_get_atom_length(env, term, &atom_length, ERL_NIF_LATIN1)) return 0;

    var.resize(atom_length+1);

    if(!enif_get_atom(env, term, &(*(var.begin())), var.size(), ERL_NIF_LATIN1)) return 0;

    var.resize(atom_length);

    return 1;
  }

  int get_type(ErlNifEnv* env, ERL_NIF_TERM term, xla::PrimitiveType &type){
    std::string type_str;
    if(!get(env, term, type_str)) return 0;

    xla::StatusOr<xla::PrimitiveType> type_status = xla::primitive_util::StringToPrimitiveType(type_str);
    if(!type_status.ok()) {
      return 0;
    }
    type = type_status.ConsumeValueOrDie();
    return 1;
  }

  ERL_NIF_TERM make(ErlNifEnv* env, ErlNifBinary &var) {
    return enif_make_binary(env, &var);
  }

  ERL_NIF_TERM make(ErlNifEnv* env, int &var) { return enif_make_int(env, var); }
  ERL_NIF_TERM make(ErlNifEnv* env, std::string &var){ return enif_make_string(env, var.c_str(), ERL_NIF_LATIN1); }
  ERL_NIF_TERM make(ErlNifEnv* env, const char* string){ return enif_make_string(env, string, ERL_NIF_LATIN1); }
}
