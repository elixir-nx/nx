#include "exla/exla_nif_util.h"

ERL_NIF_TERM ExlaNifUtil::error(ErlNifEnv* env, const char* msg){
  return enif_make_tuple2(env, enif_make_atom(env, "error"), enif_make_string(env, msg, ERL_NIF_LATIN1));
}

template <typename T>
int ExlaNifUtil::open_resource(ErlNifEnv* env, const char* module, const char* name, ErlNifResourceDtor* dtor){
  ErlNifResourceFlags flags = ErlNifResourceFlags(ERL_NIF_RT_CREATE|ERL_NIF_RT_TAKEOVER);
  ErlNifResourceFlags* tried = nullptr;

  ErlNifResourceType* type = enif_open_resource_type(env, module, name, dtor, flags, tried);

  if(!type){
    ExlaNifUtil::resource_object<T>::type = 0;
    return 0;
  } else {
    ExlaNifUtil::resource_object<T>::type = type;
    return 1;
  }
}

int ExlaNifUtil::get(ErlNifEnv* env, ERL_NIF_TERM term, int &var){
  return enif_get_int(env, term, &var);
}

int ExlaNifUtil::get(ErlNifEnv* env, ERL_NIF_TERM term, long int &var){
  return enif_get_int64(env, term, &var);
}

int ExlaNifUtil::get(ErlNifEnv* env, ERL_NIF_TERM term, long long int &var){
  return enif_get_int64(env, term, (long int*) &var);
}

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

int ExlaNifUtil::get_options(ErlNifEnv* env, ERL_NIF_TERM term, xla::ExecutableRunOptions &options){ return 0; }
int ExlaNifUtil::get_options(ErlNifEnv* env, ERL_NIF_TERM term, xla::ExecutableBuildOptions &options){ return 0; }
int ExlaNifUtil::get_options(ErlNifEnv* env, ERL_NIF_TERM term, xla::LocalClientOptions &options){ return 0; }

int ExlaNifUtil::get_atom(ErlNifEnv* env, ERL_NIF_TERM term, std::string &var){
  unsigned atom_length;
  if(!enif_get_atom_length(env, term, &atom_length, ERL_NIF_LATIN1)) return 0;

  var.resize(atom_length+1);

  if(!enif_get_atom(env, term, &(*(var.begin())), var.size(), ERL_NIF_LATIN1)) return 0;

  var.resize(atom_length);

  return 1;
}

template <typename T>
int ExlaNifUtil::get(ErlNifEnv* env, ERL_NIF_TERM term, T* var){
  return enif_get_resource(env, term, ExlaNifUtil::resource_object<T>::type, (void **) &var);
}

// TODO: Handle statusor gracefully.
int ExlaNifUtil::get_platform(ErlNifEnv* env, ERL_NIF_TERM term, stream_executor::Platform* platform){
  std::string platform_str;

  if(!ExlaNifUtil::get_atom(env, term, platform_str)){
    return 0;
  }

  if(platform_str.compare("host") == 0){
    platform = xla::PlatformUtil::GetPlatform("Host").ConsumeValueOrDie();
    return 1;
  } else if(platform_str.compare("cuda") == 0){
    platform = xla::PlatformUtil::GetPlatform("Cuda").ConsumeValueOrDie();
    return 1;
  } else {
    return 0;
  }
}

int ExlaNifUtil::get_type(ErlNifEnv* env, ERL_NIF_TERM term, xla::PrimitiveType &type){
  std::string type_str;

  if(!ExlaNifUtil::get_atom(env, term, type_str)) {
    std::cout << type_str << std::endl;
    type = xla::PrimitiveType::PRIMITIVE_TYPE_INVALID;
    return 1;
  }

  if(type_str.compare("pred") == 0){
    type = xla::PrimitiveType::PRED;
  } else if(type_str.compare("int8") == 0){
    type = xla::PrimitiveType::S8;
  } else if(type_str.compare("int16") == 0){
    type = xla::PrimitiveType::S16;
  } else if(type_str.compare("int32") == 0){
    type = xla::PrimitiveType::S32;
  } else if(type_str.compare("int64") == 0){
    type = xla::PrimitiveType::S64;
  } else if(type_str.compare("uint8") == 0){
    type = xla::PrimitiveType::U8;
  } else if(type_str.compare("uint16") == 0){
    type = xla::PrimitiveType::U16;
  } else if(type_str.compare("uint32") == 0){
    type = xla::PrimitiveType::U32;
  } else if(type_str.compare("uint64") == 0){
    type = xla::PrimitiveType::U64;
  } else if(type_str.compare("float16") == 0){
    type = xla::PrimitiveType::F16;
  } else if(type_str.compare("bfloat16") == 0){
    type = xla::PrimitiveType::BF16;
  } else if(type_str.compare("float32") == 0){
    type = xla::PrimitiveType::F32;
  } else if(type_str.compare("float64") == 0){
    type = xla::PrimitiveType::F64;
  } else if(type_str.compare("complex64") == 0){
    type = xla::PrimitiveType::C64;
  } else if(type_str.compare("complex128") == 0){
    type = xla::PrimitiveType::C128;
  } else if(type_str.compare("tuple") == 0){
    type = xla::PrimitiveType::TUPLE;
  } else if(type_str.compare("opaque") == 0){
    type = xla::PrimitiveType::OPAQUE_TYPE;
  } else if(type_str.compare("token") == 0){
    type = xla::PrimitiveType::TOKEN;
  } else {
    type = xla::PrimitiveType::PRIMITIVE_TYPE_INVALID;
  }
  return 1;
}

template <typename T>
int get_span(ErlNifEnv* env, ERL_NIF_TERM tuple, absl::Span<T> &span){
  const ERL_NIF_TERM* elems;
  int num_elems;
  if(!enif_get_tuple(env, tuple, &num_elems, &elems)) return 0;
  T data[num_elems];
  for(int i=0;i<num_elems;i++){
    T elem;
    if(!ExlaNifUtil::get(env, elems[i], elem)) return 0;
    data[i] = elem;
  }
  span = absl::Span<T>(data, num_elems);
  return 1;
}

ERL_NIF_TERM ExlaNifUtil::make(ErlNifEnv* env, std::string &var){ return enif_make_string(env, var.c_str(), ERL_NIF_LATIN1); }

template <typename T>
ERL_NIF_TERM ExlaNifUtil::make(ErlNifEnv* env, T &var){
  void* ptr = enif_alloc_resource(ExlaNifUtil::resource_object<T>::type, sizeof(T));
  new(ptr) T(std::move(var));
  ERL_NIF_TERM ret = enif_make_resource(env, ptr);
  return ret;
}

template <typename T>
ERL_NIF_TERM ExlaNifUtil::make(ErlNifEnv* env, std::unique_ptr<T> &var){
  void* ptr = enif_alloc_resource(ExlaNifUtil::resource_object<T>::type, sizeof(T));
  T* value = var.release();
  new(ptr) T(std::move(*value));
  return enif_make_resource(env, ptr);
}