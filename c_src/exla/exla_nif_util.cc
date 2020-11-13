#include "tensorflow/compiler/xla/exla/exla_nif_util.h"

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

  int get(ErlNifEnv* env, ERL_NIF_TERM term, int &var){
    return enif_get_int(env, term, &var);
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, long int &var){
    return enif_get_int64(env, term, &var);
  }

  int get(ErlNifEnv* env, ERL_NIF_TERM term, long long int &var){
    return enif_get_int64(env, term, (long int*) &var);
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

  int get_options(ErlNifEnv* env, const ERL_NIF_TERM terms[], xla::ExecutableRunOptions &options){
    // TODO: Allow better specification of allocator configuration.
    auto* allocator = new xla::ExlaAllocator(xla::PlatformUtil::GetPlatform("host").ConsumeValueOrDie());
    options.set_allocator(allocator);
    return 1;
  }
  int get_options(ErlNifEnv* env, const ERL_NIF_TERM terms[], xla::ExecutableBuildOptions &options){ return 1; }
  int get_options(ErlNifEnv* env, const ERL_NIF_TERM terms[], xla::LocalClientOptions &options){

    stream_executor::Platform* platform;
    int number_of_replicas, intra_op_parallelism_threads;

    // TODO: Adjust this to accept `int` which is passed from Elixir
    if(!exla::get_platform(env, terms[0], platform)) return 0;
    if(!exla::get(env, terms[1], number_of_replicas)) return 0;
    if(!exla::get(env, terms[2], intra_op_parallelism_threads)) return 0;

    options.set_platform(platform);
    options.set_number_of_replicas(number_of_replicas);
    options.set_intra_op_parallelism_threads(intra_op_parallelism_threads);

    return 1;
  }

  // TODO: Accept list, not tuple
  int get_argument_layouts(ErlNifEnv* env, ERL_NIF_TERM tuple, absl::Span<xla::Shape*> &span){
    const ERL_NIF_TERM* elems;
    int num_elems;
    if(!enif_get_tuple(env, tuple, &num_elems, &elems)) return 0;
    xla::Shape* data[num_elems];
    for(int i=0;i<num_elems;i++){
      xla::Shape* elem;
      if(!get<xla::Shape>(env, elems[i], elem)) return 0;
      data[i] = elem;
    }
    span = absl::Span<xla::Shape*>(data, num_elems);
    return 1;
  }

  int get_arguments(ErlNifEnv* env, ERL_NIF_TERM tuple, std::vector<xla::ShapedBuffer*> &input){
    const ERL_NIF_TERM* elems;
    int num_elems;
    if(!enif_get_tuple(env, tuple, &num_elems, &elems)) return 0;
    input.clear();
    for(int i=0;i<num_elems;i++){
      xla::ShapedBuffer* elem;
      if(!get<xla::ShapedBuffer>(env, elems[i], elem)) return 0;
      input.push_back(elem);
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

  // TODO: Handle statusor gracefully.
  int get_platform(ErlNifEnv* env, ERL_NIF_TERM term, stream_executor::Platform* platform){
    std::string platform_str;

    if(!get_atom(env, term, platform_str)){
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

  // TODO: No need for this. Match in Elixir.
  int get_type(ErlNifEnv* env, ERL_NIF_TERM term, xla::PrimitiveType &type){
    std::string type_str;

    if(!get_atom(env, term, type_str)) {
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

  ERL_NIF_TERM make(ErlNifEnv* env, int &var) { return enif_make_int(env, var); }
  ERL_NIF_TERM make(ErlNifEnv* env, std::string &var){ return enif_make_string(env, var.c_str(), ERL_NIF_LATIN1); }
  ERL_NIF_TERM make(ErlNifEnv* env, const char* string){ return enif_make_string(env, string, ERL_NIF_LATIN1); }
}
