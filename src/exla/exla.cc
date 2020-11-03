#include "absl/types/span.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include <erl_nif.h>

ErlNifResourceType *OP_RES_TYPE, *SHAPE_RES_TYPE, *COMPUTATION_RES_TYPE, *LITERAL_RES_TYPE, *GLOBAL_DATA_RES_TYPE, *LOCAL_EXECUTABLE_RES_TYPE;

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

typedef struct {
  std::shared_ptr<xla::GlobalData> data;
} Data;

typedef struct {
  std::shared_ptr<xla::LocalExecutable> local_executable;
} LocalExecutable;

// Leaving these here for the time being.
void free_op(ErlNifEnv* env, void* obj){return;}
void free_shape(ErlNifEnv* env, void* obj){return;}
void free_computation(ErlNifEnv* env, void* obj){return;}
void free_literal(ErlNifEnv* env, void* obj){return;}
void free_global_data(ErlNifEnv* env, void* obj){return;}
void free_local_executable(ErlNifEnv* env, void* obj){return;}

static int open_resources(ErlNifEnv* env) {
  const char* mod = "XLA";
  const char* name_op = "Op";
  const char* name_shape = "Shape";
  const char* name_computation = "Computation";
  const char* name_literal = "Literal";
  const char* name_global_data = "GlobalData";
  const char* name_local_executable = "LocalExectuable";

  int flags = ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER;

  OP_RES_TYPE = enif_open_resource_type(env, mod, name_op, free_op, (ErlNifResourceFlags) flags, NULL);
  SHAPE_RES_TYPE = enif_open_resource_type(env, mod, name_shape, free_shape, (ErlNifResourceFlags) flags, NULL);
  COMPUTATION_RES_TYPE = enif_open_resource_type(env, mod, name_computation, free_computation, (ErlNifResourceFlags) flags, NULL);
  LITERAL_RES_TYPE = enif_open_resource_type(env, mod, name_literal, free_literal, (ErlNifResourceFlags) flags, NULL);
  GLOBAL_DATA_RES_TYPE = enif_open_resource_type(env, mod, name_global_data, free_global_data, (ErlNifResourceFlags) flags, NULL);
  LOCAL_EXECUTABLE_RES_TYPE = enif_open_resource_type(env, mod, name_local_executable, free_local_executable, (ErlNifResourceFlags) flags, NULL);

  if(OP_RES_TYPE == NULL || SHAPE_RES_TYPE == NULL || COMPUTATION_RES_TYPE == NULL || LITERAL_RES_TYPE == NULL || LOCAL_EXECUTABLE_RES_TYPE == NULL) return -1;
  return 0;
}

static int load(ErlNifEnv* env, void** priv, ERL_NIF_TERM load_info){
  if(open_resources(env) == -1) return -1;

  ok = enif_make_atom(env, "ok");
  bad = enif_make_atom(env, "error");

  XLA* xla_objects;
  xla_objects = (XLA*) enif_alloc(sizeof(XLA));
  xla_objects->builder = new xla::XlaBuilder("Elixir");
  xla_objects->client = NULL;

  *priv = (void*) xla_objects;

  return 0;
}

xla::PrimitiveType enif_get_primitive_type(ErlNifEnv* env, ERL_NIF_TERM term){
  unsigned atom_length;
  if(!enif_get_atom_length(env, term, &atom_length, ERL_NIF_LATIN1)) return xla::PrimitiveType::PRIMITIVE_TYPE_INVALID;

  std::string atom_str;
  atom_str.resize(atom_length+1);

  if(!enif_get_atom(env, term, &(*(atom_str.begin())), atom_str.size(), ERL_NIF_LATIN1)) return xla::PrimitiveType::PRIMITIVE_TYPE_INVALID;

  atom_str.resize(atom_length);

  if(atom_str.compare("pred") == 0){
    return xla::PrimitiveType::PRED;
  } else if(atom_str.compare("int8") == 0){
    return xla::PrimitiveType::S8;
  } else if(atom_str.compare("int16") == 0){
    return xla::PrimitiveType::S16;
  } else if(atom_str.compare("int32") == 0){
    return xla::PrimitiveType::S32;
  } else if(atom_str.compare("int64") == 0){
    return xla::PrimitiveType::S64;
  } else if(atom_str.compare("uint8") == 0){
    return xla::PrimitiveType::U8;
  } else if(atom_str.compare("uint16") == 0){
    return xla::PrimitiveType::U16;
  } else if(atom_str.compare("uint32") == 0){
    return xla::PrimitiveType::U32;
  } else if(atom_str.compare("uint64") == 0){
    return xla::PrimitiveType::U64;
  } else if(atom_str.compare("float16") == 0){
    return xla::PrimitiveType::F16;
  } else if(atom_str.compare("bfloat16") == 0){
    return xla::PrimitiveType::BF16;
  } else if(atom_str.compare("float32") == 0){
    return xla::PrimitiveType::F32;
  } else if(atom_str.compare("float64") == 0){
    return xla::PrimitiveType::F64;
  } else if(atom_str.compare("complex64") == 0){
    return xla::PrimitiveType::C64;
  } else if(atom_str.compare("complex128") == 0){
    return xla::PrimitiveType::C128;
  } else if(atom_str.compare("tuple") == 0){
    return xla::PrimitiveType::TUPLE;
  } else if(atom_str.compare("opaque") == 0){
    return xla::PrimitiveType::OPAQUE_TYPE;
  } else if(atom_str.compare("token") == 0){
    return xla::PrimitiveType::TOKEN;
  } else {
    return xla::PrimitiveType::PRIMITIVE_TYPE_INVALID;
  }
}

int enif_get_std_string(ErlNifEnv* env, ERL_NIF_TERM term, std::string &var){
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

ERL_NIF_TERM enif_make_op(ErlNifEnv* env, xla::XlaOp value){
  void* ptr = enif_alloc_resource(OP_RES_TYPE, sizeof(xla::XlaOp));
  new(ptr) xla::XlaOp(value);
  return enif_make_resource(env, ptr);
}

ERL_NIF_TERM enif_make_shape(ErlNifEnv* env, xla::Shape value){
  void* ptr = enif_alloc_resource(SHAPE_RES_TYPE, sizeof(xla::Shape));
  new(ptr) xla::Shape(value);
  return enif_make_resource(env, ptr);
}

ERL_NIF_TERM enif_make_computation(ErlNifEnv* env, xla::XlaComputation value){
  void* ptr = enif_alloc_resource(COMPUTATION_RES_TYPE, sizeof(xla::XlaComputation));
  new(ptr) xla::XlaComputation(std::move(value));
  return enif_make_resource(env, ptr);
}

ERL_NIF_TERM enif_make_literal(ErlNifEnv* env, xla::Literal& value){
  void* ptr = enif_alloc_resource(LITERAL_RES_TYPE, sizeof(xla::Literal));
  new(ptr) xla::Literal(std::move(value));
  return enif_make_resource(env, ptr);
}

ERL_NIF_TERM enif_make_global_data(ErlNifEnv* env, std::unique_ptr<xla::GlobalData>& value){
  xla::GlobalData* ptr = (xla::GlobalData*) enif_alloc_resource(GLOBAL_DATA_RES_TYPE, sizeof(xla::GlobalData*));
  std::shared_ptr<xla::GlobalData> sptr = std::move(value);
  std::memmove(ptr, sptr.get(), sizeof(xla::GlobalData*));
  return enif_make_resource(env, ptr);
}

ERL_NIF_TERM enif_make_local_executable(ErlNifEnv* env, std::unique_ptr<xla::LocalExecutable>& value){
  LocalExecutable* ptr = (LocalExecutable*) enif_alloc_resource(LOCAL_EXECUTABLE_RES_TYPE, sizeof(LocalExecutable));
  ptr->local_executable = std::move(value);
  return enif_make_resource(env, ptr);
}

// TODO: Template this.
// TODO: This should return an integer status instead of the span and rather accept a reference to the Span.
// TODO: We need to exit gracefully on badarg.
absl::Span<long long int> enif_get_span(ErlNifEnv* env, ERL_NIF_TERM list){
  ERL_NIF_TERM head, tail;
  std::vector<long long int> values;
  int i = 0;
  while(enif_get_list_cell(env, list, &head, &tail)){
    long int placeholder;
    enif_get_int64(env, head, &placeholder);
    values.insert(values.begin() + (i++), placeholder);
    list = tail;
  }
  return absl::Span<long long int>(values);
}
// TODO: Template this with above!
absl::Span<xla::ShapedBuffer*> enif_get_arguments(ErlNifEnv* env, ERL_NIF_TERM tuple){
  return absl::Span<xla::ShapedBuffer*>();
}

// TODO: Template this with above!
absl::Span<xla::Shape*> enif_get_argument_layouts(ErlNifEnv* env, ERL_NIF_TERM tuple){
  const ERL_NIF_TERM* arg_layouts;
  int num_arg_layouts;
  enif_get_tuple(env, tuple, &num_arg_layouts, &arg_layouts);
  xla::Shape* argument_layouts[num_arg_layouts];
  for(int i=0;i<num_arg_layouts;i++){
    xla::Shape* shape;
    enif_get_resource(env, arg_layouts[i], GLOBAL_DATA_RES_TYPE, (void **) &shape);
    argument_layouts[i] = shape;
  }
  return absl::Span<xla::Shape*>(argument_layouts, num_arg_layouts);
}

xla::ExecutableBuildOptions enif_get_executable_build_options(ErlNifEnv* env, ERL_NIF_TERM options){
  return xla::ExecutableBuildOptions();
}

xla::ExecutableRunOptions enif_get_executable_run_options(ErlNifEnv* env, ERL_NIF_TERM options){
  return xla::ExecutableRunOptions();
}

/************************ xla::Shape Functions ***************************/
ERL_NIF_TERM make_scalar_shape(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 1){
    return enif_make_badarg(env);
  }

  xla::PrimitiveType element_type = enif_get_primitive_type(env, argv[0]);
  xla::Shape shape = xla::ShapeUtil::MakeScalarShape(element_type);
  return enif_make_shape(env, shape);
}

ERL_NIF_TERM human_string(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 1){
    return enif_make_badarg(env);
  }

  xla::Shape* shape;
  if(!enif_get_resource(env, argv[0], SHAPE_RES_TYPE, (void **) &shape)) return enif_make_badarg(env);
  std::string result = xla::ShapeUtil::HumanString(*shape);
  return enif_make_string(env, result.c_str(), ERL_NIF_LATIN1);
}

/*********************** xla::LiteralUtil Functions *************************/
ERL_NIF_TERM create_r0(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 1){
    return enif_make_badarg(env);
  }
  // TODO: Handle other types.
  int data;
  enif_get_int(env, argv[0], &data);
  xla::Literal literal = xla::LiteralUtil::CreateR0(data);
  return enif_make_literal(env, literal);
}

// This is mainly for testing purposes
ERL_NIF_TERM literal_to_string(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 1){
    return enif_make_badarg(env);
  }

  xla::Literal* literal;
  enif_get_resource(env, argv[0], LITERAL_RES_TYPE, (void **) &literal);
  std::string literal_str = (*literal).ToString();
  return enif_make_string(env, literal_str.c_str(), ERL_NIF_LATIN1);
}

/************************ xla::XlaOp Functions ***************************/
ERL_NIF_TERM parameter(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 3){
    return enif_make_badarg(env);
  }

  XLA* xla_objects = (XLA*) enif_priv_data(env);

  long int param_num;
  xla::Shape* shape;
  std::string name;

  if(!enif_get_int64(env, argv[0], &param_num)) return enif_make_badarg(env);
  if(!enif_get_resource(env, argv[1], SHAPE_RES_TYPE, (void **) &shape)) return enif_make_badarg(env);
  if(!enif_get_std_string(env, argv[2], name)) return enif_make_badarg(env);

  xla::XlaOp op = xla::Parameter(xla_objects->builder, param_num, *shape, name);
  return enif_make_op(env, op);
}

/* Stub for element-wise binary functions */
ERL_NIF_TERM xla_binary_op(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[], xla::XlaOp(*lambda)(xla::XlaOp, xla::XlaOp, absl::Span<const long long int>)){
  if(argc != 3){
    return enif_make_badarg(env);
  }

  xla::XlaOp *lhs, *rhs;
  if(!enif_get_resource(env, argv[0], OP_RES_TYPE, (void **) &lhs)) return enif_make_badarg(env);
  if(!enif_get_resource(env, argv[1], OP_RES_TYPE, (void **) &rhs)) return enif_make_badarg(env);

  absl::Span<const long long int> broadcast_dims = enif_get_span(env, argv[2]);
  xla::XlaOp result = lambda(*lhs, *rhs, broadcast_dims);
  return enif_make_op(env, result);
}

ERL_NIF_TERM add(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::Add);}
ERL_NIF_TERM sub(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::Sub);}
ERL_NIF_TERM mul(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::Mul);}
ERL_NIF_TERM div(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::Div);}
ERL_NIF_TERM rem(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::Rem);}
ERL_NIF_TERM min(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::Min);}
ERL_NIF_TERM max(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::Max);}
ERL_NIF_TERM logical_and(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::And);}
ERL_NIF_TERM logical_or(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::Or);}
ERL_NIF_TERM logical_xor(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::Xor);}
ERL_NIF_TERM shift_left(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::ShiftLeft);}
ERL_NIF_TERM shift_right_logical(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::ShiftRightLogical);}
ERL_NIF_TERM shift_right_arithmetic(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::ShiftRightArithmetic);}
ERL_NIF_TERM eq(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::Eq);}
ERL_NIF_TERM eq_total_order(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::EqTotalOrder);}
ERL_NIF_TERM ne(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::Ne);}
ERL_NIF_TERM ne_total_order(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::NeTotalOrder);}
ERL_NIF_TERM ge(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::Ge);}
ERL_NIF_TERM ge_total_order(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::GeTotalOrder);}
ERL_NIF_TERM gt(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::Gt);}
ERL_NIF_TERM gt_total_order(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::GtTotalOrder);}
ERL_NIF_TERM lt(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::Lt);}
ERL_NIF_TERM lt_total_order(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::LtTotalOrder);}
ERL_NIF_TERM le(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::Le);}
ERL_NIF_TERM le_total_order(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::LeTotalOrder);}
ERL_NIF_TERM pow(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::Pow);}
ERL_NIF_TERM complex(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::Complex);}
ERL_NIF_TERM atan2(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_binary_op(env, argc, argv, xla::Atan2);}

/* Stub for element-wise unary functions */
ERL_NIF_TERM xla_unary_op(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[], xla::XlaOp(*lambda)(xla::XlaOp)){
  if(argc != 1){
    return enif_make_badarg(env);
  }

  xla::XlaOp *op;
  if(!enif_get_resource(env, argv[0], OP_RES_TYPE, (void **) &op)) return enif_make_badarg(env);

  xla::XlaOp result = lambda(*op);
  return enif_make_op(env, result);
}

ERL_NIF_TERM abs(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::Abs);}
ERL_NIF_TERM exp(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::Exp);}
ERL_NIF_TERM expm1(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::Expm1);}
ERL_NIF_TERM floor(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::Floor);}
ERL_NIF_TERM ceil(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::Ceil);}
ERL_NIF_TERM round(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::Round);}
ERL_NIF_TERM log(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::Log);}
ERL_NIF_TERM log1p(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::Log1p);}
ERL_NIF_TERM logistic(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::Logistic);}
ERL_NIF_TERM sign(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::Sign);}
ERL_NIF_TERM clz(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::Clz);}
ERL_NIF_TERM cos(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::Cos);}
ERL_NIF_TERM sin(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::Sin);}
ERL_NIF_TERM tanh(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::Tanh);}
ERL_NIF_TERM real(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::Real);}
ERL_NIF_TERM imag(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::Imag);}
ERL_NIF_TERM sqrt(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::Sqrt);}
ERL_NIF_TERM cbrt(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::Cbrt);}
ERL_NIF_TERM rsqrt(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::Rsqrt);}
ERL_NIF_TERM is_finite(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::IsFinite);}
ERL_NIF_TERM logical_not(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::Not);}
ERL_NIF_TERM neg(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::Neg);}
ERL_NIF_TERM conj(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::Conj);}
ERL_NIF_TERM population_count(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::PopulationCount);}
ERL_NIF_TERM copy(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){return xla_unary_op(env, argc, argv, xla::Copy);}

// Constant Creation Methods
ERL_NIF_TERM constant_r0(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 1){
    return enif_make_badarg(env);
  }

  XLA* xla_objects = (XLA*) enif_priv_data(env);
  int value;
  enif_get_int(env, argv[0], &value);
  xla::XlaOp op = xla::ConstantR0(xla_objects->builder, value);
  return enif_make_op(env, op);
}

ERL_NIF_TERM constant_r1_from_list(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 1){
    return enif_make_badarg(env);
  }

  XLA* xla_objects = (XLA*) enif_priv_data(env);
  absl::Span<const long long int> values = enif_get_span(env, argv[0]);
  xla::XlaOp op = xla::ConstantR1(xla_objects->builder, values);
  return enif_make_op(env, op);
}

ERL_NIF_TERM constant_r1_fill(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 2){
    return enif_make_badarg(env);
  }

  XLA* xla_objects = (XLA*) enif_priv_data(env);
  int length, value;
  enif_get_int(env, argv[0], &length);
  enif_get_int(env, argv[1], &value);
  xla::XlaOp op = xla::ConstantR1(xla_objects->builder, length, value);
  return enif_make_op(env, op);
}

ERL_NIF_TERM dot(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  xla::XlaOp *lhs, *rhs;
  enif_get_resource(env, argv[0], OP_RES_TYPE, (void **) &lhs);
  enif_get_resource(env, argv[1], OP_RES_TYPE, (void **) &rhs);
  // TODO: Handle Precision Configuration
  xla::XlaOp result = xla::Dot(*lhs, *rhs);
  return enif_make_op(env, result);
}

/************************ xla::ClientLibrary Functions ***************************/
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

ERL_NIF_TERM get_device_count(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  XLA* xla_objects = (XLA*) enif_priv_data(env);
  if(xla_objects->client == NULL){
    return enif_make_tuple2(env, bad, enif_make_string(env, "No client available. Try creating one with get_or_create_local_client/0.", ERL_NIF_LATIN1));
  }

  int device_count = xla_objects->client->device_count();
  return enif_make_int(env, device_count);
}

/************ Build, Compilation, Execution *************/
ERL_NIF_TERM build(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  XLA* xla_objects = (XLA*) enif_priv_data(env);
  xla::StatusOr<xla::XlaComputation> computation_status = xla_objects->builder->Build();
  // TODO: Handle StatusOr more gracefully.
  return enif_make_computation(env, computation_status.ConsumeValueOrDie());
}

ERL_NIF_TERM compile(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  XLA* xla_objects = (XLA*) enif_priv_data(env);
  if(xla_objects->client == NULL){
    return enif_make_tuple2(env, bad, enif_make_string(env, "No client found. Try creating one with get_or_create_local_client/0.", ERL_NIF_LATIN1));
  }

  xla::XlaComputation* computation;
  // TODO: Handle Args
  enif_get_resource(env, argv[0], COMPUTATION_RES_TYPE, (void **) &computation);
  absl::Span<xla::Shape*> argument_layouts = enif_get_argument_layouts(env, argv[1]);
  xla::ExecutableBuildOptions options = enif_get_executable_build_options(env, argv[2]);

  xla::StatusOr<std::vector<std::unique_ptr<xla::LocalExecutable>>> exec_status = xla_objects->client->Compile(*computation, argument_layouts, options);
  // TODO: Handle this gracefully.
  std::vector<std::unique_ptr<xla::LocalExecutable>> executables = exec_status.ConsumeValueOrDie();
  ERL_NIF_TERM exec_refs[executables.size()];
  int i = 0;
  for(auto it=std::begin(executables);it!=std::end(executables);++it){
    ERL_NIF_TERM exec_term = enif_make_local_executable(env, *it);
    exec_refs[i++] = exec_term;
  }
  return enif_make_list_from_array(env, exec_refs, i);
}

ERL_NIF_TERM run(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  LocalExecutable* local_executable;
  // TODO: Handle Args
  enif_get_resource(env, argv[0], LOCAL_EXECUTABLE_RES_TYPE, (void **) &local_executable);
  absl::Span<xla::ShapedBuffer*> arguments = enif_get_arguments(env, argv[1]);
  xla::ExecutableRunOptions run_options = enif_get_executable_run_options(env, argv[2]);

  xla::StatusOr<xla::ScopedShapedBuffer> run_status = local_executable->local_executable->Run(arguments, run_options);
  return ok;
}

/*********** HLO Methods *************/
xla::StatusOr<std::unique_ptr<xla::HloModule>> get_hlo_module(const xla::XlaComputation& computation){
  xla::StatusOr<xla::HloModuleConfig> module_config = xla::HloModule::CreateModuleConfigFromProto(computation.proto(), xla::GetDebugOptionsFromFlags());
  // TODO: Handle this gracefully
  xla::StatusOr<std::unique_ptr<xla::HloModule>> module = xla::HloModule::CreateFromProto(computation.proto(), module_config.ConsumeValueOrDie());
  // TODO: Handle this gracefully.
  return module.ConsumeValueOrDie();
}

ERL_NIF_TERM get_computation_hlo_text(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  // TODO: Handle this gracefully
  xla::XlaComputation* computation;
  enif_get_resource(env, argv[0], COMPUTATION_RES_TYPE, (void **) &computation);
  xla::StatusOr<std::unique_ptr<xla::HloModule>> hlo_module_status = get_hlo_module(*computation);
  // // TODO: Handle this gracefully
  std::unique_ptr<xla::HloModule> hlo_module = hlo_module_status.ConsumeValueOrDie();

  xla::HloPrintOptions options;
  options = xla::HloPrintOptions::ShortParsable();
  options.set_print_large_constants(false);
  std::string result = hlo_module->ToString(options);
  return enif_make_string(env, result.c_str(), ERL_NIF_LATIN1);
}

ERL_NIF_TERM get_computation_hlo_proto(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  xla::XlaComputation* computation;
  enif_get_resource(env, argv[0], COMPUTATION_RES_TYPE, (void **) &computation);
  std::string result;
  (*computation).proto().SerializeToString(&result);
  return enif_make_string(env, result.c_str(), ERL_NIF_LATIN1);
}

static ErlNifFunc exla_funcs[] = {
  /****** xla::Client ******/
  {"get_or_create_local_client", 0, get_or_create_local_client},
  {"get_device_count", 0, get_device_count},
  /****** xla::Shape ******/
  {"human_string", 1, human_string},
  {"make_scalar_shape", 1, make_scalar_shape},
  {"parameter", 3, parameter},
  /****** xla::Literal ******/
  {"create_r0", 1, create_r0},
  {"literal_to_string", 1, literal_to_string},
  /****** Binary Ops ******/
  {"add", 3, add},
  {"sub", 3, sub},
  {"mul", 3, mul},
  {"div", 3, div},
  {"rem", 3, rem},
  {"min", 3, min},
  {"max", 3, max},
  {"logical_and", 3, logical_and},
  {"logical_or", 3, logical_or},
  {"logical_xor", 3, logical_xor},
  {"shift_left", 3, shift_left},
  {"shift_right_logical", 3, shift_right_logical},
  {"shift_right_arithmetic", 3, shift_right_arithmetic},
  {"eq", 3, eq},
  {"eq_total_order", 3, eq_total_order},
  {"ne", 3, ne},
  {"ne_total_order", 3, ne_total_order},
  {"gt", 3, gt},
  {"gt_total_order", 3, gt_total_order},
  {"ge", 3, ge},
  {"ge_total_order", 3, ge_total_order},
  {"lt", 3, lt},
  {"lt_total_order", 3, lt_total_order},
  {"le", 3, le},
  {"le_total_order", 3, le_total_order},
  {"pow", 3, pow},
  {"complex", 3, complex},
  {"atan2", 3, atan2},
  /****** Unary Ops ******/
  {"abs", 1, abs},
  {"exp", 1, exp},
  {"expm1", 1, expm1},
  {"floor", 1, floor},
  {"ceil", 1, ceil},
  {"round", 1, round},
  {"log", 1, log},
  {"log1p", 1, log1p},
  {"logistic", 1, logistic},
  {"sign", 1, sign},
  {"clz", 1, clz},
  {"cos", 1, cos},
  {"sin", 1, sin},
  {"tanh", 1, tanh},
  {"real", 1, real},
  {"imag", 1, imag},
  {"sqrt", 1, sqrt},
  {"rsqrt", 1, rsqrt},
  {"cbrt", 1, cbrt},
  {"is_finite", 1, is_finite},
  {"logical_not", 1, logical_not},
  {"neg", 1, neg},
  {"conj", 1, conj},
  {"population_count", 1, population_count},
  /******** Constant Creation Methods *******/
  {"constant_r0", 1, constant_r0},
  {"constant_r1", 1, constant_r1_from_list},
  {"constant_r1", 2, constant_r1_fill},
  /******** Other XLA Ops *******/
  {"dot", 2, dot},
  /******* Compilation, Execution, Etc. ******/
  {"build", 0, build},
  {"compile", 3, compile},
  {"run", 3, run},
  /******** HLO Functions ********/
  {"get_computation_hlo_proto", 1, get_computation_hlo_proto},
  {"get_computation_hlo_text", 1, get_computation_hlo_text}
};

ERL_NIF_INIT(Elixir.Exla, exla_funcs, &load, NULL, NULL, NULL);