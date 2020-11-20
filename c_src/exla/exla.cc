#include "tensorflow/compiler/xla/exla/exla_allocator.h"
#include "tensorflow/compiler/xla/exla/exla_nif_util.h"
#include "tensorflow/compiler/xla/exla/exla_macros.h"
#include "tensorflow/compiler/xla/exla/exla_client.h"

#include "absl/types/span.h"

#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/literal_util.h"

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/client_library.h"

// TODO: It might be more informative on the Elixir side to replace `enif_make_badarg` with something like `{:error, reason}`. Thoughts?
// TODO: In the same respect as above we could wrap essentially each value returning from a NIF with `ok`.
// TODO: Remove global instance of builder.
typedef struct {
  xla::XlaBuilder* builder;
} XLA;

// Once the resource is garbage collected, this should also deallocate the C++ object.
void free_op(ErlNifEnv* env, void* obj){delete (xla::XlaOp*) obj;}
void free_shape(ErlNifEnv* env, void* obj){delete (xla::Shape*) obj;}
void free_computation(ErlNifEnv* env, void* obj){delete (xla::XlaComputation*) obj;}
void free_literal(ErlNifEnv* env, void* obj){delete (xla::Literal*) obj;}
void free_local_executable(ErlNifEnv* env, void* obj){delete (xla::LocalExecutable*) obj;}
void free_shaped_buffer(ErlNifEnv* env, void* obj){delete (xla::ShapedBuffer*) obj;}
void free_builder(ErlNifEnv* env, void* obj){delete (xla::XlaBuilder*) obj;}
void free_exla_client(ErlNifEnv* env, void* obj){delete (exla::ExlaClient*) obj;}

static int open_resources(ErlNifEnv* env) {
  const char* mod = "EXLA";

  if(!exla::open_resource<xla::XlaOp>(env, mod, "Op", free_op)) return -1;
  if(!exla::open_resource<xla::Shape>(env, mod, "Shape", free_shape)) return -1;
  if(!exla::open_resource<xla::XlaComputation>(env, mod, "Computation", free_computation)) return -1;
  if(!exla::open_resource<xla::Literal>(env, mod, "Literal", free_literal)) return -1;
  if(!exla::open_resource<xla::LocalExecutable>(env, mod, "LocalExecutable", free_local_executable)) return -1;
  if(!exla::open_resource<xla::ShapedBuffer>(env, mod, "ShapedBuffer", free_shaped_buffer)) return -1;
  if(!exla::open_resource<xla::XlaBuilder*>(env, mod, "Builder", free_builder)) return -1;
  if(!exla::open_resource<exla::ExlaClient*>(env, mod, "ExlaClient", free_exla_client)) return -1;

  return 1;
}

static int load(ErlNifEnv* env, void** priv, ERL_NIF_TERM load_info){
  if(open_resources(env) == -1) return -1;

  XLA* xla_objects;
  xla_objects = (XLA*) enif_alloc(sizeof(XLA));

  xla_objects->builder = new xla::XlaBuilder("Elixir");

  *priv = (void*) xla_objects;

  return 0;
}

/************************* xla::XlaBuilder Functions ***********************/
ERL_NIF_TERM new_builder(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 1){
    return enif_make_badarg(env);
  }

  // TODO: Get this from binary
  std::string name;
  if(!exla::get(env, argv[0], name)) return enif_make_badarg(env);

  xla::XlaBuilder* builder = new xla::XlaBuilder(name);

  return exla::ok(env, exla::make<xla::XlaBuilder*>(env, builder));
}

ERL_NIF_TERM create_sub_builder(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 2){
    return exla::error(env, "Bad argument count.");
  }

  xla::XlaBuilder** builder;
  // TODO: Get this from binary
  std::string name;

  if(!exla::get<xla::XlaBuilder*>(env, argv[0], builder)) return exla::error(env, "Unable to get builder.");
  if(!exla::get(env, argv[1], name)) return exla::error(env, "Unable to get name.");

  std::unique_ptr<xla::XlaBuilder> uniq_sub_builder = (*builder)->CreateSubBuilder(name);
  xla::XlaBuilder* sub_builder = uniq_sub_builder.release();
  return exla::ok(env, exla::make<xla::XlaBuilder*>(env, sub_builder));
}

/************************ xla::ShapedBuffer Functions *********************/
// TODO: This function doesn't behave in the way it should. Rather than placing the binary directly on the device,
// it stages the binary in a ShapedBuffer and does the device transfer when passed as an argument to an `xla::LocalExecutable`
// We need to investigate `ScopedShapedBuffer` and `ExecutionInput` instead.
// TODO: Now that we have `ExlaClient`, we can implement the logic of this function in a function in that.
ERL_NIF_TERM binary_to_shaped_buffer(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 4){
    return enif_make_badarg(env);
  }

  ErlNifBinary bin;
  xla::Shape* shape;
  exla::ExlaClient **client;
  int device_ordinal;

  if(!exla::get<exla::ExlaClient*>(env, argv[0], client)) return enif_make_badarg(env);
  if(!enif_inspect_binary(env, argv[1], &bin)) return enif_make_badarg(env);
  if(!exla::get<xla::Shape>(env, argv[2], shape)) return enif_make_badarg(env);
  if(!exla::get(env, argv[3], device_ordinal)) return enif_make_badarg(env);

  stream_executor::DeviceMemoryAllocator* allocator = (*client)->allocator();

  const char *data_ptr = (char *) bin.data;
  int64_t data_size = bin.size;

  xla::BorrowingLiteral literal = xla::BorrowingLiteral(const_cast<char *>(data_ptr), *shape);

  EXLA_ASSIGN_OR_RETURN(auto scoped_buffer, (*client)->client()->LiteralToShapedBuffer(literal, device_ordinal, allocator), env);

  return exla::ok(env, exla::make<xla::ShapedBuffer>(env, scoped_buffer));
}

ERL_NIF_TERM shaped_buffer_to_binary(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 2) {
    return enif_make_badarg(env);
  }

  exla::ExlaClient **client;
  xla::ShapedBuffer *buffer;

  if(!exla::get<exla::ExlaClient*>(env, argv[0], client)) return enif_make_badarg(env);
  if(!exla::get<xla::ShapedBuffer>(env, argv[1], buffer)) return enif_make_badarg(env);

  const stream_executor::DeviceMemoryBase& mem_buffer = buffer->root_buffer();

  int64_t data_size = mem_buffer.size();
  const void *data_ptr = mem_buffer.opaque();

  ErlNifBinary binary;
  enif_alloc_binary(data_size, &binary);
  binary.data = (unsigned char *) malloc(data_size);
  std::memcpy((void *) binary.data, data_ptr, data_size);

  return exla::ok(env, enif_make_binary(env, &binary));
}

ERL_NIF_TERM on_host_shape(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 1){
    return enif_make_badarg(env);
  }

  xla::ShapedBuffer* buffer;

  if(!exla::get<xla::ShapedBuffer>(env, argv[0], buffer)) return exla::error(env, "Unable to get buffer.");

  xla::Shape shape = buffer->on_host_shape();

  return exla::ok(env, exla::make(env, shape));
}
/************************ xla::Shape Functions ***************************/
ERL_NIF_TERM make_shape(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 2){
    return enif_make_badarg(env);
  }

  xla::PrimitiveType element_type;
  std::vector<long long int> dims;

  if(!exla::get_type(env, argv[0], element_type)) return enif_make_badarg(env);
  if(!exla::get_vector<long long int>(env, argv[1], dims)) return enif_make_badarg(env);

  xla::Shape shape = xla::ShapeUtil::MakeShape(element_type, dims);
  return exla::ok(env, exla::make<xla::Shape>(env, shape));
}

ERL_NIF_TERM make_scalar_shape(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 1){
    return enif_make_badarg(env);
  }

  xla::PrimitiveType element_type;
  if(!exla::get_type(env, argv[0], element_type)) return enif_make_badarg(env);

  xla::Shape shape = xla::ShapeUtil::MakeScalarShape(element_type);
  return exla::make<xla::Shape>(env, shape);
}

ERL_NIF_TERM human_string(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 1){
    return enif_make_badarg(env);
  }

  xla::Shape* shape;
  if(!exla::get<xla::Shape>(env, argv[0], shape)) return enif_make_badarg(env);

  std::string result = xla::ShapeUtil::HumanString(*shape);
  return exla::make(env, result);
}

/*********************** xla::LiteralUtil Functions *************************/
ERL_NIF_TERM create_r0(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 1){
    return enif_make_badarg(env);
  }
  // TODO: Handle other types.
  int data;

  if(!exla::get(env, argv[0], data)) return enif_make_badarg(env);

  xla::Literal literal = xla::LiteralUtil::CreateR0(data);
  return exla::make<xla::Literal>(env, literal);
}

ERL_NIF_TERM literal_to_string(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 1){
    return enif_make_badarg(env);
  }

  xla::Literal* literal;

  if(!exla::get<xla::Literal>(env, argv[0], literal)) return enif_make_badarg(env);
  std::string literal_str = (*literal).ToString();
  return exla::make(env, literal_str);
}

/************************ xla::XlaOp Functions ***************************/
ERL_NIF_TERM parameter(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 4){
    return enif_make_badarg(env);
  }

  xla::XlaBuilder** builder;
  long int param_num;
  xla::Shape* shape;
  std::string name;

  if(!exla::get<xla::XlaBuilder*>(env, argv[0], builder)) return enif_make_badarg(env);
  if(!exla::get(env, argv[1], param_num)) return enif_make_badarg(env);
  if(!exla::get<xla::Shape>(env, argv[2], shape)) return enif_make_badarg(env);
  if(!exla::get(env, argv[3], name)) return enif_make_badarg(env);

  xla::XlaOp op = xla::Parameter((*builder), param_num, *shape, name);

  return exla::ok(env, exla::make<xla::XlaOp>(env, op));
}

/************************ Slicing Ops *****************************/
ERL_NIF_TERM slice(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if(argc != 4) {
    return exla::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  std::vector<long long int> start_indices;
  std::vector<long long int> limit_indices;
  std::vector<long long int> strides;

  if(!exla::get<xla::XlaOp>(env, argv[0], operand)) return exla::error(env, "Unable to get operand.");
  if(!exla::get_vector<long long int>(env, argv[1], start_indices)) return exla::error(env, "Unable to get start indices.");
  if(!exla::get_vector<long long int>(env, argv[2], limit_indices)) return exla::error(env, "Unable to get limit indices.");
  if(!exla::get_vector<long long int>(env, argv[3], strides)) return exla::error(env, "Unable to get strides.");

  xla::XlaOp op = xla::Slice(*operand, start_indices, limit_indices, strides);

  return exla::ok(env, exla::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM slice_in_dim(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 5) {
    return exla::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  long int start_index;
  long int end_index;
  long int stride;
  long int dimno;

  if(!exla::get<xla::XlaOp>(env, argv[0], operand)) return exla::error(env, "Unable to get operand.");
  if(!exla::get(env, argv[1], start_index)) return exla::error(env, "Unable to get start index.");
  if(!exla::get(env, argv[2], end_index)) return exla::error(env, "Unable to get end index.");
  if(!exla::get(env, argv[3], stride)) return exla::error(env, "Unable to get stride.");
  if(!exla::get(env, argv[4], dimno)) return exla::error(env, "Unable to get dimension number.");

  xla::XlaOp op = xla::SliceInDim(*operand, start_index, end_index, stride, dimno);

  return exla::ok(env, exla::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM dynamic_slice(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 3) {
    return exla::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  std::vector<xla::XlaOp> start_indices;
  std::vector<long long int> sizes;

  if(!exla::get(env, argv[0], operand)) return exla::error(env, "Unable to get operand.");
  if(!exla::get_vector_ops(env, argv[1], start_indices)) return exla::error(env, "Unable to get start index ops.");
  if(!exla::get_vector<long long int>(env, argv[2], sizes)) return exla::error(env, "Unable to get sizes.");

  xla::XlaOp op = xla::DynamicSlice(*operand, start_indices, sizes);

  return exla::ok(env, exla::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM dynamic_update_slice(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 3) {
    return exla::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  xla::XlaOp* update;
  std::vector<xla::XlaOp> start_indices;

  if(!exla::get<xla::XlaOp>(env, argv[0], operand)) return exla::error(env, "Unable to get operand.");
  if(!exla::get<xla::XlaOp>(env, argv[1], update)) return exla::error(env, "Unable to get update.");
  if(!exla::get_vector_ops(env, argv[2], start_indices)) return exla::error(env, "Unable to get start indices.");

  xla::XlaOp op = xla::DynamicUpdateSlice(*operand, *update, start_indices);

  return exla::ok(env, exla::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM xla_binary_op(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[], xla::XlaOp(*lambda)(xla::XlaOp, xla::XlaOp, absl::Span<const long long int>)){
  if(argc != 3){
    return enif_make_badarg(env);
  }

  xla::XlaOp *lhs, *rhs;
  std::vector<long long int> broadcast_dims;

  if(!exla::get<xla::XlaOp>(env, argv[0], lhs)) return enif_make_badarg(env);
  if(!exla::get<xla::XlaOp>(env, argv[1], rhs)) return enif_make_badarg(env);
  if(!exla::get_vector<long long int>(env, argv[2], broadcast_dims)) return enif_make_badarg(env);

  xla::XlaOp result = lambda(*lhs, *rhs, broadcast_dims);
  return exla::ok(env, exla::make<xla::XlaOp>(env, result));
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

ERL_NIF_TERM xla_unary_op(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[], xla::XlaOp(*lambda)(xla::XlaOp)){
  if(argc != 1){
    return enif_make_badarg(env);
  }

  xla::XlaOp *op;

  if(!exla::get<xla::XlaOp>(env, argv[0], op)) return enif_make_badarg(env);

  xla::XlaOp result = lambda(*op);
  return exla::ok(env, exla::make<xla::XlaOp>(env, result));
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

ERL_NIF_TERM zero(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 2) {
    return exla::error(env, "Bad argument count.");
  }

  xla::XlaBuilder** builder;
  xla::PrimitiveType element_type;

  if(!exla::get<xla::XlaBuilder*>(env, argv[0], builder)) return exla::error(env, "Unable to get builder.");
  if(!exla::get_type(env, argv[1], element_type)) return exla::error(env, "Unable to get element type.");

  xla::XlaOp op = xla::ConstantLiteral(*builder, xla::LiteralUtil::Zero(element_type));

  return exla::ok(env, exla::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM constant_r0(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 2){
    return enif_make_badarg(env);
  }

  xla::XlaBuilder** builder;
  int value;

  if(!exla::get<xla::XlaBuilder*>(env, argv[0], builder)) return enif_make_badarg(env);
  if(!exla::get(env, argv[1], value)) return enif_make_badarg(env);

  xla::XlaOp op = xla::ConstantR0((*builder), value);
  return exla::ok(env, exla::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM constant_r1_fill(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 3){
    return enif_make_badarg(env);
  }

  xla::XlaBuilder** builder;
  int value;
  int length;

  if(!exla::get<xla::XlaBuilder*>(env, argv[0], builder)) return enif_make_badarg(env);
  if(!exla::get(env, argv[1], length)) return enif_make_badarg(env);
  if(!exla::get(env, argv[2], value)) return enif_make_badarg(env);

  xla::XlaOp op = xla::ConstantR1((*builder), length, value);
  return exla::ok(env, exla::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM dot(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 2){
    return enif_make_badarg(env);
  }

  xla::XlaOp *lhs, *rhs;

  if(!exla::get<xla::XlaOp>(env, argv[0], lhs)) return enif_make_badarg(env);
  if(!exla::get<xla::XlaOp>(env, argv[1], rhs)) return enif_make_badarg(env);
  // TODO: Handle Precision Configuration
  xla::XlaOp result = xla::Dot(*lhs, *rhs);
  return exla::ok(env, exla::make<xla::XlaOp>(env, result));
}

ERL_NIF_TERM reduce(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 4){
    return exla::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  xla::XlaOp* init_value;
  xla::XlaComputation* computation;
  std::vector<long long int> dimensions_to_reduce;

  if(!exla::get<xla::XlaOp>(env, argv[0], operand)) return exla::error(env, "Unable to get operand.");
  if(!exla::get<xla::XlaOp>(env, argv[1], init_value)) return exla::error(env, "Unable to get initial value.");
  if(!exla::get<xla::XlaComputation>(env, argv[2], computation)) return exla::error(env, "Unable to get computation.");
  if(!exla::get_vector<long long int>(env, argv[3], dimensions_to_reduce)) return exla::error(env, "Unable to get reduction dimensions.");

  xla::XlaOp result = xla::Reduce(*operand, *init_value, *computation, dimensions_to_reduce);

  return exla::ok(env, exla::make<xla::XlaOp>(env, result));
}

/************************ xla::ClientLibrary Functions ***************************/
// TODO: This function generates mildly annoying and poorly formatted log messages from the TensorFlow side...
// We can either make the logging stricter or we can somehow get the log messages to the Elixir Logger?? I'm
// not sure what the best solution is...
ERL_NIF_TERM get_cpu_client(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  int num_replicas, intra_op_parallelism_threads;

  if(!exla::get(env, argv[0], num_replicas)) return exla::error(env, "Unable to get num_replicas.");
  if(!exla::get(env, argv[1], intra_op_parallelism_threads)) return exla::error(env, "Unable to get intra_op_parallelism_threads.");
  EXLA_ASSIGN_OR_RETURN(exla::ExlaClient* client, exla::GetCpuClient(num_replicas, intra_op_parallelism_threads), env);

  return exla::ok(env, exla::make<exla::ExlaClient*>(env, client));
}

ERL_NIF_TERM get_gpu_client(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  int num_replicas, intra_op_parallelism_threads;

  if(!exla::get(env, argv[0], num_replicas)) return exla::error(env, "Unable to get num_replicas.");
  if(!exla::get(env, argv[1], intra_op_parallelism_threads)) return exla::error(env, "Unable to get intra_op_parallelism_threads.");

  EXLA_ASSIGN_OR_RETURN(exla::ExlaClient* client, exla::GetGpuClient(num_replicas, intra_op_parallelism_threads), env);

  return exla::ok(env, exla::make<exla::ExlaClient*>(env, client));
}

ERL_NIF_TERM get_default_device_ordinal(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 1){
    return enif_make_badarg(env);
  }

  exla::ExlaClient **client;
  if(!exla::get<exla::ExlaClient*>(env, argv[0], client)) return enif_make_badarg(env);

  int device_ordinal = (*client)->client()->default_device_ordinal();
  return exla::ok(env, exla::make(env, device_ordinal));
}

ERL_NIF_TERM get_device_count(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 1){
    return enif_make_badarg(env);
  }

  exla::ExlaClient **client;
  if(!exla::get<exla::ExlaClient*>(env, argv[0], client)) return enif_make_badarg(env);

  int device_count = (*client)->client()->device_count();

  return exla::ok(env, exla::make(env, device_count));
}

/************ Build, Compilation, Execution *************/
ERL_NIF_TERM build(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 2){
    return enif_make_badarg(env);
  }

  xla::XlaBuilder** builder;
  xla::XlaOp* root;

  if(!exla::get<xla::XlaBuilder*>(env, argv[0], builder)) return exla::error(env, "Bad argument passed to build.");
  if(!exla::get<xla::XlaOp>(env, argv[1], root)) return exla::error(env, "Bad argument passed to build.");

  EXLA_ASSIGN_OR_RETURN(xla::XlaComputation computation, (*builder)->Build(*root), env);

  return exla::ok(env, exla::make<xla::XlaComputation>(env, computation));
}

// TODO: Most of this logic can move to `exla::ExlaClient`
ERL_NIF_TERM compile(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){

  exla::ExlaClient** client;
  xla::XlaComputation* computation;
  absl::Span<xla::Shape*> argument_layouts;
  xla::ExecutableBuildOptions build_options;
  int device_ordinal, num_replicas, num_partitions;

  if(!exla::get<exla::ExlaClient*>(env, argv[0], client)) return enif_make_badarg(env);
  if(!exla::get<xla::XlaComputation>(env, argv[1], computation)) return enif_make_badarg(env);
  if(!exla::get_argument_layouts(env, argv[2], argument_layouts)) return exla::error(env, "Unable to get argument layouts.");
  if(!exla::get(env, argv[3], device_ordinal)) return exla::error(env, "Unable to get device ordinal.");
  if(!exla::get(env, argv[4], num_replicas)) return exla::error(env, "Unable to get Number of Replicas.");
  if(!exla::get(env, argv[5], num_partitions)) return exla::error(env, "Unable to get Number of Partitions.");

  build_options.set_device_allocator((*client)->allocator());
  build_options.set_num_replicas(num_replicas);
  build_options.set_num_partitions(num_partitions);
  // TODO: Used in conjunction with replicas and partitions.
  // build_options.set_device_assignment(device_assignment);
  build_options.set_device_ordinal(device_ordinal);
  // TODO: Single Partition Multi-Device vs. Multi-Partition Multi-Device
  // build_options.set_use_spmd_partitioning(use_spmd);

  EXLA_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<xla::LocalExecutable>> executables,
                        (*client)->client()->Compile(*computation, argument_layouts, build_options), env);

  ERL_NIF_TERM exec_refs[executables.size()];
  int i = 0;
  for(auto it=std::begin(executables);it!=std::end(executables);++it){
    exec_refs[i++] = exla::make<xla::LocalExecutable>(env, executables.at(i));
  }
  // TODO: This should return the vector. There is an executable for every partition, usually 1.
  return exla::ok(env, exec_refs[0]);
}

// TODO: Most of this logic should be moved to `exla::ExlaClient`
ERL_NIF_TERM run(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 7){
    return enif_make_badarg(env);
  }

  exla::ExlaClient** client;
  xla::LocalExecutable* local_executable;
  std::vector<xla::ShapedBuffer*> arguments;
  xla::ExecutableRunOptions run_options;
  int run_id, rng_seed, launch_id, device_ordinal;

  if(!exla::get<exla::ExlaClient*>(env, argv[0], client)) return exla::error(env, "Unable to get client.");
  if(!exla::get<xla::LocalExecutable>(env, argv[1], local_executable)) return exla::error(env, "Unable to get executable.");
  if(!exla::get_arguments(env, argv[2], arguments)) return exla::error(env, "Unable to get arguments.");
  if(!exla::get(env, argv[3], device_ordinal)) return exla::error(env, "Unable to get device ordinal.");
  if(!exla::get(env, argv[4], run_id)) return exla::error(env, "Unable to get Run ID.");
  if(!exla::get(env, argv[5], rng_seed)) return exla::error(env, "Unable to get RNG Seed.");
  if(!exla::get(env, argv[6], launch_id)) return exla::error(env, "Unable to get Launch ID.");

  xla::RunId run_id_obj(run_id);
  exla::ExlaDevice* device = (*client)->device(device_ordinal);

  run_options.set_stream(device->compute_stream());
  run_options.set_host_to_device_stream(device->host_to_device_stream());
  run_options.set_allocator((*client)->allocator());
  run_options.set_intra_op_thread_pool((*client)->client()->backend().eigen_intra_op_thread_pool_device());
  // TODO: This is for executing multiple computations in parallel across multiple replicas.
  // run_options.set_device_assignment(device_assignment.get());
  run_options.set_run_id(run_id_obj);
  run_options.set_rng_seed(rng_seed);
  run_options.set_gpu_executable_run_options((*client)->gpu_run_options());
  run_options.set_launch_id(launch_id);

  EXLA_ASSIGN_OR_RETURN(xla::ScopedShapedBuffer result, local_executable->Run(arguments, run_options), env);

  return exla::ok(env, exla::make<xla::ShapedBuffer>(env, result));
}

ERL_NIF_TERM literal_to_shaped_buffer(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 3){
    return enif_make_badarg(env);
  }

  exla::ExlaClient** client;
  xla::Literal* literal;
  int device_ordinal;

  if(!exla::get<exla::ExlaClient*>(env, argv[0], client)) return enif_make_badarg(env);
  if(!exla::get<xla::Literal>(env, argv[1], literal)) return enif_make_badarg(env);
  if(!exla::get(env, argv[2], device_ordinal)) return enif_make_badarg(env);

  EXLA_ASSIGN_OR_RETURN(xla::ScopedShapedBuffer buffer, (*client)->client()->LiteralToShapedBuffer(*literal, device_ordinal, (*client)->allocator()), env);

  return exla::make<xla::ShapedBuffer>(env, buffer);
}

ERL_NIF_TERM shaped_buffer_to_literal(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 2){
    return enif_make_badarg(env);
  }

  exla::ExlaClient** client;
  xla::ShapedBuffer* buffer;

  if(!exla::get<exla::ExlaClient*>(env, argv[0], client)) return enif_make_badarg(env);
  if(!exla::get<xla::ShapedBuffer>(env, argv[1], buffer)) return enif_make_badarg(env);

  EXLA_ASSIGN_OR_RETURN(xla::Literal literal, (*client)->client()->ShapedBufferToLiteral(*buffer), env);

  return exla::make<xla::Literal>(env, literal);
}

/*********** HLO Methods *************/
std::unique_ptr<xla::HloModule> get_hlo_module(const xla::XlaComputation& computation){
  xla::HloModuleConfig module_config = xla::HloModule::CreateModuleConfigFromProto(computation.proto(), xla::GetDebugOptionsFromFlags()).ConsumeValueOrDie();
  std::unique_ptr<xla::HloModule> module = xla::HloModule::CreateFromProto(computation.proto(), module_config).ConsumeValueOrDie();

  return module;
}

ERL_NIF_TERM get_computation_hlo_text(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 1){
    return enif_make_badarg(env);
  }

  xla::XlaComputation* computation;

  if(!exla::get<xla::XlaComputation>(env, argv[0], computation)) return enif_make_badarg(env);

  std::unique_ptr<xla::HloModule> hlo_module = get_hlo_module(*computation);

  xla::HloPrintOptions options;
  options = xla::HloPrintOptions::ShortParsable();
  options.set_print_large_constants(false);
  std::string result = hlo_module->ToString(options);
  return exla::make(env, result);
}

ERL_NIF_TERM get_computation_hlo_proto(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 1){
    return enif_make_badarg(env);
  }

  xla::XlaComputation* computation;

  if(!exla::get<xla::XlaComputation>(env, argv[0], computation)) return enif_make_badarg(env);

  std::string result;
  (*computation).proto().SerializeToString(&result);
  return exla::make(env, result);
}

static ErlNifFunc exla_funcs[] = {
  /***** xla::XlaBuilder *****/
  {"new_builder", 1, new_builder},
  {"create_sub_builder", 2, create_sub_builder},
  /****** xla::Client ******/
  {"get_cpu_client", 2, get_cpu_client},
  {"get_gpu_client", 2, get_gpu_client},
  {"get_device_count", 1, get_device_count},
  {"get_default_device_ordinal", 1, get_default_device_ordinal},
  /****** xla::ShapedBuffer ******/
  {"binary_to_shaped_buffer", 4, binary_to_shaped_buffer, ERL_NIF_DIRTY_JOB_IO_BOUND},
  {"shaped_buffer_to_binary", 2, shaped_buffer_to_binary},
  {"on_host_shape", 1, on_host_shape},
  /****** xla::Shape ******/
  {"human_string", 1, human_string},
  {"make_shape", 2, make_shape},
  {"make_scalar_shape", 1, make_scalar_shape},
  {"parameter", 4, parameter},
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
  {"zero", 2, zero},
  {"constant_r0", 2, constant_r0},
  {"constant_r1", 3, constant_r1_fill},
  /******** Slicing Ops ********/
  {"slice", 4, slice},
  {"slice_in_dim", 5, slice_in_dim},
  {"dynamic_slice", 3, dynamic_slice},
  {"dynamic_update_slice", 3, dynamic_update_slice},
  /******** Other XLA Ops *******/
  {"dot", 2, dot},
  {"reduce", 4, reduce},
  /******* Compilation, Execution, Etc. ******/
  {"build", 2, build},
  {"compile", 6, compile},
  {"run", 7, run},
  {"literal_to_shaped_buffer", 3, literal_to_shaped_buffer},
  {"shaped_buffer_to_literal", 2, shaped_buffer_to_literal},
  /******** HLO Functions ********/
  {"get_computation_hlo_proto", 1, get_computation_hlo_proto},
  {"get_computation_hlo_text", 1, get_computation_hlo_text}
};

ERL_NIF_INIT(Elixir.Exla.NIF, exla_funcs, &load, NULL, NULL, NULL);
