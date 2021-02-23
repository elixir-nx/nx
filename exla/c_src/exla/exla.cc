#include <map>

#include "tensorflow/compiler/xla/exla/exla_nif_util.h"
#include "tensorflow/compiler/xla/exla/exla_client.h"
#include "tensorflow/compiler/xla/exla/exla_log_sink.h"
#include "tensorflow/compiler/xla/exla/exla_aot_compilation.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/lib/lu_decomposition.h"
#include "tensorflow/compiler/xla/client/lib/qr.h"
#include "tensorflow/compiler/xla/client/lib/self_adjoint_eig.h"
#include "tensorflow/compiler/xla/client/lib/svd.h"
#include "tensorflow/compiler/xla/primitive_util.h"

// All of these are created with calls to `new` and subsequently
// passed to the VM as pointers-to-pointers so we balance it out
// with calls to delete rather than just using the default destructor.

void free_exla_executable(ErlNifEnv* env, void * obj) {
  exla::ExlaExecutable** executable = reinterpret_cast<exla::ExlaExecutable**>(obj);
  if (*executable != nullptr) {
    delete *executable;
    *executable = nullptr;
  }
}

void free_xla_builder(ErlNifEnv* env, void * obj) {
  xla::XlaBuilder** builder = reinterpret_cast<xla::XlaBuilder**>(obj);
  if (*builder != nullptr) {
    delete *builder;
    *builder = nullptr;
  }
}

void free_exla_client(ErlNifEnv* env, void * obj) {
  exla::ExlaClient** client = reinterpret_cast<exla::ExlaClient**>(obj);
  if (*client != nullptr) {
    delete *client;
    *client = nullptr;
  }
}

void free_exla_buffer(ErlNifEnv* env, void * obj) {
  exla::ExlaBuffer** buffer = reinterpret_cast<exla::ExlaBuffer**>(obj);
  if (*buffer != nullptr) {
    delete *buffer;
    *buffer = nullptr;
  }
}

static int open_resources(ErlNifEnv* env) {
  const char* mod = "EXLA";

  if (!exla::nif::open_resource<xla::XlaOp>(env, mod, "Op")) {
    return -1;
  }
  if (!exla::nif::open_resource<xla::Shape>(env, mod, "Shape")) {
    return -1;
  }
  if (!exla::nif::open_resource<xla::XlaComputation>(env, mod, "Computation")) {
    return -1;
  }
  if (!exla::nif::open_resource<exla::ExlaExecutable*>(env, mod, "Executable", free_exla_executable)) {
    return -1;
  }
  if (!exla::nif::open_resource<xla::XlaBuilder*>(env, mod, "Builder", free_xla_builder)) {
    return -1;
  }
  if (!exla::nif::open_resource<exla::ExlaClient*>(env, mod, "ExlaClient", free_exla_client)) {
    return -1;
  }
  if (!exla::nif::open_resource<exla::ExlaBuffer*>(env, mod, "ExlaBuffer", free_exla_buffer)) {
    return -1;
  }
  return 1;
}

static int load(ErlNifEnv* env, void** priv, ERL_NIF_TERM load_info) {
  if (open_resources(env) == -1) return -1;

  return 0;
}

// XlaBuilder Functions

ERL_NIF_TERM new_builder(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  std::string name;
  if (!exla::nif::get(env, argv[0], name)) {
    return exla::nif::error(env, "Unable to get builder name.");
  }

  xla::XlaBuilder* builder = new xla::XlaBuilder(name);

  return exla::nif::ok(env, exla::nif::make<xla::XlaBuilder*>(env, builder));
}

ERL_NIF_TERM create_sub_builder(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaBuilder** builder;
  std::string name;

  if (!exla::nif::get<xla::XlaBuilder*>(env, argv[0], builder)) {
    return exla::nif::error(env, "Unable to get builder.");
  }
  if (!exla::nif::get(env, argv[1], name)) {
    return exla::nif::error(env, "Unable to get name.");
  }

  auto uniq_sub_builder = (*builder)->CreateSubBuilder(name);
  xla::XlaBuilder* sub_builder = uniq_sub_builder.release();
  return exla::nif::ok(env, exla::nif::make<xla::XlaBuilder*>(env, sub_builder));
}

ERL_NIF_TERM build(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaBuilder** builder;
  xla::XlaOp* root;

  if (!exla::nif::get<xla::XlaBuilder*>(env, argv[0], builder)) {
    return exla::nif::error(env, "Bad argument passed to build.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[1], root)) {
    return exla::nif::error(env, "Bad argument passed to build.");
  }

  EXLA_ASSIGN_OR_RETURN_NIF(xla::XlaComputation computation,
    (*builder)->Build(*root), env);

  return exla::nif::ok(env, exla::nif::make<xla::XlaComputation>(env, computation));
}

ERL_NIF_TERM parameter(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaBuilder** builder;
  exla::int64 param_num;
  xla::Shape* shape;
  std::string name;

  if (!exla::nif::get<xla::XlaBuilder*>(env, argv[0], builder)) {
    return exla::nif::error(env, "Unable to get builder.");
  }
  if (!exla::nif::get(env, argv[1], &param_num)) {
    return exla::nif::error(env, "Unable to get parameter number.");
  }
  if (!exla::nif::get<xla::Shape>(env, argv[2], shape)) {
    return exla::nif::error(env, "Unable to get parameter shape.");
  }
  if (!exla::nif::get(env, argv[3], name)) {
    return exla::nif::error(env, "Unable to get parameter name.");
  }

  xla::XlaOp op = xla::Parameter((*builder), param_num, *shape, name);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

// ExlaBuffer Functions

ERL_NIF_TERM binary_to_device_mem(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return exla::nif::error(env, "Bad argument count.");
  }

  ErlNifBinary bin;
  xla::Shape* shape;
  exla::ExlaClient** client;
  int device_ordinal;

  if (!exla::nif::get<exla::ExlaClient*>(env, argv[0], client)) {
    return exla::nif::error(env, "Unable to get client.");
  }
  if (!exla::nif::get_binary(env, argv[1], &bin)) {
    return exla::nif::error(env, "Unable to get data.");
  }
  if (!exla::nif::get<xla::Shape>(env, argv[2], shape)) {
    return exla::nif::error(env, "Unable to get shape.");
  }
  if (!exla::nif::get(env, argv[3], &device_ordinal)) {
    return exla::nif::error(env, "Unable to get device ordinal.");
  }

  exla::ExlaDevice* device = (*client)->device(device_ordinal);

  EXLA_ASSIGN_OR_RETURN_NIF(exla::ExlaBuffer* buffer,
    (*client)->BufferFromBinary(bin, *shape, device, false, false), env);

  return exla::nif::ok(env, exla::nif::make<exla::ExlaBuffer*>(env, buffer));
}

ERL_NIF_TERM read_device_mem(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::ExlaClient** client;
  exla::ExlaBuffer** buffer;

  if (!exla::nif::get<exla::ExlaClient*>(env, argv[0], client)) {
    return exla::nif::error(env, "Unable to get client.");
  }
  if (!exla::nif::get<exla::ExlaBuffer*>(env, argv[1], buffer)) {
    return exla::nif::error(env, "Unable to get buffer.");
  }

  if ((*buffer)->is_tuple()) {
    return exla::nif::ok(env);
  }

  EXLA_ASSIGN_OR_RETURN_NIF(ErlNifBinary binary, (*buffer)->ToBinary(), env);

  return exla::nif::ok(env, exla::nif::make(env, binary));
}

ERL_NIF_TERM deallocate_device_mem(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::ExlaBuffer** buffer;

  if (!exla::nif::get<exla::ExlaBuffer*>(env, argv[0], buffer)) {
    return exla::nif::error(env, "Unable to get buffer.");
  }

  xla::Status dealloc_status = (*buffer)->Deallocate();

  if (!dealloc_status.ok()) {
    return exla::nif::atom(env, "already_deallocated");
  } else {
    return exla::nif::ok(env);
  }
}

// Shape Functions

ERL_NIF_TERM make_shape(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::PrimitiveType element_type;
  std::vector<exla::int64> dims;

  if (!exla::nif::get_primitive_type(env, argv[0], &element_type)) {
    return exla::nif::error(env, "Unable to get type.");
  }
  if (!exla::nif::get_tuple(env, argv[1], dims)) {
    return exla::nif::error(env, "Unable to get dimensions.");
  }

  xla::Shape shape = xla::ShapeUtil::MakeShape(element_type, dims);

  return exla::nif::ok(env, exla::nif::make<xla::Shape>(env, shape));
}

ERL_NIF_TERM make_tuple_shape(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if(argc != 1){
    return exla::nif::error(env, "Bad argument count.");
  }

  std::vector<xla::Shape> shapes;

  if (!exla::nif::get_list<xla::Shape>(env, argv[0], shapes)) {
    return exla::nif::error(env, "Unable to get shapes.");
  }

  xla::Shape shape = xla::ShapeUtil::MakeTupleShape(shapes);

  return exla::nif::ok(env, exla::nif::make<xla::Shape>(env, shape));
}

ERL_NIF_TERM get_shape_info(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::Shape* shape;

  if (!exla::nif::get<xla::Shape>(env, argv[0], shape)) {
    return exla::nif::error(env, "Unable to get shape.");
  }

  return exla::nif::ok(env, exla::nif::make_shape_info(env, *shape));
}

// XlaOp Functions

// Tuples

ERL_NIF_TERM tuple(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaBuilder** builder;
  std::vector<xla::XlaOp> elements;

  if (!exla::nif::get<xla::XlaBuilder*>(env, argv[0], builder)) {
    return exla::nif::error(env, "Unable to get builder.");
  }
  if (!exla::nif::get_list<xla::XlaOp>(env, argv[1], elements)) {
    return exla::nif::error(env, "Unable to get tuple elements.");
  }

  xla::XlaOp op = xla::Tuple(*builder, elements);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM get_tuple_element(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  exla::int64 index;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get(env, argv[1], &index)) {
    return exla::nif::error(env, "Unable to get index.");
  }

  xla::XlaOp op = xla::GetTupleElement(*operand, index);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

// Conditionals

ERL_NIF_TERM conditional_if(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 5) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* pred;
  xla::XlaOp* true_op;
  xla::XlaOp* false_op;
  xla::XlaComputation* true_comp;
  xla::XlaComputation* false_comp;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], pred)) {
    return exla::nif::error(env, "Unable to get predicate.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[1], true_op)) {
    return exla::nif::error(env, "Unable to get true operand.");
  }
  if (!exla::nif::get<xla::XlaComputation>(env, argv[2], true_comp)) {
    return exla::nif::error(env, "Unable to get true computation.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[3], false_op)) {
    return exla::nif::error(env, "Unable to get false operand.");
  }
  if (!exla::nif::get<xla::XlaComputation>(env, argv[4], false_comp)) {
    return exla::nif::error(env, "Unable to get false computation.");
  }

  xla::XlaOp op = xla::Conditional(*pred, *true_op,
                                   *true_comp, *false_op,
                                   *false_comp);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM conditional_multi(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* index;
  std::vector<xla::XlaComputation*> branches;
  std::vector<xla::XlaOp> operands;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], index)) {
    return exla::nif::error(env, "Unable to get index.");
  }
  if (!exla::nif::get_list<xla::XlaComputation*>(env, argv[1], branches)) {
    return exla::nif::error(env, "Unable to get branches.");
  }
  if (!exla::nif::get_list<xla::XlaOp>(env, argv[2], operands)) {
    return exla::nif::error(env, "Unable to get operands.");
  }

  xla::XlaOp op = xla::Conditional(*index, branches, operands);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM select(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* pred;
  xla::XlaOp* on_true;
  xla::XlaOp* on_false;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], pred)) {
    return exla::nif::error(env, "Unable to get predicate.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[1], on_true)) {
    return exla::nif::error(env, "Unable to get predicate.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[2], on_false)) {
    return exla::nif::error(env, "Unable to get predicate.");
  }

  xla::XlaOp op = xla::Select(*pred, *on_true, *on_false);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

// Slicing

ERL_NIF_TERM slice(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  std::vector<exla::int64> start_indices;
  std::vector<exla::int64> limit_indices;
  std::vector<exla::int64> strides;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get_list(env, argv[1], start_indices)) {
    return exla::nif::error(env, "Unable to get start indices.");
  }
  if (!exla::nif::get_list(env, argv[2], limit_indices)) {
    return exla::nif::error(env, "Unable to get limit indices.");
  }
  if (!exla::nif::get_list(env, argv[3], strides)) {
    return exla::nif::error(env, "Unable to get strides.");
  }

  xla::XlaOp op = xla::Slice(*operand, start_indices, limit_indices, strides);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM dynamic_slice(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  std::vector<xla::XlaOp> start_indices;
  std::vector<exla::int64> sizes;

  if (!exla::nif::get(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get_list<xla::XlaOp>(env, argv[1], start_indices)) {
    return exla::nif::error(env, "Unable to get start index ops.");
  }
  if (!exla::nif::get_list(env, argv[2], sizes)) {
    return exla::nif::error(env, "Unable to get sizes.");
  }

  xla::XlaOp op = xla::DynamicSlice(*operand, start_indices, sizes);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM dynamic_update_slice(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  xla::XlaOp* update;
  std::vector<xla::XlaOp> start_indices;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[1], update)) {
    return exla::nif::error(env, "Unable to get update.");
  }
  if (!exla::nif::get_list<xla::XlaOp>(env, argv[2], start_indices)) {
    return exla::nif::error(env, "Unable to get start indices.");
  }

  xla::XlaOp op = xla::DynamicUpdateSlice(*operand, *update, start_indices);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

// Creation

ERL_NIF_TERM rng_normal(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* mu;
  xla::XlaOp* sigma;
  xla::Shape* shape;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], mu)) {
    return exla::nif::error(env, "Unable to get mu.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[1], sigma)) {
    return exla::nif::error(env, "Unable to get sigma.");
  }
  if (!exla::nif::get<xla::Shape>(env, argv[2], shape)) {
    return exla::nif::error(env, "Unable to get shape.");
  }

  xla::XlaOp op = xla::RngNormal(*mu, *sigma, *shape);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM rng_uniform(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* a;
  xla::XlaOp* b;
  xla::Shape* shape;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], a)) {
    return exla::nif::error(env, "Unable to get mu.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[1], b)) {
    return exla::nif::error(env, "Unable to get sigma.");
  }
  if (!exla::nif::get<xla::Shape>(env, argv[2], shape)) {
    return exla::nif::error(env, "Unable to get shape.");
  }

  xla::XlaOp op = xla::RngUniform(*a, *b, *shape);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM iota(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaBuilder** builder;
  xla::Shape* shape;
  exla::int64 dimension;

  if (!exla::nif::get<xla::XlaBuilder*>(env, argv[0], builder)) {
    return exla::nif::error(env, "Unable to get builder.");
  }
  if (!exla::nif::get<xla::Shape>(env, argv[1], shape)) {
    return exla::nif::error(env, "Unable to get shape.");
  }
  if (!exla::nif::get(env, argv[2], &dimension)) {
    return exla::nif::error(env, "Unable to get dimension");
  }

  xla::XlaOp op = xla::Iota(*builder, *shape, dimension);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

// Binary Ops

ERL_NIF_TERM xla_binary_op(ErlNifEnv* env,
                           int argc,
                           const ERL_NIF_TERM argv[],
                           xla::XlaOp(*lambda)(xla::XlaOp, xla::XlaOp, absl::Span<const exla::int64>)) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp *lhs;
  xla::XlaOp *rhs;
  std::vector<exla::int64> broadcast_dims;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], lhs)) {
    return exla::nif::error(env, "Unable to get left-hand side.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[1], rhs)) {
    return exla::nif::error(env, "Unable to get right-hand side.");
  }
  if (!exla::nif::get_tuple(env, argv[2], broadcast_dims)) {
    return exla::nif::error(env, "Unable to get broadcast dimensions.");
  }

  xla::XlaOp op = lambda(*lhs, *rhs, broadcast_dims);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM add(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::Add);
}

ERL_NIF_TERM sub(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::Sub);
}

ERL_NIF_TERM mul(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::Mul);
}

ERL_NIF_TERM div(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::Div);
}

ERL_NIF_TERM rem(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::Rem);
}

ERL_NIF_TERM min(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::Min);
}

ERL_NIF_TERM max(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::Max);
}

ERL_NIF_TERM bitwise_and(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::And);
}

ERL_NIF_TERM bitwise_or(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::Or);
}

ERL_NIF_TERM bitwise_xor(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::Xor);
}

ERL_NIF_TERM shift_left(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::ShiftLeft);
}

ERL_NIF_TERM shift_right_logical(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::ShiftRightLogical);
}

ERL_NIF_TERM shift_right_arithmetic(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::ShiftRightArithmetic);
}

ERL_NIF_TERM equal(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::Eq);
}

ERL_NIF_TERM eq_total_order(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::EqTotalOrder);
}

ERL_NIF_TERM not_equal(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::Ne);
}

ERL_NIF_TERM ne_total_order(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::NeTotalOrder);
}

ERL_NIF_TERM greater_equal(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::Ge);
}

ERL_NIF_TERM greater(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::Gt);
}

ERL_NIF_TERM less(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::Lt);
}

ERL_NIF_TERM less_equal(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::Le);
}

ERL_NIF_TERM pow(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::Pow);
}

ERL_NIF_TERM complex(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::Complex);
}

ERL_NIF_TERM atan2(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::Atan2);
}

// Unary Ops

ERL_NIF_TERM xla_unary_op(ErlNifEnv* env,
                          int argc,
                          const ERL_NIF_TERM argv[],
                          xla::XlaOp(*lambda)(xla::XlaOp)) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp *operand;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }

  xla::XlaOp op = lambda(*operand);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM abs(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Abs);
}

ERL_NIF_TERM exp(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Exp);
}

ERL_NIF_TERM expm1(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Expm1);
}

ERL_NIF_TERM floor(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Floor);
}

ERL_NIF_TERM ceil(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Ceil);
}

ERL_NIF_TERM round(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Round);
}

ERL_NIF_TERM log(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Log);
}

ERL_NIF_TERM log1p(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Log1p);
}

ERL_NIF_TERM logistic(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Logistic);
}

ERL_NIF_TERM sign(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Sign);
}

ERL_NIF_TERM clz(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Clz);
}

ERL_NIF_TERM cos(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Cos);
}

ERL_NIF_TERM sin(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Sin);
}

ERL_NIF_TERM acos(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Acos);
}

ERL_NIF_TERM asin(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Asin);
}

ERL_NIF_TERM atan(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Atan);
}

ERL_NIF_TERM cosh(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Cosh);
}

ERL_NIF_TERM sinh(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Sinh);
}

ERL_NIF_TERM tanh(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Tanh);
}

ERL_NIF_TERM acosh(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Acosh);
}

ERL_NIF_TERM asinh(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Asinh);
}

ERL_NIF_TERM atanh(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Atanh);
}

ERL_NIF_TERM real(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Real);
}

ERL_NIF_TERM imag(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Imag);
}

ERL_NIF_TERM sqrt(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Sqrt);
}

ERL_NIF_TERM cbrt(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Cbrt);
}

ERL_NIF_TERM rsqrt(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Rsqrt);
}

ERL_NIF_TERM erf(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Erf);
}

ERL_NIF_TERM erfc(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Erfc);
}

ERL_NIF_TERM erf_inv(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::ErfInv);
}

ERL_NIF_TERM is_finite(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::IsFinite);
}

ERL_NIF_TERM bitwise_not(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Not);
}

ERL_NIF_TERM neg(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Neg);
}

ERL_NIF_TERM conj(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::Conj);
}

ERL_NIF_TERM population_count(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::PopulationCount);
}

// Constants

ERL_NIF_TERM constant_r0(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaBuilder** builder;
  xla::PrimitiveType type;

  ERL_NIF_TERM term = argv[1];

  if (!exla::nif::get<xla::XlaBuilder*>(env, argv[0], builder)) {
    return exla::nif::error(env, "Unable to get builder.");
  }
  if (!exla::nif::get_primitive_type(env, argv[2], &type)) {
    return exla::nif::error(env, "Unable to cast scalar to type.");
  }

  xla::XlaOp op;

  switch (type) {
    case xla::PrimitiveType::PRED:
      op = xla::ConstantR0(*builder, exla::nif::get_value<xla::PrimitiveType::PRED>(env, term));
      break;
    case xla::PrimitiveType::U8:
      op = xla::ConstantR0(*builder, exla::nif::get_value<xla::PrimitiveType::U8>(env, term));
      break;
    case xla::PrimitiveType::U16:
      op = xla::ConstantR0(*builder, exla::nif::get_value<xla::PrimitiveType::U16>(env, term));
      break;
    case xla::PrimitiveType::U32:
      op = xla::ConstantR0(*builder, exla::nif::get_value<xla::PrimitiveType::U32>(env, term));
      break;
    case xla::PrimitiveType::U64:
      op = xla::ConstantR0(*builder, exla::nif::get_value<xla::PrimitiveType::U64>(env, term));
      break;
    case xla::PrimitiveType::S8:
      op = xla::ConstantR0(*builder, exla::nif::get_value<xla::PrimitiveType::S8>(env, term));
      break;
    case xla::PrimitiveType::S16:
      op = xla::ConstantR0(*builder, exla::nif::get_value<xla::PrimitiveType::S16>(env, term));
      break;
    case xla::PrimitiveType::S32:
      op = xla::ConstantR0(*builder, exla::nif::get_value<xla::PrimitiveType::S32>(env, term));
      break;
    case xla::PrimitiveType::S64:
      op = xla::ConstantR0(*builder, exla::nif::get_value<xla::PrimitiveType::S64>(env, term));
      break;
    case xla::PrimitiveType::F16:
      return exla::nif::error(env, "Unsupported constant type.");
    case xla::PrimitiveType::BF16:
      op = xla::ConstantR0(*builder, exla::nif::get_value<xla::PrimitiveType::BF16>(env, term));
      break;
    case xla::PrimitiveType::F32:
      op = xla::ConstantR0(*builder, exla::nif::get_value<xla::PrimitiveType::F32>(env, term));
      break;
    case xla::PrimitiveType::F64:
      op = xla::ConstantR0(*builder, exla::nif::get_value<xla::PrimitiveType::F64>(env, term));
      break;
    case xla::PrimitiveType::C64:
      return exla::nif::error(env, "Unsupported constant type.");
    case xla::PrimitiveType::C128:
      return exla::nif::error(env, "Unsupported constant type.");
    default:
      return exla::nif::error(env, "Invalid type.");
  }

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM constant_from_binary(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaBuilder** builder;
  ErlNifBinary binary;
  xla::Shape* shape;

  if (!exla::nif::get<xla::XlaBuilder*>(env, argv[0], builder)) {
    return exla::nif::error(env, "Unable to get builder.");
  }
  if (!exla::nif::get_binary(env, argv[1], &binary)) {
    return exla::nif::error(env, "Unable to get data.");
  }
  if (!exla::nif::get<xla::Shape>(env, argv[2], shape)) {
    return exla::nif::error(env, "Unable to get shape.");
  }

  char * data = const_cast<char*>(reinterpret_cast<char*>(binary.data));
  xla::BorrowingLiteral literal(data, *shape);

  xla::XlaOp op = xla::ConstantLiteral(*builder, literal);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

// Functional Ops

ERL_NIF_TERM reduce(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  xla::XlaOp* init_value;
  xla::XlaComputation* computation;
  std::vector<exla::int64> dimensions_to_reduce;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[1], init_value)) {
    return exla::nif::error(env, "Unable to get initial value.");
  }
  if (!exla::nif::get<xla::XlaComputation>(env, argv[2], computation)) {
    return exla::nif::error(env, "Unable to get computation.");
  }
  if (!exla::nif::get_tuple(env, argv[3], dimensions_to_reduce)) {
    return exla::nif::error(env, "Unable to get reduction dimensions.");
  }

  xla::XlaOp op = xla::Reduce(*operand, *init_value,
                              *computation, dimensions_to_reduce);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM variadic_reduce(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 5) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaBuilder** builder;
  std::vector<xla::XlaOp> operands;
  std::vector<xla::XlaOp> init_values;
  xla::XlaComputation* computation;
  std::vector<exla::int64> dimensions_to_reduce;

  if (!exla::nif::get<xla::XlaBuilder*>(env, argv[0], builder)) {
    return exla::nif::error(env, "Unable to get builder.");
  }
  if (!exla::nif::get_list<xla::XlaOp>(env, argv[1], operands)) {
    return exla::nif::error(env, "Unable to get operands.");
  }
  if (!exla::nif::get_list<xla::XlaOp>(env, argv[2], init_values)) {
    return exla::nif::error(env, "Unable to get initial values.");
  }
  if (!exla::nif::get<xla::XlaComputation>(env, argv[3], computation)) {
    return exla::nif::error(env, "Unable to get computation.");
  }
  if (!exla::nif::get_tuple(env, argv[4], dimensions_to_reduce)) {
    return exla::nif::error(env, "Unable to get dimensions.");
  }

  xla::XlaOp op = xla::Reduce(*builder, operands,
                              init_values, *computation,
                              dimensions_to_reduce);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM reduce_window(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 7) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  xla::XlaOp* initial_value;
  xla::XlaComputation* computation;
  std::vector<exla::int64> window_dimensions;
  std::vector<exla::int64> window_strides;
  // TODO(seanmor5): Not yet supported in Nx
  // std::vector<exla::int64> base_dilations;
  std::vector<exla::int64> window_dilations;
  std::vector<std::pair<exla::int64, exla::int64>> padding_config;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[1], initial_value)) {
    return exla::nif::error(env, "Unable to get initial value.");
  }
  if (!exla::nif::get<xla::XlaComputation>(env, argv[2], computation)) {
    return exla::nif::error(env, "Unable to get computation.");
  }
  if (!exla::nif::get_tuple(env, argv[3], window_dimensions)) {
    return exla::nif::error(env, "Unable to get window dimensions.");
  }
  if (!exla::nif::get_list(env, argv[4], window_strides)) {
    return exla::nif::error(env, "Unable to get window strides.");
  }
  if (!exla::nif::get_list(env, argv[5], window_dilations)) {
    return exla::nif::error(env, "Unable to get window dilations.");
  }
  if (!exla::nif::get_general_padding(env, argv[6], padding_config)) {
    return exla::nif::error(env, "Unable to get padding configuration.");
  }

  xla::XlaOp op = xla::ReduceWindowWithGeneralPadding(*operand,
                                                      *initial_value,
                                                      *computation,
                                                      window_dimensions,
                                                      window_strides,
                                                      {},
                                                      window_dilations,
                                                      padding_config);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM map(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaBuilder** builder;
  xla::XlaOp* operand;
  xla::XlaComputation* computation;
  std::vector<exla::int64> dimensions;

  if (!exla::nif::get<xla::XlaBuilder*>(env, argv[0], builder)) {
    return exla::nif::error(env, "Unable to get builder.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[1], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get<xla::XlaComputation>(env, argv[2], computation)) {
    return exla::nif::error(env, "Unable to get computation.");
  }
  if (!exla::nif::get_list(env, argv[3], dimensions)) {
    return exla::nif::error(env, "Unable to get map dimensions.");
  }

  xla::XlaOp op = xla::Map(*builder, {*operand}, *computation, dimensions);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

// Shape/Type Manipulation

ERL_NIF_TERM reshape(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  std::vector<exla::int64> new_shape;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get_tuple(env, argv[1], new_shape)) {
    return exla::nif::error(env, "Unable to get dimensions.");
  }

  xla::XlaOp op = xla::Reshape(*operand, new_shape);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM broadcast_in_dim(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  std::vector<exla::int64> new_shape;
  std::vector<exla::int64> broadcast_dims;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get_tuple(env, argv[1], new_shape)) {
    return exla::nif::error(env, "Unable to get dimensions.");
  }
  if (!exla::nif::get_tuple(env, argv[2], broadcast_dims)) {
    return exla::nif::error(env, "Unable to get broadcast dimensions.");
  }

  xla::XlaOp op = xla::BroadcastInDim(*operand, new_shape, broadcast_dims);
  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM get_shape_op(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaBuilder** builder;
  xla::XlaOp* operand;

  if (!exla::nif::get<xla::XlaBuilder*>(env, argv[0], builder)) {
    return exla::nif::error(env, "Unable to get builder.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[1], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }

  EXLA_ASSIGN_OR_RETURN_NIF(xla::Shape shape,
    (*builder)->GetShape(*operand), env);

  return exla::nif::ok(env, exla::nif::make<xla::Shape>(env, shape));
}

ERL_NIF_TERM convert_element_type(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  xla::PrimitiveType type;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get_primitive_type(env, argv[1], &type)) {
    return exla::nif::error(env, "Unable to get type string.");
  }

  xla::XlaOp op = xla::ConvertElementType(*operand, type);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM transpose(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  std::vector<exla::int64> permutation;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get_tuple(env, argv[1], permutation)) {
    return exla::nif::error(env, "Unable to get permutation.");
  }

  xla::XlaOp op = xla::Transpose(*operand, permutation);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

// Other Ops

ERL_NIF_TERM dot(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* lhs;
  xla::XlaOp* rhs;
  xla::PrecisionConfig config;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], lhs)) {
    return exla::nif::error(env, "Unable to get left-hand side operand.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[1], rhs)) {
    return exla::nif::error(env, "Unable to get right-hand side operand.");
  }
  if (!exla::nif::get_precision_config(env, argv[2], 2, &config)) {
    return exla::nif::error(env, "Unable to get precision configuration.");
  }

  xla::XlaOp op = xla::Dot(*lhs, *rhs, &config);
  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM dot_general(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* lhs;
  xla::XlaOp* rhs;
  xla::DotDimensionNumbers dnums;
  xla::PrecisionConfig config;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], lhs)) {
    return exla::nif::error(env, "Unable to get left-hand side operand.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[1], rhs)) {
    return exla::nif::error(env, "Unable to get right-hand side operand.");
  }
  if (!exla::nif::get_dot_dimension_numbers(env, argv[2], &dnums)) {
    return exla::nif::error(env, "Unable to get contraction dimensions.");
  }
  if (!exla::nif::get_precision_config(env, argv[3], 2, &config)) {
    return exla::nif::error(env, "Unable to get precision configuration.");
  }

  xla::XlaOp op = xla::DotGeneral(*lhs, *rhs, dnums, &config);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM conv_general_dilated(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 9) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  xla::XlaOp* kernel;
  std::vector<exla::int64> strides;
  std::vector<std::pair<exla::int64, exla::int64>> padding;
  std::vector<exla::int64> lhs_dilation;
  std::vector<exla::int64> rhs_dilation;
  xla::ConvolutionDimensionNumbers dimension_numbers;
  exla::int64 feature_group_count;
  xla::PrecisionConfig config;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[1], kernel)) {
    return exla::nif::error(env, "Unable to get kernel.");
  }
  if (!exla::nif::get_list(env, argv[2], strides)) {
    return exla::nif::error(env, "Unable to get strides.");
  }
  if (!exla::nif::get_general_padding(env, argv[3], padding)) {
    return exla::nif::error(env, "Unable to get padding.");
  }
  if (!exla::nif::get_list(env, argv[4], lhs_dilation)) {
    return exla::nif::error(env, "Unable to get operand dilation.");
  }
  if (!exla::nif::get_list(env, argv[5], rhs_dilation)) {
    return exla::nif::error(env, "Unable to get kernel dilation.");
  }
  if (!exla::nif::get_conv_dimension_numbers(env, argv[6], &dimension_numbers)) {
    return exla::nif::error(env, "Unable to get conv dimension numbers.");
  }
  if (!exla::nif::get(env, argv[7], &feature_group_count)) {
    return exla::nif::error(env, "Unable to get feature groups");
  }
  if (!exla::nif::get_precision_config(env, argv[8], 2, &config)) {
    return exla::nif::error(env, "Unable to get precision config");
  }

  xla::XlaOp op = xla::ConvGeneralDilated(*operand, *kernel, strides,
                                          padding, lhs_dilation, rhs_dilation,
                                          dimension_numbers, feature_group_count, 1, &config);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM pad(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  xla::XlaOp* pad_value;
  xla::PaddingConfig padding_config;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[1], pad_value)) {
    return exla::nif::error(env, "Unable to get value.");
  }
  if (!exla::nif::get_padding_config(env, argv[2], &padding_config)) {
    return exla::nif::error(env, "Unable to get padding configuration.");
  }

  xla::XlaOp op = xla::Pad(*operand, *pad_value, padding_config);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM clamp(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  xla::XlaOp* min;
  xla::XlaOp* max;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[1], min)) {
    return exla::nif::error(env, "Unable to get min value.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[2], max)) {
    return exla::nif::error(env, "Unable to get max value");
  }

  xla::XlaOp op = xla::Clamp(*min, *operand, *max);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM reverse(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  std::vector<exla::int64> dimensions;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get_list(env, argv[1], dimensions)) {
    return exla::nif::error(env, "Unable to get dimensions.");
  }

  xla::XlaOp op = xla::Rev(*operand, dimensions);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM concatenate(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaBuilder** builder;
  std::vector<xla::XlaOp> operands;
  exla::int64 dimension;

  if (!exla::nif::get<xla::XlaBuilder*>(env, argv[0], builder)) {
    return exla::nif::error(env, "Unable to get builder.");
  }
  if (!exla::nif::get_list<xla::XlaOp>(env, argv[1], operands)) {
    return exla::nif::error(env, "Unable to get operands.");
  }
  if (!exla::nif::get(env, argv[2], &dimension)) {
    return exla::nif::error(env, "Unable to get dimension.");
  }

  xla::XlaOp op = xla::ConcatInDim(*builder, operands, dimension);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM sort(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  xla::XlaComputation* comparator;
  exla::int64 dimension;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get<xla::XlaComputation>(env, argv[1], comparator)) {
    return exla::nif::error(env, "Unable to get comparator.");
  }
  if (!exla::nif::get(env, argv[2], &dimension)) {
    return exla::nif::error(env, "Unable to get dimension.");
  }

  xla::XlaOp op = xla::Sort({*operand}, *comparator, dimension);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

// LinAlg Functions

ERL_NIF_TERM cholesky(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }

  xla::XlaOp op = xla::Cholesky(*operand, true);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM eigh(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  bool lower;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get(env, argv[1], &lower)) {
    return exla::nif::error(env, "Unable to get lower flag.");
  }

  xla::SelfAdjointEigResult eigh_result = xla::SelfAdjointEig(*operand,
                                                              lower,
                                                              /*max_iter=*/15,
                                                              /*epsilon=*/1.0e-6);

  ERL_NIF_TERM v = exla::nif::make<xla::XlaOp>(env, eigh_result.v);
  ERL_NIF_TERM w = exla::nif::make<xla::XlaOp>(env, eigh_result.w);

  return exla::nif::ok(env, enif_make_tuple2(env, v, w));
}

ERL_NIF_TERM lu(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }

  xla::LuDecompositionResult lu_result = xla::LuDecomposition(*operand);

  ERL_NIF_TERM lu = exla::nif::make<xla::XlaOp>(env, lu_result.lu);
  ERL_NIF_TERM pivots = exla::nif::make<xla::XlaOp>(env, lu_result.pivots);
  ERL_NIF_TERM permutation = exla::nif::make<xla::XlaOp>(env, lu_result.permutation);

  return exla::nif::ok(env, enif_make_tuple3(env, lu, pivots, permutation));
}

ERL_NIF_TERM qr(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  bool full_matrices;
  int config_int;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get(env, argv[1], &full_matrices)) {
    return exla::nif::error(env, "Unable to get full matrices flag.");
  }
  if (!exla::nif::get(env, argv[2], &config_int)) {
    return exla::nif::error(env, "Unable to get precision configuration.");
  }

  xla::PrecisionConfig::Precision precision;
  switch (config_int) {
    case 0:
      precision = xla::PrecisionConfig::DEFAULT;
      break;
    case 1:
      precision = xla::PrecisionConfig::HIGH;
      break;
    case 2:
      precision = xla::PrecisionConfig::HIGHEST;
      break;
    default:
      LOG(ERROR) << "Invalid precision configuration";
      return exla::nif::error(env, "Invalid precision configuration");
  }

  EXLA_ASSIGN_OR_RETURN_NIF(xla::QRDecompositionResult qr_result,
    xla::QRDecomposition(*operand, full_matrices, 128, precision), env);

  ERL_NIF_TERM q = exla::nif::make<xla::XlaOp>(env, qr_result.q);
  ERL_NIF_TERM r = exla::nif::make<xla::XlaOp>(env, qr_result.r);

  return exla::nif::ok(env, enif_make_tuple2(env, q, r));
}

ERL_NIF_TERM svd(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  int config_int;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get(env, argv[1], &config_int)) {
    return exla::nif::error(env, "Unable to get precision config flag.");
  }

  xla::PrecisionConfig::Precision precision;
  switch (config_int) {
    case 0:
      precision = xla::PrecisionConfig::DEFAULT;
      break;
    case 1:
      precision = xla::PrecisionConfig::HIGH;
      break;
    case 2:
      precision = xla::PrecisionConfig::HIGHEST;
      break;
    default:
      LOG(ERROR) << "Invalid precision configuration";
      return exla::nif::error(env, "Invalid precision configuration");
  }

  xla::SVDResult svd_result = xla::SVD(*operand, /*max_iter=*/100, /*epsilon=*/1.0e-6, precision);

  ERL_NIF_TERM u = exla::nif::make<xla::XlaOp>(env, svd_result.u);
  ERL_NIF_TERM d = exla::nif::make<xla::XlaOp>(env, svd_result.d);
  ERL_NIF_TERM v = exla::nif::make<xla::XlaOp>(env, svd_result.v);

  return exla::nif::ok(env, enif_make_tuple3(env, u, d, v));
}

ERL_NIF_TERM triangular_solve(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 6) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* a;
  xla::XlaOp* b;
  bool left_side;
  bool lower;
  bool unit_diagonal;
  int transpose_a_int;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], a)) {
    return exla::nif::error(env, "Unable to get A.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[1], b)) {
    return exla::nif::error(env, "Unable to get B.");
  }
  if (!exla::nif::get(env, argv[2], &left_side)) {
    return exla::nif::error(env, "Unable to get left side flag.");
  }
  if (!exla::nif::get(env, argv[3], &lower)) {
    return exla::nif::error(env, "Unable to get lower flag.");
  }
  if (!exla::nif::get(env, argv[4], &unit_diagonal)) {
    return exla::nif::error(env, "Unable to get unit diagonal flag.");
  }
  if (!exla::nif::get(env, argv[5], &transpose_a_int)) {
    return exla::nif::error(env, "Unable to get triangular solve transpose options.");
  }

  xla::TriangularSolveOptions::Transpose transpose_a;
  switch (transpose_a_int) {
    case 0:
      transpose_a = xla::TriangularSolveOptions::NO_TRANSPOSE;
      break;
    case 1:
      transpose_a = xla::TriangularSolveOptions::TRANSPOSE;
      break;
    case 2:
      transpose_a = xla::TriangularSolveOptions::ADJOINT;
      break;
    default:
      LOG(ERROR) << "Invalid triangular solve options.";
      return 0;
  }

  xla::XlaOp op = xla::TriangularSolve(*a, *b,
                                       left_side,
                                       lower,
                                       unit_diagonal,
                                       transpose_a);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

// ExlaClient Functions

ERL_NIF_TERM get_host_client(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  int num_replicas;
  int intra_op_parallelism_threads;

  if (!exla::nif::get(env, argv[0], &num_replicas)) {
    return exla::nif::error(env, "Unable to get num_replicas.");
  }
  if (!exla::nif::get(env, argv[1], &intra_op_parallelism_threads)) {
    return exla::nif::error(env, "Unable to get intra_op_parallelism_threads.");
  }
  EXLA_ASSIGN_OR_RETURN_NIF(exla::ExlaClient* client,
    exla::GetHostClient(num_replicas, intra_op_parallelism_threads), env);

  return exla::nif::ok(env, exla::nif::make<exla::ExlaClient*>(env, client));
}

ERL_NIF_TERM get_cuda_client(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  int num_replicas;
  int intra_op_parallelism_threads;

  if (!exla::nif::get(env, argv[0], &num_replicas)) {
    return exla::nif::error(env, "Unable to get number of replicas.");
  }
  if (!exla::nif::get(env, argv[1], &intra_op_parallelism_threads)) {
    return exla::nif::error(env, "Unable to get number of parallelism threads.");
  }
  EXLA_ASSIGN_OR_RETURN_NIF(exla::ExlaClient* client,
    exla::GetGpuClient(num_replicas,
                      intra_op_parallelism_threads,
                      "CUDA"), env);

  return exla::nif::ok(env, exla::nif::make<exla::ExlaClient*>(env, client));
}

ERL_NIF_TERM get_rocm_client(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  int num_replicas;
  int intra_op_parallelism_threads;

  if (!exla::nif::get(env, argv[0], &num_replicas)) {
    return exla::nif::error(env, "Unable to get number of replicas.");
  }
  if (!exla::nif::get(env, argv[1], &intra_op_parallelism_threads)) {
    return exla::nif::error(env, "Unable to get number of parallelism threads.");
  }
  EXLA_ASSIGN_OR_RETURN_NIF(exla::ExlaClient* client,
    exla::GetGpuClient(num_replicas,
                       intra_op_parallelism_threads,
                       "ROCM"), env);

  return exla::nif::ok(env, exla::nif::make<exla::ExlaClient*>(env, client));
}

ERL_NIF_TERM get_default_device_ordinal(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::ExlaClient** client;

  if (!exla::nif::get<exla::ExlaClient*>(env, argv[0], client)) {
    return exla::nif::error(env, "Unable to get client.");
  }

  int device_ordinal = (*client)->client()->default_device_ordinal();

  return exla::nif::ok(env, exla::nif::make(env, device_ordinal));
}

ERL_NIF_TERM get_device_count(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::ExlaClient** client;

  if (!exla::nif::get<exla::ExlaClient*>(env, argv[0], client)) {
    return exla::nif::error(env, "Unable to get client.");
  }

  int device_count = (*client)->client()->device_count();

  return exla::nif::ok(env, exla::nif::make(env, device_count));
}

ERL_NIF_TERM get_supported_platforms(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 0) {
    return exla::nif::error(env, "Bad argument count.");
  }

  EXLA_ASSIGN_OR_RETURN_NIF(
    std::vector<stream_executor::Platform*> platforms,
    xla::PlatformUtil::GetSupportedPlatforms(),
    env);

  std::vector<std::string> platform_names;
  std::map<std::string, int> platform_info;

  for (auto& platform : platforms) {
    std::string key = platform->Name();
    int device_count = platform->VisibleDeviceCount();
    platform_info.insert({key, device_count});
  }

  return exla::nif::ok(env, exla::nif::make_map(env, platform_info));
}

ERL_NIF_TERM compile(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 6) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::ExlaClient** client;
  xla::XlaComputation* computation;
  std::vector<xla::Shape*> argument_layouts;
  xla::ExecutableBuildOptions build_options;
  int num_replicas;
  int num_partitions;
  bool use_spmd;

  if (!exla::nif::get<exla::ExlaClient*>(env, argv[0], client)) {
    return exla::nif::error(env, "Unable to get client.");
  }
  if (!exla::nif::get<xla::XlaComputation>(env, argv[1], computation)) {
    return exla::nif::error(env, "Unable to get computation.");
  }
  if (!exla::nif::get_list<xla::Shape>(env, argv[2], argument_layouts)) {
    return exla::nif::error(env, "Unable to get argument layouts.");
  }
  if (!exla::nif::get(env, argv[3], &num_replicas)) {
    return exla::nif::error(env, "Unable to get Number of Replicas.");
  }
  if (!exla::nif::get(env, argv[4], &num_partitions)) {
    return exla::nif::error(env, "Unable to get Number of Partitions.");
  }
  if (!exla::nif::get(env, argv[5], &use_spmd)) {
    return exla::nif::error(env, "Unable to get SPMD Partitioning Flag.");
  }

  build_options.set_num_replicas(num_replicas);
  build_options.set_num_partitions(num_partitions);
  build_options.set_use_spmd_partitioning(use_spmd);

  EXLA_ASSIGN_OR_RETURN_NIF(exla::ExlaExecutable* executable,
    (*client)->Compile(*computation, argument_layouts, build_options, false), env);

  return exla::nif::ok(env, exla::nif::make<exla::ExlaExecutable*>(env, executable));
}

ERL_NIF_TERM await_streams(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::ExlaClient** client;
  exla::ExlaBuffer** buffer;
  bool keep_on_device;

  if (!exla::nif::get<exla::ExlaClient*>(env, argv[0], client)) {
    return exla::nif::error(env, "Unable to get client.");
  }
  if (!exla::nif::get(env, argv[1], buffer)) {
    return exla::nif::error(env, "Unable to get buffer.");
  }
  if (!exla::nif::get(env, argv[2], &keep_on_device)) {
    return exla::nif::error(env, "Unable to get keep on device flag.");
  }

  exla::ExlaDevice* device = (*buffer)->device();

  xla::Status status = device->SynchronizeAllActivity();

  if (!status.ok()) {
    return exla::nif::error(env, status.error_message().c_str());
  }

  EXLA_ASSIGN_OR_RETURN_NIF(ERL_NIF_TERM term,
                            exla::ExlaBuffer::DecomposeBufferToTerm(env, *buffer, keep_on_device),
                            env);

  // We've already created a reference to this buffer
  // but we're creating one again in the above, so
  // we need to ensure this buffer doesn't get double
  // freed. This is probably bad design
  if (keep_on_device) {
    *buffer = nullptr;
  }

  return exla::nif::ok(env, term);
}

// ExlaExecutable Functions

ERL_NIF_TERM run(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 11) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::ExlaClient** client;
  exla::ExlaExecutable** executable;
  xla::Shape* output_shape;
  int run_id;
  int rng_seed;
  int launch_id;
  int replica;
  int partition;
  bool async_run;
  bool keep_on_device;

  ERL_NIF_TERM arguments = argv[2];

  if (!exla::nif::get<exla::ExlaClient*>(env, argv[0], client)) {
    return exla::nif::error(env, "Unable to get client.");
  }
  if (!exla::nif::get<exla::ExlaExecutable*>(env, argv[1], executable)) {
    return exla::nif::error(env, "Unable to get executable.");
  }
  if (!exla::nif::get<xla::Shape>(env, argv[3], output_shape)) {
    return exla::nif::error(env, "Unable to get output shape.");
  }
  if (!exla::nif::get(env, argv[4], &run_id)) {
    return exla::nif::error(env, "Unable to get Run ID.");
  }
  if (!exla::nif::get(env, argv[5], &rng_seed)) {
    return exla::nif::error(env, "Unable to get RNG Seed.");
  }
  if (!exla::nif::get(env, argv[6], &launch_id)) {
    return exla::nif::error(env, "Unable to get Launch ID.");
  }
  if (!exla::nif::get(env, argv[7], &replica)) {
    return exla::nif::error(env, "Unable to get replica.");
  }
  if (!exla::nif::get(env, argv[8], &partition)) {
    return exla::nif::error(env, "Unable to get partition.");
  }
  if (!exla::nif::get(env, argv[9], &async_run)) {
    return exla::nif::error(env, "Unable to get async run flag.");
  }
  if (!exla::nif::get(env, argv[10], &keep_on_device)) {
    return exla::nif::error(env, "Unable to get keep on device flag.");
  }

  EXLA_ASSIGN_OR_RETURN_NIF(ERL_NIF_TERM term,
    (*executable)->Run(env, arguments, *output_shape,
                       replica, partition,
                       run_id, rng_seed,
                       launch_id, async_run, keep_on_device), env);

  return term;
}

// Logging Functions

ERL_NIF_TERM start_log_sink(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::make(env, "Bad argument count.");
  }

  ErlNifPid logger_pid;

  if (!enif_get_local_pid(env, argv[0], &logger_pid)) {
    return exla::nif::error(env, "Unable to get logger pid");
  }

  exla::ExlaLogSink* sink = new exla::ExlaLogSink(logger_pid);

  // NO_DEFAULT_LOGGER doesn't behave right
  for (auto *log_sink : tensorflow::TFGetLogSinks()) {
    tensorflow::TFRemoveLogSink(log_sink);
  }

  tensorflow::TFAddLogSink(sink);

  return exla::nif::ok(env);
}

ERL_NIF_TERM compile_aot(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]){
  if (argc != 6) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaComputation* computation;
  std::string aot_path, function_name, pbtext_path, class_name, target_triple;

  if (!exla::nif::get<xla::XlaComputation>(env, argv[0], computation)) {
    return exla::nif::error(env, "Unable to get computation.");
  }
  if (!exla::nif::get(env, argv[1], pbtext_path)) {
    return exla::nif::error(env, "Unable to get Graph Config Path.");
  }
  if (!exla::nif::get(env, argv[2], aot_path)) {
    return exla::nif::error(env, "Unable to get TF Path.");
  }
  if (!exla::nif::get(env, argv[3], function_name)) {
    return exla::nif::error(env, "Unable to get function name.");
  }
  if (!exla::nif::get(env, argv[4], class_name)) {
    return exla::nif::error(env, "Unable to get class name.");
  }
  if (!exla::nif::get(env, argv[5], target_triple)) {
    return exla::nif::error(env, "Unable to get target triple.");
  }

  xla::Status compile_status =
    exla::CompileComputation(*computation,
                             pbtext_path,
                             aot_path,
                             function_name,
                             class_name,
                             target_triple);

  if(!compile_status.ok()) {
    return exla::nif::error(env, compile_status.error_message().c_str());
  }

  std::string object_path = absl::StrCat(aot_path, function_name, ".o");
  std::string header_path = absl::StrCat(aot_path, function_name, ".h");
  ERL_NIF_TERM object_path_term = exla::nif::make(env, object_path.c_str());
  ERL_NIF_TERM header_path_term = exla::nif::make(env, header_path.c_str());

  return exla::nif::ok(env, enif_make_tuple2(env, object_path_term, header_path_term));
}

static ErlNifFunc exla_funcs[] = {
  // XlaBuilder
  {"new_builder", 1, new_builder},
  {"create_sub_builder", 2, create_sub_builder},
  {"build", 2, build},
  {"parameter", 4, parameter},
  // ExlaClient
  {"get_host_client", 2, get_host_client},
  {"get_cuda_client", 2, get_cuda_client},
  {"get_rocm_client", 2, get_rocm_client},
  {"get_device_count", 1, get_device_count},
  {"get_default_device_ordinal", 1, get_default_device_ordinal},
  {"get_supported_platforms", 0, get_supported_platforms},
  {"compile", 6, compile, ERL_NIF_DIRTY_JOB_CPU_BOUND},
  {"await_streams_cpu", 3, await_streams, ERL_NIF_DIRTY_JOB_CPU_BOUND},
  {"await_streams_io", 3, await_streams, ERL_NIF_DIRTY_JOB_IO_BOUND},
  // ExlaBuffer
  {"binary_to_device_mem", 4, binary_to_device_mem, ERL_NIF_DIRTY_JOB_IO_BOUND},
  {"read_device_mem", 2, read_device_mem, ERL_NIF_DIRTY_JOB_IO_BOUND},
  {"deallocate_device_mem", 1, deallocate_device_mem, ERL_NIF_DIRTY_JOB_IO_BOUND},
  // ExlaExecutable
  {"run_io", 11, run, ERL_NIF_DIRTY_JOB_IO_BOUND},
  {"run_cpu", 11, run, ERL_NIF_DIRTY_JOB_CPU_BOUND},
  // Shape
  {"make_shape", 2, make_shape},
  {"make_tuple_shape", 1, make_tuple_shape},
  {"get_shape_info", 1, get_shape_info},
  // Element-wise Binary
  {"add", 3, add},
  {"subtract", 3, sub},
  {"multiply", 3, mul},
  {"divide", 3, div},
  {"remainder", 3, rem},
  {"min", 3, min},
  {"max", 3, max},
  {"bitwise_and", 3, bitwise_and},
  {"bitwise_or", 3, bitwise_or},
  {"bitwise_xor", 3, bitwise_xor},
  {"left_shift", 3, shift_left},
  {"right_shift_logical", 3, shift_right_logical},
  {"right_shift_arithmetic", 3, shift_right_arithmetic},
  {"power", 3, pow},
  {"complex", 3, complex},
  {"atan2", 3, atan2},
  // Element-wise Binary comparison
  {"equal", 3, equal},
  {"not_equal", 3, not_equal},
  {"greater", 3, greater},
  {"greater_equal", 3, greater_equal},
  {"less", 3, less},
  {"less_equal", 3, less_equal},
  // Element-wise Unary
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
  {"cos", 1, cos},
  {"sin", 1, sin},
  {"acos", 1, acos},
  {"asin", 1, asin},
  {"atan", 1, atan},
  {"cosh", 1, cosh},
  {"sinh", 1, sinh},
  {"tanh", 1, tanh},
  {"acosh", 1, acosh},
  {"asinh", 1, asinh},
  {"atanh", 1, atanh},
  {"real", 1, real},
  {"imag", 1, imag},
  {"sqrt", 1, sqrt},
  {"rsqrt", 1, rsqrt},
  {"cbrt", 1, cbrt},
  {"erf", 1, erf},
  {"erfc", 1, erfc},
  {"erf_inv", 1, erf_inv},
  {"is_finite", 1, is_finite},
  {"negate", 1, neg},
  {"conj", 1, conj},
  {"bitwise_not", 1, bitwise_not},
  {"count_leading_zeros", 1, clz},
  {"population_count", 1, population_count},
  // Constant Creation
  {"constant_r0", 3, constant_r0},
  {"constant_from_binary", 3, constant_from_binary},
  // Tuples
  {"tuple", 2, tuple},
  {"get_tuple_element", 2, get_tuple_element},
  // Conditionals
  {"conditional", 5, conditional_if},
  {"conditional", 3, conditional_multi},
  {"select", 3, select},
  // Slicing
  {"slice", 4, slice},
  {"dynamic_slice", 3, dynamic_slice},
  {"dynamic_update_slice", 3, dynamic_update_slice},
  // Tensor Creation
  {"rng_normal", 3, rng_normal},
  {"rng_uniform", 3, rng_uniform},
  {"iota", 3, iota},
  // Functional Ops
  {"reduce", 4, reduce},
  {"variadic_reduce", 5, variadic_reduce},
  {"reduce_window", 7, reduce_window},
  {"map", 4, map},
  // Shape/Type Manipulation
  {"broadcast_in_dim", 3, broadcast_in_dim},
  {"reshape", 2, reshape},
  {"get_shape", 2, get_shape_op},
  {"convert_element_type", 2, convert_element_type},
  {"transpose", 2, transpose},
  // Other
  {"dot", 3, dot},
  {"dot_general", 4, dot_general},
  {"conv_general_dilated", 9, conv_general_dilated},
  {"pad", 3, pad},
  {"clamp", 3, clamp},
  {"reverse", 2, reverse},
  {"concatenate", 3, concatenate},
  {"sort", 3, sort},
  // LinAlg
  {"cholesky", 1, cholesky},
  {"eigh", 2, eigh},
  {"lu", 1, lu},
  {"qr", 3, qr},
  {"triangular_solve", 6, triangular_solve},
  {"svd", 2, svd},
  // Log Sink
  {"start_log_sink", 1, start_log_sink},
  // HLO Functions
  {"compile_aot", 6, compile_aot}
};

ERL_NIF_INIT(Elixir.EXLA.NIF, exla_funcs, &load, NULL, NULL, NULL);
