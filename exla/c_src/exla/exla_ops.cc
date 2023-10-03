#include "exla_ops.h"

#include "exla_nif_util.h"
#include "xla/client/lib/lu_decomposition.h"
#include "xla/client/lib/math.h"
#include "xla/client/lib/qr.h"
#include "xla/client/lib/self_adjoint_eig.h"
#include "xla/client/lib/sorting.h"
#include "xla/client/lib/svd.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"

// Parameters

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

// Control Flow

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

ERL_NIF_TERM while_loop(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaComputation* cond_fn;
  xla::XlaComputation* body_fn;
  xla::XlaOp* init_value;

  if (!exla::nif::get<xla::XlaComputation>(env, argv[0], cond_fn)) {
    return exla::nif::error(env, "Unable to get condition computation.");
  }
  if (!exla::nif::get<xla::XlaComputation>(env, argv[1], body_fn)) {
    return exla::nif::error(env, "Unable to get body computation.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[2], init_value)) {
    return exla::nif::error(env, "Unable to get initial value.");
  }

  xla::XlaOp op = xla::While(*cond_fn, *body_fn, *init_value);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM call(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaBuilder** builder;
  xla::XlaComputation* fn;
  // We will always wrap fn to expect a single tuple-arg
  // which contains all args because we can't call xla::Call
  // in a dynamically variadic way.
  std::vector<xla::XlaOp> args;

  if (!exla::nif::get<xla::XlaBuilder*>(env, argv[0], builder)) {
    return exla::nif::error(env, "Unable to get call builder.");
  }
  if (!exla::nif::get_list<xla::XlaOp>(env, argv[1], args)) {
    return exla::nif::error(env, "Unable to get call args.");
  }
  if (!exla::nif::get<xla::XlaComputation>(env, argv[2], fn)) {
    return exla::nif::error(env, "Unable to get body computation.");
  }

  xla::XlaOp op = xla::Call(*builder, *fn, args);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

// Indexing

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

ERL_NIF_TERM gather(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 7) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  xla::XlaOp* start_indices;
  exla::int64 index_vector_dim;
  std::vector<exla::int64> slice_sizes;
  std::vector<exla::int64> offset_dims;
  std::vector<exla::int64> collapsed_slice_dims;
  std::vector<exla::int64> start_index_map;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[1], start_indices)) {
    return exla::nif::error(env, "Unable to get start indices.");
  }
  if (!exla::nif::get(env, argv[2], &index_vector_dim)) {
    return exla::nif::error(env, "Unable to get index vector dim.");
  }
  if (!exla::nif::get_list(env, argv[3], slice_sizes)) {
    return exla::nif::error(env, "Unable to get slice sizes.");
  }
  if (!exla::nif::get_list(env, argv[4], offset_dims)) {
    return exla::nif::error(env, "Unable to get offset dims.");
  }
  if (!exla::nif::get_list(env, argv[5], collapsed_slice_dims)) {
    return exla::nif::error(env, "Unable to get collapsed slice dims.");
  }
  if (!exla::nif::get_list(env, argv[6], start_index_map)) {
    return exla::nif::error(env, "Unable to get start index map.");
  }

  xla::GatherDimensionNumbers dim_numbers;

  dim_numbers.set_index_vector_dim(index_vector_dim);

  for (auto dim : offset_dims) {
    dim_numbers.add_offset_dims(dim);
  }
  for (auto dim : collapsed_slice_dims) {
    dim_numbers.add_collapsed_slice_dims(dim);
  }
  for (auto dim : start_index_map) {
    dim_numbers.add_start_index_map(dim);
  }

  xla::XlaOp op = xla::Gather(*operand, *start_indices, dim_numbers, slice_sizes);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM select_and_scatter(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 8) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  xla::XlaComputation* select_fn;
  std::vector<exla::int64> window_dimensions;
  std::vector<exla::int64> window_strides;
  std::vector<std::pair<exla::int64, exla::int64>> padding_config;
  xla::XlaOp* source;
  xla::XlaOp* init_value;
  xla::XlaComputation* scatter_fn;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get<xla::XlaComputation>(env, argv[1], select_fn)) {
    return exla::nif::error(env, "Unable to get select function.");
  }
  if (!exla::nif::get_tuple(env, argv[2], window_dimensions)) {
    return exla::nif::error(env, "Unable to get window dimensions.");
  }
  if (!exla::nif::get_list(env, argv[3], window_strides)) {
    return exla::nif::error(env, "Unable to get window strides.");
  }
  if (!exla::nif::get_general_padding(env, argv[4], padding_config)) {
    return exla::nif::error(env, "Unable to get padding configuration.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[5], source)) {
    return exla::nif::error(env, "Unable to get source.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[6], init_value)) {
    return exla::nif::error(env, "Unable to get initial value.");
  }
  if (!exla::nif::get<xla::XlaComputation>(env, argv[7], scatter_fn)) {
    return exla::nif::error(env, "Unable to get scatter function.");
  }

  xla::XlaOp op = xla::SelectAndScatterWithGeneralPadding(*operand,
                                                          *select_fn,
                                                          window_dimensions,
                                                          window_strides,
                                                          padding_config,
                                                          *source,
                                                          *init_value,
                                                          *scatter_fn);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM scatter(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 8) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* target;
  xla::XlaOp* indices;
  xla::XlaOp* updates;
  xla::XlaComputation* scatter_fn;
  exla::int64 index_vector_dim;
  std::vector<exla::int64> update_window_dims;
  std::vector<exla::int64> inserted_window_dims;
  std::vector<exla::int64> scatter_dims_to_operand_dims;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], target)) {
    return exla::nif::error(env, "Unable to get target.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[1], indices)) {
    return exla::nif::error(env, "Unable to get indices.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[2], updates)) {
    return exla::nif::error(env, "Unable to get updates.");
  }
  if (!exla::nif::get<xla::XlaComputation>(env, argv[3], scatter_fn)) {
    return exla::nif::error(env, "Unable to get scatter function.");
  }
  if (!exla::nif::get(env, argv[4], &index_vector_dim)) {
    return exla::nif::error(env, "Unable to get index vector dim.");
  }
  if (!exla::nif::get_list(env, argv[5], update_window_dims)) {
    return exla::nif::error(env, "Unable to get update window dims.");
  }
  if (!exla::nif::get_list(env, argv[6], inserted_window_dims)) {
    return exla::nif::error(env, "Unable to get update window dims.");
  }
  if (!exla::nif::get_list(env, argv[7], scatter_dims_to_operand_dims)) {
    return exla::nif::error(env, "Unable to get update window dims.");
  }

  xla::ScatterDimensionNumbers scatter_dim_numbers;

  scatter_dim_numbers.set_index_vector_dim(index_vector_dim);

  for (size_t i = 0; i < update_window_dims.size(); i++) {
    // scatter_dim_numbers.set_update_window_dims(i, update_window_dims[i]);
    scatter_dim_numbers.add_update_window_dims(update_window_dims[i]);
  }

  for (size_t i = 0; i < inserted_window_dims.size(); i++) {
    scatter_dim_numbers.add_inserted_window_dims(inserted_window_dims[i]);
  }

  for (size_t i = 0; i < scatter_dims_to_operand_dims.size(); i++) {
    scatter_dim_numbers.add_scatter_dims_to_operand_dims(scatter_dims_to_operand_dims[i]);
  }

  xla::XlaOp op = xla::Scatter(
      *target,
      *indices,
      *updates,
      *scatter_fn,
      scatter_dim_numbers,
      false);

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
    case xla::PrimitiveType::F16: {
      // Not sure why the pattern breaks here
      double var;
      if (!enif_get_double(env, term, &var)) return exla::nif::error(env, "Unable to get constant");
      exla::float16 val = static_cast<exla::float16>(var);
      op = xla::ConstantR0(*builder, val);
      break;
    }
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

  char* data = const_cast<char*>(reinterpret_cast<char*>(binary.data));
  xla::BorrowingLiteral literal(data, *shape);

  xla::XlaOp op = xla::ConstantLiteral(*builder, literal);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

// Binary Ops

ERL_NIF_TERM xla_binary_op(ErlNifEnv* env,
                           int argc,
                           const ERL_NIF_TERM argv[],
                           xla::XlaOp (*lambda)(xla::XlaOp, xla::XlaOp, absl::Span<const exla::int64>)) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* lhs;
  xla::XlaOp* rhs;
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

ERL_NIF_TERM not_equal(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::Ne);
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

ERL_NIF_TERM atan2(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_binary_op(env, argc, argv, xla::Atan2);
}

// Unary Ops

ERL_NIF_TERM xla_unary_op(ErlNifEnv* env,
                          int argc,
                          const ERL_NIF_TERM argv[],
                          xla::XlaOp (*lambda)(xla::XlaOp)) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;

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

ERL_NIF_TERM sigmoid(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
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

ERL_NIF_TERM is_nan(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, xla::IsNan);
}

ERL_NIF_TERM is_infinity(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return xla_unary_op(env, argc, argv, [](xla::XlaOp op) { return xla::IsInf(op); });
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

// LinAlg

ERL_NIF_TERM execute_fft(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[], const xla::FftType fft_type) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  std::vector<exla::int64> fft_sizes;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }

  if (!exla::nif::get_list(env, argv[1], fft_sizes)) {
    return exla::nif::error(env, "Unable to get fft_sizes.");
  }

  xla::XlaOp op = xla::Fft(*operand, fft_type, fft_sizes);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM fft(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return execute_fft(env, argc, argv, xla::FftType::FFT);
}

ERL_NIF_TERM ifft(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return execute_fft(env, argc, argv, xla::FftType::IFFT);
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
  if (argc != 4) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  bool lower;
  double eps;
  int64_t max_iter;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get(env, argv[1], &lower)) {
    return exla::nif::error(env, "Unable to get lower flag.");
  }
  if (!exla::nif::get(env, argv[2], &eps)) {
    return exla::nif::error(env, "Unable to get eps.");
  }
  if (!exla::nif::get(env, argv[3], &max_iter)) {
    return exla::nif::error(env, "Unable to get max_iter.");
  }

  xla::SelfAdjointEigResult eigh_result = xla::SelfAdjointEig(*operand,
                                                              lower,
                                                              max_iter,
                                                              eps);

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
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  bool full_matrices;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get(env, argv[1], &full_matrices)) {
    return exla::nif::error(env, "Unable to get full matrices flag.");
  }

  xla::XlaOp q, r;

  QrExplicit(*operand, full_matrices, q, r);

  ERL_NIF_TERM q_term = exla::nif::make<xla::XlaOp>(env, q);
  ERL_NIF_TERM r_term = exla::nif::make<xla::XlaOp>(env, r);

  return exla::nif::ok(env, enif_make_tuple2(env, q_term, r_term));
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

ERL_NIF_TERM window_reduce(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 7) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  xla::XlaOp* initial_value;
  xla::XlaComputation* computation;
  std::vector<exla::int64> window_dimensions;
  std::vector<exla::int64> window_strides;
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

ERL_NIF_TERM bitcast_convert_type(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
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

  xla::XlaOp op = xla::BitcastConvertType(*operand, type);

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

// NDim

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
  if (argc != 10) {
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
  exla::int64 batch_group_count;
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
    return exla::nif::error(env, "Unable to get feature groups.");
  }
  if (!exla::nif::get(env, argv[8], &batch_group_count)) {
    return exla::nif::error(env, "Unable to get batch groups.");
  }
  if (!exla::nif::get_precision_config(env, argv[9], 2, &config)) {
    return exla::nif::error(env, "Unable to get precision config.");
  }

  xla::XlaOp op = xla::ConvGeneralDilated(*operand, *kernel, strides,
                                          padding, lhs_dilation, rhs_dilation,
                                          dimension_numbers, feature_group_count,
                                          batch_group_count, &config);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

// Other

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

  xla::XlaOp op = xla::Sort({*operand}, *comparator, dimension, true);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM top_k(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  exla::int64 k;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get(env, argv[1], &k)) {
    return exla::nif::error(env, "Unable to get k.");
  }

  xla::XlaOp op = xla::TopK(*operand, k, xla::PrimitiveType::S64);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM variadic_sort(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  std::vector<xla::XlaOp> operands;
  xla::XlaComputation* comparator;
  exla::int64 dimension;

  if (!exla::nif::get_list<xla::XlaOp>(env, argv[0], operands)) {
    return exla::nif::error(env, "Unable to get operands.");
  }
  if (!exla::nif::get<xla::XlaComputation>(env, argv[1], comparator)) {
    return exla::nif::error(env, "Unable to get comparator.");
  }
  if (!exla::nif::get(env, argv[2], &dimension)) {
    return exla::nif::error(env, "Unable to get dimension.");
  }

  xla::XlaOp op = xla::Sort(operands, *comparator, dimension, true);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

// Infeed/Outfeed

ERL_NIF_TERM infeed(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* token;
  xla::Shape* shape;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], token)) {
    return exla::nif::error(env, "Unable to get token.");
  }
  if (!exla::nif::get<xla::Shape>(env, argv[1], shape)) {
    return exla::nif::error(env, "Unable to get shape.");
  }

  // TODO(seanmor5): Determine if default config is optimal
  xla::XlaOp op = xla::InfeedWithToken(*token, *shape);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM outfeed(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;
  xla::XlaOp* token;
  xla::Shape* shape_with_layout;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get<xla::XlaOp>(env, argv[1], token)) {
    return exla::nif::error(env, "Unable to get token.");
  }
  if (!exla::nif::get<xla::Shape>(env, argv[2], shape_with_layout)) {
    return exla::nif::error(env, "Unable to get shape.");
  }

  // TODO(seanmor5): Determine if default config is optimal
  xla::XlaOp op = xla::OutfeedWithToken(*operand, *token, *shape_with_layout, "");

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}

ERL_NIF_TERM create_token(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaBuilder** builder;

  if (!exla::nif::get<xla::XlaBuilder*>(env, argv[0], builder)) {
    return exla::nif::error(env, "Unable to get builder.");
  }

  xla::XlaOp token = xla::CreateToken(*builder);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, token));
}

// Checkpointing

ERL_NIF_TERM optimization_barrier(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  xla::XlaOp* operand;

  if (!exla::nif::get<xla::XlaOp>(env, argv[0], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }

  xla::XlaOp op = xla::OptimizationBarrier(*operand);

  return exla::nif::ok(env, exla::nif::make<xla::XlaOp>(env, op));
}