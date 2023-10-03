
#include "ops.h"

#include <llvm/Support/raw_os_ostream.h>

#include "../exla_client.h"
#include "../exla_nif_util.h"
#include "xla/shape_util.h"

// MLIR Builder Functions

ERL_NIF_TERM mlir_compile(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 7) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::ExlaClient** client;
  exla::MLIRModule** module;
  std::vector<xla::Shape*> argument_layouts;
  xla::ExecutableBuildOptions build_options;
  int num_replicas;
  int num_partitions;
  bool use_spmd;
  int device_id;

  if (!exla::nif::get<exla::ExlaClient*>(env, argv[0], client)) {
    return exla::nif::error(env, "Unable to get client.");
  }
  if (!exla::nif::get<exla::MLIRModule*>(env, argv[1], module)) {
    return exla::nif::error(env, "Unable to get module.");
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
  if (!exla::nif::get(env, argv[6], &device_id)) {
    return exla::nif::error(env, "Unable to get device ID.");
  }

  build_options.set_num_replicas(num_replicas);
  build_options.set_num_partitions(num_partitions);
  build_options.set_use_spmd_partitioning(use_spmd);

  bool compile_portable_executable = false;
  if (device_id >= 0) {
    compile_portable_executable = true;
    build_options.set_device_ordinal(device_id);
  }

  EXLA_ASSIGN_OR_RETURN_NIF(exla::ExlaExecutable * executable,
                            (*client)->Compile((*module)->module(), argument_layouts, build_options, compile_portable_executable), env);

  return exla::nif::ok(env, exla::nif::make<exla::ExlaExecutable*>(env, executable));
}

ERL_NIF_TERM new_mlir_module(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 0) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRModule* module = new exla::MLIRModule();

  return exla::nif::ok(env, exla::nif::make<exla::MLIRModule*>(env, module));
}

ERL_NIF_TERM create_mlir_function(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRModule** module;
  std::string func_name;
  std::vector<std::pair<std::vector<exla::int64>, xla::PrimitiveType>> arg_types;
  std::pair<std::vector<exla::int64>, xla::PrimitiveType> ret_type;
  std::vector<xla::Shape*> arg_shapes;

  if (!exla::nif::get<exla::MLIRModule*>(env, argv[0], module)) {
    return exla::nif::error(env, "Unable to get module.");
  }
  if (!exla::nif::get(env, argv[1], func_name)) {
    return exla::nif::error(env, "Unable to get function name.");
  }

  if (!exla::nif::get_list<xla::Shape>(env, argv[2], arg_shapes)) {
    return exla::nif::error(env, "Unable to get args.");
  }

  absl::Span<const int64_t> span;

  for (xla::Shape* shape : arg_shapes) {
    xla::PrimitiveType type = shape->element_type();
    if (type == -1) {
      return exla::nif::error(env, "Invalid argument type received.");
    }
    span = shape->dimensions();
    std::vector<exla::int64> dims(span.begin(), span.end());

    arg_types.emplace_back(dims, type);
  }

  xla::Shape* ret_shape;
  if (!exla::nif::get<xla::Shape>(env, argv[3], ret_shape)) {
    return exla::nif::error(env, "Unable to get return.");
  }

  xla::PrimitiveType type = ret_shape->element_type();
  if (type == -1) {
    return exla::nif::error(env, "Invalid output type received.");
  }

  span = ret_shape->dimensions();
  std::vector<exla::int64> ret_dims(span.begin(), span.end());

  ret_type = std::make_pair(ret_dims, type);

  exla::MLIRFunction* func = (*module)->CreateFunction(func_name, arg_types, ret_type);

  return exla::nif::ok(env, exla::nif::make<exla::MLIRFunction*>(env, func));
}

ERL_NIF_TERM get_mlir_function_arguments(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }

  llvm::MutableArrayRef<mlir::BlockArgument> args = (*function)->get_arguments();
  std::vector<ERL_NIF_TERM> terms;
  terms.reserve(args.size());

  for (auto arg : args) {
    ERL_NIF_TERM term = exla::nif::make<mlir::Value>(env, arg);
    terms.push_back(term);
  }

  return exla::nif::ok(env, enif_make_list_from_array(env, terms.data(), terms.size()));
}

ERL_NIF_TERM mlir_tuple(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  std::vector<mlir::Value> vals;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get_list<mlir::Value>(env, argv[1], vals)) {
    return exla::nif::error(env, "Unable to get values.");
  }

  mlir::Value res = (*function)->TupleOp(vals);

  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, res));
}

ERL_NIF_TERM mlir_get_tuple_element(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  mlir::Value* tuple;
  exla::int64 index;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], tuple)) {
    return exla::nif::error(env, "Unable to get tuple.");
  }
  if (!exla::nif::get(env, argv[2], &index)) {
    return exla::nif::error(env, "Unable to get index.");
  }

  mlir::Value res = (*function)->GetTupleElementOp(*tuple, index);

  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, res));
}

ERL_NIF_TERM mlir_binary_op(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[], std::function<mlir::Value(exla::MLIRFunction*, mlir::Value*, mlir::Value*)> op) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  mlir::Value* lhs;
  mlir::Value* rhs;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], lhs)) {
    return exla::nif::error(env, "Unable to get lhs.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[2], rhs)) {
    return exla::nif::error(env, "Unable to get rhs.");
  }

  mlir::Value res = op(*function, lhs, rhs);

  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, res));
}

#define MLIR_BIN_OP(OP) mlir_binary_op(env, argc, argv, [](exla::MLIRFunction* f, mlir::Value* lhs, mlir::Value* rhs) -> mlir::Value { return f->OP(*lhs, *rhs); })

ERL_NIF_TERM mlir_add(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_BIN_OP(AddOp);
}

ERL_NIF_TERM mlir_subtract(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_BIN_OP(SubtractOp);
}

ERL_NIF_TERM mlir_multiply(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_BIN_OP(MulOp);
}

ERL_NIF_TERM mlir_min(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_BIN_OP(MinOp);
}

ERL_NIF_TERM mlir_max(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_BIN_OP(MaxOp);
}

ERL_NIF_TERM mlir_remainder(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_BIN_OP(RemOp);
}

ERL_NIF_TERM mlir_pow(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_BIN_OP(PowOp);
}

ERL_NIF_TERM mlir_divide(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_BIN_OP(DivOp);
}

ERL_NIF_TERM mlir_atan2(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_BIN_OP(Atan2Op);
}

ERL_NIF_TERM mlir_equal(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_BIN_OP(EqualOp);
}

ERL_NIF_TERM mlir_not_equal(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_BIN_OP(NotEqualOp);
}

ERL_NIF_TERM mlir_less(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_BIN_OP(LessOp);
}

ERL_NIF_TERM mlir_less_equal(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_BIN_OP(LessEqualOp);
}

ERL_NIF_TERM mlir_greater(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_BIN_OP(GreaterOp);
}

ERL_NIF_TERM mlir_greater_equal(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_BIN_OP(GreaterEqualOp);
}

ERL_NIF_TERM mlir_bitwise_and(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_BIN_OP(BitwiseAndOp);
}

ERL_NIF_TERM mlir_bitwise_or(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_BIN_OP(BitwiseOrOp);
}

ERL_NIF_TERM mlir_bitwise_xor(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_BIN_OP(BitwiseXorOp);
}

ERL_NIF_TERM mlir_shift_left(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_BIN_OP(ShiftLeftOp);
}

ERL_NIF_TERM mlir_shift_right_logical(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_BIN_OP(ShiftRightLogicalOp);
}

ERL_NIF_TERM mlir_shift_right_arithmetic(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_BIN_OP(ShiftRightArithmeticOp);
}

ERL_NIF_TERM mlir_unary_op(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[], std::function<mlir::Value(exla::MLIRFunction*, mlir::Value*)> op) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  mlir::Value* operand;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }

  mlir::Value res = op(*function, operand);

  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, res));
}

#define MLIR_UNARY_OP(OP) mlir_unary_op(env, argc, argv, [](exla::MLIRFunction* f, mlir::Value* operand) -> mlir::Value { return f->OP(*operand); })

ERL_NIF_TERM mlir_abs(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(AbsOp);
}
ERL_NIF_TERM mlir_exp(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(ExpOp);
}
ERL_NIF_TERM mlir_expm1(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(Expm1Op);
}
ERL_NIF_TERM mlir_floor(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(FloorOp);
}
ERL_NIF_TERM mlir_ceil(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(CeilOp);
}
ERL_NIF_TERM mlir_round(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(RoundOp);
}
ERL_NIF_TERM mlir_log(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(LogOp);
}
ERL_NIF_TERM mlir_sigmoid(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(LogisticOp);
}
ERL_NIF_TERM mlir_log1p(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(Log1pOp);
}
ERL_NIF_TERM mlir_sign(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(SignOp);
}
ERL_NIF_TERM mlir_cos(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(CosOp);
}
ERL_NIF_TERM mlir_sin(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(SinOp);
}
ERL_NIF_TERM mlir_acos(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(AcosOp);
}
ERL_NIF_TERM mlir_asin(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(AsinOp);
}
ERL_NIF_TERM mlir_atan(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(AtanOp);
}
ERL_NIF_TERM mlir_cosh(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(CoshOp);
}
ERL_NIF_TERM mlir_sinh(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(SinhOp);
}
ERL_NIF_TERM mlir_tanh(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(TanhOp);
}
ERL_NIF_TERM mlir_acosh(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(AcoshOp);
}
ERL_NIF_TERM mlir_asinh(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(AsinhOp);
}
ERL_NIF_TERM mlir_atanh(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(AtanhOp);
}
ERL_NIF_TERM mlir_sqrt(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(SqrtOp);
}
ERL_NIF_TERM mlir_cbrt(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(CbrtOp);
}

ERL_NIF_TERM mlir_bitwise_not(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(BitwiseNotOp);
}

ERL_NIF_TERM mlir_negate(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(NegateOp);
}

ERL_NIF_TERM mlir_erf(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(ErfOp);
}

ERL_NIF_TERM mlir_erfc(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(ErfcOp);
}

ERL_NIF_TERM mlir_erf_inv(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(ErfInvOp);
}

ERL_NIF_TERM mlir_is_infinity(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(IsInfOp);
}
ERL_NIF_TERM mlir_is_nan(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(IsNanOp);
}
ERL_NIF_TERM mlir_rsqrt(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(RsqrtOp);
}
ERL_NIF_TERM mlir_clz(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(ClzOp);
}
ERL_NIF_TERM mlir_population_count(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(PopulationCountOp);
}
ERL_NIF_TERM mlir_real(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(RealOp);
}
ERL_NIF_TERM mlir_imag(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(ImagOp);
}
ERL_NIF_TERM mlir_conjugate(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(ConjOp);
}

ERL_NIF_TERM mlir_iota(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  xla::Shape* shape;
  exla::int64 dimension;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<xla::Shape>(env, argv[1], shape)) {
    return exla::nif::error(env, "Unable to get shape.");
  }
  if (!exla::nif::get(env, argv[2], &dimension)) {
    return exla::nif::error(env, "Unable to get dimension");
  }

  mlir::Value res = (*function)->IotaOp(*shape, dimension);
  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, res));
}
ERL_NIF_TERM mlir_reshape(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  std::vector<int64_t> shape;
  mlir::Value* operand;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get_tuple(env, argv[2], shape)) {
    return exla::nif::error(env, "Unable to get shape.");
  }

  mlir::Value res = (*function)->ReshapeOp(*operand, shape);
  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, res));
}

ERL_NIF_TERM mlir_reverse(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  std::vector<int64_t> dims;
  mlir::Value* operand;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get_list(env, argv[2], dims)) {
    return exla::nif::error(env, "Unable to get dims.");
  }

  mlir::Value res = (*function)->ReverseOp(*operand, dims);
  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, res));
}

ERL_NIF_TERM mlir_transpose(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  std::vector<int64_t> axes;
  mlir::Value* operand;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get_list(env, argv[2], axes)) {
    return exla::nif::error(env, "Unable to get axes.");
  }

  mlir::Value res = (*function)->TransposeOp(*operand, axes);
  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, res));
}

ERL_NIF_TERM mlir_slice(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 5) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  std::vector<int64_t> starts, limits, strides;
  mlir::Value* operand;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get_list(env, argv[2], starts)) {
    return exla::nif::error(env, "Unable to get starts.");
  }
  if (!exla::nif::get_list(env, argv[3], limits)) {
    return exla::nif::error(env, "Unable to get lengths.");
  }
  if (!exla::nif::get_list(env, argv[4], strides)) {
    return exla::nif::error(env, "Unable to get strides.");
  }

  mlir::Value res = (*function)->SliceOp(*operand, starts, limits, strides);
  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, res));
}

ERL_NIF_TERM mlir_dynamic_slice(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  std::vector<mlir::Value> starts;
  std::vector<int64_t> lengths;
  mlir::Value* operand;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get_list<mlir::Value>(env, argv[2], starts)) {
    return exla::nif::error(env, "Unable to get starts.");
  }
  if (!exla::nif::get_list(env, argv[3], lengths)) {
    return exla::nif::error(env, "Unable to get lengths.");
  }

  mlir::Value res = (*function)->DynamicSliceOp(*operand, starts, lengths);
  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, res));
}

ERL_NIF_TERM mlir_constant_r0(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  mlir::Type type;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if ((*function)->get_mlir_type(env, argv[2], &type)) {
    return exla::nif::error(env, "Unable to get type string.");
  }

  return (*function)->ConstantOp(type, env, argv[1]);
}
ERL_NIF_TERM mlir_constant_from_binary(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  mlir::Type type;
  std::vector<exla::int64> dims;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if ((*function)->get_mlir_type(env, argv[2], &type)) {
    return exla::nif::error(env, "Unable to get type string.");
  }
  if (!exla::nif::get_tuple(env, argv[3], dims)) {
    return exla::nif::error(env, "Unable to get dims.");
  }

  return (*function)->ConstantOp(type, env, argv[1], dims);
}

ERL_NIF_TERM mlir_dot_general(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 6) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  xla::Shape* output_shape;
  mlir::Value* lhs;
  mlir::Value* rhs;
  xla::DotDimensionNumbers dnums;
  xla::PrecisionConfig config;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<xla::Shape>(env, argv[1], output_shape)) {
    return exla::nif::error(env, "Unable to get shape.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[2], lhs)) {
    return exla::nif::error(env, "Unable to get lhs.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[3], rhs)) {
    return exla::nif::error(env, "Unable to get rhs.");
  }
  if (!exla::nif::get_dot_dimension_numbers(env, argv[4], &dnums)) {
    return exla::nif::error(env, "Unable to get dot dimensions.");
  }
  if (!exla::nif::get_precision_config(env, argv[5], 2, &config)) {
    return exla::nif::error(env, "Unable to get precision configuration.");
  }

  mlir::Value res = (*function)->DotGeneralOp(*output_shape, *lhs, *rhs, dnums, config);
  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, res));
}

ERL_NIF_TERM mlir_select(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  mlir::Value* pred;
  mlir::Value* on_true;
  mlir::Value* on_false;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], pred)) {
    return exla::nif::error(env, "Unable to get pred.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[2], on_true)) {
    return exla::nif::error(env, "Unable to get on true.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[3], on_false)) {
    return exla::nif::error(env, "Unable to get on false.");
  }

  mlir::Value res = (*function)->SelectOp(*pred, *on_true, *on_false);
  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, res));
}

ERL_NIF_TERM mlir_build(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  mlir::Value* root;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], root)) {
    return exla::nif::error(env, "Unable to get root.");
  }

  (*function)->Build(*root);

  return exla::nif::ok(env);
}

ERL_NIF_TERM mlir_convert(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  mlir::Value* t;
  mlir::Type type;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], t)) {
    return exla::nif::error(env, "Unable to get tensor.");
  }
  if ((*function)->get_mlir_type(env, argv[2], &type)) {
    return exla::nif::error(env, "Unable to get type string.");
  }

  mlir::Value result = (*function)->ConvertOp(*t, type);

  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, result));
}

ERL_NIF_TERM mlir_sort(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  std::vector<mlir::Value> operands;
  exla::int64 axis;
  bool desc;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get_list(env, argv[1], operands)) {
    return exla::nif::error(env, "Unable to get operands.");
  }
  if (!exla::nif::get(env, argv[2], &axis)) {
    return exla::nif::error(env, "Unable to get axis.");
  }
  if (!exla::nif::get(env, argv[3], &desc)) {
    return exla::nif::error(env, "Unable to get desc.");
  }
  std::vector<mlir::Value> res = (*function)->SortOp(operands, axis, desc);
  size_t n = res.size();

  std::vector<ERL_NIF_TERM> nif_terms;
  nif_terms.reserve(n);

  for (size_t i = 0; i < n; i++) {
    nif_terms[i] = exla::nif::make<mlir::Value>(env, res[i]);
  }

  auto data = nif_terms.data();
  auto list = enif_make_list_from_array(env, &data[0], n);
  return exla::nif::ok(env, list);
}

ERL_NIF_TERM mlir_bitcast_convert(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  mlir::Value* t;
  mlir::Type type;
  xla::PrimitiveType element_type;
  std::vector<exla::int64> dims;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], t)) {
    return exla::nif::error(env, "Unable to get tensor.");
  }
  if (!exla::nif::get_primitive_type(env, argv[2], &element_type)) {
    return exla::nif::error(env, "Unable to get type.");
  }
  if (!exla::nif::get_tuple(env, argv[3], dims)) {
    return exla::nif::error(env, "Unable to get dimensions.");
  }

  xla::Shape shape = xla::ShapeUtil::MakeShape(element_type, dims);

  mlir::Value result = (*function)->BitcastConvertOp(*t, shape);

  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, result));
}

ERL_NIF_TERM mlir_pad(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 6) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  std::vector<int64_t> padding_high, padding_low, padding_mid;
  mlir::Value *operand, *pad_value;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[2], pad_value)) {
    return exla::nif::error(env, "Unable to get pad value.");
  }
  if (!exla::nif::get_list(env, argv[3], padding_low)) {
    return exla::nif::error(env, "Unable to get padding_low.");
  }
  if (!exla::nif::get_list(env, argv[4], padding_high)) {
    return exla::nif::error(env, "Unable to get padding_high.");
  }
  if (!exla::nif::get_list(env, argv[5], padding_mid)) {
    return exla::nif::error(env, "Unable to get padding_mid.");
  }

  mlir::Value res = (*function)->PadOp(*operand, *pad_value, padding_low, padding_high, padding_mid);
  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, res));
}

ERL_NIF_TERM mlir_optimization_barrier(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  mlir::Value* t;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], t)) {
    return exla::nif::error(env, "Unable to get tensor.");
  }

  mlir::Value result = (*function)->OptimizationBarrierOp(*t);
  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, result));
}

ERL_NIF_TERM mlir_clamp(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  mlir::Value* operand;
  mlir::Value* min;
  mlir::Value* max;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[2], min)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[3], max)) {
    return exla::nif::error(env, "Unable to get operand.");
  }

  mlir::Value result = (*function)->ClampOp(*min, *operand, *max);

  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, result));
}

ERL_NIF_TERM mlir_get_shape(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  mlir::Value* t;

  if (!exla::nif::get<mlir::Value>(env, argv[0], t)) {
    return exla::nif::error(env, "Unable to get tensor.");
  }

  auto mlir_shape = mlir::ValueShapeRange({}).getShape(*t);

  mlir::Type type = mlir_shape.getElementType();
  xla::PrimitiveType element_type = exla::MLIRTypeToPrimitiveType(type);
  mlir::ShapedTypeComponents shape_dims(mlir_shape);

  xla::Shape shape = xla::ShapeUtil::MakeShape(element_type, shape_dims.getDims());

  return exla::nif::ok(env, exla::nif::make<xla::Shape>(env, shape));
}

ERL_NIF_TERM mlir_broadcast_in_dim(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  std::vector<int64_t> axes;
  xla::Shape* output_shape;
  mlir::Value* operand;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<xla::Shape>(env, argv[1], output_shape)) {
    return exla::nif::error(env, "Unable to get shape.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[2], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get_tuple(env, argv[3], axes)) {
    return exla::nif::error(env, "Unable to get broadcast dimensions.");
  }

  mlir::Value res = (*function)->BroadcastInDimOp(*operand, *output_shape, axes);

  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, res));
}

ERL_NIF_TERM mlir_concatenate(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  std::vector<mlir::Value> vals;
  exla::int64 dimension;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get_list<mlir::Value>(env, argv[1], vals)) {
    return exla::nif::error(env, "Unable to get values.");
  }
  if (!exla::nif::get(env, argv[2], &dimension)) {
    return exla::nif::error(env, "Unable to get dimension");
  }

  mlir::Value res = (*function)->ConcatenateOp(vals, dimension);

  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, res));
}

ERL_NIF_TERM dump_mlir_module(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRModule** builder;

  if (!exla::nif::get<exla::MLIRModule*>(env, argv[0], builder)) {
    return exla::nif::error(env, "Unable to get builder.");
  }

  (*builder)->module().dump();

  return exla::nif::ok(env);
}

ERL_NIF_TERM mlir_scatter(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 5) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  mlir::Value *target, *indices, *updates;
  bool add_or_put;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], target)) {
    return exla::nif::error(env, "Unable to get target.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[2], indices)) {
    return exla::nif::error(env, "Unable to get indices.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[3], updates)) {
    return exla::nif::error(env, "Unable to get updates.");
  }
  if (!exla::nif::get(env, argv[4], &add_or_put)) {
    return exla::nif::error(env, "Unable to get add_or_put.");
  }

  mlir::Value res = (*function)->ScatterOp(*target, *indices, *updates, add_or_put);
  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, res));
}

ERL_NIF_TERM mlir_select_and_scatter(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 8) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  mlir::Value *target, *source, *init_value;
  bool add_or_put, gt_or_lt;

  std::vector<int64_t> window_dimensions, window_strides;
  std::vector<std::pair<exla::int64, exla::int64>> padding_config;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], target)) {
    return exla::nif::error(env, "Unable to get target.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[2], source)) {
    return exla::nif::error(env, "Unable to get source.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[3], init_value)) {
    return exla::nif::error(env, "Unable to get init_value.");
  }
  if (!exla::nif::get(env, argv[4], &gt_or_lt)) {
    return exla::nif::error(env, "Unable to get gt_or_lt.");
  }
  if (!exla::nif::get_list(env, argv[5], window_dimensions)) {
    return exla::nif::error(env, "Unable to get window_dimensions.");
  }
  if (!exla::nif::get_list(env, argv[6], window_strides)) {
    return exla::nif::error(env, "Unable to get window_strides.");
  }
  if (!exla::nif::get_general_padding(env, argv[7], padding_config)) {
    return exla::nif::error(env, "Unable to get padding configuration.");
  }

  std::vector<int64_t> padding;

  for (std::pair<exla::int64, exla::int64> item : padding_config) {
    padding.push_back(item.first);
    padding.push_back(item.second);
  }

  mlir::Value res = (*function)->SelectAndScatterOp(*target, *source, *init_value, gt_or_lt, window_dimensions, window_strides, padding);
  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, res));
}

ERL_NIF_TERM mlir_fft(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  mlir::Value* operand;
  bool forward_fft;

  std::vector<int64_t> fft_length;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get(env, argv[2], &forward_fft)) {
    return exla::nif::error(env, "Unable to get forward_fft.");
  }
  if (!exla::nif::get_list(env, argv[3], fft_length)) {
    return exla::nif::error(env, "Unable to get fft_length.");
  }

  mlir::Value res = (*function)->FFTOp(*operand, forward_fft, fft_length);
  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, res));
}

ERL_NIF_TERM mlir_convolution(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 12) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  mlir::Value *tensor, *kernel;
  std::vector<int64_t> strides;
  std::vector<std::pair<int64_t, int64_t>> padding_config;
  std::vector<int64_t> tensor_dilation;
  std::vector<int64_t> kernel_dilation;
  xla::ConvolutionDimensionNumbers dimension_numbers;
  uint64_t feature_group_count, batch_group_count, precision_config;
  std::vector<int64_t> output_dims;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], tensor)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[2], kernel)) {
    return exla::nif::error(env, "Unable to get kernel.");
  }
  if (!exla::nif::get_list(env, argv[3], strides)) {
    return exla::nif::error(env, "Unable to get strides.");
  }
  if (!exla::nif::get_general_padding(env, argv[4], padding_config)) {
    return exla::nif::error(env, "Unable to get padding_config.");
  }
  if (!exla::nif::get_list(env, argv[5], tensor_dilation)) {
    return exla::nif::error(env, "Unable to get operand dilation.");
  }
  if (!exla::nif::get_list(env, argv[6], kernel_dilation)) {
    return exla::nif::error(env, "Unable to get kernel dilation.");
  }
  if (!exla::nif::get_conv_dimension_numbers(env, argv[7], &dimension_numbers)) {
    return exla::nif::error(env, "Unable to get conv dimension numbers.");
  }
  if (!exla::nif::get(env, argv[8], &feature_group_count)) {
    return exla::nif::error(env, "Unable to get feature groups.");
  }
  if (!exla::nif::get(env, argv[9], &batch_group_count)) {
    return exla::nif::error(env, "Unable to get batch groups.");
  }
  if (!exla::nif::get(env, argv[10], &precision_config)) {
    return exla::nif::error(env, "Unable to get precision config.");
  }
  if (!exla::nif::get_list(env, argv[11], output_dims)) {
    return exla::nif::error(env, "Unable to get output_dims.");
  }

  std::vector<int64_t> padding;

  for (std::pair<exla::int64, exla::int64> item : padding_config) {
    padding.push_back(item.first);
    padding.push_back(item.second);
  }

  mlir::Value res = (*function)->ConvOp(
      *tensor,
      *kernel,
      strides,
      padding,
      tensor_dilation,
      kernel_dilation,
      dimension_numbers,
      feature_group_count,
      batch_group_count,
      precision_config,
      output_dims);

  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, res));
}