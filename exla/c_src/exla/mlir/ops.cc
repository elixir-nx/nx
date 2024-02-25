
#include "ops.h"

#include <llvm/Support/raw_os_ostream.h>

#include "../exla_client.h"
#include "../exla_nif_util.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
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

ERL_NIF_TERM new_mlir_context(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 0) {
    return exla::nif::error(env, "Bad argument count.");
  }

  mlir::MLIRContext* context = new mlir::MLIRContext();
  context->getOrLoadDialect<mlir::func::FuncDialect>();
  context->getOrLoadDialect<mlir::stablehlo::StablehloDialect>();
  context->getOrLoadDialect<mlir::mhlo::MhloDialect>();
  context->getOrLoadDialect<mlir::chlo::ChloDialect>();

  auto ret = exla::nif::make<mlir::MLIRContext*>(env, context);
  return exla::nif::ok(env, ret);
}

ERL_NIF_TERM new_mlir_module(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  mlir::MLIRContext** ctx;

  if (!exla::nif::get<mlir::MLIRContext*>(env, argv[0], ctx)) {
    return exla::nif::error(env, "Unable to get context.");
  }

  exla::MLIRModule* module = new exla::MLIRModule(*ctx);

  return exla::nif::ok(env, exla::nif::make<exla::MLIRModule*>(env, module));
}

ERL_NIF_TERM create_mlir_function(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 5) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRModule** module;
  std::string func_name;
  std::vector<std::pair<std::vector<exla::int64>, xla::PrimitiveType>> arg_types;
  std::pair<std::vector<exla::int64>, xla::PrimitiveType> ret_type;
  std::vector<xla::Shape*> arg_shapes;
  std::vector<xla::Shape*> ret_shapes;
  bool is_public;

  if (!exla::nif::get<exla::MLIRModule*>(env, argv[0], module)) {
    return exla::nif::error(env, "Unable to get module.");
  }
  if (!exla::nif::get(env, argv[1], func_name)) {
    return exla::nif::error(env, "Unable to get function name.");
  }
  if (!exla::nif::get_list<xla::Shape>(env, argv[2], arg_shapes)) {
    return exla::nif::error(env, "Unable to get args.");
  }
  if (!exla::nif::get_list<xla::Shape>(env, argv[3], ret_shapes)) {
    return exla::nif::error(env, "Unable to get return.");
  }
  if (!exla::nif::get(env, argv[4], &is_public)) {
    return exla::nif::error(env, "Unable to get is_public.");
  }

  exla::MLIRFunction* func = (*module)->CreateFunction(func_name, arg_shapes, ret_shapes, is_public);

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
ERL_NIF_TERM mlir_tan(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return MLIR_UNARY_OP(TanOp);
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
ERL_NIF_TERM mlir_top_k(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  mlir::Value* operand;
  int64_t k;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get(env, argv[1], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get(env, argv[2], &k)) {
    return exla::nif::error(env, "Unable to get k.");
  }

  std::vector<mlir::Value> result = (*function)->TopKOp(*operand, k);
  return exla::nif::ok(env, exla::nif::make_list<mlir::Value>(env, result));
}

ERL_NIF_TERM mlir_sort(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 5) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  std::vector<mlir::Value> operands;
  exla::int64 axis;
  exla::MLIRFunction** comparator;
  bool stable;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get_list(env, argv[1], operands)) {
    return exla::nif::error(env, "Unable to get operands.");
  }
  if (!exla::nif::get(env, argv[2], &axis)) {
    return exla::nif::error(env, "Unable to get axis.");
  }
  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[3], comparator)) {
    return exla::nif::error(env, "Unable to get comparator.");
  }
  if (!exla::nif::get(env, argv[4], &stable)) {
    return exla::nif::error(env, "Unable to get stable flag.");
  }

  std::vector<mlir::Value> res = (*function)->SortOp(*comparator, operands, axis, stable);
  return exla::nif::ok(env, exla::nif::make_list<mlir::Value>(env, res));
}

ERL_NIF_TERM mlir_reduce(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 5) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  exla::MLIRFunction** reducer;
  std::vector<mlir::Value> init_values;
  std::vector<mlir::Value> inputs;
  std::vector<exla::int64> dimensions;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[1], reducer)) {
    return exla::nif::error(env, "Unable to get reducer.");
  }
  if (!exla::nif::get_list(env, argv[2], init_values)) {
    return exla::nif::error(env, "Unable to get init_values.");
  }
  if (!exla::nif::get_list(env, argv[3], inputs)) {
    return exla::nif::error(env, "Unable to get inputs.");
  }
  if (!exla::nif::get_tuple(env, argv[4], dimensions)) {
    return exla::nif::error(env, "Unable to get dimensions.");
  }

  std::vector<mlir::Value> res = (*function)->ReduceOp(*reducer, init_values, inputs, dimensions);
  return exla::nif::ok(env, exla::nif::make_list<mlir::Value>(env, res));
}

ERL_NIF_TERM mlir_window_reduce(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 9) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  exla::MLIRFunction** reducer;
  std::vector<mlir::Value> init_values;
  std::vector<mlir::Value> inputs;
  std::vector<exla::int64> window_dimensions;
  std::vector<exla::int64> window_strides;
  std::vector<exla::int64> input_dilations;
  std::vector<exla::int64> window_dilations;
  std::vector<std::pair<exla::int64, exla::int64>> padding;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[1], reducer)) {
    return exla::nif::error(env, "Unable to get reducer.");
  }
  if (!exla::nif::get_list(env, argv[2], init_values)) {
    return exla::nif::error(env, "Unable to get init_values.");
  }
  if (!exla::nif::get_list(env, argv[3], inputs)) {
    return exla::nif::error(env, "Unable to get inputs.");
  }
  if (!exla::nif::get_tuple(env, argv[4], window_dimensions)) {
    return exla::nif::error(env, "Unable to get window_dimensions.");
  }
  if (!exla::nif::get_tuple(env, argv[5], window_strides)) {
    return exla::nif::error(env, "Unable to get window_strides.");
  }
  if (!exla::nif::get_tuple(env, argv[6], input_dilations)) {
    return exla::nif::error(env, "Unable to get input_dilations.");
  }
  if (!exla::nif::get_tuple(env, argv[7], window_dilations)) {
    return exla::nif::error(env, "Unable to get window_dilations.");
  }
  if (!exla::nif::get_general_padding(env, argv[8], padding)) {
    return exla::nif::error(env, "Unable to get padding.");
  }

  std::vector<mlir::Value> res = (*function)->WindowReduceOp(*reducer,
                                                             init_values,
                                                             inputs,
                                                             window_dimensions,
                                                             window_strides,
                                                             input_dilations,
                                                             window_dilations,
                                                             padding);

  return exla::nif::ok(env, exla::nif::make_list<mlir::Value>(env, res));
}

ERL_NIF_TERM mlir_map(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  exla::MLIRFunction** mapper;
  std::vector<mlir::Value> inputs;
  std::vector<exla::int64> dimensions;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[1], mapper)) {
    return exla::nif::error(env, "Unable to get mapper.");
  }
  if (!exla::nif::get_list(env, argv[2], inputs)) {
    return exla::nif::error(env, "Unable to get inputs.");
  }
  if (!exla::nif::get_tuple(env, argv[3], dimensions)) {
    return exla::nif::error(env, "Unable to get dimensions.");
  }

  mlir::Value result = (*function)->MapOp(*mapper, inputs, dimensions);

  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, result));
}

ERL_NIF_TERM mlir_if(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  mlir::Value* pred;
  std::vector<xla::Shape> output_shapes;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], pred)) {
    return exla::nif::error(env, "Unable to get pred.");
  }
  if (!exla::nif::get_list<xla::Shape>(env, argv[2], output_shapes)) {
    return exla::nif::error(env, "Unable to get output shapes.");
  }

  std::vector<mlir::Value> result = (*function)->IfOp(*pred, output_shapes);
  return exla::nif::ok(env, exla::nif::make_list<mlir::Value>(env, result));
}

ERL_NIF_TERM mlir_set_if_block(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  mlir::Value* node;
  bool true_or_false_branch;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], node)) {
    return exla::nif::error(env, "Unable to get node.");
  }
  if (!exla::nif::get(env, argv[2], &true_or_false_branch)) {
    return exla::nif::error(env, "Unable to get true_or_false_branch.");
  }

  (*function)->SetIfOpBlock(*node, true_or_false_branch);
  return exla::nif::ok(env);
}

ERL_NIF_TERM mlir_pop_region(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }

  (*function)->PopRegion();
  return exla::nif::ok(env);
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

xla::Shape mlir_type_to_xla_shape(mlir::Type type) {
  if (type.isa<mlir::RankedTensorType>()) {
    auto tensorType = type.cast<mlir::RankedTensorType>();
    // Get the shape (dimensions) of the tensor
    std::vector<int64_t> dims = tensorType.getShape();
    auto element_type = tensorType.getElementType();
    return xla::ShapeUtil::MakeShape(exla::MLIRTypeToPrimitiveType(element_type), dims);
  }

  if (type.isa<mlir::TupleType>()) {
    auto tupleType = type.cast<mlir::TupleType>();
    std::vector<xla::Shape> subshapes;

    for (mlir::Type subType : tupleType.getTypes()) {
      // Handle each sub-type in the tuple
      subshapes.push_back(mlir_type_to_xla_shape(subType));
    }
    return xla::ShapeUtil::MakeTupleShape(subshapes);
  }

  auto element_type = exla::MLIRTypeToPrimitiveType(type);

  if (element_type == xla::PrimitiveType::TOKEN) {
    return xla::ShapeUtil::MakeTokenShape();
  }

  return xla::ShapeUtil::MakeShape(element_type, {});
}

ERL_NIF_TERM mlir_get_shape(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  mlir::Value* t;

  if (!exla::nif::get<mlir::Value>(env, argv[0], t)) {
    return exla::nif::error(env, "Unable to get tensor.");
  }

  mlir::Type type = t->getType();
  xla::Shape shape = mlir_type_to_xla_shape(type);

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
  if (argc != 3) {
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
  if (argc != 9) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  mlir::Value *target, *indices, *updates;
  bool add_or_put;
  int64_t indices_rank;
  std::vector<int64_t> update_window_dims, inserted_window_dims, index_dims_to_window_dims;

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
  if (!exla::nif::get(env, argv[5], &indices_rank)) {
    return exla::nif::error(env, "Unable to get indices_rank.");
  }
  if (!exla::nif::get_list(env, argv[6], update_window_dims)) {
    return exla::nif::error(env, "Unable to get update_window_dims.");
  }
  if (!exla::nif::get_list(env, argv[7], inserted_window_dims)) {
    return exla::nif::error(env, "Unable to get inserted_window_dims.");
  }
  if (!exla::nif::get_list(env, argv[8], index_dims_to_window_dims)) {
    return exla::nif::error(env, "Unable to get index_dims_to_window_dims.");
  }

  mlir::Value res = (*function)->ScatterOp(
      *target,
      *indices,
      *updates,
      add_or_put,
      indices_rank,
      update_window_dims,
      inserted_window_dims,
      index_dims_to_window_dims);
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

ERL_NIF_TERM mlir_gather(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 8) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  mlir::Value *source, *indices;

  int64_t index_vector_dim;
  std::vector<int64_t> slice_sizes, offset_dims, collapsed_slice_dims, start_index_map;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], source)) {
    return exla::nif::error(env, "Unable to get source.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[2], indices)) {
    return exla::nif::error(env, "Unable to get indices.");
  }
  if (!exla::nif::get_list(env, argv[3], slice_sizes)) {
    return exla::nif::error(env, "Unable to get slice_sizes.");
  }
  if (!exla::nif::get_list(env, argv[4], offset_dims)) {
    return exla::nif::error(env, "Unable to get offset_dims.");
  }
  if (!exla::nif::get_list(env, argv[5], collapsed_slice_dims)) {
    return exla::nif::error(env, "Unable to get collapsed_slice_dims.");
  }
  if (!exla::nif::get_list(env, argv[6], start_index_map)) {
    return exla::nif::error(env, "Unable to get start_index_map.");
  }
  if (!exla::nif::get(env, argv[7], &index_vector_dim)) {
    return exla::nif::error(env, "Unable to get index_vector_dim.");
  }

  mlir::Value res = (*function)->GatherOp(*source, *indices, offset_dims, collapsed_slice_dims, start_index_map, slice_sizes, index_vector_dim);
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

ERL_NIF_TERM mlir_create_token(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 1) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }

  mlir::Value token = (*function)->CreateTokenOp();

  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, token));
}

ERL_NIF_TERM mlir_triangular_solve(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 6) {
    return exla::nif::error(env, "Bad argument count.");
  }
  // mlir::Value TriangularSolveOp(mlir::Value a, mlir::Value b, bool left_side, bool lower, bool transpose_a);

  exla::MLIRFunction** function;
  mlir::Value *a, *b;
  bool left_side, lower, transpose_a;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], a)) {
    return exla::nif::error(env, "Unable to get a.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[2], b)) {
    return exla::nif::error(env, "Unable to get b.");
  }
  if (!exla::nif::get(env, argv[3], &left_side)) {
    return exla::nif::error(env, "Unable to get left_side.");
  }
  if (!exla::nif::get(env, argv[4], &lower)) {
    return exla::nif::error(env, "Unable to get lower.");
  }
  if (!exla::nif::get(env, argv[5], &transpose_a)) {
    return exla::nif::error(env, "Unable to get transpose_a.");
  }

  mlir::Value res = (*function)->TriangularSolveOp(*a, *b, left_side, lower, transpose_a);

  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, res));
}

ERL_NIF_TERM mlir_dynamic_update_slice(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  mlir::Value *operand, *updates;
  std::vector<mlir::Value> starts;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], operand)) {
    return exla::nif::error(env, "Unable to get operand.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[2], updates)) {
    return exla::nif::error(env, "Unable to get updates.");
  }
  if (!exla::nif::get_list<mlir::Value>(env, argv[3], starts)) {
    return exla::nif::error(env, "Unable to get starts.");
  }

  mlir::Value res = (*function)->DynamicUpdateSliceOp(*operand, *updates, starts);

  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, res));
}

ERL_NIF_TERM mlir_infeed(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  mlir::Value* token;
  xla::Shape* shape;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], token)) {
    return exla::nif::error(env, "Unable to get token.");
  }
  if (!exla::nif::get<xla::Shape>(env, argv[2], shape)) {
    return exla::nif::error(env, "Unable to get shape.");
  }

  mlir::Value infeed = (*function)->InfeedOp(*token, shape);

  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, infeed));
}

ERL_NIF_TERM mlir_outfeed(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  mlir::Value* token;
  std::vector<mlir::Value> inputs;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<mlir::Value>(env, argv[1], token)) {
    return exla::nif::error(env, "Unable to get token.");
  }
  if (!exla::nif::get_list<mlir::Value>(env, argv[2], inputs)) {
    return exla::nif::error(env, "Unable to get inputs.");
  }

  mlir::Value result = (*function)->OutfeedOp(inputs, *token);

  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, result));
}

ERL_NIF_TERM mlir_call(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 3) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction **function, **computation;
  std::vector<mlir::Value> arguments;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get_list<mlir::Value>(env, argv[1], arguments)) {
    return exla::nif::error(env, "Unable to get arguments.");
  }
  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[2], computation)) {
    return exla::nif::error(env, "Unable to get computation.");
  }

  std::vector<mlir::Value> result = (*function)->CallOp(arguments, *computation);

  return exla::nif::ok(env, exla::nif::make_list<mlir::Value>(env, result));
}

ERL_NIF_TERM mlir_while(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 4) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction **function, **pred, **body;
  std::vector<mlir::Value> initial;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[1], pred)) {
    return exla::nif::error(env, "Unable to get pred.");
  }
  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[2], body)) {
    return exla::nif::error(env, "Unable to get body.");
  }
  if (!exla::nif::get_list<mlir::Value>(env, argv[3], initial)) {
    return exla::nif::error(env, "Unable to get initial.");
  }

  std::vector<mlir::Value> result = (*function)->WhileOp(*pred, *body, initial);

  return exla::nif::ok(env, exla::nif::make_list<mlir::Value>(env, result));
}

ERL_NIF_TERM mlir_return(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  if (argc != 2) {
    return exla::nif::error(env, "Bad argument count.");
  }

  exla::MLIRFunction** function;
  std::vector<mlir::Value> operands;

  if (!exla::nif::get<exla::MLIRFunction*>(env, argv[0], function)) {
    return exla::nif::error(env, "Unable to get function.");
  }
  if (!exla::nif::get_list<mlir::Value>(env, argv[1], operands)) {
    return exla::nif::error(env, "Unable to get operands.");
  }

  std::vector<mlir::Value> res = (*function)->ReturnOp(operands);
  return exla::nif::ok(env, exla::nif::make_list<mlir::Value>(env, res));
}