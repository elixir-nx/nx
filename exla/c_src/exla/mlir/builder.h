#ifndef EXLA_MLIR_BUILDER_H_
#define EXLA_MLIR_BUILDER_H_

#include "../exla_nif_util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "xla/shape.h"
#include "xla/types.h"

namespace exla {

class MLIRModule;

class MLIRFunction {
 public:
  MLIRFunction(MLIRModule *module, std::unique_ptr<mlir::func::FuncOp> func);

  mlir::Value AddOp(mlir::Value lhs, mlir::Value rhs);
  mlir::Value SubtractOp(mlir::Value lhs, mlir::Value rhs);
  mlir::Value TupleOp(std::vector<mlir::Value> vals);
  mlir::Value GetTupleElementOp(mlir::Value tuple, tsl::int64 index);
  mlir::Value MulOp(mlir::Value lhs, mlir::Value rhs);
  mlir::Value MinOp(mlir::Value lhs, mlir::Value rhs);
  mlir::Value MaxOp(mlir::Value lhs, mlir::Value rhs);
  mlir::Value RemOp(mlir::Value lhs, mlir::Value rhs);
  mlir::Value PowOp(mlir::Value lhs, mlir::Value rhs);
  mlir::Value DivOp(mlir::Value lhs, mlir::Value rhs);
  mlir::Value Atan2Op(mlir::Value lhs, mlir::Value rhs);
  mlir::Value EqualOp(mlir::Value lhs, mlir::Value rhs);
  mlir::Value NotEqualOp(mlir::Value lhs, mlir::Value rhs);
  mlir::Value LessOp(mlir::Value lhs, mlir::Value rhs);
  mlir::Value LessEqualOp(mlir::Value lhs, mlir::Value rhs);
  mlir::Value GreaterOp(mlir::Value lhs, mlir::Value rhs);
  mlir::Value GreaterEqualOp(mlir::Value lhs, mlir::Value rhs);
  mlir::Value BitwiseAndOp(mlir::Value lhs, mlir::Value rhs);
  mlir::Value BitwiseOrOp(mlir::Value lhs, mlir::Value rhs);
  mlir::Value BitwiseXorOp(mlir::Value lhs, mlir::Value rhs);
  mlir::Value BitwiseNotOp(mlir::Value operand);
  mlir::Value ShiftLeftOp(mlir::Value lhs, mlir::Value rhs);
  mlir::Value ShiftRightLogicalOp(mlir::Value lhs, mlir::Value rhs);
  mlir::Value ShiftRightArithmeticOp(mlir::Value lhs, mlir::Value rhs);
  mlir::Value ConvertOp(mlir::Value operand, mlir::Type type);
  mlir::Value BitcastConvertOp(mlir::Value operand, xla::Shape shape);
  mlir::Value PadOp(mlir::Value op, mlir::Value pad, std::vector<int64_t> padding_low, std::vector<int64_t> padding_high, std::vector<int64_t> padding_mid);
  mlir::Value AbsOp(mlir::Value operand);
  mlir::Value RealOp(mlir::Value operand);
  mlir::Value ImagOp(mlir::Value operand);
  mlir::Value ConjOp(mlir::Value operand);
  mlir::Value ExpOp(mlir::Value operand);
  mlir::Value Expm1Op(mlir::Value operand);
  mlir::Value FloorOp(mlir::Value operand);
  mlir::Value CeilOp(mlir::Value operand);
  mlir::Value RoundOp(mlir::Value operand);
  mlir::Value LogOp(mlir::Value operand);
  mlir::Value LogisticOp(mlir::Value operand);
  mlir::Value Log1pOp(mlir::Value operand);
  mlir::Value SignOp(mlir::Value operand);
  mlir::Value CosOp(mlir::Value operand);
  mlir::Value SinOp(mlir::Value operand);
  mlir::Value AcosOp(mlir::Value operand);
  mlir::Value AsinOp(mlir::Value operand);
  mlir::Value AtanOp(mlir::Value operand);
  mlir::Value CoshOp(mlir::Value operand);
  mlir::Value SinhOp(mlir::Value operand);
  mlir::Value TanhOp(mlir::Value operand);
  mlir::Value AcoshOp(mlir::Value operand);
  mlir::Value AsinhOp(mlir::Value operand);
  mlir::Value AtanhOp(mlir::Value operand);
  mlir::Value SqrtOp(mlir::Value operand);
  mlir::Value CbrtOp(mlir::Value operand);
  mlir::Value NegateOp(mlir::Value operand);
  mlir::Value ErfOp(mlir::Value operand);
  mlir::Value ErfInvOp(mlir::Value operand);
  mlir::Value ErfcOp(mlir::Value operand);
  mlir::Value IsFiniteOp(mlir::Value operand);
  mlir::Value IsInfOp(mlir::Value operand);
  mlir::Value IsNanOp(mlir::Value operand);
  mlir::Value RsqrtOp(mlir::Value operand);
  mlir::Value ClzOp(mlir::Value operand);
  mlir::Value PopulationCountOp(mlir::Value operand);
  mlir::Value IotaOp(xla::Shape shape, int64_t dimension);
  mlir::Value TransposeOp(mlir::Value operand, std::vector<int64_t> axes);
  mlir::Value ReshapeOp(mlir::Value operand, std::vector<int64_t> target_shape);
  mlir::Value ReverseOp(mlir::Value operand, std::vector<int64_t> dims);
  mlir::Value SliceOp(mlir::Value operand, std::vector<int64_t> starts, std::vector<int64_t> limites, std::vector<int64_t> strides);
  std::vector<mlir::Value> SortOp(std::vector<mlir::Value> operand, int64_t dim, bool desc);
  mlir::Value DynamicSliceOp(mlir::Value operand, std::vector<mlir::Value> starts, std::vector<int64_t> lengths);
  mlir::Value BroadcastInDimOp(mlir::Value operand, xla::Shape result_shape, std::vector<int64_t> axes);
  mlir::Value DotGeneralOp(xla::Shape output_shape, mlir::Value lhs, mlir::Value rhs, xla::DotDimensionNumbers dnums, xla::PrecisionConfig config);
  mlir::Value ConcatenateOp(std::vector<mlir::Value> operands, int64_t dimension);
  mlir::Value OptimizationBarrierOp(mlir::Value operand);
  mlir::Value ClampOp(mlir::Value min, mlir::Value operand, mlir::Value max);
  mlir::Value SelectOp(mlir::Value pred, mlir::Value on_true, mlir::Value on_false);
  mlir::Value ScatterOp(mlir::Value target, mlir::Value indices, mlir::Value updates, bool add_or_put);
  mlir::Value SelectAndScatterOp(mlir::Value target, mlir::Value source, mlir::Value init_value, bool gt_or_lt, std::vector<int64_t> window_dimensions, std::vector<int64_t> window_strides, std::vector<int64_t> padding);
  mlir::Value FFTOp(mlir::Value tensor, bool forward_fft, std::vector<int64_t> fft_length);
  mlir::Value ConvOp(mlir::Value tensor, mlir::Value kernel, std::vector<int64_t> window_strides, std::vector<int64_t> padding, std::vector<int64_t> tensor_dilation, std::vector<int64_t> kernel_dilation, xla::ConvolutionDimensionNumbers dimension_numbers, uint64_t feature_group_count, uint64_t batch_group_count, uint64_t precision_config, std::vector<int64_t> output_dims);
  mlir::Value CreateTokenOp();
  mlir::Value TriangularSolveOp(mlir::Value a, mlir::Value b, bool left_side, bool lower, bool transpose_a);
  mlir::Value DynamicUpdateSliceOp(mlir::Value operand, mlir::Value update, std::vector<mlir::Value> start_indices);
  ERL_NIF_TERM ConstantOp(mlir::Type type, ErlNifEnv *env, ERL_NIF_TERM value_ptr, std::vector<int64_t> dims = {});
  int get_mlir_type(ErlNifEnv *env, ERL_NIF_TERM term, mlir::Type *type);

  void Build(mlir::Value root);

  llvm::MutableArrayRef<mlir::BlockArgument> get_arguments() { return func_->getBody().front().getArguments(); }

 private:
  std::shared_ptr<MLIRModule> module_;
  std::unique_ptr<mlir::func::FuncOp> func_;
};

class MLIRModule {
 public:
  MLIRModule();

  MLIRFunction *CreateFunction(
      std::string name,
      std::vector<xla::Shape *> arg_shapes,
      xla::Shape *ret_shape);

  mlir::ModuleOp module() { return module_.get(); }
  mlir::OpBuilder *builder() { return builder_.get(); }
  mlir::MLIRContext *context() { return context_.get(); }

 private:
  std::unique_ptr<mlir::MLIRContext> context_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  std::unique_ptr<mlir::OpBuilder> builder_;

  std::vector<mlir::Type> input_types_;
};

mlir::Type TypeIntToMLIRType(mlir::OpBuilder *builder, xla::PrimitiveType type_int);

xla::PrimitiveType MLIRTypeToPrimitiveType(mlir::Type);
}  // namespace exla

#endif