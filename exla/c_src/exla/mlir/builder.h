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
  mlir::Value ShiftLeftOp(mlir::Value lhs, mlir::Value rhs);
  mlir::Value ConvertOp(mlir::Value operand, mlir::Type type);
  mlir::Value AbsOp(mlir::Value operand);
  mlir::Value ExpOp(mlir::Value operand);
  mlir::Value Expm1Op(mlir::Value operand);
  mlir::Value FloorOp(mlir::Value operand);
  mlir::Value CeilOp(mlir::Value operand);
  mlir::Value RoundOp(mlir::Value operand);
  mlir::Value LogOp(mlir::Value operand);
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
  mlir::Value IotaOp(xla::Shape shape, int64_t dimension);

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
      std::vector<std::pair<std::vector<tsl::int64>, int>> arg_types,
      std::pair<std::vector<tsl::int64>, int> ret_type);

  mlir::ModuleOp module() { return module_.get(); }
  mlir::OpBuilder *builder() { return builder_.get(); }
  mlir::MLIRContext *context() { return context_.get(); }

 private:
  std::unique_ptr<mlir::MLIRContext> context_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  std::unique_ptr<mlir::OpBuilder> builder_;

  std::vector<mlir::Type> input_types_;
};

mlir::Type TypeIntToMLIRType(mlir::OpBuilder *builder, int type_int);

xla::PrimitiveType MLIRTypeToPrimitiveType(mlir::Type);
}  // namespace exla

#endif