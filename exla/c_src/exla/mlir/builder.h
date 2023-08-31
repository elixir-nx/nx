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
  mlir::Value ShiftLeftOp(mlir::Value lhs, mlir::Value rhs);

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

int get_mlir_type_for_xla_shape(xla::Shape *);
std::vector<exla::int64> get_xla_shape_dims(xla::Shape *);

}  // namespace exla

#endif