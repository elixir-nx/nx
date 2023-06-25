#ifndef EXLA_MLIR_BUILDER_H_
#define EXLA_MLIR_BUILDER_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltInOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "xla/types.h"
#include "xla/shape.h"

namespace exla {

class MLIRModule;

class MLIRFunction {
 public:
  MLIRFunction(MLIRModule * module, std::unique_ptr<mlir::func::FuncOp> func);

  mlir::Value AddOp(mlir::Value lhs, mlir::Value rhs);
  mlir::Value SubtractOp(mlir::Value lhs, mlir::Value rhs);
  mlir::Value TupleOp(std::vector<mlir::Value> vals);
  mlir::Value GetTupleElementOp(mlir::Value tuple, tsl::int64 index);

  // TODO: Somehow protect this resource so we cannot
  // specify more operations after return?
  void Build(mlir::Value root);

  llvm::MutableArrayRef<mlir::BlockArgument> get_arguments() { return func_->getBody().front().getArguments(); }

 private:
  std::shared_ptr<MLIRModule> module_;
  std::unique_ptr<mlir::func::FuncOp> func_;
};

class MLIRModule {
 public:
  MLIRModule();

  MLIRFunction * CreateFunction(
    std::string name,
    std::vector<std::pair<std::vector<tsl::int64>, int>> arg_types,
    std::pair<std::vector<tsl::int64>, int> ret_type
  );

  mlir::ModuleOp module() { return module_.get(); }
  mlir::OpBuilder * builder() { return builder_.get(); }
  mlir::MLIRContext * context() { return context_.get(); }

 private:
  std::unique_ptr<mlir::MLIRContext> context_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  std::unique_ptr<mlir::OpBuilder> builder_;

  std::vector<mlir::Type> input_types_;
};

} // namespace exla
#endif