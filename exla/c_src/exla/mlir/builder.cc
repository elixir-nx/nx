#include "builder.h"

#include <memory>

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltInOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

#include "xla/types.h"

namespace exla {

MLIRFunction::MLIRFunction(MLIRModule * module, std::unique_ptr<mlir::func::FuncOp> func)
  : func_(std::move(func)),
    module_(module) {}

mlir::Value MLIRFunction::SubtractOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::mhlo::SubtractOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
  return op;
}

mlir::Value MLIRFunction::AddOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::mhlo::AddOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
  return op;
}

mlir::Value MLIRFunction::TupleOp(std::vector<mlir::Value> vals) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::mhlo::TupleOp>(module_->builder()->getUnknownLoc(), vals);
  return op;
}

mlir::Value MLIRFunction::GetTupleElementOp(mlir::Value tuple, tsl::int64 index) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::mhlo::GetTupleElementOp>(module_->builder()->getUnknownLoc(), tuple, index);
  return op;
}

void MLIRFunction::Build(mlir::Value root) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::func::ReturnOp>(module_->builder()->getUnknownLoc(), root);
  return;
}

MLIRModule::MLIRModule() {
  context_ = std::make_unique<mlir::MLIRContext>();

  // TODO: Should we load all these up front, infer them, or make it
  // manual?
  context_->loadDialect<mlir::func::FuncDialect>();
  context_->loadDialect<mlir::mhlo::MhloDialect>();

  module_ = mlir::OwningOpRef<mlir::ModuleOp>(mlir::ModuleOp::create(mlir::UnknownLoc::get(context_.get())));
  builder_ = std::make_unique<mlir::OpBuilder>(context_.get());
  builder_->setInsertionPointToStart(module_->getBody());
}

mlir::Type TypeIntToMLIRType(mlir::OpBuilder * builder, int type_int) {
  switch (type_int) {
    case 9:
      return builder->getF32Type();
  }
}

mlir::TensorType GetMLIRType(mlir::OpBuilder * builder, std::vector<tsl::int64> dims, int type_int) {
  auto type = TypeIntToMLIRType(builder, type_int);
  return mlir::RankedTensorType::get(dims, type);
}

MLIRFunction * MLIRModule::CreateFunction(
  std::string name,
  std::vector<std::pair<std::vector<tsl::int64>, int>> arg_types,
  std::pair<std::vector<tsl::int64>, int> ret_type
) {
  std::vector<mlir::Type> types;
  types.reserve(arg_types.size());
  for (auto arg_type : arg_types) {
    mlir::Type type = GetMLIRType(builder_.get(), arg_type.first, arg_type.second);
    types.push_back(type);
  }

  mlir::Type return_type = GetMLIRType(builder_.get(), ret_type.first, ret_type.second);

  auto funcType = builder_->getFunctionType(types, return_type);
  auto loc = builder_->getUnknownLoc();
  auto funcOp = std::make_unique<mlir::func::FuncOp>(mlir::func::FuncOp::create(loc, name, funcType));
  module_->push_back(*funcOp);
  funcOp->addEntryBlock();
  builder_->setInsertionPointToStart(&funcOp->getBody().front());
  return new MLIRFunction(this, std::move(funcOp));
}

} // namespace exla