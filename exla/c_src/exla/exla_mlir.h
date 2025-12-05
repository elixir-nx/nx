#ifndef EXLA_MLIR_BUILDER_H_
#define EXLA_MLIR_BUILDER_H_

#include <fine.hpp>
#include <stack>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

namespace exla {

class MLIRModule;

class MLIRFunction {
 public:
  MLIRFunction(fine::ResourcePtr<MLIRModule> module, std::unique_ptr<mlir::func::FuncOp> func);

  std::vector<fine::ResourcePtr<mlir::Value>> Op(
      std::string op_name, std::vector<fine::ResourcePtr<mlir::Value>> operands,
      std::vector<mlir::Type> result_types,
      std::vector<std::tuple<std::string, mlir::Attribute>> attributes,
      std::vector<fine::ResourcePtr<mlir::Region>> regions);

  std::tuple<fine::ResourcePtr<mlir::Region>, std::vector<fine::ResourcePtr<mlir::Value>>> PushRegion(std::vector<mlir::Type> types);
  void PopRegion();

  llvm::MutableArrayRef<mlir::BlockArgument> GetArguments() { return func_->getBody().front().getArguments(); }

  mlir::func::FuncOp function() { return *func_; }

  fine::ResourcePtr<MLIRModule> module() { return module_; }

 private:
  fine::ResourcePtr<MLIRModule> module_;
  std::unique_ptr<mlir::func::FuncOp> func_;

  std::stack<fine::ResourcePtr<mlir::Region>> region_stack;

  void setInsertionPoint();
};

class MLIRModule {
 public:
  MLIRModule(fine::ResourcePtr<mlir::MLIRContext> context);

  std::unique_ptr<mlir::func::FuncOp> CreateFunction(
      std::string name,
      std::vector<mlir::Type> arg_types,
      std::vector<mlir::Type> ret_types,
      bool is_public);

  std::string ToString();

  // Note: ParseAttribute and ParseType return nullptr if the parsing fails
  mlir::Type ParseType(std::string);
  mlir::Attribute ParseAttribute(std::string);

  mlir::ModuleOp module() { return module_.get(); }
  mlir::OpBuilder *builder() { return builder_.get(); }

 private:
  fine::ResourcePtr<mlir::MLIRContext> context_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  std::unique_ptr<mlir::OpBuilder> builder_;
};

}  // namespace exla

#endif
