#ifndef EXLA_MLIR_BUILDER_H_
#define EXLA_MLIR_BUILDER_H_

#include <stack>

#include "exla_nif_util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/reference/Types.h"
#include "xla/shape.h"
#include "xla/types.h"

namespace exla {

class MLIRModule;

class MLIRFunction {
 public:
  MLIRFunction(MLIRModule *module, std::unique_ptr<mlir::func::FuncOp> func);

  std::vector<mlir::Value> Op(
    std::string op_name,
    std::vector<mlir::Value> operands,
    std::vector<mlir::Type> result_types,
    std::vector<std::pair<std::string, mlir::Attribute>> attributes,
    std::vector<mlir::Region *> regions);

  std::pair<mlir::Region *, std::vector<mlir::Value>> PushRegion(std::vector<mlir::Type> types);
  void PopRegion();

  llvm::MutableArrayRef<mlir::BlockArgument> GetArguments() { return func_->getBody().front().getArguments(); }

  std::shared_ptr<MLIRModule> module() { return module_; }

 private:
  std::shared_ptr<MLIRModule> module_;
  std::unique_ptr<mlir::func::FuncOp> func_;

  std::stack<mlir::Region *> region_stack;

  void setInsertionPoint();
};

class MLIRModule {
 public:
  MLIRModule(mlir::MLIRContext *context);

  MLIRFunction *CreateFunction(
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
  mlir::MLIRContext *context_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  std::unique_ptr<mlir::OpBuilder> builder_;
};

}  // namespace exla

#endif
