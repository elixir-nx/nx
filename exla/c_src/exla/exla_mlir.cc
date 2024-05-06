#include "exla_mlir.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/AsmParser/AsmParser.h"

namespace exla {
MLIRFunction::MLIRFunction(MLIRModule *module, std::unique_ptr<mlir::func::FuncOp> func)
    : func_(std::move(func)),
      module_(module) {}

std::vector<mlir::Value> MLIRFunction::Op(
    std::string op_name, std::vector<mlir::Value> operands,
    std::vector<mlir::Type> result_types,
    std::vector<std::pair<std::string, mlir::Attribute>> attributes,
    std::vector<mlir::Region *> regions) {
  auto builder = module_->builder();
  auto context = builder->getContext();

  auto types_range = mlir::TypeRange{llvm::ArrayRef<mlir::Type>{result_types}};

  auto named_attributes = std::vector<mlir::NamedAttribute>{};
  for (auto const &pair : attributes) {
    auto attribute = builder->getNamedAttr(pair.first, pair.second);
    named_attributes.push_back(attribute);
  }

  auto operands_range = mlir::ValueRange{
      llvm::ArrayRef<mlir::Value>(operands.data(), operands.size())};
  auto attributes_array = llvm::ArrayRef<mlir::NamedAttribute>{named_attributes};

  setInsertionPoint();

  auto op_state = mlir::OperationState{mlir::UnknownLoc::get(context),
                                        builder->getStringAttr(op_name),
                                        operands_range,
                                        types_range,
                                        attributes_array,
                                        {},
                                        {}};

  for (auto region : regions) {
    auto new_region = op_state.addRegion();
    new_region->getBlocks().splice(new_region->end(), region->getBlocks());
  }

  auto op = builder->create(op_state);

  auto results = op->getResults();
  return std::vector<mlir::Value>(results.begin(), results.end());
}

std::pair<mlir::Region *, std::vector<mlir::Value>> MLIRFunction::PushRegion(std::vector<mlir::Type> types) {
  auto context = module_->builder()->getContext();

  auto region = new mlir::Region();
  auto & block = region->emplaceBlock();

  for (mlir::Type type : types) {
    block.addArgument(type, mlir::UnknownLoc::get(context));
  }

  auto args = std::vector<mlir::Value>{};
  for (auto &arg : block.getArguments()) {
    args.push_back(arg);
  }

  region_stack.push(std::move(region));
  setInsertionPoint();

  return {region, args};
}

void MLIRFunction::PopRegion() {
  region_stack.pop();
  setInsertionPoint();
}

void MLIRFunction::setInsertionPoint() {
  if (region_stack.size() == 0) {
    module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  } else {
    module_->builder()->setInsertionPointToEnd(&region_stack.top()->back());
  }
}

MLIRModule::MLIRModule(mlir::MLIRContext *context) {
  context_ = context;
  module_ = mlir::OwningOpRef<mlir::ModuleOp>(mlir::ModuleOp::create(mlir::UnknownLoc::get(context_)));
  builder_ = std::make_unique<mlir::OpBuilder>(context_);
  builder_->setInsertionPointToStart(module_->getBody());
}

MLIRFunction *MLIRModule::CreateFunction(
    std::string name,
    std::vector<mlir::Type> arg_types,
    std::vector<mlir::Type> ret_types,
    bool is_public) {
  auto visibility = is_public ? "public" : "nested";

  auto funcType = builder_->getFunctionType(arg_types, ret_types);
  auto loc = builder_->getUnknownLoc();
  auto funcOp = std::make_unique<mlir::func::FuncOp>(mlir::func::FuncOp::create(loc, name, funcType));
  funcOp->setSymVisibility(visibility);
  module_->push_back(*funcOp);
  funcOp->addEntryBlock();
  builder_->setInsertionPointToStart(&funcOp->getBody().front());

  return new MLIRFunction(this, std::move(funcOp));
}

std::string MLIRModule::ToString() {
  auto output_string = std::string{};
  auto output_stream = llvm::raw_string_ostream{output_string};
  module_->print(output_stream);
  return output_string;
}

mlir::Type MLIRModule::ParseType(std::string string) {
  return mlir::parseType(string, context_);
}

mlir::Attribute MLIRModule::ParseAttribute(std::string string) {
  return mlir::parseAttribute(string, context_);
}

}  // namespace exla
