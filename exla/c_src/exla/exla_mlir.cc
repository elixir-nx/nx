#include "exla_mlir.h"

#include <fine.hpp>
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Region.h"

namespace exla {
MLIRFunction::MLIRFunction(fine::ResourcePtr<MLIRModule> module, std::unique_ptr<mlir::func::FuncOp> func)
    : func_(std::move(func)),
      module_(module) {}

std::vector<fine::ResourcePtr<mlir::Value>> MLIRFunction::Op(
    std::string op_name, std::vector<fine::ResourcePtr<mlir::Value>> operands,
    std::vector<mlir::Type> result_types,
    std::vector<std::tuple<std::string, mlir::Attribute>> attributes,
    std::vector<fine::ResourcePtr<mlir::Region>> regions) {
  auto builder = module_->builder();
  auto context = builder->getContext();

  auto types_range = mlir::TypeRange{llvm::ArrayRef<mlir::Type>{result_types}};

  auto named_attributes = std::vector<mlir::NamedAttribute>{};
  for (auto const &[key, value] : attributes) {
    auto attribute = builder->getNamedAttr(key, value);
    named_attributes.push_back(attribute);
  }


  auto operand_values = std::vector<mlir::Value>();
  operand_values.reserve(operands.size());
  for (const auto &operand : operands) {
    operand_values.push_back(*operand);
  }

  auto operands_range = mlir::ValueRange{llvm::ArrayRef<mlir::Value>{operand_values}};
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

  auto result_values = op->getResults();

  auto results = std::vector<fine::ResourcePtr<mlir::Value>>();
  results.reserve(result_values.size());
  for (const auto &result : result_values) {
    results.push_back(fine::make_resource<mlir::Value>(result));
  }

  return results;
}

std::tuple<fine::ResourcePtr<mlir::Region>, std::vector<fine::ResourcePtr<mlir::Value>>>
MLIRFunction::PushRegion(std::vector<mlir::Type> types) {
  auto context = module_->builder()->getContext();

  auto region = fine::make_resource<mlir::Region>();
  auto & block = region->emplaceBlock();

  for (mlir::Type type : types) {
    block.addArgument(type, mlir::UnknownLoc::get(context));
  }

  auto args = std::vector<fine::ResourcePtr<mlir::Value>>();
  for (auto &arg : block.getArguments()) {
    args.push_back(fine::make_resource<mlir::Value>(arg));
  }

  region_stack.push(region);
  setInsertionPoint();

  return std::make_tuple(region, args);
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

MLIRModule::MLIRModule(fine::ResourcePtr<mlir::MLIRContext> context) {
  context_ = context;
  module_ = mlir::OwningOpRef<mlir::ModuleOp>(mlir::ModuleOp::create(mlir::UnknownLoc::get(context_.get())));
  builder_ = std::make_unique<mlir::OpBuilder>(context_.get());
  builder_->setInsertionPointToStart(module_->getBody());
}

std::unique_ptr<mlir::func::FuncOp> MLIRModule::CreateFunction(
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

  return funcOp;
}

std::string MLIRModule::ToString() {
  auto output_string = std::string{};
  auto output_stream = llvm::raw_string_ostream{output_string};
  module_->print(output_stream);
  return output_string;
}

mlir::Type MLIRModule::ParseType(std::string string) {
  auto type = mlir::parseType(string, context_.get());

  if (type == nullptr) {
    throw std::runtime_error("unable to parse MLIR type: " + string);
  }

  return type;
}

mlir::Attribute MLIRModule::ParseAttribute(std::string string) {
  auto attribute = mlir::parseAttribute(string, context_.get());

  if (attribute == nullptr) {
    throw std::runtime_error("unable to parse MLIR type: " + string);
  }

  return attribute;
}

}  // namespace exla
