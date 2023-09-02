#include "builder.h"

#include <memory>

#include "../exla_nif_util.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/primitive_util.h"
#include "xla/types.h"

namespace exla {

int MLIRFunction::get_mlir_type(ErlNifEnv* env, ERL_NIF_TERM term, mlir::Type* type) {
  auto builder = module_->builder();
  std::string type_str;
  if (!exla::nif::get(env, term, type_str)) return 1;

  if (type_str == "u8") {
    *type = builder->getIntegerType(8, false);
    return 0;
  }
  if (type_str == "u16") {
    *type = builder->getIntegerType(16, false);
    return 0;
  }
  if (type_str == "u32") {
    *type = builder->getIntegerType(32, false);
    return 0;
  }
  if (type_str == "u64") {
    *type = builder->getIntegerType(64, false);
    return 0;
  }
  if (type_str == "s8") {
    *type = builder->getIntegerType(8);
    return 0;
  }
  if (type_str == "s16") {
    *type = builder->getIntegerType(16);
    return 0;
  }
  if (type_str == "s32") {
    *type = builder->getIntegerType(32);
    return 0;
  }
  if (type_str == "s64") {
    *type = builder->getIntegerType(64);
    return 0;
  }
  if (type_str == "f16") {
    *type = builder->getF16Type();
    return 0;
  }
  if (type_str == "f32") {
    *type = builder->getF32Type();
    return 0;
  }
  if (type_str == "f64") {
    *type = builder->getF64Type();
    return 0;
  }
  if (type_str == "bf16") {
    *type = builder->getBF16Type();
    return 0;
  }

  return 1;
}

MLIRFunction::MLIRFunction(MLIRModule* module, std::unique_ptr<mlir::func::FuncOp> func)
    : func_(std::move(func)),
      module_(module) {}

mlir::Value MLIRFunction::SubtractOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::mhlo::SubtractOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
  return op;
}

mlir::Value MLIRFunction::ConvertOp(mlir::Value operand, mlir::Type type) {
  mlir::OpBuilder* builder = module_->builder();
  builder->setInsertionPointToEnd(&func_->getBody().back());

  auto op = builder->create<mlir::mhlo::ConvertOp>(builder->getUnknownLoc(), operand, type);
  return op;
}

mlir::Value MLIRFunction::AddOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::mhlo::AddOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
  return op;
}

mlir::Value MLIRFunction::MulOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::mhlo::MulOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
  return op;
}

mlir::Value MLIRFunction::MinOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::mhlo::MinOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
  return op;
}

mlir::Value MLIRFunction::MaxOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::mhlo::MaxOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
  return op;
}

mlir::Value MLIRFunction::RemOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::mhlo::RemOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
  return op;
}

mlir::Value MLIRFunction::PowOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::mhlo::PowOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
  return op;
}

mlir::Value MLIRFunction::DivOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::mhlo::DivOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
  return op;
}

mlir::Value MLIRFunction::Atan2Op(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::mhlo::Atan2Op>(module_->builder()->getUnknownLoc(), lhs, rhs);
  return op;
}

mlir::Value compare_and_return_bool(mlir::OpBuilder* builder, mlir::Value lhs, mlir::Value rhs, mlir::mhlo::ComparisonDirection direction) {
  auto op = builder->create<mlir::mhlo::CompareOp>(builder->getUnknownLoc(), lhs, rhs, direction);
  mlir::Type mlir_bool = builder->getIntegerType(8, false);
  return builder->create<mlir::mhlo::ConvertOp>(builder->getUnknownLoc(), op, mlir_bool);
}

mlir::Value MLIRFunction::EqualOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return compare_and_return_bool(module_->builder(), lhs, rhs, mlir::mhlo::ComparisonDirection::EQ);
}

mlir::Value MLIRFunction::NotEqualOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return compare_and_return_bool(module_->builder(), lhs, rhs, mlir::mhlo::ComparisonDirection::NE);
}

mlir::Value MLIRFunction::LessOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return compare_and_return_bool(module_->builder(), lhs, rhs, mlir::mhlo::ComparisonDirection::LT);
}

mlir::Value MLIRFunction::LessEqualOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return compare_and_return_bool(module_->builder(), lhs, rhs, mlir::mhlo::ComparisonDirection::LE);
}

mlir::Value MLIRFunction::GreaterOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return compare_and_return_bool(module_->builder(), lhs, rhs, mlir::mhlo::ComparisonDirection::GT);
}

mlir::Value MLIRFunction::GreaterEqualOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return compare_and_return_bool(module_->builder(), lhs, rhs, mlir::mhlo::ComparisonDirection::GE);
}

mlir::Value MLIRFunction::AbsOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::AbsOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::ExpOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::ExpOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::Expm1Op(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::Expm1Op>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::FloorOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::FloorOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::CeilOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::CeilOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::RoundOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::RoundOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::LogOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::LogOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::Log1pOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::Log1pOp>(module_->builder()->getUnknownLoc(), operand);
}

// mlir::Value MLIRFunction::SigmoidOp(mlir::Value operand) {
//   module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
//   return module_->builder()->create<mlir::mhlo::SigmoidOp>(module_->builder()->getUnknownLoc(), operand);
// }

mlir::Value MLIRFunction::SignOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::SignOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::ClzOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::ClzOp>(module_->builder()->getUnknownLoc(), operand);
}

// mlir::Value MLIRFunction::CosOp(mlir::Value operand) {
//   module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
//   return module_->builder()->create<mlir::math::CosOp>(module_->builder()->getUnknownLoc(), operand);
// }

// mlir::Value MLIRFunction::SinOp(mlir::Value operand) {
//   module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
//   return module_->builder()->create<mlir::math::SinOp>(module_->builder()->getUnknownLoc(), operand);
// }

// mlir::Value MLIRFunction::AcosOp(mlir::Value operand) {
//   module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
//   return module_->builder()->create<mlir::math::AcosOp>(module_->builder()->getUnknownLoc(), operand);
// }

// mlir::Value MLIRFunction::AsinOp(mlir::Value operand) {
//   module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
//   return module_->builder()->create<mlir::math::AsinOp>(module_->builder()->getUnknownLoc(), operand);
// }

// mlir::Value MLIRFunction::AtanOp(mlir::Value operand) {
//   module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
//   return module_->builder()->create<mlir::math::AtanOp>(module_->builder()->getUnknownLoc(), operand);
// }

// mlir::Value MLIRFunction::CoshOp(mlir::Value operand) {
//   module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
//   return module_->builder()->create<mlir::math::CoshOp>(module_->builder()->getUnknownLoc(), operand);
// }

// mlir::Value MLIRFunction::SinhOp(mlir::Value operand) {
//   module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
//   return module_->builder()->create<mlir::math::SinhOp>(module_->builder()->getUnknownLoc(), operand);
// }

// mlir::Value MLIRFunction::TanhOp(mlir::Value operand) {
//   module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
//   return module_->builder()->create<mlir::math::TanhOp>(module_->builder()->getUnknownLoc(), operand);
// }

// mlir::Value MLIRFunction::AcoshOp(mlir::Value operand) {
//   module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
//   return module_->builder()->create<mlir::math::AcoshOp>(module_->builder()->getUnknownLoc(), operand);
// }

// mlir::Value MLIRFunction::AsinhOp(mlir::Value operand) {
//   module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
//   return module_->builder()->create<mlir::math::AsinhOp>(module_->builder()->getUnknownLoc(), operand);
// }

// mlir::Value MLIRFunction::AtanhOp(mlir::Value operand) {
//   module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
//   return module_->builder()->create<mlir::math::AtanhOp>(module_->builder()->getUnknownLoc(), operand);
// }

mlir::Value MLIRFunction::RealOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::RealOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::ImagOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::ImagOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::SqrtOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::SqrtOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::CbrtOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::CbrtOp>(module_->builder()->getUnknownLoc(), operand);
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

mlir::Type TypeIntToMLIRType(mlir::OpBuilder* builder, int type_int) {
  switch (type_int) {
    case 2:
      return builder->getIntegerType(8);
    case 3:
      return builder->getIntegerType(16);
    case 4:
      return builder->getIntegerType(32);
    case 5:
      return builder->getIntegerType(64);
    case 6:
      return builder->getIntegerType(8, false);
    case 7:
      return builder->getIntegerType(16, false);
    case 8:
      return builder->getIntegerType(32, false);
    case 9:
      return builder->getIntegerType(64, false);
    case 10:
      return builder->getF16Type();
    case 11:
      return builder->getF32Type();
    case 12:
      return builder->getF64Type();
    case 16:
      return builder->getBF16Type();
  }
}

xla::PrimitiveType MLIRTypeToPrimitiveType(mlir::Type type) {
  if (type.isUnsignedInteger(8)) {
    return xla::primitive_util::NativeToPrimitiveType<uint8_t>();
  }
  if (type.isUnsignedInteger(16)) {
    return xla::primitive_util::NativeToPrimitiveType<uint16_t>();
  }
  if (type.isUnsignedInteger(32)) {
    return xla::primitive_util::NativeToPrimitiveType<uint32_t>();
  }
  if (type.isUnsignedInteger(64)) {
    return xla::primitive_util::NativeToPrimitiveType<uint64_t>();
  }
  if (type.isSignlessInteger(8)) {
    return xla::primitive_util::NativeToPrimitiveType<int8_t>();
  }
  if (type.isSignlessInteger(16)) {
    return xla::primitive_util::NativeToPrimitiveType<int16_t>();
  }
  if (type.isSignlessInteger(32)) {
    return xla::primitive_util::NativeToPrimitiveType<int32_t>();
  }
  if (type.isSignlessInteger(64)) {
    return xla::primitive_util::NativeToPrimitiveType<int64_t>();
  }
  if (type.isBF16()) {
    return xla::primitive_util::NativeToPrimitiveType<xla::bfloat16>();
  }
  if (type.isF16()) {
    return xla::primitive_util::NativeToPrimitiveType<xla::half>();
  }
  if (type.isF32()) {
    return xla::primitive_util::NativeToPrimitiveType<float>();
  }
  // type.isF64()
  return xla::primitive_util::NativeToPrimitiveType<double>();
}

mlir::TensorType GetMLIRType(mlir::OpBuilder* builder, std::vector<tsl::int64> dims, int type_int) {
  auto type = TypeIntToMLIRType(builder, type_int);
  return mlir::RankedTensorType::get(dims, type);
}

MLIRFunction* MLIRModule::CreateFunction(
    std::string name,
    std::vector<std::pair<std::vector<tsl::int64>, int>> arg_types,
    std::pair<std::vector<tsl::int64>, int> ret_type) {
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

}  // namespace exla