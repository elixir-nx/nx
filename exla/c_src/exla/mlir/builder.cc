#include "builder.h"

#include <memory>

#include "../exla_nif_util.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/rewriters.h"
#include "mhlo/utils/type_conversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/primitive_util.h"
#include "xla/types.h"

namespace exla {
mlir::Type TypeIntToMLIRType(mlir::OpBuilder *builder, xla::PrimitiveType type_int) {
  // type_int comes from the xla::PrimitiveType enum
  using xla::PrimitiveType;
  switch (type_int) {
    case PrimitiveType::S8:
      return builder->getIntegerType(8);
    case PrimitiveType::S16:
      return builder->getIntegerType(16);
    case PrimitiveType::S32:
      return builder->getIntegerType(32);
    case PrimitiveType::S64:
      return builder->getIntegerType(64);
    case PrimitiveType::U8:
      return builder->getIntegerType(8, false);
    case PrimitiveType::U16:
      return builder->getIntegerType(16, false);
    case PrimitiveType::U32:
      return builder->getIntegerType(32, false);
    case PrimitiveType::U64:
      return builder->getIntegerType(64, false);
    case PrimitiveType::F16:
      return builder->getF16Type();
    case PrimitiveType::F32:
      return builder->getF32Type();
    case PrimitiveType::F64:
      return builder->getF64Type();
    case PrimitiveType::BF16:
      return builder->getBF16Type();
    case PrimitiveType::C64:
      return mlir::ComplexType::get(builder->getF32Type());
    case PrimitiveType::C128:
      return mlir::ComplexType::get(builder->getF64Type());
  }
}

mlir::TensorType GetMLIRType(mlir::OpBuilder *builder, std::vector<tsl::int64> dims, xla::PrimitiveType type_int) {
  auto type = TypeIntToMLIRType(builder, type_int);
  return mlir::RankedTensorType::get(dims, type);
}

mlir::Type GetMLIRFunctionType(mlir::OpBuilder *builder, xla::Shape *shape) {
  if (shape->IsTuple()) {
    // iterate through tuple types
    std::vector<mlir::Type> element_types;
    for (xla::Shape element : shape->tuple_shapes()) {
      mlir::Type element_type;
      if (element.IsTuple()) {
        element_type = GetMLIRFunctionType(builder, &element);
      } else {
        auto span = element.dimensions();
        std::vector<tsl::int64> dims(span.begin(), span.end());
        element_type = GetMLIRType(builder, dims, element.element_type());
      }
      element_types.push_back(element_type);
    }

    mlir::TupleType tuple = mlir::TupleType::get(builder->getContext(), mlir::TypeRange(element_types));
    return tuple;
  }

  auto span = shape->dimensions();
  std::vector<tsl::int64> dims(span.begin(), span.end());
  return GetMLIRType(builder, dims, shape->element_type());
}

mlir::stablehlo::DotDimensionNumbersAttr ConvertDotDimensionNumbersToAttr(mlir::OpBuilder *builder, const xla::DotDimensionNumbers &dotDimNumbers) {
  std::vector<int64_t> lhsContractingVec(dotDimNumbers.lhs_contracting_dimensions().begin(),
                                         dotDimNumbers.lhs_contracting_dimensions().end());
  std::vector<int64_t> rhsContractingVec(dotDimNumbers.rhs_contracting_dimensions().begin(),
                                         dotDimNumbers.rhs_contracting_dimensions().end());
  std::vector<int64_t> lhsBatchVec(dotDimNumbers.lhs_batch_dimensions().begin(),
                                   dotDimNumbers.lhs_batch_dimensions().end());
  std::vector<int64_t> rhsBatchVec(dotDimNumbers.rhs_batch_dimensions().begin(),
                                   dotDimNumbers.rhs_batch_dimensions().end());

  return mlir::stablehlo::DotDimensionNumbersAttr::get(
      builder->getContext(),
      lhsBatchVec,
      rhsBatchVec,
      lhsContractingVec,
      rhsContractingVec);
}

int MLIRFunction::get_mlir_type(ErlNifEnv *env, ERL_NIF_TERM term, mlir::Type *type) {
  auto builder = module_->builder();
  std::string type_str;
  if (!exla::nif::get(env, term, type_str)) return 1;

  if (type_str == "pred") {
    *type = builder->getIntegerType(1);
    return 0;
  }
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
  if (type_str == "c64") {
    *type = mlir::ComplexType::get(builder->getF32Type());
    return 0;
  }
  if (type_str == "c128") {
    *type = mlir::ComplexType::get(builder->getF64Type());
    return 0;
  }

  return 1;
}

mlir::DenseIntElementsAttr Int64ToDenseIntElementsAttr(mlir::OpBuilder *builder, std::vector<int64_t> vec) {
  int64_t num_entries[] = {static_cast<int64_t>(vec.size())};
  auto type = mlir::RankedTensorType::get(llvm::ArrayRef(num_entries, 1), builder->getIntegerType(64));
  auto dense_attr = mlir::DenseElementsAttr::get<int64_t>(type, llvm::ArrayRef<int64_t>(vec.data(), vec.size()));
  return llvm::cast<mlir::DenseIntElementsAttr>(dense_attr);
}

MLIRFunction::MLIRFunction(MLIRModule *module, std::unique_ptr<mlir::func::FuncOp> func)
    : func_(std::move(func)),
      module_(module) {}

mlir::Value MLIRFunction::SubtractOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::stablehlo::SubtractOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
  return op;
}

mlir::Value MLIRFunction::ConvertOp(mlir::Value operand, mlir::Type type) {
  mlir::OpBuilder *builder = module_->builder();
  builder->setInsertionPointToEnd(&func_->getBody().back());

  if (operand.getType().isa<mlir::ComplexType>() && !type.isa<mlir::ComplexType>()) {
    // get the real part of the operand in case we're downcasting from complex to something else
    operand = builder->create<mlir::stablehlo::RealOp>(builder->getUnknownLoc(), operand);
  }

  auto op = builder->create<mlir::stablehlo::ConvertOp>(builder->getUnknownLoc(), operand, type);
  return op;
}

mlir::Value MLIRFunction::BitcastConvertOp(mlir::Value operand, xla::Shape shape) {
  mlir::OpBuilder *builder = module_->builder();
  builder->setInsertionPointToEnd(&func_->getBody().back());

  absl::Span<const int64_t> dimensions_span = shape.dimensions();
  std::vector<int64_t> dimensions(dimensions_span.begin(), dimensions_span.end());

  mlir::TensorType type = GetMLIRType(module_->builder(), dimensions, shape.element_type());

  auto op = builder->create<mlir::stablehlo::BitcastConvertOp>(builder->getUnknownLoc(), type, operand);
  return op;
}

mlir::Value MLIRFunction::AddOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::stablehlo::AddOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
  return op;
}

mlir::Value MLIRFunction::MulOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::stablehlo::MulOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
  return op;
}

mlir::Value MLIRFunction::MinOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::stablehlo::MinOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
  return op;
}

mlir::Value MLIRFunction::MaxOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::stablehlo::MaxOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
  return op;
}

mlir::Value MLIRFunction::RemOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::stablehlo::RemOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
  return op;
}

mlir::Value MLIRFunction::PowOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::stablehlo::PowOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
  return op;
}

mlir::Value MLIRFunction::DivOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::stablehlo::DivOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
  return op;
}

mlir::Value MLIRFunction::Atan2Op(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::stablehlo::Atan2Op>(module_->builder()->getUnknownLoc(), lhs, rhs);
  return op;
}

mlir::Value MLIRFunction::PadOp(mlir::Value op, mlir::Value pad, std::vector<int64_t> padding_low, std::vector<int64_t> padding_high, std::vector<int64_t> padding_mid) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());

  auto padding_low_attr = Int64ToDenseIntElementsAttr(module_->builder(), padding_low);
  auto padding_high_attr = Int64ToDenseIntElementsAttr(module_->builder(), padding_high);
  auto padding_mid_attr = Int64ToDenseIntElementsAttr(module_->builder(), padding_mid);

  return module_->builder()->create<mlir::stablehlo::PadOp>(module_->builder()->getUnknownLoc(), op, pad, padding_low_attr, padding_high_attr, padding_mid_attr);
}

mlir::Value compare_and_return_bool(mlir::OpBuilder *builder, mlir::Value lhs, mlir::Value rhs, mlir::stablehlo::ComparisonDirection direction) {
  auto op = builder->create<mlir::stablehlo::CompareOp>(builder->getUnknownLoc(), lhs, rhs, direction);
  mlir::Type mlir_bool = builder->getIntegerType(8, false);
  return builder->create<mlir::stablehlo::ConvertOp>(builder->getUnknownLoc(), op, mlir_bool);
}

mlir::Value MLIRFunction::EqualOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return compare_and_return_bool(module_->builder(), lhs, rhs, mlir::stablehlo::ComparisonDirection::EQ);
}

mlir::Value MLIRFunction::NotEqualOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return compare_and_return_bool(module_->builder(), lhs, rhs, mlir::stablehlo::ComparisonDirection::NE);
}

mlir::Value MLIRFunction::LessOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return compare_and_return_bool(module_->builder(), lhs, rhs, mlir::stablehlo::ComparisonDirection::LT);
}

mlir::Value MLIRFunction::LessEqualOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return compare_and_return_bool(module_->builder(), lhs, rhs, mlir::stablehlo::ComparisonDirection::LE);
}

mlir::Value MLIRFunction::GreaterOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return compare_and_return_bool(module_->builder(), lhs, rhs, mlir::stablehlo::ComparisonDirection::GT);
}

mlir::Value MLIRFunction::GreaterEqualOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return compare_and_return_bool(module_->builder(), lhs, rhs, mlir::stablehlo::ComparisonDirection::GE);
}

mlir::Value MLIRFunction::ShiftLeftOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::ShiftLeftOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
}

mlir::Value MLIRFunction::ShiftRightLogicalOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::ShiftRightLogicalOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
}

mlir::Value MLIRFunction::ShiftRightArithmeticOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::ShiftRightArithmeticOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
}

mlir::Value MLIRFunction::BitwiseAndOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::AndOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
}

mlir::Value MLIRFunction::BitwiseOrOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::OrOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
}

mlir::Value MLIRFunction::BitwiseNotOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::NotOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::BitwiseXorOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::XorOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
}

mlir::Value MLIRFunction::AbsOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::AbsOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::ExpOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::ExpOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::Expm1Op(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::Expm1Op>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::FloorOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::FloorOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::CeilOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::CeilOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::RoundOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::RoundOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::LogOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::LogOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::LogisticOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::LogisticOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::Log1pOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::Log1pOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::SignOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::SignOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::CosOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::CosineOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::SinOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::SineOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::AcosOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  module_->context()->getOrLoadDialect<mlir::chlo::ChloDialect>();
  return module_->builder()->create<mlir::chlo::AcosOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::AsinOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  module_->context()->getOrLoadDialect<mlir::chlo::ChloDialect>();
  return module_->builder()->create<mlir::chlo::AsinOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::AtanOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  module_->context()->getOrLoadDialect<mlir::chlo::ChloDialect>();
  return module_->builder()->create<mlir::chlo::AtanOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::CoshOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  module_->context()->getOrLoadDialect<mlir::chlo::ChloDialect>();
  return module_->builder()->create<mlir::chlo::CoshOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::SinhOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  module_->context()->getOrLoadDialect<mlir::chlo::ChloDialect>();
  return module_->builder()->create<mlir::chlo::SinhOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::TanhOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::TanhOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::AcoshOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  module_->context()->getOrLoadDialect<mlir::chlo::ChloDialect>();
  return module_->builder()->create<mlir::chlo::AcoshOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::AsinhOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  module_->context()->getOrLoadDialect<mlir::chlo::ChloDialect>();
  return module_->builder()->create<mlir::chlo::AsinhOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::AtanhOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  module_->context()->getOrLoadDialect<mlir::chlo::ChloDialect>();
  return module_->builder()->create<mlir::chlo::AtanhOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::SqrtOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::SqrtOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::CbrtOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::CbrtOp>(module_->builder()->getUnknownLoc(), operand);
}
mlir::Value MLIRFunction::NegateOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::NegOp>(module_->builder()->getUnknownLoc(), operand);
}
mlir::Value MLIRFunction::ErfOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  module_->context()->getOrLoadDialect<mlir::chlo::ChloDialect>();
  return module_->builder()->create<mlir::chlo::ErfOp>(module_->builder()->getUnknownLoc(), operand);
}
mlir::Value MLIRFunction::ErfInvOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  module_->context()->getOrLoadDialect<mlir::chlo::ChloDialect>();
  return module_->builder()->create<mlir::chlo::ErfInvOp>(module_->builder()->getUnknownLoc(), operand);
}
mlir::Value MLIRFunction::ErfcOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  module_->context()->getOrLoadDialect<mlir::chlo::ChloDialect>();
  return module_->builder()->create<mlir::chlo::ErfcOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::IsInfOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  module_->context()->getOrLoadDialect<mlir::chlo::ChloDialect>();
  mlir::Value op = module_->builder()->create<mlir::chlo::IsInfOp>(module_->builder()->getUnknownLoc(), operand);
  mlir::Type mlir_bool = module_->builder()->getIntegerType(8, false);
  return module_->builder()->create<mlir::stablehlo::ConvertOp>(module_->builder()->getUnknownLoc(), op, mlir_bool);
}

mlir::Value MLIRFunction::IsNanOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  module_->context()->getOrLoadDialect<mlir::chlo::ChloDialect>();
  mlir::Type mlir_bool = module_->builder()->getI1Type();

  mlir::Value is_finite_op = module_->builder()->create<mlir::stablehlo::IsFiniteOp>(module_->builder()->getUnknownLoc(), operand);
  is_finite_op = module_->builder()->create<mlir::stablehlo::ConvertOp>(module_->builder()->getUnknownLoc(), is_finite_op, mlir_bool);

  mlir::Value is_inf_op = this->IsInfOp(operand);
  is_inf_op = module_->builder()->create<mlir::stablehlo::ConvertOp>(module_->builder()->getUnknownLoc(), is_inf_op, mlir_bool);

  mlir::Value is_nan_op = this->BitwiseAndOp(this->BitwiseNotOp(is_inf_op), this->BitwiseNotOp(is_finite_op));
  mlir_bool = module_->builder()->getIntegerType(8, false);

  return module_->builder()->create<mlir::stablehlo::ConvertOp>(module_->builder()->getUnknownLoc(), is_nan_op, mlir_bool);
}
mlir::Value MLIRFunction::RsqrtOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::RsqrtOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::RealOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::RealOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::ImagOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::ImagOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::ConjOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  module_->context()->getOrLoadDialect<mlir::chlo::ChloDialect>();
  return module_->builder()->create<mlir::chlo::ConjOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::TransposeOp(mlir::Value operand, std::vector<int64_t> axes) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto axes_attr = Int64ToDenseIntElementsAttr(module_->builder(), axes);
  return module_->builder()->create<mlir::stablehlo::TransposeOp>(module_->builder()->getUnknownLoc(), operand, axes_attr);
}

mlir::Value MLIRFunction::ReshapeOp(mlir::Value operand, std::vector<int64_t> target_shape) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  mlir::RankedTensorType t_in = llvm::cast<mlir::RankedTensorType>(operand.getType());
  mlir::RankedTensorType type = mlir::RankedTensorType::get(target_shape, t_in.getElementType());
  return module_->builder()->create<mlir::stablehlo::ReshapeOp>(module_->builder()->getUnknownLoc(), type, operand);
}

mlir::Value MLIRFunction::ReverseOp(mlir::Value operand, std::vector<int64_t> dims) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto dims_attr = Int64ToDenseIntElementsAttr(module_->builder(), dims);
  return module_->builder()->create<mlir::stablehlo::ReverseOp>(module_->builder()->getUnknownLoc(), operand, dims_attr);
}

class PublicPatternRewriter : public mlir::PatternRewriter {
 public:
  PublicPatternRewriter(mlir::MLIRContext *context) : mlir::PatternRewriter(context) {}
};

static void buildSortComparisonBody(llvm::ArrayRef<mlir::Type> elementTypes,
                                    mlir::stablehlo::ComparisonDirection direction,
                                    std::optional<mlir::StringRef> compare_type,
                                    mlir::Region *body, mlir::OpBuilder *builder) {
  mlir::OpBuilder::InsertionGuard insertionPointGuard(*builder);
  mlir::Location loc = body->getLoc();
  mlir::Block *block = builder->createBlock(body);
  // Add two arguments for each element type.
  for (mlir::Type elementType : elementTypes) {
    // mlir::ShapedType shapedType = mlir::RankedTensorType::get({}, elementType);
    block->addArguments({elementType, elementType}, {loc, loc});
  }
  mlir::stablehlo::ComparisonType type_attr;
  if (compare_type) {
    type_attr = mlir::stablehlo::symbolizeComparisonType(*compare_type).value();
  } else {
    type_attr = mlir::stablehlo::ComparisonType::NOTYPE;
  }
  mlir::BlockArgument arg0 = block->getArgument(0);
  mlir::BlockArgument arg1 = block->getArgument(1);
  mlir::Value compare = builder->create<mlir::stablehlo::CompareOp>(loc, arg0, arg1, direction);
  builder->create<mlir::stablehlo::ReturnOp>(loc, compare);
}

std::vector<mlir::Value> MLIRFunction::SortOp(std::vector<mlir::Value> operands, int64_t dim, bool desc, bool stable) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  std::vector<mlir::Type> element_types;
  element_types.reserve(operands.size());
  std::optional<mlir::StringRef> compare_type = std::nullopt;

  for (auto element : operands) {
    mlir::RankedTensorType ranked_type = llvm::cast<mlir::RankedTensorType>(element.getType());
    mlir::Type type = mlir::RankedTensorType::get({}, ranked_type.getElementType());
    element_types.push_back(type);
    if (type.isa<mlir::FloatType>()) {
      compare_type.emplace("TOTALORDER");
    }
  }

  mlir::stablehlo::ComparisonDirection direction = desc ? mlir::stablehlo::ComparisonDirection::GT : mlir::stablehlo::ComparisonDirection::LT;

  mlir::OpBuilder *builder = module_->builder();

  mlir::ValueRange value_range(operands);
  mlir::stablehlo::SortOp sort_op = builder->create<mlir::stablehlo::SortOp>(
      builder->getUnknownLoc(),
      value_range,
      dim,
      stable);
  buildSortComparisonBody(element_types, direction, compare_type,
                          &sort_op.getComparator(), builder);

  mlir::Operation::result_range results = sort_op.getResults();
  return std::vector<mlir::Value>(results.begin(), results.end());
}

mlir::Value MLIRFunction::SliceOp(mlir::Value operand, std::vector<int64_t> starts, std::vector<int64_t> limits, std::vector<int64_t> strides) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto idx_attr = Int64ToDenseIntElementsAttr(module_->builder(), starts);
  auto lim_attr = Int64ToDenseIntElementsAttr(module_->builder(), limits);
  auto strides_attr = Int64ToDenseIntElementsAttr(module_->builder(), strides);

  return module_->builder()->create<mlir::stablehlo::SliceOp>(
      module_->builder()->getUnknownLoc(),
      operand,
      idx_attr,
      lim_attr,
      strides_attr);
}

mlir::Value MLIRFunction::DynamicSliceOp(mlir::Value operand, std::vector<mlir::Value> starts, std::vector<int64_t> lengths) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto len_attr = Int64ToDenseIntElementsAttr(module_->builder(), lengths);
  mlir::ValueRange starts_range(llvm::ArrayRef<mlir::Value>(starts.data(), starts.size()));

  return module_->builder()
      ->create<mlir::stablehlo::DynamicSliceOp>(
          module_->builder()->getUnknownLoc(),
          operand,
          starts_range,
          len_attr);
}

mlir::Value MLIRFunction::ClzOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::ClzOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::PopulationCountOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::stablehlo::PopulationCountOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::TupleOp(std::vector<mlir::Value> vals) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::stablehlo::TupleOp>(module_->builder()->getUnknownLoc(), vals);
  return op;
}

mlir::Value MLIRFunction::GetTupleElementOp(mlir::Value tuple, tsl::int64 index) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::stablehlo::GetTupleElementOp>(module_->builder()->getUnknownLoc(), tuple, index);
  return op;
}

mlir::Value MLIRFunction::IotaOp(xla::Shape shape, int64_t dimension) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());

  absl::Span<const int64_t> dimensions_span = shape.dimensions();
  std::vector<int64_t> dimensions(dimensions_span.begin(), dimensions_span.end());

  mlir::TensorType type = GetMLIRType(module_->builder(), dimensions, shape.element_type());

  return module_->builder()->create<mlir::stablehlo::IotaOp>(module_->builder()->getUnknownLoc(), type, dimension);
}

mlir::Value MLIRFunction::DotGeneralOp(
    xla::Shape output_shape,
    mlir::Value lhs,
    mlir::Value rhs,
    xla::DotDimensionNumbers dnums,
    xla::PrecisionConfig config) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());

  absl::Span<const int64_t> dimensions_span = output_shape.dimensions();
  std::vector<int64_t> dimensions(dimensions_span.begin(), dimensions_span.end());

  mlir::TensorType output_type = GetMLIRType(module_->builder(), dimensions, output_shape.element_type());
  auto mlir_dnums = ConvertDotDimensionNumbersToAttr(module_->builder(), dnums);

  auto op = module_->builder()->create<mlir::stablehlo::DotGeneralOp>(
      module_->builder()->getUnknownLoc(),
      output_type,
      lhs,
      rhs,
      mlir_dnums,
      nullptr);

  return op;
}

mlir::Value MLIRFunction::BroadcastInDimOp(mlir::Value operand, xla::Shape shape, std::vector<int64_t> axes) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());

  absl::Span<const int64_t> dimensions_span = shape.dimensions();
  std::vector<int64_t> dimensions(dimensions_span.begin(), dimensions_span.end());
  mlir::TensorType result_type = GetMLIRType(module_->builder(), dimensions, shape.element_type());

  auto axes_attr = Int64ToDenseIntElementsAttr(module_->builder(), axes);

  auto op = module_->builder()->create<mlir::stablehlo::BroadcastInDimOp>(module_->builder()->getUnknownLoc(), result_type, operand, axes_attr);
  return op;
}

mlir::Value MLIRFunction::ConcatenateOp(std::vector<mlir::Value> operands, int64_t dimension) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  mlir::ValueRange operands_range(llvm::ArrayRef<mlir::Value>(operands.data(), operands.size()));
  auto op = module_->builder()->create<mlir::stablehlo::ConcatenateOp>(module_->builder()->getUnknownLoc(), operands_range, dimension);
  return op;
}

mlir::Value MLIRFunction::OptimizationBarrierOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::stablehlo::OptimizationBarrierOp>(module_->builder()->getUnknownLoc(), operand);
  return op.getResult()[0];
}

mlir::Value MLIRFunction::ClampOp(mlir::Value min, mlir::Value operand, mlir::Value max) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::stablehlo::ClampOp>(module_->builder()->getUnknownLoc(), min, operand, max);
  return op;
}

mlir::Value MLIRFunction::SelectOp(mlir::Value pred, mlir::Value on_true, mlir::Value on_false) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::stablehlo::SelectOp>(module_->builder()->getUnknownLoc(), pred, on_true, on_false);
  return op;
}

static void buildScatterComputation(mlir::Type element_type, bool add_or_put, mlir::Region *body, mlir::OpBuilder *builder) {
  mlir::OpBuilder::InsertionGuard insertionPointGuard(*builder);
  mlir::Location loc = body->getLoc();
  mlir::Block *block = builder->createBlock(body);
  // Add two arguments for each element type.
  block->addArguments({element_type, element_type}, {loc, loc});

  if (add_or_put) {
    mlir::BlockArgument arg0 = block->getArgument(0);
    mlir::BlockArgument arg1 = block->getArgument(1);
    mlir::Value add = builder->create<mlir::stablehlo::AddOp>(loc, arg0, arg1);
    builder->create<mlir::stablehlo::ReturnOp>(loc, add);
  } else {
    mlir::BlockArgument arg1 = block->getArgument(1);
    builder->create<mlir::stablehlo::ReturnOp>(loc, arg1);
  }
}

mlir::Value MLIRFunction::ScatterOp(mlir::Value target, mlir::Value indices, mlir::Value updates, bool add_or_put, int64_t indices_rank, std::vector<int64_t> update_window_dims, std::vector<int64_t> inserted_window_dims, std::vector<int64_t> index_dims_to_window_dims) {
  auto builder = module_->builder();
  builder->setInsertionPointToEnd(&func_->getBody().back());
  mlir::RankedTensorType type = llvm::cast<mlir::RankedTensorType>(target.getType());
  auto scatter_dimension_numbers = mlir::stablehlo::ScatterDimensionNumbersAttr::get(builder->getContext(), update_window_dims, inserted_window_dims, index_dims_to_window_dims, indices_rank);

  mlir::stablehlo::ScatterOp scatter_op = builder->create<mlir::stablehlo::ScatterOp>(builder->getUnknownLoc(), target, indices, updates, scatter_dimension_numbers);
  mlir::Type computation_operand_type = mlir::RankedTensorType::get({}, type.getElementType());
  buildScatterComputation(computation_operand_type, add_or_put, &scatter_op.getUpdateComputation(), builder);
  return scatter_op.getResult(0);
}

std::vector<mlir::Value> MLIRFunction::ReduceOp(
    MLIRFunction *reducer,
    std::vector<mlir::Value> init_values,
    std::vector<mlir::Value> inputs,
    std::vector<int64_t> dimensions) {
  auto builder = module_->builder();
  builder->setInsertionPointToEnd(&func_->getBody().back());

  mlir::ValueRange init_values_range(init_values);
  mlir::ValueRange inputs_range(inputs);
  mlir::DenseIntElementsAttr dimensions_attr = Int64ToDenseIntElementsAttr(builder, dimensions);

  mlir::stablehlo::ReduceOp reduce_op = builder->create<mlir::stablehlo::ReduceOp>(builder->getUnknownLoc(), inputs_range, init_values_range, dimensions_attr);
  mlir::Region &reduceBody = reduce_op.getRegion();
  mlir::Region &funcBody = reducer->function()->getBody();
  reduceBody.getBlocks().splice(reduceBody.end(), funcBody.getBlocks());

  mlir::Operation::result_range results = reduce_op.getResults();
  return std::vector<mlir::Value>(results.begin(), results.end());
}

mlir::Value MLIRFunction::MapOp(
    MLIRFunction *mapper,
    std::vector<mlir::Value> inputs,
    std::vector<int64_t> dimensions) {
  auto builder = module_->builder();
  builder->setInsertionPointToEnd(&func_->getBody().back());

  mlir::ValueRange inputs_range(inputs);
  mlir::DenseIntElementsAttr dimensions_attr = Int64ToDenseIntElementsAttr(builder, dimensions);

  mlir::stablehlo::MapOp map_op = builder->create<mlir::stablehlo::MapOp>(builder->getUnknownLoc(), inputs[0].getType(), inputs_range, dimensions_attr);

  mlir::Region &mapBody = map_op.getComputation();
  mlir::Region &funcBody = mapper->function()->getBody();
  mapBody.getBlocks().splice(mapBody.end(), funcBody.getBlocks());

  return map_op;
}

mlir::Value MLIRFunction::SelectAndScatterOp(
    mlir::Value target,
    mlir::Value source,
    mlir::Value init_value,
    bool gt_or_lt,
    std::vector<int64_t> window_dimensions,
    std::vector<int64_t> window_strides,
    std::vector<int64_t> padding) {
  auto builder = module_->builder();
  builder->setInsertionPointToEnd(&func_->getBody().back());
  mlir::RankedTensorType type = llvm::cast<mlir::RankedTensorType>(target.getType());
  int64_t rank = type.getShape().size();
  std::vector<int64_t> axes(rank);
  for (int64_t i = 0; i < rank; i++) {
    axes[i] = i;
  }
  auto scatter_dimension_numbers = mlir::stablehlo::ScatterDimensionNumbersAttr::get(builder->getContext(), {}, axes, axes, rank);

  mlir::DenseIntElementsAttr window_dimensions_attr = Int64ToDenseIntElementsAttr(module_->builder(), window_dimensions);
  mlir::DenseIntElementsAttr window_strides_attr = Int64ToDenseIntElementsAttr(module_->builder(), window_strides);

  auto dense_attr_type = mlir::RankedTensorType::get({static_cast<int64_t>(padding.size() / 2), 2}, builder->getIntegerType(64));
  auto dense_attr = mlir::DenseElementsAttr::get<int64_t>(dense_attr_type, llvm::ArrayRef<int64_t>(padding.data(), padding.size()));
  auto padding_attr = llvm::cast<mlir::DenseIntElementsAttr>(dense_attr);

  mlir::stablehlo::SelectAndScatterOp op = builder->create<mlir::stablehlo::SelectAndScatterOp>(
      builder->getUnknownLoc(),
      target,
      source,
      init_value,
      window_dimensions_attr,
      window_strides_attr,
      padding_attr);

  mlir::Type computation_operand_type = mlir::RankedTensorType::get({}, type.getElementType());
  buildScatterComputation(computation_operand_type, true, &op.getScatter(), builder);

  mlir::stablehlo::ComparisonDirection direction = gt_or_lt ? mlir::stablehlo::ComparisonDirection::GT : mlir::stablehlo::ComparisonDirection::LT;
  std::optional<mlir::StringRef> compare_type = std::nullopt;
  if (type.isa<mlir::FloatType>()) {
    compare_type.emplace("TOTALORDER");
  }

  buildSortComparisonBody({computation_operand_type}, direction, compare_type, &op.getSelect(), builder);
  return op.getResult();
}

mlir::Value MLIRFunction::GatherOp(mlir::Value source, mlir::Value indices, std::vector<int64_t> offset_dims, std::vector<int64_t> collapsed_slice_dims, std::vector<int64_t> start_index_map, std::vector<int64_t> slice_sizes, int64_t index_vector_dim) {
  auto builder = module_->builder();
  builder->setInsertionPointToEnd(&func_->getBody().back());
  auto gather_dimension_numbers = mlir::stablehlo::GatherDimensionNumbersAttr::get(builder->getContext(), offset_dims, collapsed_slice_dims, start_index_map, index_vector_dim);
  auto slice_sizes_attr = Int64ToDenseIntElementsAttr(module_->builder(), slice_sizes);
  return builder->create<mlir::stablehlo::GatherOp>(builder->getUnknownLoc(), source, indices, gather_dimension_numbers, slice_sizes_attr, false);
}

mlir::Value MLIRFunction::FFTOp(mlir::Value tensor, bool forward_fft, std::vector<int64_t> fft_length) {
  auto builder = module_->builder();
  builder->setInsertionPointToEnd(&func_->getBody().back());

  auto fft_type = mlir::stablehlo::FftTypeAttr::get(builder->getContext(), forward_fft ? mlir::stablehlo::FftType::FFT : mlir::stablehlo::FftType::IFFT);
  return builder->create<mlir::stablehlo::FftOp>(builder->getUnknownLoc(), tensor, fft_type, Int64ToDenseIntElementsAttr(builder, fft_length));
}

template <typename T>
ERL_NIF_TERM ConstantOpImpl(mlir::OpBuilder *builder, mlir::Type type, ErlNifEnv *env, ERL_NIF_TERM term, std::vector<int64_t> dims) {
  mlir::RankedTensorType ty = mlir::RankedTensorType::get(dims, type);
  mlir::DenseElementsAttr attr;

  if (dims.size() == 0) {
    // this is the scalar case
    T value;
    if (!exla::nif::get(env, term, &value)) {
      return exla::nif::error(env, "Unable to cast scalar to type.");
    }
    attr = mlir::DenseElementsAttr::get(ty, value);
  } else {
    // non-scalar case. we'll assume our data
    // is in the form of a raw buffer
    ErlNifBinary binary;
    if (!exla::nif::get_binary(env, term, &binary)) {
      return exla::nif::error(env, "Unable to get binary data.");
    }
    char *data = const_cast<char *>(reinterpret_cast<char *>(binary.data));
    llvm::ArrayRef<char> values(data, binary.size);

    attr = mlir::DenseElementsAttr::getFromRawBuffer(ty, values);
  }

  // We set a fixed scalar shape because we're using single values here.
  mlir::Value op = builder->create<mlir::stablehlo::ConstantOp>(builder->getUnknownLoc(), attr);
  return exla::nif::ok(env, exla::nif::make<mlir::Value>(env, op));
}

ERL_NIF_TERM MLIRFunction::ConstantOp(mlir::Type type, ErlNifEnv *env, ERL_NIF_TERM term, std::vector<int64_t> dims) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());

  if (type.isUnsignedInteger(8)) {
    return ConstantOpImpl<exla::uint8>(module_->builder(), type, env, term, dims);
  }

  if (type.isUnsignedInteger(16)) {
    return ConstantOpImpl<exla::uint16>(module_->builder(), type, env, term, dims);
  }

  if (type.isUnsignedInteger(32)) {
    return ConstantOpImpl<exla::uint32>(module_->builder(), type, env, term, dims);
  }

  if (type.isUnsignedInteger(64)) {
    return ConstantOpImpl<exla::uint64>(module_->builder(), type, env, term, dims);
  }

  if (type.isSignlessInteger(8)) {
    return ConstantOpImpl<exla::int8>(module_->builder(), type, env, term, dims);
  }

  if (type.isSignlessInteger(16)) {
    return ConstantOpImpl<exla::int16>(module_->builder(), type, env, term, dims);
  }

  if (type.isSignlessInteger(32)) {
    return ConstantOpImpl<exla::int32>(module_->builder(), type, env, term, dims);
  }

  if (type.isSignlessInteger(64)) {
    return ConstantOpImpl<exla::int64>(module_->builder(), type, env, term, dims);
  }

  if (type.isBF16()) {
    return ConstantOpImpl<exla::bfloat16>(module_->builder(), type, env, term, dims);
  }

  if (type.isF16()) {
    return ConstantOpImpl<exla::float16>(module_->builder(), type, env, term, dims);
  }

  if (type.isa<mlir::ComplexType>()) {
    mlir::ComplexType complex_type = llvm::cast<mlir::ComplexType>(type);
    if (complex_type.getElementType().isF32()) {
      return ConstantOpImpl<exla::complex64>(module_->builder(), complex_type, env, term, dims);
    } else {
      return ConstantOpImpl<exla::complex128>(module_->builder(), complex_type, env, term, dims);
    }
  }

  if (type.isF32()) {
    return ConstantOpImpl<exla::float32>(module_->builder(), type, env, term, dims);
  }

  if (type.isF64()) {
    return ConstantOpImpl<exla::float64>(module_->builder(), type, env, term, dims);
  }

  return exla::nif::error(env, "invalid type received");
}

void MLIRFunction::Build(mlir::Value root, bool use_stablehlo_return) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());

  if (use_stablehlo_return) {
    module_->builder()->create<mlir::stablehlo::ReturnOp>(module_->builder()->getUnknownLoc(), root);
  } else {
    module_->builder()->create<mlir::func::ReturnOp>(module_->builder()->getUnknownLoc(), root);
  }

  module_->LowerPatterns();
}

MLIRModule::MLIRModule() {
  context_ = std::make_unique<mlir::MLIRContext>();

  // TODO: Should we load all these up front, infer them, or make it
  // manual?
  context_->loadDialect<mlir::func::FuncDialect>();
  context_->loadDialect<mlir::stablehlo::StablehloDialect>();
  context_->loadDialect<mlir::mhlo::MhloDialect>();

  module_ = mlir::OwningOpRef<mlir::ModuleOp>(mlir::ModuleOp::create(mlir::UnknownLoc::get(context_.get())));
  builder_ = std::make_unique<mlir::OpBuilder>(context_.get());
  builder_->setInsertionPointToStart(module_->getBody());
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
  if (type.isF64()) {
    return xla::primitive_util::NativeToPrimitiveType<double>();
  }
  if (type.isa<mlir::ComplexType>()) {
    mlir::ComplexType complex_type = llvm::cast<mlir::ComplexType>(type);
    if (complex_type.getElementType().isF32()) {
      return xla::primitive_util::NativeToPrimitiveType<xla::complex64>();
    } else {
      return xla::primitive_util::NativeToPrimitiveType<xla::complex128>();
    }
  }
}

MLIRFunction *MLIRModule::CreateFunction(
    std::string name,
    std::vector<xla::Shape *> arg_shapes,
    xla::Shape *ret_shape, bool is_public) {
  std::vector<mlir::Type> types;
  types.reserve(arg_shapes.size());
  for (auto arg_shape : arg_shapes) {
    mlir::Type type = GetMLIRFunctionType(builder_.get(), arg_shape);
    types.push_back(type);
  }

  mlir::Type return_type = GetMLIRFunctionType(builder_.get(), ret_shape);

  auto visibility = is_public ? "public" : "nested";

  auto funcType = builder_->getFunctionType(types, return_type);
  auto loc = builder_->getUnknownLoc();
  auto funcOp = std::make_unique<mlir::func::FuncOp>(mlir::func::FuncOp::create(loc, name, funcType));
  funcOp->setSymVisibility(visibility);
  module_->push_back(*funcOp);
  funcOp->addEntryBlock();
  builder_->setInsertionPointToStart(&funcOp->getBody().front());
  return new MLIRFunction(this, std::move(funcOp));
}

mlir::Value MLIRFunction::ConvOp(
    mlir::Value tensor,
    mlir::Value kernel,
    std::vector<int64_t> window_strides,
    std::vector<int64_t> padding,
    std::vector<int64_t> tensor_dilation,
    std::vector<int64_t> kernel_dilation,
    xla::ConvolutionDimensionNumbers dimension_numbers,
    uint64_t feature_group_count,
    uint64_t batch_group_count,
    uint64_t precision_config,
    std::vector<int64_t> output_dims) {
  auto builder = module_->builder();
  builder->setInsertionPointToEnd(&func_->getBody().back());

  mlir::RankedTensorType t_in = llvm::cast<mlir::RankedTensorType>(tensor.getType());
  mlir::RankedTensorType result_type = mlir::RankedTensorType::get(output_dims, t_in.getElementType());

  auto window_strides_attr = Int64ToDenseIntElementsAttr(module_->builder(), window_strides);
  auto tensor_dilation_attr = Int64ToDenseIntElementsAttr(module_->builder(), tensor_dilation);
  auto kernel_dilation_attr = Int64ToDenseIntElementsAttr(module_->builder(), kernel_dilation);
  auto dimension_numbers_attr = mlir::stablehlo::ConvDimensionNumbersAttr::get(
      builder->getContext(),
      dimension_numbers.input_batch_dimension(),
      dimension_numbers.input_feature_dimension(),
      llvm::ArrayRef<int64_t>(dimension_numbers.input_spatial_dimensions().data(), dimension_numbers.input_spatial_dimensions_size()),
      dimension_numbers.kernel_input_feature_dimension(),
      dimension_numbers.kernel_output_feature_dimension(),
      llvm::ArrayRef<int64_t>(dimension_numbers.kernel_spatial_dimensions().data(), dimension_numbers.kernel_spatial_dimensions_size()),
      dimension_numbers.output_batch_dimension(),
      dimension_numbers.output_feature_dimension(),
      llvm::ArrayRef<int64_t>(dimension_numbers.output_spatial_dimensions().data(), dimension_numbers.output_spatial_dimensions_size()));

  auto dense_attr_type = mlir::RankedTensorType::get({static_cast<int64_t>(padding.size() / 2), 2}, builder->getIntegerType(64));
  auto dense_attr = mlir::DenseElementsAttr::get<int64_t>(dense_attr_type, llvm::ArrayRef<int64_t>(padding.data(), padding.size()));
  auto padding_attr = llvm::cast<mlir::DenseIntElementsAttr>(dense_attr);

  return builder->create<mlir::stablehlo::ConvolutionOp>(
      builder->getUnknownLoc(),
      result_type,
      tensor,
      kernel,
      window_strides_attr,
      padding_attr,
      tensor_dilation_attr,
      kernel_dilation_attr,
      nullptr,
      dimension_numbers_attr,
      feature_group_count,
      batch_group_count,
      nullptr);
}

mlir::Value MLIRFunction::CreateTokenOp() {
  auto builder = module_->builder();
  builder->setInsertionPointToEnd(&func_->getBody().back());
  return builder->create<mlir::stablehlo::CreateTokenOp>(builder->getUnknownLoc());
}

mlir::Value MLIRFunction::TriangularSolveOp(mlir::Value a, mlir::Value b, bool left_side, bool lower, bool transpose_a) {
  auto builder = module_->builder();
  builder->setInsertionPointToEnd(&func_->getBody().back());
  mlir::stablehlo::Transpose transpose = mlir::stablehlo::Transpose::NO_TRANSPOSE;

  if (a.getType().isa<mlir::ComplexType>() and transpose_a) {
    transpose = mlir::stablehlo::Transpose::ADJOINT;
  } else if (transpose_a) {
    transpose = mlir::stablehlo::Transpose::TRANSPOSE;
  }

  return builder->create<mlir::stablehlo::TriangularSolveOp>(builder->getUnknownLoc(), a, b, left_side, lower, false, transpose);
}

mlir::Value MLIRFunction::DynamicUpdateSliceOp(mlir::Value operand, mlir::Value update, std::vector<mlir::Value> start_indices) {
  auto builder = module_->builder();
  builder->setInsertionPointToEnd(&func_->getBody().back());
  return builder->create<mlir::stablehlo::DynamicUpdateSliceOp>(builder->getUnknownLoc(), operand, update, mlir::ValueRange(start_indices));
}

void MLIRModule::LowerPatterns() {
  mlir::ConversionTarget target(*context());
  target.addIllegalDialect<mlir::stablehlo::StablehloDialect>();
  target.addIllegalDialect<mlir::func::FuncDialect>();
  target.addLegalDialect<mlir::mhlo::MhloDialect>();

  mlir::stablehlo::StablehloToHloTypeConverter converter;
  mlir::RewritePatternSet patterns(context());

  mlir::stablehlo::registerFuncOpsForTypeConversion(target, patterns, converter);
  mlir::stablehlo::populateStablehloToHloPatterns(&patterns, &converter, context());

  mlir::applyPartialConversion(module(), target, std::move(patterns));
}

}  // namespace exla