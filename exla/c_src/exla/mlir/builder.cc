#include "builder.h"

#include <memory>

#include "../exla_nif_util.h"
#include "_virtual_includes/chlo_ops/stablehlo/dialect/ChloOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/primitive_util.h"
#include "xla/types.h"

namespace exla {
mlir::Type TypeIntToMLIRType(mlir::OpBuilder *builder, int type_int) {
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

mlir::TensorType GetMLIRType(mlir::OpBuilder *builder, std::vector<tsl::int64> dims, int type_int) {
  auto type = TypeIntToMLIRType(builder, type_int);
  return mlir::RankedTensorType::get(dims, type);
}

mlir::mhlo::DotDimensionNumbersAttr ConvertDotDimensionNumbersToAttr(mlir::OpBuilder *builder, const xla::DotDimensionNumbers &dotDimNumbers) {
  std::vector<int64_t> lhsContractingVec(dotDimNumbers.lhs_contracting_dimensions().begin(),
                                         dotDimNumbers.lhs_contracting_dimensions().end());
  std::vector<int64_t> rhsContractingVec(dotDimNumbers.rhs_contracting_dimensions().begin(),
                                         dotDimNumbers.rhs_contracting_dimensions().end());
  std::vector<int64_t> lhsBatchVec(dotDimNumbers.lhs_batch_dimensions().begin(),
                                   dotDimNumbers.lhs_batch_dimensions().end());
  std::vector<int64_t> rhsBatchVec(dotDimNumbers.rhs_batch_dimensions().begin(),
                                   dotDimNumbers.rhs_batch_dimensions().end());

  return mlir::mhlo::DotDimensionNumbersAttr::get(
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
  auto op = module_->builder()->create<mlir::mhlo::SubtractOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
  return op;
}

mlir::Value MLIRFunction::ConvertOp(mlir::Value operand, mlir::Type type) {
  mlir::OpBuilder *builder = module_->builder();
  builder->setInsertionPointToEnd(&func_->getBody().back());

  auto op = builder->create<mlir::mhlo::ConvertOp>(builder->getUnknownLoc(), operand, type);
  return op;
}

mlir::Value MLIRFunction::BitcastConvertOp(mlir::Value operand, xla::Shape shape) {
  mlir::OpBuilder *builder = module_->builder();
  builder->setInsertionPointToEnd(&func_->getBody().back());

  absl::Span<const int64_t> dimensions_span = shape.dimensions();
  std::vector<int64_t> dimensions(dimensions_span.begin(), dimensions_span.end());

  mlir::TensorType type = GetMLIRType(module_->builder(), dimensions, shape.element_type());

  auto op = builder->create<mlir::mhlo::BitcastConvertOp>(builder->getUnknownLoc(), type, operand);
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

mlir::Value MLIRFunction::PadOp(mlir::Value op, mlir::Value pad, std::vector<int64_t> padding_low, std::vector<int64_t> padding_high, std::vector<int64_t> padding_mid) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());

  auto padding_low_attr = Int64ToDenseIntElementsAttr(module_->builder(), padding_low);
  auto padding_high_attr = Int64ToDenseIntElementsAttr(module_->builder(), padding_high);
  auto padding_mid_attr = Int64ToDenseIntElementsAttr(module_->builder(), padding_mid);

  return module_->builder()->create<mlir::mhlo::PadOp>(module_->builder()->getUnknownLoc(), op, pad, padding_low_attr, padding_high_attr, padding_mid_attr);
}

mlir::Value compare_and_return_bool(mlir::OpBuilder *builder, mlir::Value lhs, mlir::Value rhs, mlir::mhlo::ComparisonDirection direction) {
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

mlir::Value MLIRFunction::ShiftLeftOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::ShiftLeftOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
}

mlir::Value MLIRFunction::ShiftRightLogicalOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::ShiftRightLogicalOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
}

mlir::Value MLIRFunction::ShiftRightArithmeticOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::ShiftRightArithmeticOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
}

mlir::Value MLIRFunction::BitwiseAndOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::AndOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
}

mlir::Value MLIRFunction::BitwiseOrOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::OrOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
}

mlir::Value MLIRFunction::BitwiseNotOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::NotOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::BitwiseXorOp(mlir::Value lhs, mlir::Value rhs) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::XorOp>(module_->builder()->getUnknownLoc(), lhs, rhs);
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

mlir::Value MLIRFunction::LogisticOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::LogisticOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::Log1pOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::Log1pOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::SignOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::SignOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::CosOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::CosineOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::SinOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::SineOp>(module_->builder()->getUnknownLoc(), operand);
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
  return module_->builder()->create<mlir::mhlo::TanhOp>(module_->builder()->getUnknownLoc(), operand);
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
  return module_->builder()->create<mlir::mhlo::SqrtOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::CbrtOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::CbrtOp>(module_->builder()->getUnknownLoc(), operand);
}
mlir::Value MLIRFunction::NegateOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::NegOp>(module_->builder()->getUnknownLoc(), operand);
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
  return module_->builder()->create<mlir::mhlo::ConvertOp>(module_->builder()->getUnknownLoc(), op, mlir_bool);
}

mlir::Value MLIRFunction::IsNanOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  module_->context()->getOrLoadDialect<mlir::chlo::ChloDialect>();
  mlir::Type mlir_bool = module_->builder()->getI1Type();

  mlir::Value is_finite_op = module_->builder()->create<mlir::mhlo::IsFiniteOp>(module_->builder()->getUnknownLoc(), operand);
  is_finite_op = module_->builder()->create<mlir::mhlo::ConvertOp>(module_->builder()->getUnknownLoc(), is_finite_op, mlir_bool);

  mlir::Value is_inf_op = this->IsInfOp(operand);
  is_inf_op = module_->builder()->create<mlir::mhlo::ConvertOp>(module_->builder()->getUnknownLoc(), is_inf_op, mlir_bool);

  mlir::Value is_nan_op = this->BitwiseAndOp(this->BitwiseNotOp(is_inf_op), this->BitwiseNotOp(is_finite_op));
  mlir_bool = module_->builder()->getIntegerType(8, false);

  return module_->builder()->create<mlir::mhlo::ConvertOp>(module_->builder()->getUnknownLoc(), is_nan_op, mlir_bool);
}
mlir::Value MLIRFunction::RsqrtOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::RsqrtOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::TransposeOp(mlir::Value operand, std::vector<int64_t> axes) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto axes_attr = Int64ToDenseIntElementsAttr(module_->builder(), axes);
  return module_->builder()->create<mlir::mhlo::TransposeOp>(module_->builder()->getUnknownLoc(), operand, axes_attr);
}

mlir::Value MLIRFunction::ReshapeOp(mlir::Value operand, std::vector<int64_t> target_shape) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  mlir::RankedTensorType t_in = llvm::cast<mlir::RankedTensorType>(operand.getType());
  mlir::RankedTensorType type = mlir::RankedTensorType::get(target_shape, t_in.getElementType());
  return module_->builder()->create<mlir::mhlo::ReshapeOp>(module_->builder()->getUnknownLoc(), type, operand);
}

mlir::Value MLIRFunction::ReverseOp(mlir::Value operand, std::vector<int64_t> dims) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto dims_attr = Int64ToDenseIntElementsAttr(module_->builder(), dims);
  return module_->builder()->create<mlir::mhlo::ReverseOp>(module_->builder()->getUnknownLoc(), operand, dims_attr);
}

class PublicPatternRewriter : public mlir::PatternRewriter {
 public:
  PublicPatternRewriter(mlir::MLIRContext *context) : mlir::PatternRewriter(context) {}
};

static void buildSortComparisonBody(llvm::ArrayRef<mlir::Type> elementTypes,
                                    mlir::mhlo::ComparisonDirection direction,
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
  mlir::mhlo::ComparisonType type_attr;
  if (compare_type) {
    type_attr = mlir::mhlo::symbolizeComparisonType(*compare_type).value();
  } else {
    type_attr = mlir::mhlo::ComparisonType::NOTYPE;
  }
  mlir::BlockArgument arg0 = block->getArgument(0);
  mlir::BlockArgument arg1 = block->getArgument(1);
  mlir::Value compare = builder->create<mlir::mhlo::CompareOp>(loc, arg0, arg1, direction);
  builder->create<mlir::mhlo::ReturnOp>(loc, compare);
}

std::vector<mlir::Value> MLIRFunction::SortOp(std::vector<mlir::Value> operands, int64_t dim, bool desc) {
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

  mlir::mhlo::ComparisonDirection direction = desc ? mlir::mhlo::ComparisonDirection::GT : mlir::mhlo::ComparisonDirection::LT;

  mlir::OpBuilder *builder = module_->builder();

  mlir::ValueRange value_range(operands);
  mlir::mhlo::SortOp sort_op = builder->create<mlir::mhlo::SortOp>(
      builder->getUnknownLoc(),
      value_range,
      dim,
      true);

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

  return module_->builder()->create<mlir::mhlo::SliceOp>(
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
      ->create<mlir::mhlo::DynamicSliceOp>(
          module_->builder()->getUnknownLoc(),
          operand,
          starts_range,
          len_attr);
}

mlir::Value MLIRFunction::ClzOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::ClzOp>(module_->builder()->getUnknownLoc(), operand);
}

mlir::Value MLIRFunction::PopulationCountOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  return module_->builder()->create<mlir::mhlo::PopulationCountOp>(module_->builder()->getUnknownLoc(), operand);
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

mlir::Value MLIRFunction::IotaOp(xla::Shape shape, int64_t dimension) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());

  absl::Span<const int64_t> dimensions_span = shape.dimensions();
  std::vector<int64_t> dimensions(dimensions_span.begin(), dimensions_span.end());

  mlir::TensorType type = GetMLIRType(module_->builder(), dimensions, shape.element_type());

  return module_->builder()->create<mlir::mhlo::IotaOp>(module_->builder()->getUnknownLoc(), type, dimension);
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

  auto op = module_->builder()->create<mlir::mhlo::DotGeneralOp>(
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

  auto op = module_->builder()->create<mlir::mhlo::BroadcastInDimOp>(module_->builder()->getUnknownLoc(), result_type, operand, axes_attr);
  return op;
}

mlir::Value MLIRFunction::ConcatenateOp(std::vector<mlir::Value> operands, int64_t dimension) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  mlir::ValueRange operands_range(llvm::ArrayRef<mlir::Value>(operands.data(), operands.size()));
  auto op = module_->builder()->create<mlir::mhlo::ConcatenateOp>(module_->builder()->getUnknownLoc(), operands_range, dimension);
  return op;
}

mlir::Value MLIRFunction::OptimizationBarrierOp(mlir::Value operand) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::mhlo::OptimizationBarrierOp>(module_->builder()->getUnknownLoc(), operand);
  return op.getResult()[0];
}

mlir::Value MLIRFunction::ClampOp(mlir::Value min, mlir::Value operand, mlir::Value max) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::mhlo::ClampOp>(module_->builder()->getUnknownLoc(), min, operand, max);
  return op;
}

mlir::Value MLIRFunction::SelectOp(mlir::Value pred, mlir::Value on_true, mlir::Value on_false) {
  module_->builder()->setInsertionPointToEnd(&func_->getBody().back());
  auto op = module_->builder()->create<mlir::mhlo::SelectOp>(module_->builder()->getUnknownLoc(), pred, on_true, on_false);
  return op;
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
  mlir::Value op = builder->create<mlir::mhlo::ConstantOp>(builder->getUnknownLoc(), attr);
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

  if (type.isF32()) {
    return ConstantOpImpl<exla::float32>(module_->builder(), type, env, term, dims);
  }

  if (type.isF64()) {
    return ConstantOpImpl<exla::float64>(module_->builder(), type, env, term, dims);
  }
  return exla::nif::error(env, "invalid type received");
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

MLIRFunction *MLIRModule::CreateFunction(
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