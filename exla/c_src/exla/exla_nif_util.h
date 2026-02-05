#ifndef EXLA_NIF_UTIL_H_
#define EXLA_NIF_UTIL_H_

#include <fine.hpp>
#include <tuple>

#include "mlir/IR/Types.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace exla {

namespace atoms {
static auto ElixirEXLATypespec = fine::Atom("Elixir.EXLA.Typespec");
static auto __struct__ = fine::Atom("__struct__");
static auto already_deallocated = fine::Atom("already_deallocated");
static auto bf = fine::Atom("bf");
static auto c = fine::Atom("c");
static auto error = fine::Atom("error");
static auto f = fine::Atom("f");
static auto f8_e4m3fn = fine::Atom("f8_e4m3fn");
static auto info = fine::Atom("info");
static auto pred = fine::Atom("pred");
static auto s = fine::Atom("s");
static auto shape = fine::Atom("shape");
static auto token = fine::Atom("token");
static auto type = fine::Atom("type");
static auto u = fine::Atom("u");
static auto warning = fine::Atom("warning");
} // namespace atoms
} // namespace exla

namespace fine {

// Define decoding for xla::Shape from for %EXLA.Typespec{} term
template <> struct Decoder<xla::Shape> {
  static xla::Shape decode(ErlNifEnv *env, const ERL_NIF_TERM &term) {
    ERL_NIF_TERM type_term;
    ERL_NIF_TERM shape_term;

    if (!enif_get_map_value(env, term, fine::encode(env, exla::atoms::type),
                            &type_term)) {
      throw std::invalid_argument(
          "decode failed, expected EXLA.Typespec struct");
    }

    if (!enif_get_map_value(env, term, fine::encode(env, exla::atoms::shape),
                            &shape_term)) {
      throw std::invalid_argument(
          "decode failed, expected EXLA.Typespec struct");
    }

    return xla::ShapeUtil::MakeShape(decode_type(env, type_term),
                                     decode_shape(env, shape_term));
  }

private:
  static std::vector<int64_t> decode_shape(ErlNifEnv *env,
                                           const ERL_NIF_TERM &term) {
    int size;
    const ERL_NIF_TERM *terms;

    if (!enif_get_tuple(env, term, &size, &terms)) {
      throw std::invalid_argument(
          "decode failed, expected shape to be a tuple");
    }

    auto vector = std::vector<int64_t>();
    vector.reserve(size);

    for (auto i = 0; i < size; i++) {
      auto elem = fine::decode<int64_t>(env, terms[i]);
      vector.push_back(elem);
    }

    return vector;
  }

  static xla::PrimitiveType decode_type(ErlNifEnv *env,
                                        const ERL_NIF_TERM &term) {
    auto [element, size] =
        fine::decode<std::tuple<fine::Atom, uint64_t>>(env, term);

    if (element == "u") {
      switch (size) {
      case 2:
        return xla::U2;
      case 4:
        return xla::U4;
      case 8:
        return xla::U8;
      case 16:
        return xla::U16;
      case 32:
        return xla::U32;
      case 64:
        return xla::U64;
      }
    }
    if (element == "s") {
      switch (size) {
      case 2:
        return xla::S2;
      case 4:
        return xla::S4;
      case 8:
        return xla::S8;
      case 16:
        return xla::S16;
      case 32:
        return xla::S32;
      case 64:
        return xla::S64;
      }
    }
    if (element == "f") {
      switch (size) {
      case 8:
        return xla::F8E5M2;
      case 16:
        return xla::F16;
      case 32:
        return xla::F32;
      case 64:
        return xla::F64;
      }
    }
    if (element == "f8_e4m3fn") {
      return xla::F8E4M3FN;
    }
    if (element == "bf") {
      switch (size) {
      case 16:
        return xla::BF16;
      }
    }
    if (element == "c") {
      switch (size) {
      case 64:
        return xla::C64;
      case 128:
        return xla::C128;
      }
    }
    if (element == "pred") {
      return xla::PRED;
    }

    throw std::invalid_argument("decode failed, unexpected type");
  }
};

// Define encoding for mlir::Type into %EXLA.Typespec{} term
template <> struct Encoder<mlir::Type> {
  static ERL_NIF_TERM encode(ErlNifEnv *env, const mlir::Type &type) {
    ERL_NIF_TERM keys[] = {
        fine::encode(env, exla::atoms::__struct__),
        fine::encode(env, exla::atoms::type),
        fine::encode(env, exla::atoms::shape),
    };

    ERL_NIF_TERM values[] = {
        fine::encode(env, exla::atoms::ElixirEXLATypespec),
        encode_type(env, type),
        encode_shape(env, type),
    };

    ERL_NIF_TERM map;
    if (!enif_make_map_from_arrays(env, keys, values, 3, &map)) {
      throw std::runtime_error("encode: failed to make a map");
    }

    return map;
  }

private:
  static ERL_NIF_TERM encode_type(ErlNifEnv *env, const mlir::Type &type) {
    if (mlir::isa<mlir::stablehlo::TokenType>(type)) {
      return fine::encode(env, exla::atoms::token);
    }

    std::optional<fine::Atom> type_name;
    std::optional<uint64_t> type_size;

    if (mlir::isa<mlir::RankedTensorType>(type)) {
      auto tensor_type = mlir::cast<mlir::RankedTensorType>(type);
      auto element_type = tensor_type.getElementType();

      if (element_type.isSignlessInteger(1)) {
        type_name = exla::atoms::pred;
        type_size = 8;
      } else if (auto integer_type =
                     mlir::dyn_cast<mlir::IntegerType>(element_type)) {
        if (integer_type.isUnsigned()) {
          type_name = exla::atoms::u;
        } else {
          type_name = exla::atoms::s;
        }

        type_size = integer_type.getWidth();
      } else if (element_type.isBF16()) {
        type_name = exla::atoms::bf;
        type_size = 16;
      // Float8E4M3FNType doesn't have a convenience method like isBF16() on the
      // base mlir::Type class, so we use mlir::isa<> for type checking instead.
      } else if (mlir::isa<mlir::Float8E4M3FNType>(element_type)) {
        type_name = exla::atoms::f8_e4m3fn;
        type_size = 8;
      } else if (auto float_type =
                     mlir::dyn_cast<mlir::FloatType>(element_type)) {
        type_name = exla::atoms::f;
        type_size = float_type.getWidth();
      } else if (auto complex_type =
                     mlir::dyn_cast<mlir::ComplexType>(element_type)) {
        auto element_type = complex_type.getElementType();
        type_name = exla::atoms::c;
        type_size = mlir::cast<mlir::FloatType>(element_type).getWidth() * 2;
      }
    }

    if (type_name) {
      return fine::encode(
          env, std::make_tuple(type_name.value(), type_size.value()));
    } else {
      throw std::invalid_argument("encode failed, unexpected mlir type");
    }
  }

  static ERL_NIF_TERM encode_shape(ErlNifEnv *env, const mlir::Type &type) {
    if (mlir::isa<mlir::stablehlo::TokenType>(type)) {
      return enif_make_tuple(env, 0);
    }

    if (mlir::isa<mlir::RankedTensorType>(type)) {
      auto tensor_type = mlir::cast<mlir::RankedTensorType>(type);
      auto dims_array = tensor_type.getShape();

      auto dims = std::vector<ERL_NIF_TERM>{};
      dims.reserve(dims_array.size());

      for (auto dim : dims_array) {
        dims.push_back(fine::encode<int64_t>(env, dim));
      }

      return enif_make_tuple_from_array(env, dims.data(), dims.size());
    }

    throw std::invalid_argument("encode failed, unexpected mlir type");
  }
};

// Define encoding for xla::Shape into %EXLA.Typespec{} term
template <> struct Encoder<xla::Shape> {
  static ERL_NIF_TERM encode(ErlNifEnv *env, const xla::Shape &shape) {
    ERL_NIF_TERM keys[] = {
        fine::encode(env, exla::atoms::__struct__),
        fine::encode(env, exla::atoms::type),
        fine::encode(env, exla::atoms::shape),
    };

    ERL_NIF_TERM values[] = {
        fine::encode(env, exla::atoms::ElixirEXLATypespec),
        encode_type(env, shape),
        encode_shape(env, shape),
    };

    ERL_NIF_TERM map;
    if (!enif_make_map_from_arrays(env, keys, values, 3, &map)) {
      throw std::runtime_error("encode: failed to make a map");
    }

    return map;
  }

private:
  static ERL_NIF_TERM encode_type(ErlNifEnv *env, const xla::Shape &shape) {
    auto ptype = shape.element_type();

    // Tokens are encoded as the atom :token
    if (ptype == xla::TOKEN) {
      return fine::encode(env, exla::atoms::token);
    }

    std::optional<fine::Atom> type_name;
    std::optional<uint64_t> type_size;

    switch (ptype) {
    case xla::PRED:
      type_name = exla::atoms::pred;
      type_size = 8;
      break;

    case xla::U8:
      type_name = exla::atoms::u;
      type_size = 8;
      break;
    case xla::U16:
      type_name = exla::atoms::u;
      type_size = 16;
      break;
    case xla::U32:
      type_name = exla::atoms::u;
      type_size = 32;
      break;
    case xla::U64:
      type_name = exla::atoms::u;
      type_size = 64;
      break;

    case xla::S8:
      type_name = exla::atoms::s;
      type_size = 8;
      break;
    case xla::S16:
      type_name = exla::atoms::s;
      type_size = 16;
      break;
    case xla::S32:
      type_name = exla::atoms::s;
      type_size = 32;
      break;
    case xla::S64:
      type_name = exla::atoms::s;
      type_size = 64;
      break;

    case xla::F8E5M2:
      type_name = exla::atoms::f;
      type_size = 8;
      break;
    case xla::F8E4M3FN:
      type_name = exla::atoms::f8_e4m3fn;
      type_size = 8;
      break;

    case xla::F16:
      type_name = exla::atoms::f;
      type_size = 16;
      break;
    case xla::F32:
      type_name = exla::atoms::f;
      type_size = 32;
      break;
    case xla::F64:
      type_name = exla::atoms::f;
      type_size = 64;
      break;

    case xla::BF16:
      type_name = exla::atoms::bf;
      type_size = 16;
      break;

    case xla::C64:
      type_name = exla::atoms::c;
      type_size = 64;
      break;
    case xla::C128:
      type_name = exla::atoms::c;
      type_size = 128;
      break;

    default:
      break;
    }

    if (type_name && type_size) {
      return fine::encode(
          env, std::make_tuple(type_name.value(), type_size.value()));
    }

    throw std::invalid_argument("encode failed, unexpected xla::PrimitiveType");
  }

  static ERL_NIF_TERM encode_shape(ErlNifEnv *env, const xla::Shape &shape) {
    auto dims = shape.dimensions();
    std::vector<ERL_NIF_TERM> dim_terms;
    dim_terms.reserve(dims.size());

    for (int64_t dim : dims) {
      dim_terms.push_back(fine::encode(env, dim));
    }

    return enif_make_tuple_from_array(env, dim_terms.data(), dim_terms.size());
  }
};
} // namespace fine

// Helper Macros
//
// See:
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/stream_executor/lib/statusor.h

#define EXLA_STATUS_MACROS_CONCAT_NAME(x, y)                                   \
  EXLA_STATUS_MACROS_CONCAT_NAME_IMPL(x, y)

#define EXLA_STATUS_MACROS_CONCAT_NAME_IMPL(x, y) x##y

// Macro to be used to consume StatusOr. Will bind lhs
// to value if the status is OK, otherwise will return
// the status.
#define EXLA_ASSIGN_OR_RETURN(lhs, rexpr)                                      \
  EXLA_ASSIGN_OR_RETURN_IMPL(                                                  \
      EXLA_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs,      \
      rexpr)

#define EXLA_ASSIGN_OR_RETURN_IMPL(statusor, lhs, rexpr)                       \
  auto statusor = (rexpr);                                                     \
  if (!statusor.ok()) {                                                        \
    return statusor.status();                                                  \
  }                                                                            \
  lhs = std::move(statusor.value());

#endif
