#ifndef TORCHX_NIF_UTIL_H_
#define TORCHX_NIF_UTIL_H_

#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <fine.hpp>
#include <torch/torch.h>
#include <map>
#include <string>

namespace torchx {

// Atom definitions
namespace atoms {
static auto already_deallocated = fine::Atom("already_deallocated");
} // namespace atoms

// Type mappings
inline std::map<const std::string, const torch::ScalarType> dtypes = {
    {"byte", torch::kByte},
    {"char", torch::kChar},
    {"short", torch::kShort},
    {"int", torch::kInt},
    {"long", torch::kLong},
    {"float8_e5m2", torch::kFloat8_e5m2},
    {"float8_e4m3fn", torch::kFloat8_e4m3fn},
    {"half", torch::kHalf},
    {"brain", torch::kBFloat16},
    {"float", torch::kFloat},
    {"double", torch::kDouble},
    {"bool", torch::kBool},
    {"complex", at::ScalarType::ComplexFloat},
    {"complex_double", at::ScalarType::ComplexDouble}};

inline std::map<const std::string, const int> dtype_sizes = {
    {"byte", 1},
    {"char", 1},
    {"short", 2},
    {"int", 4},
    {"long", 8},
    {"float8_e5m2", 1},
    {"float8_e4m3fn", 1},
    {"half", 2},
    {"brain", 2},
    {"float", 4},
    {"double", 8},
    {"complex", 8},
    {"complex_double", 16}};

inline torch::ScalarType string2type(const std::string &atom) {
  return dtypes[atom];
}

inline const std::string *type2string(const torch::ScalarType type) {
  for (std::map<const std::string, const torch::ScalarType>::iterator i =
           dtypes.begin();
       i != dtypes.end(); ++i) {
    if (i->second == type)
      return &i->first;
  }
  return nullptr;
}

// Tensor resource wrapper with deallocation tracking
class TorchTensor {
public:
  TorchTensor(torch::Tensor tensor) : tensor_(tensor), deallocated_(false) {}

  torch::Tensor &tensor() {
    if (deallocated_) {
      throw std::runtime_error("Tensor has been deallocated");
    }
    return tensor_;
  }

  const torch::Tensor &tensor() const {
    if (deallocated_) {
      throw std::runtime_error("Tensor has been deallocated");
    }
    return tensor_;
  }

  bool deallocate() {
    if (!deallocated_) {
      deallocated_ = true;
      // Assignment to empty tensor properly handles destruction and frees
      // memory The destructor will be called automatically by the assignment
      // operator
      tensor_ = torch::Tensor();
      return true;
    }
    return false;
  }

  bool is_deallocated() const { return deallocated_; }

  // True when no other tensor or view shares the storage of this tensor, so
  // its buffer can be given to a result.
  bool is_unique() const {
    return !deallocated_ && tensor_.defined() && tensor_.has_storage() &&
           tensor_.use_count() == 1 && tensor_.storage().use_count() == 1;
  }

  // Like is_unique, and the elements can also be overwritten in place.
  bool is_writable() const {
    return is_unique() &&
           at::has_internal_overlap(tensor_) == at::MemOverlap::No;
  }

  // Moves the tensor out of this resource, which is deallocated afterwards.
  torch::Tensor take() {
    if (deallocated_) {
      throw std::runtime_error("Tensor has been deallocated");
    }
    deallocated_ = true;
    torch::Tensor result = std::move(tensor_);
    tensor_ = torch::Tensor();
    return result;
  }

  // Replace the tensor owned by this BEAM resource (e.g. after materializing
  // a contiguous copy before exporting a data pointer).
  void replace(torch::Tensor tensor) {
    if (deallocated_) {
      throw std::runtime_error("Tensor has been deallocated");
    }
    tensor_ = std::move(tensor);
  }

private:
  torch::Tensor tensor_;
  bool deallocated_;
};

} // namespace torchx

// Fine specializations for torch types
namespace fine {

// Decoder for std::vector<int64_t> from tuple (for shape parameters)
// Elixir passes shapes as tuples like {2, 3}, but we need vector<int64_t>
template <> struct Decoder<std::vector<int64_t>> {
  static DecodeResult<std::vector<int64_t>>
  try_decode(ErlNifEnv *env, const ERL_NIF_TERM &term) {
    // First try to decode as tuple (for shapes)
    int size;
    const ERL_NIF_TERM *terms;
    if (enif_get_tuple(env, term, &size, &terms)) {
      std::vector<int64_t> vec;
      vec.reserve(size);
      for (int i = 0; i < size; i++) {
        auto elem = fine::try_decode<int64_t>(env, terms[i]);
        if (!elem) {
          return std::move(elem).error();
        }
        vec.push_back(elem.value());
      }
      return vec;
    }

    // Otherwise try to decode as list
    unsigned int length;
    if (!enif_get_list_length(env, term, &length)) {
      return DecodeError("decode failed, expected a tuple or list", term);
    }

    std::vector<int64_t> vector;
    vector.reserve(length);

    auto list = term;
    ERL_NIF_TERM head, tail;
    while (enif_get_list_cell(env, list, &head, &tail)) {
      auto elem = fine::try_decode<int64_t>(env, head);
      if (!elem) {
        return std::move(elem).error();
      }
      vector.push_back(elem.value());
      list = tail;
    }

    return vector;
  }
};

// Decoder for torch::Scalar
template <> struct Decoder<torch::Scalar> {
  static DecodeResult<torch::Scalar> try_decode(ErlNifEnv *env,
                                                const ERL_NIF_TERM &term) {
    if (auto value = fine::try_decode<double>(env, term)) {
      return torch::Scalar(value.value());
    }

    if (auto value = fine::try_decode<int64_t>(env, term)) {
      return torch::Scalar(value.value());
    }

    // Complex number as a tuple of two doubles
    if (auto value = fine::try_decode<std::tuple<double, double>>(env, term)) {
      auto [real, imag] = value.value();
      return torch::Scalar(c10::complex<double>(real, imag));
    }

    return DecodeError("decode failed, expected a number or a tuple of two floats",
                       term);
  }
};

// Encoder for torch::Scalar
template <> struct Encoder<torch::Scalar> {
  static ERL_NIF_TERM encode(ErlNifEnv *env, const torch::Scalar &scalar) {
    if (scalar.isIntegral(false)) {
      return fine::encode(env, scalar.toLong());
    } else if (scalar.isFloatingPoint()) {
      return fine::encode(env, scalar.toDouble());
    } else if (scalar.isComplex()) {
      auto complex = scalar.toComplexDouble();
      return fine::encode(env, std::make_tuple(complex.real(), complex.imag()));
    } else {
      throw std::runtime_error("Unknown scalar type");
    }
  }
};

// Decoder for torch::ScalarType (from atom string)
template <> struct Decoder<torch::ScalarType> {
  static DecodeResult<torch::ScalarType> try_decode(ErlNifEnv *env,
                                                    const ERL_NIF_TERM &term) {
    auto atom = fine::try_decode<fine::Atom>(env, term);
    if (!atom) {
      return std::move(atom).error();
    }
    return torchx::string2type(atom.value().to_string());
  }
};

// Decoder for torch::Device
template <> struct Decoder<torch::Device> {
  static DecodeResult<torch::Device> try_decode(ErlNifEnv *env,
                                                const ERL_NIF_TERM &term) {
    auto device_string = fine::try_decode<std::string>(env, term);
    if (!device_string) {
      return std::move(device_string).error();
    }
    return torch::Device(device_string.value());
  }
};

// Decoder for c10::IntArrayRef (from list of int64)
template <> struct Decoder<c10::IntArrayRef> {
  static DecodeResult<c10::IntArrayRef> try_decode(ErlNifEnv *env,
                                                   const ERL_NIF_TERM &term) {
    // We need to store the vector somewhere persistent for IntArrayRef to reference
    // This is tricky - IntArrayRef is a view, so we'll decode to vector instead
    throw std::runtime_error(
        "Cannot decode directly to IntArrayRef, use std::vector<int64_t>");
  }
};

// Decoder for std::vector<torch::Tensor>
template <>
struct Decoder<std::vector<torch::Tensor>> {
  static DecodeResult<std::vector<torch::Tensor>>
  try_decode(ErlNifEnv *env, const ERL_NIF_TERM &term) {
    auto tensor_resources = fine::try_decode<
        std::vector<fine::ResourcePtr<torchx::TorchTensor>>>(env, term);
    if (!tensor_resources) {
      return std::move(tensor_resources).error();
    }
    std::vector<torch::Tensor> tensors;
    tensors.reserve(tensor_resources.value().size());
    for (const auto &res : tensor_resources.value()) {
      tensors.push_back(res->tensor());
    }
    return tensors;
  }
};

// Encoder for std::vector<torch::Tensor>
template <>
struct Encoder<std::vector<torch::Tensor>> {
  static ERL_NIF_TERM encode(ErlNifEnv *env,
                             const std::vector<torch::Tensor> &tensors) {
    std::vector<fine::ResourcePtr<torchx::TorchTensor>> tensor_resources;
    tensor_resources.reserve(tensors.size());
    for (const auto &tensor : tensors) {
      tensor_resources.push_back(fine::make_resource<torchx::TorchTensor>(tensor));
    }
    return fine::encode(env, tensor_resources);
  }
};

} // namespace fine

#endif // TORCHX_NIF_UTIL_H_

